import Foundation
import Darwin

private final class CalibrationChild: @unchecked Sendable {
    private let lock=NSLock()
    private var process:Process?
    private var cancelled=false
    func assign(_ value:Process) throws {
        try lock.withLock {guard !cancelled else{throw CancellationError()};try value.run();process=value}
    }
    func clear() {lock.withLock{process=nil}}
    func cancel() {lock.withLock{cancelled=true;if process?.isRunning == true {process?.interrupt()}}}
}

/// Explicit, optional storage tuning. Trials use one child worker at a time;
/// no GPU graph, second model, Python runtime or user transcript is loaded here.
public enum ModelCalibrator {
    public static func run(model:InstalledModel,engines:URL,progress:@escaping @Sendable (Int,Int,String)->Void) async throws -> ModelCalibrationRecord {
        guard ArchitectureRegistry.report(model.descriptor).installable,model.path.hasSuffix(".particle") else {throw ModelLibraryError.incompatible("Optimize for this Mac requires a qualified native pack.")}
        let baseline=model.profile
        guard [1,2,4].contains(baseline.readWorkers),[32,64,128].contains(baseline.prefillChunk),
              (model.descriptor.architecture == .sarvam ? 6...16 : 8...32).contains(baseline.slots),
              model.descriptor.expertRange.contains(baseline.activeExperts),baseline.context >= 4096 else {throw ModelLibraryError.invalid("The saved execution profile is invalid.")}
        let worker=engines.appendingPathComponent(ArchitectureRegistry.adapter(for:model.descriptor.architecture).worker)
        guard FileManager.default.isExecutableFile(atPath:worker.path) else {throw ModelLibraryError.invalid("The bundled model worker is missing.")}
        let identity=try ModelCalibrationIdentity.current(model:URL(fileURLWithPath:model.path),engine:worker)
        let candidates=ModelCalibrationSelection.candidates(baseline:baseline,model:model.descriptor,physicalMemory:ProcessInfo.processInfo.physicalMemory)
        let temporary=FileManager.default.temporaryDirectory.appendingPathComponent("particle-calibration-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700]);defer{try? FileManager.default.removeItem(at:temporary)}
        let configuration=temporary.appendingPathComponent("workloads.json")
        try JSONSerialization.data(withJSONObject:["histories":[512,2048],"suffixes":[16,128],"rounds":1,"decodeTokens":64,"candidateOnly":true]).write(to:configuration,options:.atomic)
        var profiles=[ModelRuntimeProfile]()
        for candidate in candidates {profiles += [baseline,candidate,candidate,baseline]}
        if profiles.isEmpty {profiles=[baseline,baseline]}
        let pressure=ModelMemoryPressure(),child=CalibrationChild()
        return try await withTaskCancellationHandler {
            var trials=[ModelCalibrationTrial]()
            for (index,profile) in profiles.enumerated() {
                try Task.checkCancellation();try pressure.check()
                guard ModelCalibrationIdentity.currentPowerSource() == identity.powerSource,ProcessInfo.processInfo.isLowPowerModeEnabled == identity.lowPowerMode else {throw ModelLibraryError.invalid("Power mode changed during calibration. Your current profile was retained; rerun with stable power conditions.")}
                progress(index,profiles.count,"Trial \(index+1) of \(profiles.count): \(profile.slots) slots, \(profile.readWorkers) readers, \(profile.prefillChunk)-token chunks. Each workload checks 64 decode steps.")
                let output=temporary.appendingPathComponent("trial-\(index).json"),errors=temporary.appendingPathComponent("trial-\(index).log")
                FileManager.default.createFile(atPath:output.path,contents:nil);FileManager.default.createFile(atPath:errors.path,contents:nil)
                let stdout=try FileHandle(forWritingTo:output),stderr=try FileHandle(forWritingTo:errors)
                defer{try? stdout.close();try? stderr.close()}
                let process=Process();process.executableURL=worker;process.standardInput=FileHandle.nullDevice;process.standardOutput=stdout;process.standardError=stderr
                var arguments=["--model",model.path,"--prompt","Mixture of experts models select specialized networks for each token. Compare two approaches, explain their tradeoffs, and give a practical example.",
                    "--max-context",String(min(8192,profile.context)),"--max-new","64","--expert-cache-slots",String(profile.slots),"--active-experts",String(profile.activeExperts),
                    "--prefill-chunk",String(profile.prefillChunk),"--benchmark-session",configuration.path]
                if model.descriptor.architecture == .sarvam {
                    arguments += ["--raw","--no-ssd-cache","--source-read-workers",String(profile.readWorkers),"--attention","reference","--projection","vector","--deferred-decode","--grouped-experts","--ffn-vector","--cooperative-quant"]
                } else {arguments += ["--read-workers",String(profile.readWorkers),"--temperature","0"]}
                process.arguments=arguments
                try child.assign(process)
                var nextPowerCheck=Date().addingTimeInterval(5)
                do {
                    while process.isRunning {
                        try await Task.sleep(for:.milliseconds(250));try Task.checkCancellation();try pressure.check()
                        let bytes=try output.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
                        guard bytes <= 32*1024*1024 else {throw ModelLibraryError.invalid("The calibration report exceeded its bound.")}
                        if Date() >= nextPowerCheck {
                            guard ModelCalibrationIdentity.currentPowerSource() == identity.powerSource,ProcessInfo.processInfo.isLowPowerModeEnabled == identity.lowPowerMode else {throw ModelLibraryError.invalid("Power mode changed during calibration. Current settings were retained.")}
                            nextPowerCheck=Date().addingTimeInterval(5)
                        }
                    }
                    try stdout.close();try stderr.close();child.clear()
                } catch {
                    child.cancel()
                    // Do not launch another worker until this one has exited.
                    let deadline=Date().addingTimeInterval(5)
                    while process.isRunning && Date() < deadline {await Task.detached{try? await Task.sleep(for:.milliseconds(100))}.value}
                    if process.isRunning {process.terminate()}
                    // Foundation's blocking wait can stall when a Process was
                    // launched on a cooperative executor and already reaped.
                    // Poll asynchronously, keeping ownership until it exits.
                    let terminationDeadline=Date().addingTimeInterval(5)
                    while process.isRunning && Date() < terminationDeadline {await Task.detached{try? await Task.sleep(for:.milliseconds(100))}.value}
                    if process.isRunning {Darwin.kill(process.processIdentifier,SIGKILL)}
                    while process.isRunning {await Task.detached{try? await Task.sleep(for:.milliseconds(100))}.value}
                    child.clear()
                    try? stdout.close();try? stderr.close();throw error
                }
                guard process.terminationStatus == 0 else {
                    let log=(try? ArchitectureRegistry.boundedData(errors,limit:2*1024*1024)).map{String(decoding:$0.suffix(2000),as:UTF8.self)} ?? "Model validation did not finish."
                    throw ModelLibraryError.invalid("Calibration stopped; the current profile is unchanged. "+log)
                }
                let data=try ArchitectureRegistry.boundedData(output,limit:32*1024*1024)
                trials.append(.init(profile:profile,samples:try samples(data,architecture:model.descriptor.architecture)))
                progress(index+1,profiles.count,"Completed and checked trial \(index+1) of \(profiles.count)")
            }
            let selected=ModelCalibrationSelection.select(baseline:baseline,trials:trials)
            // Check identity again before making any result eligible for saving.
            guard identity == (try ModelCalibrationIdentity.current(model:URL(fileURLWithPath:model.path),engine:worker)) else {throw ModelLibraryError.invalid("The model, engine or storage changed during calibration. The current profile is unchanged.")}
            return .init(identity:identity,createdAt:Date(),baseline:baseline,selected:selected,promoted:selected != baseline,trials:trials,
                note:"Storage-profile calibration: 64 actual decode steps per workload, alternating saved-profile/candidate runs. Independent adapter qualification supplies upstream reference checks; these trials compare complete output digests across profiles. Logical read bytes are not physical SSD traffic. Promotion requires unchanged full-logit/continuation digests, at least 10% faster latency on a target workload and no regression above 5% on other workloads. This is not a natural-completion speed or quality guarantee.")
        } onCancel:{child.cancel()}
    }
    private static func samples(_ data:Data,architecture:ModelArchitecture) throws -> [ModelCalibrationSample] {
        guard let rows=try JSONSerialization.jsonObject(with:data) as? [[String:Any]] else {throw ModelLibraryError.invalid("Invalid calibration output.")}
        func sample(_ row:[String:Any],history:Int,suffix:Int,executionValid:Bool)->ModelCalibrationSample? {
            guard let prefill=row["prefillSeconds"] as? Double,let decode=row["decodeSeconds"] as? Double,let steps=row["decodeSteps"] as? Int,
                  let a=row["logitSHA256"] as? String,let b=row["continuationSHA256"] as? String,let c=row["tokensSHA256"] as? String else{return nil}
            return .init(workload:"history-\(history)-suffix-\(suffix)",prefillSeconds:prefill,decodeSeconds:decode,decodeSteps:steps,
                sourceReadBytes:(row["sourceBytes"] as? Int64) ?? (row["sourceReadBytes"] as? Int64) ?? -1,
                footprintBytes:row["footprintBytes"] as? UInt64,executionValid:executionValid,outputIdentity:a+b+c)
        }
        func validExecution(_ row:[String:Any])->Bool {
            row["parityPassed"] as? Bool == true || (row["referenceCompared"] as? Bool == false && row["executionFinite"] as? Bool == true)
        }
        var result=[ModelCalibrationSample]()
        if architecture == .sarvam {
            for row in rows where row["mode"] as? String == "candidate" {
                guard let h=row["history"] as? Int,let s=row["suffix"] as? Int,let value=sample(row,history:h,suffix:s,executionValid:validExecution(row)) else {throw ModelLibraryError.invalid("Incomplete Sarvam calibration output.")};result.append(value)
            }
        } else {
            for row in rows {
                guard let h=row["history"] as? Int,let s=row["suffix"] as? Int,let report=row["report"] as? [String:Any],let passes=report["passes"] as? [[String:Any]] else {throw ModelLibraryError.invalid("Incomplete Chronos calibration output.")}
                for pass in passes where pass["mode"] as? String == "streamed-candidate" {
                    guard let value=sample(pass,history:h,suffix:s,executionValid:validExecution(report)) else {throw ModelLibraryError.invalid("Incomplete Chronos calibration output.")};result.append(value)
                }
            }
        }
        guard !result.isEmpty,result.allSatisfy(\.valid) else {throw ModelLibraryError.invalid("Calibration did not produce complete, valid measurements.")};return result
    }
}
