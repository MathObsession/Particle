import Foundation
import ParticleModelLibrary

extension WorkspaceModel {
    func optimizeLibraryModel(_ installed:InstalledModel) {
        guard !busy,!loading,!code.active,!modelInstallationActive,ArchitectureRegistry.report(installed.descriptor).installable else{return}
        library.optimizing=true;library.converting=true;nativeLibraryIntent=nil
        nativeLibraryTask=Task { [self] in
            let temporary=FileManager.default.temporaryDirectory.appendingPathComponent("particle-optimize-"+UUID().uuidString)
            var child:Process?
            defer {
                nativeLibraryProcess=nil;nativeLibraryTask=nil;library.optimizing=false;library.converting=false
                try? FileManager.default.removeItem(at:temporary)
            }
            do {
                library.message="Preparing optimization. This runs repeated 64-token checks and can take a long time; Cancel keeps your current settings."
                if connected {
                    unload()
                    for _ in 0..<150 {if !loading{break};try await Task.sleep(for:.milliseconds(100))}
                    guard !loading else {throw ModelLibraryError.invalid("The previous model is still stopping. Try again after it closes.")}
                }
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
                let profile=temporary.appendingPathComponent("profile.json"),output=temporary.appendingPathComponent("result.jsonl"),errors=temporary.appendingPathComponent("error.log")
                try JSONEncoder().encode(installed.profile).write(to:profile,options:.atomic)
                FileManager.default.createFile(atPath:output.path,contents:nil);FileManager.default.createFile(atPath:errors.path,contents:nil)
                let stdout=try FileHandle(forWritingTo:output),stderr=try FileHandle(forWritingTo:errors);defer{try? stdout.close();try? stderr.close()}
                let engine=Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/ParticleLibraryCLI")
                let process=Process();process.executableURL=engine;process.arguments=["optimize","--source",installed.path,"--profile",profile.path]
                process.standardInput=FileHandle.nullDevice;process.standardOutput=stdout;process.standardError=stderr
                try process.run();child=process;nativeLibraryProcess=process
                while process.isRunning {
                    try await Task.sleep(for:.seconds(1));try Task.checkCancellation()
                    let data=try ArchitectureRegistry.boundedData(output,limit:16*1024*1024)
                    for line in data.split(separator:10).suffix(2) {
                        if let object=try? JSONSerialization.jsonObject(with:Data(line)) as? [String:Any],object["event"] as? String == "calibration",let detail=object["detail"] as? String {library.message=detail}
                    }
                }
                try stdout.close();try stderr.close()
                guard process.terminationStatus == 0 else {
                    let detail=(try? ArchitectureRegistry.boundedData(errors,limit:2*1024*1024)).map{String(decoding:$0.suffix(2000),as:UTF8.self)} ?? "Optimization did not finish."
                    throw ModelLibraryError.invalid(detail)
                }
                struct Result:Decodable {let event:String;let record:ModelCalibrationRecord}
                let data=try ArchitectureRegistry.boundedData(output,limit:16*1024*1024)
                let result=data.split(separator:10).compactMap{try? JSONDecoder().decode(Result.self,from:Data($0))}.last
                guard let result,result.event == "calibrated" else {throw ModelLibraryError.invalid("No verified optimization report was produced.")}
                let worker=engine.deletingLastPathComponent().appendingPathComponent(ArchitectureRegistry.adapter(for:installed.descriptor.architecture).worker)
                guard result.record.identity == (try ModelCalibrationIdentity.current(model:URL(fileURLWithPath:installed.path),engine:worker)) else {throw ModelLibraryError.invalid("The engine, model or storage changed. Current settings were retained.")}
                try library.saveCalibration(result.record,model:installed)
                if settings.modelID == installed.id {
                    var next=settings;next.slots=result.record.selected.slots;next.readWorkers=result.record.selected.readWorkers;next.prefillChunk=result.record.selected.prefillChunk
                    // Complete ownership cleanup before applying validated settings.
                    library.converting=false;updateSettings(next)
                }
                library.message=result.record.promoted ? "Optimization saved a faster verified storage profile for this Mac and model location." : "Optimization complete. The current profile remains the fastest that met every validation gate."
            } catch {
                if let child,child.isRunning {
                    child.interrupt()
                    // The converter owns its worker's safe cancellation and
                    // waits for that child before releasing the model lease.
                    while child.isRunning {await Task.detached{try? await Task.sleep(for:.milliseconds(100))}.value}
                }
                library.message=error is CancellationError ? "Optimization cancelled. Your previous settings were retained." : error.localizedDescription
            }
        }
    }
    func cancelModelOptimization() {guard library.optimizing else{return};nativeLibraryTask?.cancel()}
}
