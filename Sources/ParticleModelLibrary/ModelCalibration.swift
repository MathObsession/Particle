import Foundation
import CryptoKit
import Darwin
import IOKit.ps

public struct ModelCalibrationIdentity: Codable,Equatable,Sendable {
    public let modelSHA256:String
    public let engineSHA256:String
    public let hardware:String
    public let physicalMemory:UInt64
    public let system:String
    public let storage:String
    public let powerSource:String?
    public let lowPowerMode:Bool?
    public static func currentPowerSource()->String {
        guard let snapshot=IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let source=IOPSGetProvidingPowerSourceType(snapshot)?.takeUnretainedValue() else {return "Unknown"}
        return source as String
    }

    public static func current(model:URL,engine:URL) throws -> Self {
        func hashFile(_ url:URL) throws -> String {
            let handle=try FileHandle(forReadingFrom:url);defer{try? handle.close()}
            var digest=SHA256()
            while let data=try handle.read(upToCount:1024*1024),!data.isEmpty {digest.update(data:data)}
            return digest.finalize().map{String(format:"%02x",$0)}.joined()
        }
        var size=0;sysctlbyname("hw.model",nil,&size,nil,0)
        var bytes=[CChar](repeating:0,count:max(1,size));sysctlbyname("hw.model",&bytes,&size,nil,0)
        let volume=try model.resourceValues(forKeys:[.volumeUUIDStringKey]).volumeUUIDString ?? "unknown-volume"
        let hardware=String(decoding:bytes.prefix(while:{$0 != 0}).map{UInt8(bitPattern:$0)},as:UTF8.self)
        return try .init(modelSHA256:hashFile(model.appendingPathComponent("manifest.json")),engineSHA256:hashFile(engine),hardware:hardware,
            physicalMemory:ProcessInfo.processInfo.physicalMemory,system:ProcessInfo.processInfo.operatingSystemVersionString,
            storage:volume+":"+model.standardizedFileURL.resolvingSymlinksInPath().path,powerSource:currentPowerSource(),lowPowerMode:ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}
public struct ModelCalibrationSample: Codable,Equatable,Sendable {
    public let workload:String
    public let prefillSeconds:Double
    public let decodeSeconds:Double
    public let decodeSteps:Int
    public let sourceReadBytes:Int64
    public let footprintBytes:UInt64?
    public let executionValid:Bool
    /// All vocabulary logits and the continuation token IDs, rather than a
    /// short textual answer, establish unchanged execution for storage tuning.
    public let outputIdentity:String
    public var seconds:Double {prefillSeconds+decodeSeconds}
    public var valid:Bool {prefillSeconds.isFinite && prefillSeconds >= 0 && decodeSeconds.isFinite && decodeSeconds > 0 && decodeSteps >= 64 && sourceReadBytes >= 0 && executionValid && outputIdentity.count == 192 && outputIdentity.utf8.allSatisfy{(48...57).contains($0)||(97...102).contains($0)}}
}
public struct ModelCalibrationTrial: Codable,Equatable,Sendable {
    public let profile:ModelRuntimeProfile
    public let samples:[ModelCalibrationSample]
    public init(profile:ModelRuntimeProfile,samples:[ModelCalibrationSample]) {self.profile=profile;self.samples=samples}
}
public struct ModelCalibrationRecord: Codable,Equatable,Sendable {
    public let identity:ModelCalibrationIdentity
    public let createdAt:Date
    public let baseline:ModelRuntimeProfile
    public let selected:ModelRuntimeProfile
    public let promoted:Bool
    public let trials:[ModelCalibrationTrial]
    public let note:String
}
public enum ModelCalibrationSelection {
    private static func median(_ values:[Double])->Double {
        let sorted=values.sorted(),middle=sorted.count/2
        return sorted.count.isMultiple(of:2) ? (sorted[middle-1]+sorted[middle])/2 : sorted[middle]
    }
    /// Require repeatable evidence on every workload. A faster but numerically
    /// different or under-tested profile never replaces the current setting.
    public static func select(baseline:ModelRuntimeProfile,trials:[ModelCalibrationTrial])->ModelRuntimeProfile {
        let controls=trials.filter{$0.profile == baseline}
        guard controls.count >= 2,controls.allSatisfy({!$0.samples.isEmpty && $0.samples.allSatisfy(\.valid)}) else{return baseline}
        let controlSamples=controls.flatMap(\.samples),workloads=Set(controlSamples.map(\.workload))
        guard workloads.count >= 2,controls.allSatisfy({Set($0.samples.map(\.workload)) == workloads}) else{return baseline}
        let groups=Dictionary(grouping:controlSamples,by:\.workload)
        guard groups.values.allSatisfy({Set($0.map(\.outputIdentity)).count == 1}) else{return baseline}
        var best=baseline,bestRatio=1.0
        var seen=[ModelRuntimeProfile]()
        for profile in trials.map(\.profile) where profile != baseline && !seen.contains(profile) {
            seen.append(profile)
            guard profile.activeExperts == baseline.activeExperts,profile.context == baseline.context else{continue}
            let matches=trials.filter{$0.profile == profile}
            guard matches.count >= 2,matches.allSatisfy({!$0.samples.isEmpty && $0.samples.allSatisfy(\.valid) && Set($0.samples.map(\.workload)) == workloads}) else{continue}
            let samples=Dictionary(grouping:matches.flatMap(\.samples),by:\.workload)
            var ratios=[Double](),unchanged=true
            for workload in workloads {
                guard let candidate=samples[workload],let control=groups[workload],
                      Set(candidate.map(\.outputIdentity)) == Set(control.map(\.outputIdentity)) else{unchanged=false;break}
                ratios.append(median(candidate.map(\.seconds))/median(control.map(\.seconds)))
            }
            guard unchanged,ratios.allSatisfy({$0 <= 1.05}),let target=ratios.min(),target <= 0.9 else{continue}
            let ratio=median(ratios)
            if ratio < bestRatio {best=profile;bestRatio=ratio}
        }
        return best
    }
    public static func candidates(baseline:ModelRuntimeProfile,model:ModelDescriptor,physicalMemory:UInt64)->[ModelRuntimeProfile] {
        var values=[ModelRuntimeProfile]()
        func append(_ profile:ModelRuntimeProfile) {if profile != baseline && !values.contains(profile){values.append(profile)}}
        if let stride=baseline.expertBytesPerSlot,stride > 0 {
            var cache=baseline
            cache.slots=min(model.architecture == .sarvam ? 16 : 32,baseline.slots+4)
            if Int64(cache.slots)*stride <= Int64(physicalMemory/4) {append(cache)}
        }
        for workers in [1,2,4] where workers != baseline.readWorkers {var candidate=baseline;candidate.readWorkers=workers;append(candidate)}
        for chunk in [32,64,128] where chunk != baseline.prefillChunk {var candidate=baseline;candidate.prefillChunk=chunk;append(candidate)}
        return values
    }
}
