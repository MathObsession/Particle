import Testing
import Foundation
@testable import ParticleModelLibrary

@Suite("Evidence-based profile promotion") struct ModelCalibrationTests {
    private let baseline=ModelRuntimeProfile(model:.sarvam,physicalMemory:8*1024*1024*1024)
    private func trial(_ profile:ModelRuntimeProfile,_ times:[Double],identity:String=String(repeating:"a",count:192),steps:Int=64)->ModelCalibrationTrial {
        .init(profile:profile,samples:times.enumerated().map{index,time in
            .init(workload:index == 0 ? "short" : "long",prefillSeconds:time/2,decodeSeconds:time/2,decodeSteps:steps,sourceReadBytes:100,footprintBytes:nil,executionValid:true,outputIdentity:identity)
        })
    }
    @Test func promotesRepeatedFasterUnchangedResults() {
        var candidate=baseline;candidate.readWorkers=1
        let trials=[trial(baseline,[10,20]),trial(candidate,[8,18]),trial(candidate,[8,18]),trial(baseline,[10,20])]
        #expect(ModelCalibrationSelection.select(baseline:baseline,trials:trials) == candidate)
    }
    @Test func rejectsRegressionChangedOutputsShortRunsAndChangedExpertCount() {
        var candidate=baseline;candidate.readWorkers=1
        for bad in [trial(candidate,[8,21.2]),trial(candidate,[8,18],identity:String(repeating:"b",count:192)),trial(candidate,[8,18],steps:63)] {
            let trials=[trial(baseline,[10,20]),bad,bad,trial(baseline,[10,20])]
            #expect(ModelCalibrationSelection.select(baseline:baseline,trials:trials) == baseline)
        }
        candidate.activeExperts=3
        let trials=[trial(baseline,[10,20]),trial(candidate,[5,10]),trial(candidate,[5,10]),trial(baseline,[10,20])]
        #expect(ModelCalibrationSelection.select(baseline:baseline,trials:trials) == baseline)
    }
    @Test func requiresTenPercentTargetImprovementAndCompleteControls() {
        var candidate=baseline;candidate.prefillChunk=64
        let small=[trial(baseline,[10,20]),trial(candidate,[9.1,18.2]),trial(candidate,[9.1,18.2]),trial(baseline,[10,20])]
        #expect(ModelCalibrationSelection.select(baseline:baseline,trials:small) == baseline)
        let incomplete=[trial(baseline,[10]),trial(candidate,[8,18]),trial(candidate,[8,18]),trial(baseline,[10,20])]
        #expect(ModelCalibrationSelection.select(baseline:baseline,trials:incomplete) == baseline)
    }
    @Test func tuningNeverChangesNativeRoutingOrContext() {
        let candidates=ModelCalibrationSelection.candidates(baseline:baseline,model:.sarvam,physicalMemory:8*1024*1024*1024)
        #expect(!candidates.isEmpty)
        #expect(candidates.allSatisfy{$0.activeExperts == baseline.activeExperts && $0.context == baseline.context})
    }
    @Test func orchestratesSequentialMockWorkersWithoutChangingProfile() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("particle-calibration-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer{try? FileManager.default.removeItem(at:root)}
        let pack=root.appendingPathComponent("fixture.particle"),worker=root.appendingPathComponent("ParticleCLI")
        try FileManager.default.createDirectory(at:pack,withIntermediateDirectories:false)
        try Data("metadata-only fixture".utf8).write(to:pack.appendingPathComponent("manifest.json"))
        let digest=String(repeating:"a",count:64)
        let rows=[["mode":"candidate","history":512,"suffix":16,"prefillSeconds":1,"decodeSeconds":1,"decodeSteps":64,"sourceBytes":0,"parityPassed":true,"logitSHA256":digest,"continuationSHA256":digest,"tokensSHA256":digest],
                  ["mode":"candidate","history":2048,"suffix":128,"prefillSeconds":2,"decodeSeconds":2,"decodeSteps":64,"sourceBytes":0,"parityPassed":true,"logitSHA256":digest,"continuationSHA256":digest,"tokensSHA256":digest]] as [[String:Any]]
        let json=String(decoding:try JSONSerialization.data(withJSONObject:rows),as:UTF8.self)
        try "#!/bin/sh\ncat <<'PARTICLE_FIXTURE'\n\(json)\nPARTICLE_FIXTURE\n".write(to:worker,atomically:true,encoding:.utf8)
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:worker.path)
        var model=InstalledModel(descriptor:.sarvam,path:pack.path,managed:false);model.profile=baseline
        let report=try await ModelCalibrator.run(model:model,engines:root){_,_,_ in}
        #expect(!report.promoted)
        #expect(report.selected == baseline)
        #expect(report.trials.count == 16)
        #expect(report.trials.allSatisfy{$0.samples.count == 2})
        #expect(FileManager.default.fileExists(atPath:pack.path))
    }
    @Test func cancellationWaitsForItsOwnMockWorkerToExit() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("particle-calibration-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false);defer{try? FileManager.default.removeItem(at:root)}
        let pack=root.appendingPathComponent("fixture.particle"),worker=root.appendingPathComponent("ParticleCLI"),started=root.appendingPathComponent("started")
        try FileManager.default.createDirectory(at:pack,withIntermediateDirectories:false)
        try Data("metadata-only fixture".utf8).write(to:pack.appendingPathComponent("manifest.json"))
        let quoted="'"+started.path.replacingOccurrences(of:"'",with:"'\\''")+"'"
        try "#!/bin/sh\nprintf started > \(quoted)\nexec /bin/sleep 30\n".write(to:worker,atomically:true,encoding:.utf8)
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:worker.path)
        let model=InstalledModel(descriptor:.sarvam,path:pack.path,managed:false)
        let task=Task {try await ModelCalibrator.run(model:model,engines:root){_,_,_ in}}
        for _ in 0..<100 {if FileManager.default.fileExists(atPath:started.path){break};try await Task.sleep(for:.milliseconds(50))}
        #expect(FileManager.default.fileExists(atPath:started.path))
        task.cancel()
        do {_=try await task.value;Issue.record("Expected cancellation")}
        catch is CancellationError {}
        #expect(FileManager.default.fileExists(atPath:pack.path))
    }
}
