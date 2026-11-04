import Foundation

/// Runs only Particle's bundled compiler. No model executable, shell or remote
/// module is loaded. Input/output metadata is bounded and private to the job.
public enum GraphCompilationClient {
    public static func compile(configuration:Data,tensors:[SafetensorRecord]) throws -> ParticleGraphPlan {
        let executable=Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("ParticleMLXCLI")
        guard let executable,FileManager.default.isExecutableFile(atPath:executable.path) else {throw ModelLibraryError.incompatible("The bundled graph compiler is unavailable. Use the complete Particle app or build ParticleMLXCLI beside the converter.")}
        let request=ParticleGraphRequest(configuration:configuration,tensors:tensors)
        let encoded=try JSONEncoder().encode(request)
        guard encoded.count <= 24*1024*1024 else {throw ModelLibraryError.invalid("Graph metadata exceeds the compiler request limit.")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("particle-graph-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        defer{try? FileManager.default.removeItem(at:root)}
        let input=root.appendingPathComponent("request.json"),output=root.appendingPathComponent("plan.json"),errors=root.appendingPathComponent("error.txt")
        try encoded.write(to:input,options:.atomic)
        FileManager.default.createFile(atPath:errors.path,contents:nil,attributes:[.posixPermissions:0o600])
        let errorHandle=try FileHandle(forWritingTo:errors);defer{try? errorHandle.close()}
        let process=Process();process.executableURL=executable
        process.arguments=["--compile-graph",input.path,"--graph-output",output.path]
        process.standardOutput=FileHandle.nullDevice;process.standardInput=FileHandle.nullDevice;process.standardError=errorHandle
        try process.run()
        let deadline=ProcessInfo.processInfo.systemUptime+45
        while process.isRunning {
            if Task.isCancelled {process.terminate();process.waitUntilExit();throw CancellationError()}
            if ProcessInfo.processInfo.systemUptime >= deadline {process.terminate();process.waitUntilExit();throw ModelLibraryError.incompatible("The model graph compiler timed out. No weights were downloaded.")}
            Thread.sleep(forTimeInterval:0.05)
        }
        guard process.terminationStatus == 0 else {
            let handle=try FileHandle(forReadingFrom:errors);defer{try? handle.close()}
            let detail=String(decoding:(try? handle.read(upToCount:8192)) ?? Data(),as:UTF8.self)
            throw ModelLibraryError.incompatible(detail.isEmpty ? "The bundled backend could not compile this graph. No weights were downloaded." : detail)
        }
        let plan=try JSONDecoder().decode(ParticleGraphPlan.self,from:ArchitectureRegistry.boundedData(output,limit:4*1024*1024))
        try plan.validate()
        guard plan.configurationSHA256 == ParticleGraphPlan.digest(configuration),plan.sourceTensorsSHA256 == (try ParticleGraphPlan.tensorDigest(tensors)) else {throw ModelLibraryError.invalid("The graph compiler returned a plan for different model metadata.")}
        return plan
    }
}
