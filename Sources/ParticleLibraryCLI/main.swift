import Foundation
import Darwin
import ParticleModelLibrary
import ParticleFormat
import ParticleGGUF
import ParticlePackCore

@main struct ParticleLibraryCLI {
    static func emit<T:Encodable>(_ value: T) throws {FileHandle.standardOutput.write(try JSONEncoder().encode(value)+Data([10]))}
    static func progress(_ completed:Int64,_ total:Int64,_ detail:String) throws {
        struct Progress:Encodable {let event="conversion";let completed:Int64;let total:Int64;let detail:String}
        try emit(Progress(completed:completed,total:total,detail:detail))
    }
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.isEmpty || args.contains("--help") {
            print("""
            ParticleLibraryCLI — inspect and losslessly prepare MoE models
            inspect --source <model-folder-or-gguf>
            inspect --repo <owner/model>
            convert --source <indexed-mlx-folder-or-gguf> --output <new.particle> [--resume] [--plan]
            convert --repo <owner/model> --output <new.particle> [--resume] [--plan]
            verify --source <model.particle>
            optimize --source <model.particle> --profile <profile.json>
            Experimental adapters require --allow-unqualified-conversion for conversion.
            Conversion preserves native quantization; it does not qualify a runtime or promise speed.
            """);return
        }
        do {
            let command = args[0];var values = [String:String]();var flags = Set<String>();var index = 1
            while index < args.count {
                let key = args[index]
                if ["--resume","--plan","--allow-unqualified-conversion"].contains(key) {guard flags.insert(key).inserted else {throw ModelLibraryError.invalid("Duplicate argument.")}}
                else {
                    guard ["--source","--repo","--revision","--output","--profile","--expected-source-id"].contains(key),index+1 < args.count,values[key] == nil else {throw ModelLibraryError.invalid("Unknown or incomplete argument: \(key).")}
                    index += 1;values[key] = args[index]
                }
                index += 1
            }
            guard (values["--source"] == nil) != (values["--repo"] == nil) else {throw ModelLibraryError.invalid("Specify exactly one --source or --repo.")}
            guard values["--revision"] == nil || values["--repo"] != nil else {throw ModelLibraryError.invalid("--revision requires --repo.")}
            if let expected=values["--expected-source-id"],let path=values["--source"] {
                guard try ArchitectureRegistry.inspectLocal(URL(fileURLWithPath:path)).descriptor?.id == expected else {throw ModelLibraryError.invalid("The local source changed since it was inspected. Import the current source again; existing files were preserved.")}
            }
            if command == "optimize" {
                guard let path=values["--source"],let profilePath=values["--profile"],let executable=Bundle.main.executableURL else {throw ModelLibraryError.invalid("Optimization requires a local pack and saved profile.")}
                let report=try ArchitectureRegistry.inspectLocal(URL(fileURLWithPath:path))
                guard report.installable,let descriptor=report.descriptor else {throw ModelLibraryError.incompatible(report.reason)}
                var model=InstalledModel(descriptor:descriptor,path:path,managed:false)
                model.profile=try JSONDecoder().decode(ModelRuntimeProfile.self,from:ArchitectureRegistry.boundedData(URL(fileURLWithPath:profilePath),limit:65_536))
                let task=Task {try await ModelCalibrator.run(model:model,engines:executable.deletingLastPathComponent()) {completed,total,detail in
                    struct Progress:Encodable {let event="calibration";let completed:Int;let total:Int;let detail:String}
                    try? emit(Progress(completed:completed,total:total,detail:detail))
                }}
                signal(SIGINT,SIG_IGN)
                let interrupt=DispatchSource.makeSignalSource(signal:SIGINT,queue:.global());interrupt.setEventHandler{task.cancel()};interrupt.resume();defer{interrupt.cancel()}
                let result=try await task.value
                struct Result:Encodable {let event="calibrated";let record:ModelCalibrationRecord}
                try emit(Result(record:result));return
            }
            if command == "inspect" {
                let report: CompatibilityReport
                if let path = values["--source"] {report = try ArchitectureRegistry.inspectLocal(URL(fileURLWithPath:path))}
                else {
                    let client = HuggingFaceClient(),card = try await client.model(HubRepository.parse(values["--repo"]!),revision:values["--revision"])
                    report = try await client.inspect(card)
                }
                try emit(report);return
            }
            if command == "verify",let path = values["--source"] {
                let lease=try ModelOperationLease();defer{withExtendedLifetime(lease){}}
                let url = URL(fileURLWithPath:path)
                let data = try ArchitectureRegistry.boundedData(url.appendingPathComponent("manifest.json"),limit:16*1024*1024)
                let json = try JSONSerialization.jsonObject(with:data) as? [String:Any]
                var done:Int64=0
                if json?["versionMajor"] as? Int == 1 {
                    let pack=try ParticlePackFile(path:path),total=pack.manifest.files.values.reduce(Int64(0)){$0+Int64($1.bytes)}
                    try pack.verifyAll(onFile:{name in done += Int64(pack.manifest.files[name]?.bytes ?? 0);try? progress(done,total,"Verifying "+name)})
                } else {
                    let pack=try ParticleV2Reader(path:url),total=pack.manifest.files.values.reduce(Int64(0)){$0+$1.bytes}
                    try pack.verifyAll(progress:{name in done += pack.manifest.files[name]?.bytes ?? 0;try? progress(done,total,"Verifying "+name)})
                }
                try emit(["event":"verified","path":path]);return
            }
            guard command == "convert" else {throw ModelLibraryError.invalid("Unknown library command.")}
            if let path=values["--source"],URL(fileURLWithPath:path).pathExtension.lowercased() == "gguf" {
                let file=try GGUFFile(path:path),report=try ArchitectureRegistry.inspectLocal(URL(fileURLWithPath:path))
                guard report.installable else {throw ModelLibraryError.incompatible(report.reason)}
                let plan=try ParticlePacking.plan(file:file)
                if flags.contains("--plan") {try emit(["source_bytes":plan.sourceBytes,"packed_payload_bytes":plan.payloadBytes]);return}
                guard let destination=values["--output"] else {throw ModelLibraryError.invalid("Specify --output ending in .particle.")}
                let output=URL(fileURLWithPath:destination),lease=try ModelOperationLease();defer{withExtendedLifetime(lease){}}
                if FileManager.default.fileExists(atPath:output.path) {
                    guard flags.contains("--resume") else {throw ModelLibraryError.invalid("Choose a new destination or resume the existing pack.")}
                    let pack=try ParticlePackFile(path:destination)
                    guard pack.manifest.sourceBytes == file.size,
                          try ParticleIO.hashFile(descriptor:file.descriptor,size:file.size) == pack.manifest.sourceSHA256 else {throw ModelLibraryError.invalid("The existing destination belongs to a different source.")}
                    try pack.verifyAll()
                } else {
                    try progress(0,Int64(plan.sourceBytes),"Inspecting the pinned tokenizer")
                    let tokenizer=try await SarvamTokenizerAssets.prepare(at:output.deletingLastPathComponent().appendingPathComponent(".sarvam-tokenizer-"+SarvamTokenizerAssets.revision),existing:[URL(fileURLWithPath:file.tokenizerDirectory)])
                    let pressure=ModelMemoryPressure();var completed:Int64=0
                    try ParticlePacking.pack(file:file,tokenizer:tokenizer,output:output,resume:flags.contains("--resume"),progress:{done,total in try pressure.check();completed=Int64(done);try progress(completed,Int64(total),"Converting local GGUF")},status:{detail in try? progress(completed,Int64(plan.sourceBytes),detail)})
                }
                try emit(["event":"converted","path":destination,"runtime":"qualified"]);return
            }
            var provenance = ModelProvenance()
            let source: any ModelByteSource
            if let repo = values["--repo"] {
                let card = try await HuggingFaceClient().model(HubRepository.parse(repo),revision:values["--revision"])
                provenance = .init(repository:card.repository,revision:card.revision,license:card.license)
                source = try RemoteModelByteSource(card:card)
                if card.files.contains(where:{$0.path == "manifest.json"}) {
                    let native=try await NativePackDownload.plan(source:source)
                    if flags.contains("--plan") {try emit(native.layout);return}
                    guard native.report.installable || flags.contains("--allow-unqualified-conversion") else {throw ModelLibraryError.incompatible(native.report.reason)}
                    guard let path=values["--output"] else {throw ModelLibraryError.invalid("Specify --output ending in .particle.")}
                    try await MLXPackConverter.convert(source:source,plan:native.layout,output:URL(fileURLWithPath:path),resume:flags.contains("--resume"),nativeManifest:native.manifest) {done,total,detail in
                        try? progress(done,total,detail)
                    }
                    try emit(["event":"converted","path":path,"runtime":native.report.qualification.rawValue]);return
                }
            } else {
                let root = URL(fileURLWithPath:values["--source"]!,isDirectory:true)
                let data = try ArchitectureRegistry.boundedData(root.appendingPathComponent("model.safetensors.index.json"),limit:16*1024*1024)
                guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any],let map = object["weight_map"] as? [String:String] else {throw ModelLibraryError.invalid("An indexed MLX model folder is required.")}
                let metadata = ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json","README.md","LICENSE","LICENSE.txt"].filter {FileManager.default.fileExists(atPath:root.appendingPathComponent($0).path)}
                source = try LocalModelByteSource(root:root,paths:Array(Set(map.values))+metadata)
            }
            let plan = try await MLXPackConverter.plan(source:source,provenance:provenance)
            if flags.contains("--plan") {try emit(plan);return}
            guard ArchitectureRegistry.report(plan.model).installable || flags.contains("--allow-unqualified-conversion") else {
                throw ModelLibraryError.incompatible("This runtime is not qualified. Development conversion requires --allow-unqualified-conversion; the pack will remain unavailable in the public library.")
            }
            guard let path = values["--output"] else {throw ModelLibraryError.invalid("Specify --output ending in .particle.")}
            try await MLXPackConverter.convert(source:source,plan:plan,output:URL(fileURLWithPath:path),resume:flags.contains("--resume")) {done,total,detail in
                try? progress(done,total,detail)
            }
            try emit(["event":"converted","path":path,"runtime":ArchitectureRegistry.adapter(for:plan.model.architecture).qualification.rawValue])
        } catch {FileHandle.standardError.write(Data(("Particle: "+((error as? LocalizedError)?.errorDescription ?? String(describing:error))+"\n").utf8));exit(1)}
    }
}
