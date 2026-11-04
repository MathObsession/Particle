import Foundation
import ParticleGGUF
import ParticleFormat
import CryptoKit

public struct ArchitectureAdapter: Sendable {
    public let architecture: ModelArchitecture
    public let worker: String
    public let qualification: AdapterQualification
    public let supportsTools: Bool
}

public enum ArchitectureRegistry {
    /// Validate the decoder contract structurally; model identity and payload
    /// checksums pin each checkpoint without a single-tokenizer allowlist.
    public static func validateChronosTokenizer(tokenizer:Data,configuration:Data) throws {
        guard let root=try JSONSerialization.jsonObject(with:tokenizer) as? [String:Any],
              let model=root["model"] as? [String:Any],model["type"] as? String == "BPE",
              let vocab=model["vocab"] as? [String:Int],!vocab.isEmpty,vocab.count <= 524288,
              let decoder=root["decoder"] as? [String:Any],decoder["type"] as? String == "ByteLevel",
              root["normalizer"] == nil || root["normalizer"] is NSNull || (root["normalizer"] as? [String:Any])?["type"] as? String == "NFC",
              let config=try JSONSerialization.jsonObject(with:configuration) as? [String:Any],config["auto_map"] == nil || config["auto_map"] is NSNull else {
            throw ModelLibraryError.incompatible("This tokenizer needs BPE with ByteLevel decoding, no custom code and a supported Unicode normalizer.")
        }
        let added=root["added_tokens"] as? [[String:Any]] ?? []
        let controls=Set(added.compactMap{$0["content"] as? String})
        guard ["<|im_start|>","<|im_end|>","<think>","</think>"].allSatisfy({controls.contains($0) || vocab[$0] != nil}),
              Set(vocab.values).count == vocab.count,vocab.values.allSatisfy({$0 >= 0 && $0 < 524288}) else {throw ModelLibraryError.incompatible("The tokenizer lacks the chat/reasoning control tokens required by this graph adapter.")}
    }
    // Qualification is shipped with the engine, never supplied by a remote model card.
    public static let adapters: [ArchitectureAdapter] = [
        .init(architecture: .sarvam, worker: "ParticleCLI", qualification: .qualified, supportsTools: true),
        .init(architecture: .qwen35MoE, worker: "ParticleMLXCLI", qualification: .qualified, supportsTools: false),
        .init(architecture: .qwen3MoE, worker: "ParticleMLXCLI", qualification: .qualified, supportsTools: false)
    ]
    public static func adapter(for architecture: ModelArchitecture) -> ArchitectureAdapter {
        adapters.first { $0.architecture == architecture }!
    }

    public static func inspectLocal(_ url: URL) throws -> CompatibilityReport {
        if url.pathExtension.lowercased() == "gguf" {
            let file = try GGUFFile(path: url.path)
            _ = try SarvamConfiguration(file: file)
            try SarvamConfiguration.validateTensorGraph(file: file)
            let d = ModelDescriptor.sarvam
            return .init(descriptor: ModelDescriptor(id: "local:"+ModelDescriptor.identity(repository:"local",revision:file.sourceIdentity,artifact:"gguf"), name: file.metadata["general.name"]?.text ?? url.deletingPathExtension().lastPathComponent, architecture: .sarvam,
                artifact: .gguf, provenance: .init(), quantization: .init(scheme:"GGUF K-quant"), layers: d.layers, experts: d.experts,
                activeExperts: d.activeExperts, contextLimit: d.contextLimit, sourceBytes: Int64(file.size),
                kvBytesPerToken: d.kvBytesPerToken, supportsTools: true), conversionSupported: true, runtimeAvailable: true,
                qualification: .qualified, reason: "Validated Sarvam GGUF tensor graph.")
        }
        let manifestURL = url.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            let data = try boundedData(manifestURL, limit: ParticleFormat.indexLimit)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            if json?["versionMajor"] as? Int == 1 {
                let pack = try ParticlePackFile(path: url.path)
                _ = try SarvamConfiguration(file: pack); try SarvamConfiguration.validateTensorGraph(file: pack)
                let digest = SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
                let descriptor:ModelDescriptor
                if digest == ModelDescriptor.sarvamManifestSHA256 {descriptor = .sarvam}
                else {let d = ModelDescriptor.sarvam;descriptor = .init(id:"particle.sarvam."+digest,name:pack.metadata["general.name"]?.text ?? url.deletingPathExtension().lastPathComponent,architecture:.sarvam,artifact:.particle,provenance:.init(),quantization:.init(scheme:"Native K-quant"),layers:d.layers,experts:d.experts,activeExperts:d.activeExperts,contextLimit:d.contextLimit,sourceBytes:Int64(pack.size),kvBytesPerToken:d.kvBytesPerToken,supportsTools:true)}
                return .init(descriptor: descriptor, conversionSupported: true, runtimeAvailable: true,
                             qualification: .qualified, reason: "Validated native Sarvam pack.")
            }
            try validateV2Envelope(data)
            let reader = try ParticleV2Reader(path:url)
            let manifest = reader.manifest
            return try inspectV2Metadata(manifest,
                configuration:boundedData(url.appendingPathComponent("config.json"),limit:1_048_576),
                tokenizer:boundedData(url.appendingPathComponent("tokenizer/tokenizer.json"),limit:64*1024*1024),
                tokenizerConfiguration:boundedData(url.appendingPathComponent("tokenizer/tokenizer_config.json"),limit:1_048_576),
                template:boundedData(url.appendingPathComponent("tokenizer/chat_template.jinja"),limit:262_144))
        }
        let config = try boundedData(url.appendingPathComponent("config.json"), limit: 1_048_576)
        let map:[String:String]
        if FileManager.default.fileExists(atPath:url.appendingPathComponent("model.safetensors.index.json").path) {
            let index=try boundedData(url.appendingPathComponent("model.safetensors.index.json"),limit:16*1024*1024)
            guard let object=try JSONSerialization.jsonObject(with:index) as? [String:Any],let value=object["weight_map"] as? [String:String],!value.isEmpty,value.count <= 100_000 else {throw ModelLibraryError.invalid("Invalid local checkpoint index.")};map=value
        } else {
            let shards=try FileManager.default.contentsOfDirectory(at:url,includingPropertiesForKeys:nil).filter{$0.pathExtension == "safetensors"}
            guard shards.count == 1,let shard=shards.first else {throw ModelLibraryError.invalid("Choose a single-file or indexed safetensors MoE checkpoint.")}
            let header=try SafetensorsHeader.read(shard);map=Dictionary(uniqueKeysWithValues:header.tensors.keys.map{($0,shard.lastPathComponent)})
        }
        var tensors = [String: SafetensorRecord]()
        var bytes: Int64 = 0
        for filename in Set(map.values).sorted() {
            try ParticleFormat.validatePath(filename)
            guard !filename.contains("/"), filename.hasSuffix(".safetensors") else { throw ModelLibraryError.invalid("Unsafe checkpoint shard path.") }
            let file = url.appendingPathComponent(filename)
            let header = try SafetensorsHeader.read(file)
            bytes += Int64(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            for (name, record) in header.tensors {
                guard map[name] == filename, tensors[name] == nil else { throw ModelLibraryError.invalid("Checkpoint index does not match its shards.") }
                tensors[name] = record
            }
        }
        guard Set(tensors.keys) == Set(map.keys) else { throw ModelLibraryError.invalid("Checkpoint tensors are missing.") }
        let tokenizer = try boundedData(url.appendingPathComponent("tokenizer.json"), limit: 64*1024*1024)
        try validateChronosTokenizer(tokenizer:tokenizer,configuration:boundedData(url.appendingPathComponent("tokenizer_config.json"),limit:1_048_576))
        let templateURL = url.appendingPathComponent("chat_template.jinja")
        let template:String
        if FileManager.default.fileExists(atPath:templateURL.path) {template=String(decoding:try boundedData(templateURL,limit:262_144),as:UTF8.self)}
        else {
            let c=try JSONSerialization.jsonObject(with:boundedData(url.appendingPathComponent("tokenizer_config.json"),limit:1_048_576)) as? [String:Any]
            guard let value=c?["chat_template"] as? String,value.utf8.count <= 262_144 else {throw ModelLibraryError.invalid("Missing embedded chat template.")};template=value
        }
        let extras = ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json","README.md","LICENSE","LICENSE.txt"].filter{FileManager.default.fileExists(atPath:url.appendingPathComponent($0).path)}
        let source = try LocalModelByteSource(root:url,paths:Array(Set(map.values))+extras)
        return try inspectMLX(configuration: config, tensors: tensors, template: template,
                              repository: nil, revision: nil, sourceBytes: bytes, sourceIdentity:source.identity)
    }

    public static func validateV2Envelope(_ data:Data) throws {
        guard let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],object["versionMajor"] as? Int == 2,
              let model=object["model"] as? [String:Any],[ModelArchitecture.qwen35MoE.rawValue,ModelArchitecture.qwen3MoE.rawValue].contains(model["architecture"] as? String ?? "") else {
            throw ModelLibraryError.incompatible("This pack’s format or architecture is unsupported. Sarvam uses Particle’s v1 default pack; additional native packs require a supported MLX MoE v2 graph.")
        }
    }
    public static func inspectV2Metadata(_ manifest:ParticleV2Manifest,configuration:Data,tokenizer:Data,tokenizerConfiguration:Data,template:Data) throws -> CompatibilityReport {
        try manifest.validate();try validateChronosLayout(manifest)
        try validateChronosTokenizer(tokenizer:tokenizer,configuration:tokenizerConfiguration)
        let records=Dictionary(uniqueKeysWithValues:manifest.tensors.map{($0.name,SafetensorRecord(name:$0.name,dtype:$0.dtype,shape:$0.shape,offset:0,bytes:$0.bytes))})
        let checked=try inspectMLX(configuration:configuration,tensors:records,template:String(decoding:template,as:UTF8.self),
            repository:manifest.model.provenance.repository,revision:manifest.model.provenance.revision,
            sourceBytes:manifest.model.sourceBytes,license:manifest.model.provenance.license)
        guard checked.conversionSupported,let expected=checked.descriptor,
              expected.architecture == manifest.model.architecture,expected.layers == manifest.model.layers,
              expected.experts == manifest.model.experts,expected.activeExperts == manifest.model.activeExperts,
              expected.quantization == manifest.model.quantization,expected.contextLimit == manifest.model.contextLimit,
              expected.kvBytesPerToken == manifest.model.kvBytesPerToken,!manifest.model.supportsTools,
              manifest.model.id != ModelDescriptor.sarvam.id,
              manifest.model.provenance.repository == nil || expected.id == manifest.model.id else {
            return .unsupported("The pack descriptor does not match its configuration and tensor graph.")
        }
        return report(manifest.model)
    }

    public static func validateChronosLayout(_ manifest:ParticleV2Manifest) throws {
        for tensor in manifest.tensors {
            guard tensor.layer == MoEModelSchema.routedLayer(tensor.name) else {throw ModelLibraryError.invalid("Tensor layer ownership does not match its model name: \(tensor.name).")}
        }
    }
    public static func configurationIncompatibility(_ configuration:Data) throws -> String? {
        do {_=try MoEModelSchema(configuration:configuration);return nil}
        catch let error as ModelLibraryError {return error.localizedDescription}
    }
    public static func inspectMLX(configuration: Data, tensors: [String: SafetensorRecord], template: String,
                                  repository: String?, revision: String?, sourceBytes: Int64, license: String? = nil, sourceIdentity: String? = nil) throws -> CompatibilityReport {
        do {
            let schema=try MoEModelSchema(configuration:configuration)
            let overrides=try schema.validate(tensors:tensors)
            guard template.contains("<|im_start|>"),template.contains("<|im_end|>") else {return .unsupported("A Qwen chat template with Instant/Thinking support is required.")}
            let artifact="mlx-affine-\(schema.bits)bit-g\(schema.groupSize)"
            let id=repository.flatMap {repo in revision.map{ModelDescriptor.identity(repository:repo,revision:$0,artifact:artifact)}} ?? "local:"+ModelDescriptor.identity(repository:"local",revision:sourceIdentity ?? String(decoding:configuration,as:UTF8.self),artifact:artifact)
            let descriptor=ModelDescriptor(id:id,name:repository?.split(separator:"/").last.map(String.init) ?? "Imported MoE",architecture:schema.architecture,artifact:.mlx,
                provenance:.init(repository:repository,revision:revision,license:license),quantization:.init(scheme:"MLX affine",bits:schema.bits,groupSize:schema.groupSize,overrides:overrides),
                layers:schema.layers,experts:schema.experts,activeExperts:schema.active,contextLimit:schema.context,sourceBytes:sourceBytes,
                kvBytesPerToken:Int64(schema.fullAttentionLayers.count)*Int64(schema.kvHeads)*Int64(schema.headDim)*8,supportsTools:false)
            try descriptor.validate();return report(descriptor)
        } catch let error as ModelLibraryError {return .unsupported(error.localizedDescription)}
    }

    public static func report(_ descriptor: ModelDescriptor) -> CompatibilityReport {
        let adapter = adapter(for: descriptor.architecture)
        return .init(descriptor: descriptor, conversionSupported: true,
            runtimeAvailable: adapter.qualification == .qualified, qualification: adapter.qualification,
            reason: adapter.qualification == .qualified ? "Compatible with Particle." : "The converter recognizes this checkpoint. Its streaming runtime is awaiting correctness and performance qualification.")
    }
    public static func boundedData(_ url: URL, limit: Int) throws -> Data {
        let attrs = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey])
        guard attrs.isSymbolicLink != true, attrs.isRegularFile == true, let size = attrs.fileSize, size <= limit else { throw ModelLibraryError.invalid("Missing, oversized or linked model metadata: \(url.lastPathComponent).") }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let data = try handle.read(upToCount: limit+1) ?? Data()
        guard data.count <= limit else { throw ModelLibraryError.invalid("Model metadata exceeds its safety limit.") }
        return data
    }
}
