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
    /// Initial qualification covers the pinned Qwen byte-level vocabulary and
    /// official template. New tokenizer variants need their own round-trip checks.
    public static func validateChronosTokenizer(tokenizer:Data,configuration:Data) throws {
        func hash(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
        guard hash(tokenizer) == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
              hash(configuration) == "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679" else {throw ModelLibraryError.incompatible("This tokenizer variant needs separate qualification for the Chronos byte decoder.")}
    }
    // Qualification is shipped with the engine, never supplied by a remote model card.
    public static let adapters: [ArchitectureAdapter] = [
        .init(architecture: .sarvam, worker: "ParticleCLI", qualification: .qualified, supportsTools: true),
        .init(architecture: .qwen35MoE, worker: "ParticleMLXCLI", qualification: .qualified, supportsTools: false)
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
        let index = try boundedData(url.appendingPathComponent("model.safetensors.index.json"), limit: 16*1024*1024)
        guard let object = try JSONSerialization.jsonObject(with: index) as? [String: Any],
              let map = object["weight_map"] as? [String: String], !map.isEmpty, map.count <= 100_000 else {
            throw ModelLibraryError.invalid("An indexed MLX safetensors checkpoint is required.")
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
        let template = try String(decoding: boundedData(templateURL, limit: 262_144), as: UTF8.self)
        let extras = ["config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","generation_config.json","README.md","LICENSE","LICENSE.txt"].filter{FileManager.default.fileExists(atPath:url.appendingPathComponent($0).path)}
        let source = try LocalModelByteSource(root:url,paths:Array(Set(map.values))+extras)
        return try inspectMLX(configuration: config, tensors: tensors, template: template,
                              repository: nil, revision: nil, sourceBytes: bytes, sourceIdentity:source.identity)
    }

    public static func validateV2Envelope(_ data:Data) throws {
        guard let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],object["versionMajor"] as? Int == 2,
              let model=object["model"] as? [String:Any],model["architecture"] as? String == ModelArchitecture.qwen35MoE.rawValue else {
            throw ModelLibraryError.incompatible("This pack’s format or architecture is unsupported. Sarvam uses Particle’s v1 default pack; additional native packs require the supported Qwen3.5 MoE v2 layout.")
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
            let parts=tensor.name.split(separator:".")
            let routed=parts.count >= 8 && parts[0] == "language_model" && parts[1] == "model" && parts[2] == "layers" && parts[4] == "mlp" && parts[5] == "switch_mlp"
            let expected=routed ? Int(parts[3]) : nil
            guard tensor.layer == expected else {throw ModelLibraryError.invalid("Tensor layer ownership does not match its model name: \(tensor.name).")}
        }
    }

    /// Early rejection only. A supported configuration must still pass tensor,
    /// tokenizer/template and runtime qualification before it becomes installable.
    public static func configurationIncompatibility(_ configuration:Data) throws -> String? {
        guard let root = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
              let text = root["text_config"] as? [String: Any], root["model_type"] as? String == "qwen3_5_moe" else {
            return "This model family is not supported by a Particle streaming adapter."
        }
        guard root["vision_config"] == nil || root["vision_config"] is NSNull,
              root["auto_map"] == nil, text["auto_map"] == nil else {
            return "This release supports text-only checkpoints without custom repository code."
        }
        let expected = ["hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,
                        "moe_intermediate_size":512,"shared_expert_intermediate_size":512,
                        "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,
                        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
                        "linear_num_value_heads":32,"linear_value_head_dim":128,"vocab_size":248320]
        for (key, value) in expected where (text[key] as? Int) != value {
            return "Unsupported Qwen3.5 MoE configuration: \(key). This adapter targets the Chronos 35B-A3B layout."
        }
        let layerTypes = (0..<40).map{($0+1).isMultiple(of:4) ? "full_attention" : "linear_attention"}
        guard text["layer_types"] as? [String] == layerTypes,text["attention_bias"] as? Bool == false,
              text["attn_output_gate"] as? Bool == true,text["tie_word_embeddings"] as? Bool == false,
              text["hidden_act"] as? String == "silu",text["rms_norm_eps"] as? Double == 1e-6,
              (text["norm_topk_prob"] as? Bool ?? true),text["mamba_ssm_dtype"] as? String == "float32",
              let rope = text["rope_parameters"] as? [String:Any],rope["rope_theta"] as? Double == 10_000_000,
              rope["partial_rotary_factor"] as? Double == 0.25,rope["mrope_section"] as? [Int] == [11,11,10],
              rope["mrope_interleaved"] as? Bool == true,rope["type"] as? String == "default" else {
            return "Unsupported attention, recurrent-layer schedule, normalization or positional encoding."
        }
        guard let quant = (root["quantization"] ?? root["quantization_config"]) as? [String: Any],
              quant["bits"] as? Int == 3, quant["group_size"] as? Int == 64,
              (quant["mode"] as? String ?? "affine") == "affine" else {
            return "This adapter requires MLX affine 3-bit weights with group size 64."
        }
        return nil
    }

    public static func inspectMLX(configuration: Data, tensors: [String: SafetensorRecord], template: String,
                                  repository: String?, revision: String?, sourceBytes: Int64, license: String? = nil, sourceIdentity: String? = nil) throws -> CompatibilityReport {
        if let reason=try configurationIncompatibility(configuration) {return .unsupported(reason)}
        guard let root=try JSONSerialization.jsonObject(with:configuration) as? [String:Any],
              let text=root["text_config"] as? [String:Any],
              let quant=(root["quantization"] ?? root["quantization_config"]) as? [String:Any] else {return .unsupported("Incomplete model configuration.")}
        guard !tensors.keys.contains(where:{$0.contains("visual.") || $0.contains("vision_tower")}) else {return .unsupported("This release supports text-only checkpoints.")}
        var overrides = [String: Int]()
        guard tensors.count == 1757 else {return .unsupported("This checkpoint contains missing or additional tensors outside the supported text graph.")}
        func raw(_ name:String,_ shape:[Int]) -> Bool {
            tensors[name]?.shape == shape && ["F16","BF16","F32"].contains(tensors[name]?.dtype ?? "")
        }
        func matrix(_ name:String,_ out:Int,_ input:Int,_ bits:Int = 3) -> Bool {
            tensors[name+".weight"]?.dtype == "U32" && tensors[name+".weight"]?.shape == [out,input*bits/32] &&
            tensors[name+".scales"]?.shape == [out,input/64] && tensors[name+".biases"]?.shape == [out,input/64] &&
            ["F16","BF16","F32"].contains(tensors[name+".scales"]?.dtype ?? "") && tensors[name+".scales"]?.dtype == tensors[name+".biases"]?.dtype
        }
        guard matrix("language_model.model.embed_tokens",248320,2048),matrix("language_model.lm_head",248320,2048),raw("language_model.model.norm.weight",[2048]) else {
            return .unsupported("Unsupported embedding or output-head quantization layout.")
        }
        for layer in 0..<40 {
            let prefix = "language_model.model.layers.\(layer).mlp"
            let layerPrefix = "language_model.model.layers.\(layer)"
            guard raw(layerPrefix+".input_layernorm.weight",[2048]),raw(layerPrefix+".post_attention_layernorm.weight",[2048]),
                  matrix(prefix+".shared_expert.gate_proj",512,2048),matrix(prefix+".shared_expert.up_proj",512,2048),matrix(prefix+".shared_expert.down_proj",2048,512) else {
                return .unsupported("Unsupported shared expert or normalization layout at layer \(layer).")
            }
            if (layer+1).isMultiple(of:4) {
                let attention = layerPrefix+".self_attn"
                guard matrix(attention+".q_proj",8192,2048),matrix(attention+".k_proj",512,2048),matrix(attention+".v_proj",512,2048),
                      matrix(attention+".o_proj",2048,4096),raw(attention+".q_norm.weight",[256]),raw(attention+".k_norm.weight",[256]) else {
                    return .unsupported("Unsupported full-attention tensor layout at layer \(layer).")
                }
            } else {
                let linear = layerPrefix+".linear_attn"
                guard matrix(linear+".in_proj_qkv",8192,2048),matrix(linear+".in_proj_z",4096,2048),matrix(linear+".in_proj_a",32,2048),
                      matrix(linear+".in_proj_b",32,2048),matrix(linear+".out_proj",2048,4096),raw(linear+".conv1d.weight",[8192,4,1]),
                      raw(linear+".A_log",[32]),raw(linear+".dt_bias",[32]),raw(linear+".norm.weight",[128]) else {
                    return .unsupported("Unsupported recurrent-attention tensor layout at layer \(layer).")
                }
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let out = projection == "down_proj" ? 2048 : 512
                let input = projection == "down_proj" ? 512 : 2048
                let p = prefix+".switch_mlp."+projection
                guard tensors[p+".weight"]?.dtype == "U32", tensors[p+".weight"]?.shape == [256,out,input*3/32],
                      tensors[p+".scales"]?.shape == [256,out,input/64], tensors[p+".biases"]?.shape == [256,out,input/64],
                      ["F16","BF16","F32"].contains(tensors[p+".scales"]?.dtype ?? ""),
                      tensors[p+".scales"]?.dtype == tensors[p+".biases"]?.dtype else {
                    return .unsupported("Unsupported packed expert layout at layer \(layer), \(projection).")
                }
            }
            for (name, out) in [("gate",256),("shared_expert_gate",1)] {
                let p = prefix+"."+name
                guard let q = quant[p] as? [String: Any], q["bits"] as? Int == 8, q["group_size"] as? Int == 64,
                      (q["mode"] as? String ?? "affine") == "affine",
                      tensors[p+".weight"]?.dtype == "U32", tensors[p+".weight"]?.shape == [out,512],
                      tensors[p+".scales"]?.shape == [out,32], tensors[p+".biases"]?.shape == [out,32],
                      ["F16","BF16","F32"].contains(tensors[p+".scales"]?.dtype ?? ""),tensors[p+".scales"]?.dtype == tensors[p+".biases"]?.dtype else {
                    return .unsupported("The higher-precision routing tensors must be preserved (\(p)).")
                }
                overrides[p] = 8
            }
        }
        guard template.contains("<|im_start|>"), template.contains("enable_thinking"), template.contains("<tool_call>"),
              SHA256.hash(data:Data(template.utf8)).map({String(format:"%02x",$0)}).joined() == "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e",
              tensors["language_model.model.embed_tokens.weight"] != nil, tensors["language_model.lm_head.weight"] != nil,
              tensors["language_model.model.norm.weight"] != nil else {
            return .unsupported("The tokenizer/template variant or required common tensors are not supported by this adapter.")
        }
        let id = repository.flatMap { repo in revision.map { ModelDescriptor.identity(repository: repo, revision: $0, artifact: "mlx-affine-3bit-g64") } } ?? "local:"+ModelDescriptor.identity(repository:"local",revision:sourceIdentity ?? String(decoding:configuration,as:UTF8.self),artifact:"mlx-affine-3bit-g64")
        let descriptor = ModelDescriptor(id: id, name: repository?.split(separator: "/").last.map(String.init) ?? "Chronos MoE",
            architecture: .qwen35MoE, artifact: .mlx, provenance: .init(repository: repository, revision: revision, license: license),
            quantization: .init(scheme: "MLX affine", bits: 3, groupSize: 64, overrides: overrides),
            layers: 40, experts: 256, activeExperts: 8, contextLimit: min(text["max_position_embeddings"] as? Int ?? 262144,262144),
            sourceBytes: sourceBytes, kvBytesPerToken: 40_960, supportsTools: false)
        try descriptor.validate()
        return report(descriptor)
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
