import Foundation

/// A conversion recipe derived from configuration and tensor headers, never a
/// repository name. The same recipe drives inspection, packing and execution.
public struct MoEModelSchema {
    public let architecture:ModelArchitecture
    public let root:[String:Any]
    public let text:[String:Any]
    public let quant:[String:Any]
    public let prefix:String
    public let hidden:Int,layers:Int,experts:Int,active:Int,intermediate:Int,heads:Int,kvHeads:Int,headDim:Int,vocabulary:Int,fullInterval:Int
    public var fullAttentionLayers:[Int] {(0..<layers).filter{architecture == .qwen3MoE || ($0+1).isMultiple(of:fullInterval)}}
    public var bits:Int {quant["bits"] as! Int}
    public var groupSize:Int {quant["group_size"] as! Int}
    public var context:Int {min(text["max_position_embeddings"] as? Int ?? 32768,262144)}
    public init(configuration:Data) throws {
        guard let root=try JSONSerialization.jsonObject(with:configuration) as? [String:Any],let kind=root["model_type"] as? String,
              let architecture=ModelArchitecture(rawValue:kind),architecture != .sarvam else {
            throw ModelLibraryError.incompatible("No streaming graph adapter for this architecture yet. Supported conversion families: Qwen3 MoE and Qwen3.5 MoE (MLX affine safetensors). A file conversion alone cannot implement a different model’s attention and routing.")
        }
        self.root=root;self.architecture=architecture
        let text=(root["text_config"] as? [String:Any]) ?? root
        self.text=text
        prefix=architecture == .qwen35MoE ? "language_model." : ""
        guard root["auto_map"] == nil,text["auto_map"] == nil,
              root["vision_config"] == nil || root["vision_config"] is NSNull else {throw ModelLibraryError.incompatible("Text-only checkpoints without custom repository code are required.")}
        guard let q=(root["quantization"] ?? root["quantization_config"]) as? [String:Any],
              [2,3,4,5,6,8].contains(q["bits"] as? Int ?? 0),[32,64,128].contains(q["group_size"] as? Int ?? 0),
              (q["mode"] as? String ?? "affine") == "affine" else {throw ModelLibraryError.incompatible("Choose an MLX affine-quantized checkpoint (2/3/4/5/6/8-bit, group 32/64/128). GGUF quantization needs its own compute adapter.")}
        quant=q
        func dimension(_ key:String,_ range:ClosedRange<Int>,_ fallback:Int? = nil) throws -> Int {
            guard let value=(text[key] as? Int) ?? fallback,range.contains(value) else {throw ModelLibraryError.incompatible("Invalid or unsupported model dimension: \(key).")};return value
        }
        hidden=try dimension("hidden_size",64...16384);layers=try dimension("num_hidden_layers",1...128)
        experts=try dimension("num_experts",2...4096);active=try dimension("num_experts_per_tok",1...min(64,experts))
        intermediate=try dimension("moe_intermediate_size",32...65536)
        heads=try dimension("num_attention_heads",1...256);kvHeads=try dimension("num_key_value_heads",1...heads)
        headDim=try dimension("head_dim",16...512,hidden/heads);vocabulary=try dimension("vocab_size",256...524288)
        fullInterval=architecture == .qwen35MoE ? try dimension("full_attention_interval",1...layers,4) : 1
        guard heads.isMultiple(of:kvHeads),text["attention_bias"] as? Bool != true,
              (text["hidden_act"] as? String ?? "silu") == "silu",
              (text["decoder_sparse_step"] as? Int ?? 1) == 1,(text["mlp_only_layers"] as? [Int] ?? []).isEmpty,
              (128...4_194_304).contains(text["max_position_embeddings"] as? Int ?? 32768),
              let epsilon=text["rms_norm_eps"] as? Double,epsilon.isFinite,epsilon > 0 else {throw ModelLibraryError.incompatible("Unsupported dense-layer schedule, attention bias, activation or normalization.")}
        if architecture == .qwen35MoE {
            let kh=try dimension("linear_num_key_heads",1...256),vh=try dimension("linear_num_value_heads",1...256)
            _=try dimension("linear_key_head_dim",16...256);_=try dimension("linear_value_head_dim",16...256)
            _=try dimension("linear_conv_kernel_dim",2...16);_=try dimension("shared_expert_intermediate_size",32...65536)
            let rope=(text["rope_parameters"] ?? text["rope_scaling"]) as? [String:Any] ?? [:]
            guard vh.isMultiple(of:kh),text["attn_output_gate"] as? Bool != false,
                  (text["mamba_ssm_dtype"] as? String ?? "float32") == "float32",
                  (text["norm_topk_prob"] as? Bool ?? true),
                  ["default"].contains((rope["type"] ?? rope["rope_type"]) as? String ?? "default"),
                  (rope["partial_rotary_factor"] as? Double ?? text["partial_rotary_factor"] as? Double ?? 0.25) == 0.25 else {throw ModelLibraryError.incompatible("Unsupported recurrent state or positional encoding in this Qwen3.5 graph.")}
            if let schedule=text["layer_types"] as? [String],schedule != (0..<layers).map({($0+1).isMultiple(of:fullInterval) ? "full_attention" : "linear_attention"}) {throw ModelLibraryError.incompatible("The attention schedule does not match full_attention_interval.")}
        } else if let rope=text["rope_scaling"] as? [String:Any] {
            guard ["default","linear"].contains((rope["type"] ?? rope["rope_type"]) as? String ?? "default"),
                  (rope["factor"] as? Double ?? 1) > 0 else {throw ModelLibraryError.incompatible("This Qwen3 positional encoding is not implemented by the bundled graph.")}
        }
    }
    public func quantization(for name:String)->(bits:Int,group:Int)? {
        if quant[name] as? Bool == false {return nil}
        let local=quant[name] as? [String:Any] ?? [:]
        return ((local["bits"] as? Int) ?? bits,(local["group_size"] as? Int) ?? groupSize)
    }
    public static func routedLayer(_ name:String)->Int? {
        let parts=name.split(separator:".")
        guard let p=parts.firstIndex(of:"layers"),p+4 < parts.count,parts[p+2] == "mlp",parts[p+3] == "switch_mlp" else{return nil}
        return Int(parts[p+1])
    }
    /// Exact named graph validation, including native mixed precision. No weight
    /// data is read and no tensors are silently dropped or requantized.
    public func validate(tensors:[String:SafetensorRecord]) throws -> [String:Int] {
        var required=Set<String>(),overrides=[String:Int]()
        func raw(_ name:String,_ shape:[Int]) throws {
            guard let t=tensors[name],t.shape == shape,["F16","BF16","F32"].contains(t.dtype) else {throw ModelLibraryError.incompatible("Missing or incompatible tensor: \(name).")};required.insert(name)
        }
        func matrix(_ name:String,_ out:Int,_ input:Int,_ routed:Bool=false) throws {
            let batch=routed ? [experts] : []
            if !routed,tensors[name+".scales"] == nil {try raw(name+".weight",batch+[out,input]);return}
            guard let q=quantization(for:name),[2,3,4,5,6,8].contains(q.bits),[32,64,128].contains(q.group),input.isMultiple(of:q.group), (input*q.bits).isMultiple(of:32),
                  tensors[name+".weight"]?.dtype == "U32",tensors[name+".weight"]?.shape == batch+[out,input*q.bits/32],
                  let s=tensors[name+".scales"],s.shape == batch+[out,input/q.group],
                  ["F16","BF16","F32"].contains(s.dtype),tensors[name+".biases"]?.dtype == s.dtype,tensors[name+".biases"]?.shape == s.shape else {throw ModelLibraryError.incompatible("Quantization metadata does not match tensor \(name).")}
            for suffix in [".weight",".scales",".biases"] {required.insert(name+suffix)}
            if q.bits != bits || quant[name] != nil {overrides[name]=q.bits}
            if let mode=(quant[name] as? [String:Any])?["mode"] as? String,mode != "affine" {throw ModelLibraryError.incompatible("Only affine native quantization is supported for \(name).")}
        }
        try matrix(prefix+"model.embed_tokens",vocabulary,hidden)
        if !(text["tie_word_embeddings"] as? Bool ?? false) {try matrix(prefix+"lm_head",vocabulary,hidden)}
        try raw(prefix+"model.norm.weight",[hidden])
        for layer in 0..<layers {
            let p=prefix+"model.layers.\(layer)",m=p+".mlp"
            for norm in ["input_layernorm","post_attention_layernorm"] {try raw(p+"."+norm+".weight",[hidden])}
            try matrix(m+".gate",experts,hidden)
            for (name,out,input) in [("gate_proj",intermediate,hidden),("up_proj",intermediate,hidden),("down_proj",hidden,intermediate)] {try matrix(m+".switch_mlp."+name,out,input,true)}
            if architecture == .qwen35MoE {
                let shared=text["shared_expert_intermediate_size"] as! Int
                for (name,out,input) in [("gate_proj",shared,hidden),("up_proj",shared,hidden),("down_proj",hidden,shared)] {try matrix(m+".shared_expert."+name,out,input)}
                try matrix(m+".shared_expert_gate",1,hidden)
            }
            if fullAttentionLayers.contains(layer) {
                let a=p+".self_attn"
                try matrix(a+".q_proj",heads*headDim*(architecture == .qwen35MoE ? 2 : 1),hidden)
                for name in ["k_proj","v_proj"] {try matrix(a+"."+name,kvHeads*headDim,hidden)}
                try matrix(a+".o_proj",hidden,heads*headDim)
                for norm in ["q_norm","k_norm"] {try raw(a+"."+norm+".weight",[headDim])}
            } else {
                let a=p+".linear_attn",kh=text["linear_num_key_heads"] as! Int,vh=text["linear_num_value_heads"] as! Int
                let kd=text["linear_key_head_dim"] as! Int,vd=text["linear_value_head_dim"] as! Int,conv=text["linear_conv_kernel_dim"] as! Int
                let values=vh*vd,channels=2*kh*kd+values
                for (name,out,input) in [("in_proj_qkv",channels,hidden),("in_proj_z",values,hidden),("in_proj_a",vh,hidden),("in_proj_b",vh,hidden),("out_proj",hidden,values)] {try matrix(a+"."+name,out,input)}
                try raw(a+".conv1d.weight",[channels,conv,1]);try raw(a+".A_log",[vh]);try raw(a+".dt_bias",[vh]);try raw(a+".norm.weight",[vd])
            }
        }
        guard required == Set(tensors.keys) else {throw ModelLibraryError.incompatible("The checkpoint has extra or unrecognized tensors (\(Set(tensors.keys).subtracting(required).sorted().prefix(3).joined(separator:", "))). No weights were dropped.")}
        return overrides
    }
}
