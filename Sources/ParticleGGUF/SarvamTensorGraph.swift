import Foundation

extension SarvamConfiguration {
    /// Validate the complete pinned graph before an installer spends disk
    /// space or a runtime creates any Metal mapping.
    public static func validateTensorGraph(file: any ParticleModelSource) throws {
        var required = Set<String>()
        func require(_ name: String, _ shape: [Int], fp32: Bool = false) throws {
            let tensor = try file.tensor(name,shape:shape)
            guard !fp32 || tensor.type == .f32 else { throw ParticleError.invalid("Expected FP32 tensor: \(name)") }
            required.insert(name)
        }
        try require("token_embd.weight",[4096,262144])
        try require("output.weight",[4096,262144])
        try require("output_norm.weight",[4096],fp32:true)
        for layer in 0..<19 {
            let p = "blk.\(layer)."
            try require(p+"attn_norm.weight",[4096],fp32:true)
            try require(p+"attn_q_norm.weight",[64],fp32:true)
            try require(p+"attn_k_norm.weight",[64],fp32:true)
            try require(p+"attn_qkv.weight",[4096,4608])
            try require(p+"attn_output.weight",[4096,4096])
            try require(p+"ffn_norm.weight",[4096],fp32:true)
            if layer == 0 {
                try require(p+"ffn_gate.weight",[4096,8192])
                try require(p+"ffn_up.weight",[4096,8192])
                try require(p+"ffn_down.weight",[8192,4096])
            } else {
                try require(p+"ffn_gate_inp.weight",[4096,128],fp32:true)
                try require(p+"exp_probs_b.bias",[128],fp32:true)
                for branch in ["gate","up"] {
                    try require(p+"ffn_"+branch+"_shexp.weight",[4096,1024])
                    try require(p+"ffn_"+branch+"_exps.weight",[4096,1024,128])
                }
                try require(p+"ffn_down_shexp.weight",[1024,4096])
                try require(p+"ffn_down_exps.weight",[1024,4096,128])
            }
        }
        guard Set(file.tensors.keys) == required else { throw ParticleError.invalid("Unsupported additional Sarvam tensors") }
    }
}
