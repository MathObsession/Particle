// Native ports of the Apple MLX LM Qwen2-MoE / DeepSeek graphs (MIT), with
// MiniCPM-MoE routing/scaling from openbmb/MiniCPM-MoE-8x2B (Apache-2.0).
// Repository code is read as reference during development, never executed.
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import ParticleModelLibrary

struct ClassicMoEConfiguration {
    let kind:String,hidden:Int,layers:Int,heads:Int,kvHeads:Int,vocabulary:Int,experts:Int,active:Int,intermediate:Int,expertIntermediate:Int,shared:Int,firstDense:Int,frequency:Int
    let epsilon:Float,theta:Float,ropeScale:Float,embeddingScale:Float,residualScale:Float,outputScale:Float
    let tied:Bool
    init(_ data:Data) throws {
        guard let c=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {throw ModelLibraryError.invalid("Invalid classic MoE configuration.")}
        kind=c["model_type"] as? String ?? "minicpm"
        func n(_ key:String,_ fallback:Int=0)->Int {c[key] as? Int ?? fallback}
        hidden=n("hidden_size");layers=n("num_hidden_layers");heads=n("num_attention_heads");kvHeads=n("num_key_value_heads",heads);vocabulary=n("vocab_size")
        experts=n("num_experts",n("n_routed_experts"));active=n("num_experts_per_tok");intermediate=n("intermediate_size");expertIntermediate=n("moe_intermediate_size",intermediate)
        shared=kind == "qwen2_moe" ? n("shared_expert_intermediate_size") : kind == "deepseek" ? n("n_shared_experts")*expertIntermediate : 0
        firstDense=n("first_k_dense_replace");frequency=n("moe_layer_freq",1)
        epsilon=(c["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6;theta=(c["rope_theta"] as? NSNumber)?.floatValue ?? (kind == "qwen2_moe" ? 1000000 : 10000)
        let scaling=c["rope_scaling"] as? [String:Any];ropeScale=scaling?["type"] as? String == "linear" ? 1/((scaling?["factor"] as? NSNumber)?.floatValue ?? 1) : 1
        embeddingScale=kind == "minicpm" ? (c["scale_emb"] as? NSNumber)?.floatValue ?? 1 : 1
        residualScale=kind == "minicpm" ? ((c["scale_depth"] as? NSNumber)?.floatValue ?? 1)/sqrt(Float(max(1,layers))) : 1
        outputScale=kind == "minicpm" ? Float(n("dim_model_base",256))/Float(max(1,hidden)) : 1
        tied=c["tie_word_embeddings"] as? Bool ?? (kind == "minicpm")
        guard ["qwen2_moe","deepseek","minicpm"].contains(kind),hidden>0,layers>0,heads>0,kvHeads>0,hidden.isMultiple(of:heads),heads.isMultiple(of:kvHeads),experts>1,active>0,active<=experts,frequency>0,intermediate>0,expertIntermediate>0 else {throw ModelLibraryError.invalid("Invalid classic MoE dimensions or routing.")}
        guard (c["hidden_act"] as? String ?? "silu") == "silu",(c["scoring_func"] as? String ?? "softmax") == "softmax",c["norm_topk_prob"] as? Bool != true || kind == "minicpm" else {throw ModelLibraryError.incompatible("Unsupported classic MoE activation or route normalization.")}
    }
}
private final class ClassicAttention:Module {
    @ModuleInfo(key:"q_proj") var q:Linear
    @ModuleInfo(key:"k_proj") var k:Linear
    @ModuleInfo(key:"v_proj") var v:Linear
    @ModuleInfo(key:"o_proj") var o:Linear
    let c:ClassicMoEConfiguration,rope:RoPE,scale:Float
    init(_ c:ClassicMoEConfiguration) {
        self.c=c;let dim=c.hidden/c.heads
        _q.wrappedValue=Linear(c.hidden,c.hidden,bias:c.kind == "qwen2_moe")
        _k.wrappedValue=Linear(c.hidden,c.kvHeads*dim,bias:c.kind == "qwen2_moe")
        _v.wrappedValue=Linear(c.hidden,c.kvHeads*dim,bias:c.kind == "qwen2_moe")
        _o.wrappedValue=Linear(c.hidden,c.hidden,bias:false)
        rope=RoPE(dimensions:dim,traditional:false,base:c.theta,scale:c.ropeScale);scale=pow(Float(dim),-0.5)
    }
    func callAsFunction(_ x:MLXArray,mask:MLXFast.ScaledDotProductAttentionMaskMode,cache:KVCache?)->MLXArray {
        let b=x.dim(0),l=x.dim(1)
        let query=applyRotaryPosition(rope,to:q(x).reshaped(b,l,c.heads,-1).transposed(0,2,1,3),offset:cache?.ropeOffset)
        let key=applyRotaryPosition(rope,to:k(x).reshaped(b,l,c.kvHeads,-1).transposed(0,2,1,3),offset:cache?.ropeOffset)
        let value=v(x).reshaped(b,l,c.kvHeads,-1).transposed(0,2,1,3)
        return o(attentionWithCacheUpdate(queries:query,keys:key,values:value,cache:cache,scale:scale,mask:mask).transposed(0,2,1,3).reshaped(b,l,-1))
    }
}
private final class ClassicMLP:Module,UnaryLayer {
    @ModuleInfo(key:"gate_proj") var gate:Linear
    @ModuleInfo(key:"up_proj") var up:Linear
    @ModuleInfo(key:"down_proj") var down:Linear
    init(_ hidden:Int,_ intermediate:Int) {_gate.wrappedValue=Linear(hidden,intermediate,bias:false);_up.wrappedValue=Linear(hidden,intermediate,bias:false);_down.wrappedValue=Linear(intermediate,hidden,bias:false)}
    func callAsFunction(_ x:MLXArray)->MLXArray {down(silu(gate(x))*up(x))}
}
private final class ClassicRawGate:Module {
    @ParameterInfo var weight:MLXArray
    init(_ c:ClassicMoEConfiguration) {_weight.wrappedValue=MLXArray.zeros([c.experts,c.hidden])}
    func callAsFunction(_ x:MLXArray)->MLXArray {matmul(x,weight.T)}
}
private final class ClassicExperts:Module,UnaryLayer {
    @ModuleInfo(key:"gate") var gate:Module
    @ModuleInfo(key:"switch_mlp") var bank:SwitchGLU
    @ModuleInfo(key:"shared_expert") var sharedQwen:ClassicMLP?
    @ModuleInfo(key:"shared_experts") var sharedDeepseek:ClassicMLP?
    @ModuleInfo(key:"shared_expert_gate") var sharedGate:Linear?
    let c:ClassicMoEConfiguration
    init(_ c:ClassicMoEConfiguration) {
        self.c=c
        _gate.wrappedValue=c.kind == "deepseek" ? ClassicRawGate(c) : Linear(c.hidden,c.experts,bias:false)
        _bank.wrappedValue=SwitchGLU(inputDims:c.hidden,hiddenDims:c.expertIntermediate,numExperts:c.experts)
        if c.shared>0 {
            if c.kind == "qwen2_moe" {_sharedQwen.wrappedValue=ClassicMLP(c.hidden,c.shared);_sharedGate.wrappedValue=Linear(c.hidden,1,bias:false)}
            else {_sharedDeepseek.wrappedValue=ClassicMLP(c.hidden,c.shared)}
        }
    }
    func callAsFunction(_ x:MLXArray)->MLXArray {
        let logits=(gate as? Linear).map{$0(x)} ?? (gate as! ClassicRawGate)(x)
        let scores=softmax(logits.asType(.float32),axis:-1,precise:true)
        let indices=stopGradient(argPartition(-scores,kth:c.active-1,axis:-1)[.ellipsis,..<c.active])
        var weights=takeAlong(scores,indices,axis:-1)
        if c.kind == "minicpm" {weights=weights/sum(weights,axis:-1,keepDims:true)}
        let output=bank(x,indices)
        var result=sum(output*weights.asType(output.dtype).expandedDimensions(axis:-1),axis:-2)
        if let sharedQwen,let sharedGate {result=result+sigmoid(sharedGate(x))*sharedQwen(x)}
        if let sharedDeepseek {result=result+sharedDeepseek(x)}
        return result
    }
}
private final class ClassicBlock:Module {
    @ModuleInfo(key:"self_attn") var attention:ClassicAttention
    @ModuleInfo(key:"mlp") var mlp:Module
    @ModuleInfo(key:"input_layernorm") var norm1:RMSNorm
    @ModuleInfo(key:"post_attention_layernorm") var norm2:RMSNorm
    let scale:Float
    init(_ c:ClassicMoEConfiguration,_ index:Int) {
        _attention.wrappedValue=ClassicAttention(c)
        _mlp.wrappedValue=index>=c.firstDense && index.isMultiple(of:c.frequency) ? ClassicExperts(c) : ClassicMLP(c.hidden,c.intermediate)
        _norm1.wrappedValue=RMSNorm(dimensions:c.hidden,eps:c.epsilon);_norm2.wrappedValue=RMSNorm(dimensions:c.hidden,eps:c.epsilon);scale=c.residualScale
    }
    func callAsFunction(_ x:MLXArray,mask:MLXFast.ScaledDotProductAttentionMaskMode,cache:KVCache?)->MLXArray {
        let h=x+attention(norm1(x),mask:mask,cache:cache)*scale
        let y=(mlp as? ClassicExperts).map{$0(norm2(h))} ?? (mlp as! ClassicMLP)(norm2(h))
        return h+y*scale
    }
}
private final class ClassicBackbone:Module {
    @ModuleInfo(key:"embed_tokens") var embedding:Embedding
    @ModuleInfo var layers:[ClassicBlock]
    let norm:RMSNorm
    let c:ClassicMoEConfiguration
    init(_ c:ClassicMoEConfiguration) {self.c=c;_embedding.wrappedValue=Embedding(embeddingCount:c.vocabulary,dimensions:c.hidden);_layers.wrappedValue=(0..<c.layers).map{ClassicBlock(c,$0)};norm=RMSNorm(dimensions:c.hidden,eps:c.epsilon)}
    func callAsFunction(_ tokens:MLXArray,cache:[KVCache]?)->MLXArray {
        var h=embedding(tokens)*c.embeddingScale
        let mask=createAttentionMask(h:h,cache:cache?.first)
        for (index,layer) in layers.enumerated() {h=layer(h,mask:mask,cache:cache?[index])}
        return norm(h)*c.outputScale
    }
}
final class ClassicMoEModel:Module,LLMModel,KVCacheDimensionProvider {
    var loraLayers:[Module] {backbone.layers}
    let vocabularySize:Int,kvHeads:[Int],c:ClassicMoEConfiguration
    @ModuleInfo(key:"model") private var backbone:ClassicBackbone
    @ModuleInfo(key:"lm_head") private var head:Linear?
    init(_ c:ClassicMoEConfiguration) {self.c=c;vocabularySize=c.vocabulary;kvHeads=Array(repeating:c.kvHeads,count:c.layers);_backbone.wrappedValue=ClassicBackbone(c);if !c.tied {_head.wrappedValue=Linear(c.hidden,c.vocabulary,bias:false)}}
    func callAsFunction(_ tokens:MLXArray,cache:[KVCache]?)->MLXArray {let h=backbone(tokens,cache:cache);return head.map{$0(h)} ?? backbone.embedding.asLinear(h)}
    func sanitize(weights:[String:MLXArray])->[String:MLXArray] {
        var result=weights
        if c.kind == "minicpm" {
            for name in weights.keys where name.contains(".mlp.experts.") {
                for (source,target) in [("w1","gate_proj"),("w2","down_proj"),("w3","up_proj")] where name.contains("."+source+".") {
                    result[name.replacingOccurrences(of:"."+source+".",with:"."+target+".")]=result.removeValue(forKey:name)
                }
            }
        }
        if c.tied {for key in result.keys where key.hasPrefix("lm_head.") {result[key]=nil}}
        for layer in 0..<c.layers {for projection in ["gate_proj","up_proj","down_proj"] {for suffix in ["weight","scales","biases"] {
            let prefix="model.layers.\(layer).mlp",names=(0..<c.experts).map{"\(prefix).experts.\($0).\(projection).\(suffix)"}
            if names.allSatisfy({result[$0] != nil}) {let arrays=names.map{result.removeValue(forKey:$0)!};result["\(prefix).switch_mlp.\(projection).\(suffix)"]=stacked(arrays)}
        }}}
        return result.filter{!$0.key.contains("rotary_emb.inv_freq")}
    }
}
