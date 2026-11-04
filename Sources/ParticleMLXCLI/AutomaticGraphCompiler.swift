import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import ParticleModelLibrary

/// Discovers typed expert operations in any graph supplied by the bundled MLX
/// registry. There is no model-name or architecture switch in this compiler.
@MainActor enum AutomaticGraphCompiler {
    static func configuration(_ data:Data,experts:Int? = nil,fields:[String] = []) throws -> (String,Data,[String:Any]) {
        guard data.count <= 1_048_576,var root=try JSONSerialization.jsonObject(with:data) as? [String:Any],let type=root["model_type"] as? String else {throw ModelLibraryError.invalid("Missing graph model_type.")}
        func scan(_ object:[String:Any],depth:Int=0) throws {
            guard depth <= 12 else {throw ModelLibraryError.invalid("Graph configuration nesting is excessive.")}
            for (key,value) in object {
                if key == "auto_map",!(value is NSNull) {throw ModelLibraryError.incompatible("This model requires repository code. Export a supported graph; Particle will not execute downloaded model code.")}
                if let nested=value as? [String:Any] {try scan(nested,depth:depth+1)}
                if let number=value as? Int,["hidden_size","num_attention_heads","num_key_value_heads","num_hidden_layers","head_dim","vocab_size","num_experts","num_local_experts"].contains(key) {
                    guard number > 0,number <= 1_048_576 else {throw ModelLibraryError.invalid("Invalid graph dimension: \(key).")}
                    if key == "num_hidden_layers",number > 512 {throw ModelLibraryError.incompatible("The graph exceeds the supported layer bound.")}
                    if ["num_experts","num_local_experts"].contains(key),number > 4096 {throw ModelLibraryError.incompatible("The graph exceeds the supported expert-count bound.")}
                    if key == "hidden_size",number > 32768 {throw ModelLibraryError.incompatible("The graph exceeds the supported hidden-dimension bound.")}
                }
            }
            if let heads=object["num_attention_heads"] as? Int,let kv=object["num_key_value_heads"] as? Int, !heads.isMultiple(of:kv) {throw ModelLibraryError.invalid("Query heads must be divisible by KV heads.")}
        }
        try scan(root)
        if let experts {
            func replace(_ path:ArraySlice<String>,_ object:inout [String:Any]) {
                guard let first=path.first else{return}
                if path.count == 1 {object[first]=experts}
                else if var nested=object[first] as? [String:Any] {replace(path.dropFirst(),&nested);object[first]=nested}
            }
            for field in fields {replace(field.split(separator:".").map(String.init)[...],&root)}
        }
        return (type,try JSONSerialization.data(withJSONObject:root),root)
    }
    static func compile(_ request:ParticleGraphRequest) async throws -> ParticleGraphPlan {
        guard !request.tensors.isEmpty,request.tensors.count <= 100_000,Set(request.tensors.map(\.name)).count == request.tensors.count else {throw ModelLibraryError.invalid("Invalid graph tensor inventory.")}
        let (type,config,root)=try configuration(request.configuration)
        guard await LLMTypeRegistry.shared.contains(type) else {throw ModelLibraryError.incompatible("The bundled graph backend has no implementation for \(type). A portable graph/operator extension is required; no model weights were downloaded.")}
        let model=try await LLMTypeRegistry.shared.createModel(configuration:config,modelType:type)
        let modules=Dictionary(uniqueKeysWithValues:model.namedModules())
        let parameters=Dictionary(uniqueKeysWithValues:model.parameters().flattened())
        let normalization=try SourceTensorNormalizer.normalize(model:model,records:request.tensors)
        let sources=normalization.tensors
        let base=try JSONDecoder().decode(BaseConfiguration.self,from:request.configuration)
        let expertModules=modules.filter{$0.value is SwitchLinear && ($0.value.parameters().flattened().first(where:{$0.0 == "weight"})?.1.shape.first ?? 0) > 1}.sorted{$0.key < $1.key}
        guard !expertModules.isEmpty else {throw ModelLibraryError.incompatible("This graph exposes no supported expert matrix operations. Dense models or custom expert operators cannot use this streaming pass.")}
        let parents=Array(Set(expertModules.map{String($0.key.split(separator:".").dropLast().joined(separator:"."))})).sorted()
        let bankIDs=Dictionary(uniqueKeysWithValues:parents.enumerated().map{($0.element,$0.offset)})
        var projections=[ParticleGraphPlan.Projection](),claimed=Set<String>()
        func floating(_ record:SafetensorRecord)->Bool {["F16","BF16","F32"].contains(record.dtype)}
        for (name,array) in parameters.sorted(by:{$0.key < $1.key}) {
            if name.hasSuffix(".weight"),let module=modules[String(name.dropLast(7))],module is Quantizable,let weight=sources[name],let scales=sources[String(name.dropLast(7))+".scales"] {
                let path=String(name.dropLast(7))
                guard let q=base.perLayerQuantization?.quantization(layer:path) ?? (base.perLayerQuantization == nil ? base.quantization : nil) else {throw ModelLibraryError.incompatible("Missing native quantization contract for \(path).")}
                let bits=q.bits,group=q.groupSize,mode=q.mode
                let valid = mode == .affine ? [2,3,4,5,6,8].contains(bits) && [32,64,128].contains(group) : mode == .nvfp4 ? bits == 4 && group == 16 : (bits == (mode == .mxfp4 ? 4 : 8) && group == 32)
                guard valid,array.ndim >= 2,array.dim(-1).isMultiple(of:group) else {throw ModelLibraryError.incompatible("Invalid native quantization operation at \(path).")}
                // Ask the backend primitive for its storage layout. This creates
                // only a tiny lazy shape probe, never full-size float weights.
                let probe=MLX.quantized(MLXArray.zeros([1,group],dtype:.bfloat16),groupSize:group,bits:bits,mode:mode)
                var packed=array.shape;packed[packed.count-1]=array.dim(-1)/group*probe.wq.dim(-1)
                var scaleShape=array.shape;scaleShape[scaleShape.count-1]=array.dim(-1)/group*probe.scales.dim(-1)
                let scaleTypeOK=[DType.float16,.bfloat16,.float32].contains(probe.scales.dtype) ? floating(scales) : (try? ExpertStore.dtype(scales.dtype)) == probe.scales.dtype
                guard (try? ExpertStore.dtype(weight.dtype)) == probe.wq.dtype,weight.shape == packed,scales.shape == scaleShape,scaleTypeOK else {throw ModelLibraryError.incompatible("Native packed storage does not match graph matrix \(path).")}
                claimed.formUnion([name,path+".scales"])
                if probe.biases != nil {
                    guard let biases=sources[path+".biases"],biases.shape == scaleShape,biases.dtype == scales.dtype else {throw ModelLibraryError.incompatible("Missing affine offsets for \(path).")}
                    claimed.insert(path+".biases")
                }
            } else {
                guard let source=sources[name],source.shape == array.shape,floating(source) else {throw ModelLibraryError.incompatible("Checkpoint tensor does not match the backend graph: \(name). Export the backend’s native tensor layout first.")}
                claimed.insert(name)
            }
        }
        guard claimed == Set(sources.keys) else {throw ModelLibraryError.incompatible("Checkpoint has tensors outside the compiled graph: \(Set(sources.keys).subtracting(claimed).sorted().prefix(3).joined(separator:", ")). No tensors were silently discarded.")}
        for (path,module) in expertModules {
            guard let weight=parameters[path+".weight"],weight.ndim == 3 else {throw ModelLibraryError.incompatible("An expert projection is not a rank-three bank: \(path).")}
            let parent=path.split(separator:".").dropLast().joined(separator:".")
            let q=base.perLayerQuantization?.quantization(layer:path) ?? (base.perLayerQuantization == nil ? base.quantization : nil),quantized=sources[path+".scales"] != nil
            projections.append(.init(path:path,bank:bankIDs[parent]!,input:weight.dim(2),output:weight.dim(1),experts:weight.dim(0),
                bits:quantized ? q?.bits : nil,
                groupSize:quantized ? q?.groupSize : nil,bias:module.parameters().flattened().contains{$0.0 == "bias"},mode:q?.mode.rawValue ?? "affine"))
        }
        var routing=[String:Int]()
        func findRouting(_ object:[String:Any],prefix:String="") {
            for (key,value) in object {
                if let nested=value as? [String:Any],key != "quantization",key != "quantization_config" {findRouting(nested,prefix:prefix+key+".")}
                if ["num_experts_per_tok","num_experts_per_token","experts_per_token","moe_top_k","moe_topk"].contains(key),let count=value as? Int,(1...64).contains(count) {routing[prefix+key]=count}
            }
        }
        findRouting(root)
        guard let native=routing.values.first,Set(routing.values).count == 1 else {throw ModelLibraryError.incompatible("The graph’s expert selection is not represented by a single configurable top-k field. Export its routing contract before enabling this pass.")}
        let text=root["text_config"] as? [String:Any] ?? root
        let cache=try model.newCache(parameters:nil)
        let layers=text["num_hidden_layers"] as? Int ?? text["num_layers"] as? Int ?? max(1,cache.count)
        let hidden=text["hidden_size"] as? Int ?? projections[0].input
        let heads=max(1,text["num_attention_heads"] as? Int ?? 1),kvHeads=max(1,text["num_key_value_heads"] as? Int ?? heads)
        let headDim=max(1,text["head_dim"] as? Int ?? hidden/heads)
        let full=max(1,cache.filter{$0.isTrimmable}.count)
        var expertBytes:Int64=0
        let routed=Set(projections.flatMap(\.tensors))
        for bank in Set(projections.map(\.bank)) {
            var offset:Int64=0
            let names=projections.filter{$0.bank == bank}.flatMap(\.tensors).sorted()
            for name in names {let record=sources[name]!;let count=Int64(projections.first(where:{$0.bank == bank})!.experts);offset=(offset+255)/256*256+record.bytes/count}
            expertBytes += (offset+16383)/16384*16384
        }
        let resident=sources.values.filter{!routed.contains($0.name)}.reduce(Int64(0)){$0+$1.bytes}
        let plan=ParticleGraphPlan(modelType:type,configurationSHA256:ParticleGraphPlan.digest(request.configuration),tensorsSHA256:try ParticleGraphPlan.tensorDigest(Array(sources.values)),projections:projections,routingFields:routing.keys.sorted(),nativeExperts:native,hiddenLayers:layers,
            contextLimit:min(262144,max(4096,text["max_position_embeddings"] as? Int ?? 32768)),kvBytesPerToken:Int64(full)*Int64(kvHeads)*Int64(headDim)*8,cacheTypes:cache.map{String(describing:Swift.type(of:$0))},expertBytesPerSlot:expertBytes,residentBytes:resident,sourceTensorsSHA256:try ParticleGraphPlan.tensorDigest(request.tensors),assemblies:normalization.assemblies)
        try plan.validate();return plan
    }
}
