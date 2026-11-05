import Foundation
import MLX
import MLXNN
import MLXLMCommon
import ParticleModelLibrary

/// Trace the trusted backend's weight-normalization graph without evaluating
/// any model tensor. Only byte-preserving rename/reshape/leading-concat passes
/// are lowered to file-copy ranges; other transformations are reported.
@MainActor enum SourceTensorNormalizer {
    struct Result {let tensors:[String:SafetensorRecord];let assemblies:[ParticleGraphPlan.Assembly];let archived:[String]}
    static func normalize(model:any LanguageModel,records:[SafetensorRecord]) throws -> Result {
        let original=Dictionary(uniqueKeysWithValues:records.map{($0.name,$0)})
        let namer=NodeNamer();var probes=[String:MLXArray](),symbols=[String:SafetensorRecord](),sourceSymbols=[String:String]()
        for (index,record) in records.sorted(by:{$0.name < $1.name}).enumerated() {
            let probe=MLXArray.zeros(record.shape,dtype:try ExpertStore.dtype(record.dtype)),symbol="source_\(index)"
            namer.setName(symbol,for:probe);symbols[symbol]=record;sourceSymbols[record.name]=symbol;probes[record.name]=probe
        }
        let routed=Set(model.namedModules().filter{$0.1 is SwitchLinear}.flatMap{path,_ in [".weight",".scales",".biases",".bias"].map{path+$0}})
        let normalized=model.sanitize(weights:probes)
        var result=[String:SafetensorRecord](),assemblies=[ParticleGraphPlan.Assembly](),claimed=Set<String>()
        struct Node {let operation:String;let inputs:[String];let shape:[Int]}
        let pattern=try NSRegularExpression(pattern:"([A-Za-z0-9_]+) \\[\\(([0-9, ]*)\\), [^\\]]+\\]")
        func arrays(_ text:String)->[(String,[Int])] {
            pattern.matches(in:text,range:NSRange(text.startIndex...,in:text)).compactMap{match in
                guard let name=Range(match.range(at:1),in:text),let shape=Range(match.range(at:2),in:text) else{return nil}
                return (String(text[name]),text[shape].split(separator:",").compactMap{Int($0.trimmingCharacters(in:.whitespaces))})
            }
        }
        for (name,array) in normalized.sorted(by:{$0.key < $1.key}) {
            if let source=original[name],source.shape == array.shape,(!routed.contains(name) || namer.name(for:array) == sourceSymbols[name]) {
                // Common value transformations remain deferred to the original
                // model.sanitize at load time; the pack keeps the source bytes.
                result[name]=source;claimed.insert(name);continue
            }
            let output=namer.name(for:array)!
            var nodes=[String:Node]()
            let description=graphDescription([array],namer:namer)
            guard description.utf8.count <= 8*1024*1024 else {throw ModelLibraryError.invalid("Normalization graph exceeds its metadata limit.")}
            for line in description.split(separator:"\n") {
                let parts=line.components(separatedBy:" -> ");guard parts.count == 2 else {continue}
                let operation=String(parts[0].prefix(while:{$0 != " "})),inputs=arrays(parts[0])
                for (id,shape) in arrays(parts[1]) {nodes[id]=Node(operation:operation,inputs:inputs.map(\.0),shape:shape)}
                for (id,shape) in inputs where nodes[id] == nil {nodes[id]=Node(operation:"Input",inputs:[],shape:shape)}
            }
            var visiting=Set<String>()
            func resolve(_ id:String) throws -> [SafetensorRecord] {
                if let source=symbols[id] {return [source]}
                guard visiting.insert(id).inserted,let node=nodes[id] else {throw ModelLibraryError.incompatible("The backend's normalization graph could not be resolved for \(name).")};defer{visiting.remove(id)}
                if ["Reshape","ExpandDims","Squeeze","Copy","StopGradient"].contains(node.operation),node.inputs.count == 1 {return try resolve(node.inputs[0])}
                if node.operation == "Concatenate",!node.inputs.isEmpty {
                    let shapes=node.inputs.compactMap{nodes[$0]?.shape ?? symbols[$0]?.shape}
                    guard shapes.count == node.inputs.count,!node.shape.isEmpty,shapes.allSatisfy({$0.count == node.shape.count && Array($0.dropFirst()) == Array(node.shape.dropFirst())}),shapes.reduce(0,{$0+$1[0]}) == node.shape[0] else {throw ModelLibraryError.incompatible("Normalization requires a non-leading concatenation for \(name).")}
                    return try node.inputs.flatMap{try resolve($0)}
                }
                throw ModelLibraryError.incompatible("The normalization compiler needs a byte-preserving implementation of \(node.operation) for \(name). No values were guessed or requantized.")
            }
            let sources=try resolve(output)
            guard !sources.isEmpty,Set(sources.map(\.name)).count == sources.count,sources.allSatisfy({$0.dtype == sources[0].dtype}) else {throw ModelLibraryError.incompatible("Normalization has an ambiguous source tensor mapping.")}
            let bytes=sources.reduce(Int64(0)){$0+$1.bytes},record=SafetensorRecord(name:name,dtype:sources[0].dtype,shape:array.shape,offset:0,bytes:bytes)
            guard claimed.isDisjoint(with:sources.map(\.name)) else {throw ModelLibraryError.incompatible("The normalizer aliases a source tensor into multiple destinations; a shared-storage lowering is required.")}
            result[name]=record;claimed.formUnion(sources.map(\.name))
            assemblies.append(.init(target:record,sources:sources.map{.init(name:$0.name,dtype:$0.dtype,shape:$0.shape,offset:0,bytes:$0.bytes)}))
        }
        // Keep non-text, MTP and backend-redundant source tensors in a separate
        // checksummed archive. The trusted backend deliberately omits them;
        // retaining their bytes avoids silent loss or RAM residency.
        var archived=[String]()
        for (index,name) in Set(original.keys).subtracting(claimed).sorted().enumerated() {
            let source=original[name]!,target="__particle_archive.\(index)"
            guard result[target] == nil else {throw ModelLibraryError.invalid("Reserved archive tensor name collision.")}
            let record=SafetensorRecord(name:target,dtype:source.dtype,shape:source.shape,offset:0,bytes:source.bytes)
            result[target]=record;archived.append(target);assemblies.append(.init(target:record,sources:[.init(name:source.name,dtype:source.dtype,shape:source.shape,offset:0,bytes:source.bytes)]))
        }
        return Result(tensors:result,assemblies:assemblies,archived:archived)
    }
}
