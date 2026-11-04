import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import ParticleModelLibrary
import ParticleCore

private final class ExpertReadTicket: @unchecked Sendable {
    private let condition = NSCondition()
    private var result: Result<Data,Error>?
    func finish(_ value: Result<Data,Error>) {condition.lock();result = value;condition.broadcast();condition.unlock()}
    func wait() throws -> Data {condition.lock();defer{condition.unlock()};while result == nil {condition.wait()};return try result!.get()}
}

private final class SharedExpert: Module, UnaryLayer {
    @ModuleInfo(key:"gate_proj") var gate: Linear
    @ModuleInfo(key:"up_proj") var up: Linear
    @ModuleInfo(key:"down_proj") var down: Linear
    override init() {
        _gate.wrappedValue = Linear(2048,512,bias:false)
        _up.wrappedValue = Linear(2048,512,bias:false)
        _down.wrappedValue = Linear(512,2048,bias:false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {down(silu(gate(x))*up(x))}
}

struct QuantizedExpert {
    struct Projection {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Match the reference gathered operator even for one resident
            // expert, retaining bounded native-quantized weights.
            gatherQuantizedMM(x.expandedDimensions(axes:[-2,-3]),weight.expandedDimensions(axis:0),
                scales:scales.expandedDimensions(axis:0),biases:biases.expandedDimensions(axis:0),
                rhsIndices:MLXArray.zeros([x.dim(0),1],dtype:.uint32),transpose:true,groupSize:64,bits:3,
                sortedIndices:true).squeezed(axes:[-2,-3])
        }
    }
    let gate: Projection
    let up: Projection
    let down: Projection
    func callAsFunction(_ x: MLXArray) -> MLXArray {down(silu(gate(x))*up(x))}
    static func gathered(_ x: MLXArray, experts: [Self], indices: MLXArray, totalExperts:Int? = nil) -> MLXArray {
        let sort=indices.size >= 64
        // MLX chooses its sorted GEMM from assignments / bank size. Compacting
        // selected experts must not accidentally change that arithmetic path.
        let sortedKernel=sort && (totalExperts == nil || indices.size/(totalExperts!) >= 4)
        var input=x.expandedDimensions(axes:[-2,-3]),selected=indices,inverse:MLXArray?
        if sort {let result=gatherSort(x:input,indices:indices);input=result.0;selected=result.1;inverse=result.2}
        func project(_ input: MLXArray, _ key: KeyPath<Self,Projection>) -> MLXArray {
            let projections = experts.map{$0[keyPath:key]}
            return gatherQuantizedMM(input,stacked(projections.map(\.weight)),scales:stacked(projections.map(\.scales)),
                biases:stacked(projections.map(\.biases)),rhsIndices:selected,transpose:true,groupSize:64,bits:3,sortedIndices:sortedKernel)
        }
        let hidden = compiledSiluProduct(project(input,\.gate),project(input,\.up))
        var output=project(hidden,\.down)
        if let inverse {output=scatterUnsort(x:output,invOrder:inverse,shape:indices.shape)}
        return output.squeezed(axis:-2)
    }
    static func upstream(_ x: MLXArray, experts: [Self], indices: MLXArray, bankIndices:[Int]? = nil,totalExperts:Int? = nil) throws -> MLXArray {
        let hidden = experts[0].gate.weight.dim(0),input = x.dim(-1)
        let block = SwitchGLU(inputDims:input,hiddenDims:hidden,numExperts:totalExperts ?? experts.count)
        quantize(model:block,groupSize:64,bits:3)
        var weights = [String:MLXArray]()
        for (name,key) in [("gate_proj",\Self.gate),("up_proj",\Self.up),("down_proj",\Self.down)] {
            let projections = experts.map{$0[keyPath:key]}
            func bank(_ values:[MLXArray])->MLXArray {
                guard let bankIndices,let totalExperts else{return stacked(values)}
                let result=MLXArray.zeros([totalExperts]+values[0].shape,dtype:values[0].dtype)
                result[MLXArray(bankIndices.map(Int32.init))]=stacked(values)
                return result
            }
            weights[name+".weight"] = bank(projections.map(\.weight))
            weights[name+".scales"] = bank(projections.map(\.scales))
            weights[name+".biases"] = bank(projections.map(\.biases))
        }
        try block.update(parameters:.unflattened(weights),verify:.all)
        return block(x,indices)
    }
}

/// Owns quantized arrays, not reusable pointers. Eviction cannot overwrite a GPU consumer.
final class ExpertStore {
    let reader: ParticleV2Reader
    let slots: Int
    var failure: Error?
    var readBytes: Int64 = 0
    var hits = 0
    var reference = false
    var gatheredDecode = false
    var referenceBlocks = [Int:any UnaryLayer]()
    var cancelled: @Sendable ()->Bool = {false}
    private let readQueue = DispatchQueue(label:"app.particle.chronos.demand-reads",qos:.userInitiated,attributes:.concurrent)
    private var pending = [Int:ExpertReadTicket]()
    private var values = [Int:[Int:QuantizedExpert]]()
    private var frequency = [Int:[Int:Int]]()
    init(reader: ParticleV2Reader, slots: Int) {self.reader = reader;self.slots = slots}
    static func dtype(_ value: String) throws -> DType {
        switch value {
        case "U32": .uint32
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        default: throw ModelLibraryError.invalid("Unsupported MLX tensor dtype: \(value).")
        }
    }
    func get(layer: Int, expert: Int) throws -> QuantizedExpert {
        if cancelled() {throw CancellationError()}
        if let value = values[layer]?[expert] {frequency[layer,default:[:]][expert,default:0] += 1;hits += 1;return value}
        let data = try pending.removeValue(forKey:expert)?.wait() ?? reader.expert(layer:layer,expert:expert);readBytes += Int64(data.count)
        func projection(_ name: String) throws -> QuantizedExpert.Projection {
            let prefix = "language_model.model.layers.\(layer).mlp.switch_mlp.\(name)"
            func tensor(_ suffix: String) throws -> MLXArray {
                guard let record = reader.tensorIndex[prefix+suffix],record.layer == layer else {throw ModelLibraryError.invalid("Expert projection metadata is missing.")}
                let bytes = data.subdata(in:Int(record.offset)..<Int(record.offset+record.bytes))
                return MLXArray(bytes,Array(record.shape.dropFirst()),dtype:try Self.dtype(record.dtype))
            }
            return try .init(weight:tensor(".weight"),scales:tensor(".scales"),biases:tensor(".biases"))
        }
        let value = try QuantizedExpert(gate:projection("gate_proj"),up:projection("up_proj"),down:projection("down_proj"))
        if values[layer,default:[:]].count >= slots,
           let victim = values[layer]?.keys.min(by: {(frequency[layer]?[$0] ?? 0) < (frequency[layer]?[$1] ?? 0)}) {
            values[layer]?[victim] = nil;frequency[layer]?[victim] = nil
        }
        values[layer,default:[:]][expert] = value;frequency[layer,default:[:]][expert] = 1
        return value
    }
    func isResident(layer: Int, expert: Int) -> Bool {values[layer]?[expert] != nil}
    func prefetch(layer: Int, selected: [Int], workers: Int) {
        for expert in selected where pending.count < workers && pending[expert] == nil && !isResident(layer:layer,expert:expert) {
            let ticket = ExpertReadTicket(),reader = reader;pending[expert] = ticket
            readQueue.async {ticket.finish(Result{try reader.expert(layer:layer,expert:expert)})}
        }
    }
    func finishLayer() {
        for ticket in pending.values {if let data = try? ticket.wait() {readBytes += Int64(data.count)}}
        pending.removeAll(keepingCapacity:true)
    }
}

/// Keep the upstream MoE router/shared-branch code as the reference while
/// replacing only its whole-expert-bank leaf with bounded selected weights.
/// Tiny unused superclass placeholders are never evaluated or expanded.
private final class BoundedReferenceSwitch: SwitchGLU {
    unowned let store: ExpertStore
    let layer: Int
    let readWorkers: Int
    init(store:ExpertStore,layer:Int,readWorkers:Int) {
        self.store=store;self.layer=layer;self.readWorkers=readWorkers
        super.init(inputDims:1,hiddenDims:1,numExperts:1)
    }
    override func callAsFunction(_ x:MLXArray,_ indices:MLXArray)->MLXArray {
        defer{store.finishLayer()}
        do {
            let routing=indices.asType(.int32).asArray(Int32.self)
            let selected=Set(routing.map(Int.init)).sorted()
            store.prefetch(layer:layer,selected:selected,workers:readWorkers)
            var weights=[QuantizedExpert]()
            for (position,expert) in selected.enumerated() {
                weights.append(try store.get(layer:layer,expert:expert))
                store.prefetch(layer:layer,selected:Array(selected.dropFirst(position+1)),workers:readWorkers)
            }
            let mapping=Dictionary(uniqueKeysWithValues:selected.enumerated().map{($0.element,UInt32($0.offset))})
            let compact=MLXArray(routing.map{mapping[Int($0)]!}).reshaped(indices.shape)
            let result=store.reference ? try QuantizedExpert.upstream(x,experts:weights,indices:indices,bankIndices:selected,totalExperts:256) : QuantizedExpert.gathered(x,experts:weights,indices:compact,totalExperts:256)
            eval(result);return result
        } catch {
            store.failure=error
            return MLXArray.zeros(indices.shape+[x.dim(-1)],dtype:x.dtype)
        }
    }
}

private final class StreamedMoE: Module, UnaryLayer {
    @ModuleInfo(key:"gate") var gate: Linear
    @ModuleInfo(key:"shared_expert") var shared: Module
    @ModuleInfo(key:"shared_expert_gate") var sharedGate: Linear
    private let store: ExpertStore
    private let layer: Int
    private let topK: Int
    private let readWorkers: Int
    init(store: ExpertStore, layer: Int, topK: Int, readWorkers: Int, shared:Module) {
        self.store = store;self.layer = layer;self.topK = topK
        self.readWorkers = readWorkers
        _gate.wrappedValue = Linear(2048,256,bias:false)
        _shared.wrappedValue = shared
        _sharedGate.wrappedValue = Linear(2048,1,bias:false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard store.failure == nil else {return MLXArray.zeros(x.shape,dtype:x.dtype)}
        if store.reference || store.gatheredDecode,let block=store.referenceBlocks[layer] {return block(x)}
        defer{store.finishLayer()}
        do {
            let probabilities = softmax(gate(x),axis:-1,precise:true)
            // Reference top-k chain from MLX Swift LM's MoERouterTopK.swift (MIT).
            // Its fused helper is package-private; preserve the reference semantics here.
            let kth = probabilities.dim(-1)-topK
            let indices = argPartition(probabilities,kth:kth,axis:-1)[.ellipsis,kth...]
            let selected = takeAlong(probabilities,indices,axis:-1)
            let scores = selected/selected.sum(axis:-1,keepDims:true)
            let rows = x.size/2048,flat = x.reshaped(rows,2048)
            let routing = indices.reshaped(rows,topK).asType(.int32).asArray(Int32.self)
            let rankOutputs = (0..<topK).map{_ in MLXArray.zeros([rows,2048],dtype:x.dtype)}
            let sharedOutput = sigmoid(sharedGate(x))*(shared as! any UnaryLayer)(x)
            asyncEval(sharedOutput)
            if store.reference {
                let selected = Set(routing.map(Int.init)).sorted()
                store.prefetch(layer:layer,selected:selected,workers:readWorkers)
                var weights = [QuantizedExpert]()
                for (position,expert) in selected.enumerated() {
                    weights.append(try store.get(layer:layer,expert:expert))
                    store.prefetch(layer:layer,selected:Array(selected.dropFirst(position+1)),workers:readWorkers)
                }
                let mapping = Dictionary(uniqueKeysWithValues:selected.enumerated().map{($0.element,UInt32($0.offset))})
                let compact = MLXArray(routing.map{mapping[Int($0)]!}).reshaped(rows,topK)
                let projected = try QuantizedExpert.upstream(flat,experts:weights,indices:compact)
                let reduced = weightedExpertSum(projected,scores.reshaped(rows,topK));eval(reduced)
                return reduced.reshaped(x.shape)+sharedOutput
            }
            if rows == 1 && store.gatheredDecode {
                var results = [MLXArray]()
                for row in 0..<rows {
                    let selected = (0..<topK).map{Int(routing[row*topK+$0])}
                    store.prefetch(layer:layer,selected:selected,workers:readWorkers)
                    var weights = [QuantizedExpert]()
                    for (position,expert) in selected.enumerated() {
                        weights.append(try store.get(layer:layer,expert:expert))
                        store.prefetch(layer:layer,selected:Array(selected.dropFirst(position+1)),workers:readWorkers)
                    }
                    let compact = MLXArray((0..<topK).map(UInt32.init)).reshaped(1,topK)
                    let projected = QuantizedExpert.gathered(flat[row..<(row+1)],experts:weights,indices:compact)
                    let result = weightedExpertSum(projected,scores.reshaped(rows,topK)[row..<(row+1)])
                    eval(result);results.append(result)
                }
                return concatenated(results,axis:0).reshaped(x.shape)+sharedOutput
            }
            // Group every selected row by expert. Each fetched expert serves the whole chunk.
            var groups = [Int:[(Int,Int)]]()
            for row in 0..<rows {for rank in 0..<topK {groups[Int(routing[row*topK+rank]),default:[]].append((row,rank))}}
            let resident = groups.keys.filter{store.isResident(layer:layer,expert:$0)}.sorted()
            let missing = groups.keys.filter{!store.isResident(layer:layer,expert:$0)}.sorted()
            let ordered = resident+missing
            store.prefetch(layer:layer,selected:missing,workers:readWorkers)
            for (position,expert) in ordered.enumerated() {
                let assignments = groups[expert]!,weights = try store.get(layer:layer,expert:expert)
                store.prefetch(layer:layer,selected:Array(ordered.dropFirst(position+1)),workers:readWorkers)
                let rowIDs = MLXArray(assignments.map{Int32($0.0)})
                let result = weights(flat[rowIDs]);eval(result)
                for (index,assignment) in assignments.enumerated() {rankOutputs[assignment.1][assignment.0] = result[index]}
            }
            let combined = (stacked(rankOutputs,axis:1)*scores.reshaped(rows,topK,1)).sum(axis:1).reshaped(x.shape)
            return combined+sharedOutput
        } catch {
            // UnaryLayer cannot throw. The runner checks this before accepting any logits/state.
            store.failure = error;return MLXArray.zeros(x.shape,dtype:x.dtype)
        }
    }
}

@MainActor final class ChronosModel {
    let model: Qwen35MoEModel
    let reader: ParticleV2Reader
    private let store: ExpertStore
    private(set) var cache: [KVCache] = []
    private var checkpoint: [(Int,KVCache)] = []
    private(set) var checkpointPosition = 0
    private(set) var tokens = [Int]()
    private(set) var currentLogits: MLXArray?
    private var checkpointLogits: MLXArray?
    private var evaluations=0
    var expertReadBytes: Int64 {store.readBytes}
    var expertHits: Int {store.hits}
    init(path: URL, slots: Int, activeExperts: Int, readWorkers: Int = 2, cancellation: ParticleCancellation? = nil) throws {
        guard (8...32).contains(slots),(2...8).contains(activeExperts),[1,2,4].contains(readWorkers) else {throw ModelLibraryError.invalid("Invalid Chronos cache, expert count or read-worker setting.")}
        reader = try ParticleV2Reader(path:path)
        guard reader.manifest.model.architecture == .qwen35MoE else {throw ModelLibraryError.incompatible("This worker requires a Qwen3.5 MoE pack.")}
        try ArchitectureRegistry.validateChronosLayout(reader.manifest)
        try reader.verifyResidentFiles()
        try ArchitectureRegistry.validateChronosTokenizer(tokenizer:ArchitectureRegistry.boundedData(path.appendingPathComponent("tokenizer/tokenizer.json"),limit:64*1024*1024),configuration:ArchitectureRegistry.boundedData(path.appendingPathComponent("tokenizer/tokenizer_config.json"),limit:1_048_576))
        let config = try ArchitectureRegistry.boundedData(path.appendingPathComponent("config.json"),limit:1_048_576)
        let records = Dictionary(uniqueKeysWithValues:reader.manifest.tensors.map{record in
            (record.name,SafetensorRecord(name:record.name,dtype:record.dtype,shape:record.shape,offset:0,bytes:record.bytes))
        })
        let template = try String(contentsOf:path.appendingPathComponent("tokenizer/chat_template.jinja"),encoding:.utf8)
        let report = try ArchitectureRegistry.inspectMLX(configuration:config,tensors:records,template:template,repository:nil,revision:nil,sourceBytes:reader.manifest.model.sourceBytes)
        guard report.conversionSupported else {throw ModelLibraryError.incompatible(report.reason)}
        var runtimeConfig=config
        if activeExperts != 8 {
            guard var object=try JSONSerialization.jsonObject(with:config) as? [String:Any],var text=object["text_config"] as? [String:Any] else {throw ModelLibraryError.invalid("Missing runtime routing configuration.")}
            // Explicit experimental routing preference; native source metadata
            // and quantized checkpoint bytes remain unchanged.
            text["num_experts_per_tok"]=activeExperts;object["text_config"]=text
            runtimeConfig=try JSONSerialization.data(withJSONObject:object)
        }
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self,from:runtimeConfig)
        // Host demand reads and evaluated router IDs cannot be captured inside a static graph.
        MLX.compile(enable:false)
        model = Qwen35MoEModel(configuration)
        store = ExpertStore(reader:reader,slots:slots)
        store.cancelled = {cancellation?.isCancelled ?? false}
        let originals=Dictionary(uniqueKeysWithValues:model.namedModules())
        var referenceModules=[Int:Module]()
        for layer in 0..<40 {
            let key="language_model.model.layers.\(layer).mlp"
            guard let original=originals[key],let unary=original as? any UnaryLayer else {throw ModelLibraryError.invalid("The upstream reference MoE graph changed.")}
            try original.update(modules:.unflattened([("switch_mlp",BoundedReferenceSwitch(store:store,layer:layer,readWorkers:readWorkers) as Module)]),verify:.none)
            referenceModules[layer]=original;store.referenceBlocks[layer]=unary
        }
        let replacements: [(String,Module)] = try (0..<40).map{layer in
            let key="language_model.model.layers.\(layer).mlp"
            guard let shared=originals[key+".shared_expert"],shared is any UnaryLayer else {throw ModelLibraryError.invalid("The upstream shared-expert graph changed.")}
            return(key,StreamedMoE(store:store,layer:layer,topK:activeExperts,readWorkers:readWorkers,shared:shared))
        }
        try model.update(modules:.unflattened(replacements),verify:.none)
        guard !model.parameters().flattened().contains(where: {$0.0.contains(".switch_mlp.")}) else {throw ModelLibraryError.invalid("Routed weights were not replaced before model evaluation.")}
        var weights = [String:MLXArray]()
        for tensor in reader.manifest.tensors where tensor.layer == nil {
            let (_,data) = try reader.residentTensor(tensor.name)
            weights[tensor.name] = MLXArray(data,tensor.shape,dtype:try ExpertStore.dtype(tensor.dtype))
        }
        weights = model.sanitize(weights:weights)
        let overrides = reader.manifest.model.quantization.overrides
        quantize(model:model,filter:{name,_ -> (groupSize:Int,bits:Int,mode:QuantizationMode)? in
            weights[name+".scales"] == nil ? nil : (64,overrides[name] ?? 3,.affine)
        })
        try model.update(parameters:.unflattened(weights),verify:.all)
        let residentModules=Dictionary(uniqueKeysWithValues:model.namedModules())
        for (layer,original) in referenceModules {
            let prefix="language_model.model.layers.\(layer).mlp."
            let replacements=try ["gate","shared_expert","shared_expert_gate"].map {name -> (String,Module) in
                guard let module=residentModules[prefix+name] else {throw ModelLibraryError.invalid("A resident reference module is missing.")};return(name,module)
            }
            try original.update(modules:.unflattened(replacements),verify:.none)
        }
        eval(model.parameters());try model.prepare();cache = try model.newCache(parameters:nil)
    }
    func reset() throws {cache = try model.newCache(parameters:nil);checkpoint = [];tokens = [];checkpointPosition = 0;store.failure = nil;currentLogits = nil;checkpointLogits = nil}
    func setReference(_ enabled: Bool) {store.reference = enabled}
    func setGatheredExecution(_ enabled: Bool) {store.gatheredDecode = enabled}
    func rewindForPrompt(_ prompt: [Int]) throws -> Int {
        let common = zip(prompt,tokens).prefix(while:{$0 == $1}).count
        if common == tokens.count {return common}
        if checkpointPosition > 0,common >= checkpointPosition {
            for (index,state) in checkpoint {cache[index] = state.copy()}
            for (index,state) in cache.enumerated() where (index+1).isMultiple(of:4) {
                let amount = state.offset-checkpointPosition
                guard amount >= 0,state.trim(amount) == amount else {try reset();return 0}
            }
            tokens = Array(tokens.prefix(checkpointPosition));currentLogits = checkpointLogits;return checkpointPosition
        }
        try reset();return 0
    }
    func saveCheckpoint() {
        checkpoint.removeAll(keepingCapacity:true)
        checkpoint = cache.enumerated().filter{!($0.offset+1).isMultiple(of:4)}.map{($0.offset,$0.element.copy())}
        checkpointPosition = tokens.count
        checkpointLogits = currentLogits
    }
    func evaluate(_ input: [Int]) throws -> MLXArray {
        guard !input.isEmpty,input.count <= 128 else {throw ModelLibraryError.invalid("Invalid prompt chunk size.")}
        return try autoreleasepool {
            let output=model(MLXArray(input.map(Int32.init)).expandedDimensions(axis:0),cache:cache)
            let last=output[0,-1]
            // Match upstream TokenIterator: settle functional cache updates as
            // well as logits, so old activation graphs cannot stay reachable.
            // Reference module constructors also advance otherwise-unused lazy
            // random keys; evaluate the key without changing its sequence.
            eval([last]+cache.flatMap{$0.state}+MLXRandom.globalState.innerState())
            if let error=store.failure {throw error}
            tokens += input;currentLogits=last;evaluations += 1
            if evaluations.isMultiple(of:256) {Memory.clearCache()}
            return last
        }
    }
}
