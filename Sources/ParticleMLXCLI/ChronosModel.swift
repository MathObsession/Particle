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

struct QuantizedExpert {
    struct Projection {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        var bits:Int = 3
        var groupSize:Int = 64
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Match the reference gathered operator even for one resident
            // expert, retaining bounded native-quantized weights.
            gatherQuantizedMM(x.expandedDimensions(axes:[-2,-3]),weight.expandedDimensions(axis:0),
                scales:scales.expandedDimensions(axis:0),biases:biases.expandedDimensions(axis:0),
                rhsIndices:MLXArray.zeros([x.dim(0),1],dtype:.uint32),transpose:true,groupSize:groupSize,bits:bits,
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
                biases:stacked(projections.map(\.biases)),rhsIndices:selected,transpose:true,groupSize:projections[0].groupSize,bits:projections[0].bits,sortedIndices:sortedKernel)
        }
        let hidden = compiledSiluProduct(project(input,\.gate),project(input,\.up))
        var output=project(hidden,\.down)
        if let inverse {output=scatterUnsort(x:output,invOrder:inverse,shape:indices.shape)}
        return output.squeezed(axis:-2)
    }
    static func upstream(_ x: MLXArray, experts: [Self], indices: MLXArray, bankIndices:[Int]? = nil,totalExperts:Int? = nil) throws -> MLXArray {
        let hidden = experts[0].gate.weight.dim(0),input = x.dim(-1)
        let block = SwitchGLU(inputDims:input,hiddenDims:hidden,numExperts:totalExperts ?? experts.count)
        var modules=[(String,Module)]()
        for (name,inputDims,outputDims) in [("gate_proj",input,hidden),("up_proj",input,hidden),("down_proj",hidden,input)] {
            // Preserve upstream projection types and shape metadata without
            // abandoning unevaluated full-size quantization graphs. Those
            // retained ~600 small Metal buffers per reference model step.
            let placeholder=SwitchLinear(inputDims:inputDims,outputDims:outputDims,numExperts:totalExperts ?? experts.count,
                weight:MLXArray.zeros([1,128],dtype:.bfloat16),bias:nil)
            let source=experts[0][keyPath:name == "gate_proj" ? \Self.gate : name == "up_proj" ? \Self.up : \Self.down]
            let projection=QuantizedSwitchLinear(placeholder,groupSize:source.groupSize,bits:source.bits)
            eval(projection.parameters());modules.append((name,projection))
        }
        try block.update(modules:.unflattened(modules),verify:.all)
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
        // The bounded placeholders intentionally change shape here. Native
        // tensor dimensions are validated by the pack reader before use.
        try block.update(parameters:.unflattened(weights),verify:[.allModelKeysSet,.noUnusedKeys])
        return block(x,indices)
    }
}

/// Owns quantized arrays, not reusable pointers. Eviction cannot overwrite a GPU consumer.
final class ExpertStore {
    let reader: ParticleV2Reader
    let slots: Int
    var prefix:String {reader.manifest.model.architecture == .qwen3MoE ? "" : "language_model."}
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
            let prefix = "\(self.prefix)model.layers.\(layer).mlp.switch_mlp.\(name)"
            func tensor(_ suffix: String) throws -> MLXArray {
                guard let record = reader.tensorIndex[prefix+suffix],record.layer == layer else {throw ModelLibraryError.invalid("Expert projection metadata is missing.")}
                let bytes = data.subdata(in:Int(record.offset)..<Int(record.offset+record.bytes))
                return MLXArray(bytes,Array(record.shape.dropFirst()),dtype:try Self.dtype(record.dtype))
            }
            let w=try tensor(".weight"),scale=try tensor(".scales"),bias=try tensor(".biases")
            let defaultBits=reader.manifest.model.quantization.bits ?? 3
            let bits=reader.manifest.model.quantization.overrides[prefix] ?? defaultBits
            let input=w.dim(-1)*32/bits
            return .init(weight:w,scales:scale,biases:bias,bits:bits,groupSize:input/scale.dim(-1))
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
    override func items() -> ModuleItems {ModuleItems(values:[:])}
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
            let result=store.reference ? try QuantizedExpert.upstream(x,experts:weights,indices:indices,bankIndices:selected,totalExperts:store.reader.manifest.model.experts) : QuantizedExpert.gathered(x,experts:weights,indices:compact,totalExperts:store.reader.manifest.model.experts)
            eval(result);return result
        } catch {
            store.failure=error
            return MLXArray.zeros(indices.shape+[x.dim(-1)],dtype:x.dtype)
        }
    }
}

@MainActor final class ChronosModel {
    let model: any LLMModel
    let schema:MoEModelSchema
    let reader: ParticleV2Reader
    private let store: ExpertStore
    private(set) var cache: [KVCache] = []
    private var checkpoint: [(Int,KVCache)] = []
    private(set) var checkpointPosition = 0
    private(set) var tokens = [Int]()
    private(set) var currentLogits: MLXArray?
    private var checkpointLogits: MLXArray?
    private var evaluations=0
    private var promptRowsLimit=128
    var expertReadBytes: Int64 {store.readBytes}
    var expertHits: Int {store.hits}
    init(path: URL, slots: Int, activeExperts: Int, readWorkers: Int = 2, cancellation: ParticleCancellation? = nil) throws {
        guard (8...32).contains(slots),(1...64).contains(activeExperts),[1,2,4].contains(readWorkers) else {throw ModelLibraryError.invalid("Invalid Chronos cache, expert count or read-worker setting.")}
        reader = try ParticleV2Reader(path:path)
        guard [.qwen35MoE,.qwen3MoE].contains(reader.manifest.model.architecture),activeExperts <= reader.manifest.model.activeExperts else {throw ModelLibraryError.incompatible("This worker needs a supported MLX MoE pack and a valid expert selection.")}
        let commonBytes=reader.manifest.tensors.filter{$0.layer == nil}.reduce(Int64(0)){$0+$1.bytes}
        let strides=Dictionary(grouping:reader.manifest.tensors.filter{$0.layer != nil},by:{$0.layer!}).values.compactMap{$0.first?.expertStride}.reduce(0,+)
        guard commonBytes+strides*Int64(slots)+512*1024*1024 < Int64(ProcessInfo.processInfo.physicalMemory)*3/4 else {throw ModelLibraryError.incompatible("The resident weights and selected expert cache exceed this Mac’s memory allowance. Choose a smaller model/quantization or fewer cache slots.")}
        let maximumStride=reader.manifest.tensors.compactMap(\.expertStride).max() ?? 1
        let transientBudget=min(Int64(2_147_483_648),Int64(ProcessInfo.processInfo.physicalMemory)/4)
        let selectedCapacity=Int(transientBudget/(2*maximumStride))
        guard selectedCapacity >= activeExperts else {throw ModelLibraryError.incompatible("This expert selection exceeds the temporary buffer budget. Choose fewer active experts or a smaller model.")}
        if selectedCapacity < reader.manifest.model.experts {promptRowsLimit=max(1,min(128,selectedCapacity/activeExperts))}
        try ArchitectureRegistry.validateChronosLayout(reader.manifest)
        try reader.verifyResidentFiles()
        try ArchitectureRegistry.validateChronosTokenizer(tokenizer:ArchitectureRegistry.boundedData(path.appendingPathComponent("tokenizer/tokenizer.json"),limit:64*1024*1024),configuration:ArchitectureRegistry.boundedData(path.appendingPathComponent("tokenizer/tokenizer_config.json"),limit:1_048_576))
        let config = try ArchitectureRegistry.boundedData(path.appendingPathComponent("config.json"),limit:1_048_576)
        let records = Dictionary(uniqueKeysWithValues:reader.manifest.tensors.map{record in
            (record.name,SafetensorRecord(name:record.name,dtype:record.dtype,shape:record.shape,offset:0,bytes:record.bytes))
        })
        let template = try String(contentsOf:path.appendingPathComponent("tokenizer/chat_template.jinja"),encoding:.utf8)
        let report = try ArchitectureRegistry.inspectMLX(configuration:config,tensors:records,template:template,repository:nil,revision:nil,sourceBytes:reader.manifest.model.sourceBytes)
        guard report.conversionSupported,let checked=report.descriptor,
              checked.architecture == reader.manifest.model.architecture,checked.layers == reader.manifest.model.layers,
              checked.experts == reader.manifest.model.experts,checked.activeExperts == reader.manifest.model.activeExperts,
              checked.quantization == reader.manifest.model.quantization else {throw ModelLibraryError.incompatible(report.reason)}
        schema=try MoEModelSchema(configuration:config)
        var object=schema.root,text=schema.text
        text["num_experts_per_tok"]=activeExperts
        if schema.architecture == .qwen3MoE {
            text["head_dim"]=schema.headDim
            text["intermediate_size"]=text["intermediate_size"] ?? schema.intermediate
            text["decoder_sparse_step"]=text["decoder_sparse_step"] ?? 1
            text["mlp_only_layers"]=text["mlp_only_layers"] ?? [Int]()
        }
        if schema.architecture == .qwen35MoE {object["text_config"]=text} else {object=text}
        if schema.architecture == .qwen3MoE,var rope=object["rope_scaling"] as? [String:Any],rope["type"] == nil,let kind=rope["rope_type"] {rope["type"]=kind;object["rope_scaling"]=rope}
        let runtimeConfig=try JSONSerialization.data(withJSONObject:object)
        MLX.compile(enable:false)
        switch schema.architecture {
        case .qwen35MoE:model=Qwen35MoEModel(try JSONDecoder().decode(Qwen35Configuration.self,from:runtimeConfig))
        case .qwen3MoE:model=Qwen3MoEModel(try JSONDecoder().decode(Qwen3MoEConfiguration.self,from:runtimeConfig))
        default:throw ModelLibraryError.incompatible("No MLX streaming graph for this pack.")
        }
        store=ExpertStore(reader:reader,slots:slots)
        store.cancelled={cancellation?.isCancelled ?? false}
        let originals=Dictionary(uniqueKeysWithValues:model.namedModules())
        for layer in 0..<schema.layers {
            let key=schema.prefix+"model.layers.\(layer).mlp"
            guard let original=originals[key],let unary=original as? any UnaryLayer else {throw ModelLibraryError.invalid("The bundled MoE graph does not expose the required router.")}
            try original.update(modules:.unflattened([("switch_mlp",BoundedReferenceSwitch(store:store,layer:layer,readWorkers:readWorkers) as Module)]),verify:.none)
            store.referenceBlocks[layer]=unary
        }
        // Keep the original router, normalization, shared branch and reduction.
        // Replace only the expert-bank leaf with demand-streamed quantized rows.
        guard !model.parameters().flattened().contains(where:{$0.0.contains(".switch_mlp.")}) else {throw ModelLibraryError.invalid("Routed tensors must not remain resident.")}
        var weights = [String:MLXArray]()
        for tensor in reader.manifest.tensors where tensor.layer == nil {
            let (_,data) = try reader.residentTensor(tensor.name)
            weights[tensor.name] = MLXArray(data,tensor.shape,dtype:try ExpertStore.dtype(tensor.dtype))
        }
        weights = model.sanitize(weights:weights)
        quantize(model:model,filter:{name,_ -> (groupSize:Int,bits:Int,mode:QuantizationMode)? in
            guard weights[name+".scales"] != nil,let q=self.schema.quantization(for:name) else{return nil}
            return (q.group,q.bits,.affine)
        })
        try model.update(parameters:.unflattened(weights),verify:.all)
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
            for (index,state) in cache.enumerated() where schema.fullAttentionLayers.contains(index) {
                let amount = state.offset-checkpointPosition
                guard amount >= 0,state.trim(amount) == amount else {try reset();return 0}
            }
            tokens = Array(tokens.prefix(checkpointPosition));currentLogits = checkpointLogits;return checkpointPosition
        }
        try reset();return 0
    }
    func saveCheckpoint() {
        checkpoint.removeAll(keepingCapacity:true)
        checkpoint = cache.enumerated().filter{!schema.fullAttentionLayers.contains($0.offset)}.map{($0.offset,$0.element.copy())}
        checkpointPosition = tokens.count
        checkpointLogits = currentLogits
    }
    func evaluate(_ input: [Int]) throws -> MLXArray {
        guard !input.isEmpty,input.count <= 128 else {throw ModelLibraryError.invalid("Invalid prompt chunk size.")}
        if input.count > promptRowsLimit {
            var last:MLXArray?
            for offset in stride(from:0,to:input.count,by:promptRowsLimit) {last=try evaluate(Array(input[offset..<min(input.count,offset+promptRowsLimit)]))}
            return last!
        }
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
