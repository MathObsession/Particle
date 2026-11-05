import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import ParticleModelLibrary
import ParticleCore

@MainActor protocol ParticleStreamedModel:AnyObject {
    var reader:ParticleV2Reader {get}
    var tokens:[Int] {get}
    var currentLogits:MLXArray? {get}
    var checkpointPosition:Int {get}
    var expertReadBytes:Int64 {get}
    func reset() throws
    func setReference(_ enabled:Bool)
    func setGatheredExecution(_ enabled:Bool)
    func rewindForPrompt(_ prompt:[Int]) throws -> Int
    func saveCheckpoint()
    func evaluate(_ input:[Int]) throws -> MLXArray
}
extension ChronosModel:ParticleStreamedModel {}

@MainActor final class CompiledGraphModel:ParticleStreamedModel {
    let reader:ParticleV2Reader
    let model:any LanguageModel
    let store:GraphExpertStore
    private var cache=[KVCache](),checkpoint=[Int:KVCache](),trimmed=Set<Int>()
    private(set) var checkpointPosition=0
    private(set) var tokens=[Int]()
    private(set) var currentLogits:MLXArray?
    private var checkpointLogits:MLXArray?
    private var rowLimit=128
    var expertReadBytes:Int64 {store.readBytes}
    init(path:URL,slots:Int,activeExperts:Int,readWorkers:Int,cancellation:ParticleCancellation?) async throws {
        reader=try ParticleV2Reader(path:path)
        guard let plan=reader.manifest.model.graph,reader.manifest.versionMajor == 3,reader.manifest.model.expertRange.contains(activeExperts) else {throw ModelLibraryError.invalid("The graph pack has no valid execution contract or expert preference.")}
        try plan.validate()
        let config=try ArchitectureRegistry.boundedData(path.appendingPathComponent("config.json"),limit:1_048_576)
        guard plan.configurationSHA256 == ParticleGraphPlan.digest(config) else {throw ModelLibraryError.checksum("Graph configuration")}
        let headers=reader.manifest.tensors.map{t in SafetensorRecord(name:t.name,dtype:t.dtype,shape:t.shape,offset:0,bytes:t.bytes*Int64(t.layer.flatMap{plan.banks[$0]} ?? 1))}
        guard plan.tensorsSHA256 == (try ParticleGraphPlan.tensorDigest(headers)) else {throw ModelLibraryError.checksum("Graph tensor inventory")}
        let strides=Dictionary(grouping:reader.manifest.tensors.filter{$0.layer != nil},by:{$0.layer!}).values.compactMap{$0.first?.expertStride}
        let common=reader.manifest.tensors.filter{$0.layer == nil && !(plan.archivedTensors ?? []).contains($0.name)}.reduce(Int64(0)){$0+$1.bytes}
        let memory=Int64(ProcessInfo.processInfo.physicalMemory)
        guard common+Dictionary(grouping:reader.manifest.tensors.filter{$0.layer != nil},by:{$0.layer!}).reduce(Int64(0)){sum,pair in sum+(pair.value.first?.expertStride ?? 0)*Int64(min(slots,plan.banks[pair.key] ?? slots))}+512*1024*1024 < memory*3/4 else {throw ModelLibraryError.incompatible("Resident graph weights and expert slots exceed this Mac's memory allowance. Choose fewer slots or a smaller model.")}
        let maximum=strides.max() ?? 1,budget=min(Int64(2_147_483_648),memory/4),capacity=Int(budget/(2*maximum))
        guard capacity >= activeExperts else {throw ModelLibraryError.incompatible("The graph's selected expert matrices exceed its staging budget.")}
        if capacity < (plan.banks.values.max() ?? 0) {rowLimit=max(1,min(128,capacity/activeExperts))}
        try reader.verifyResidentFiles()
        let (type,runtime,_)=try AutomaticGraphCompiler.configuration(config,experts:activeExperts,fields:plan.routingFields)
        guard type == plan.modelType,await ParticleGraphFactory.contains(type) else {throw ModelLibraryError.incompatible("The graph backend is unavailable in this engine version.")}
        MLX.compile(enable:false)
        model=try await ParticleGraphFactory.create(runtime,type:type)
        let boundedReaders=min(readWorkers,max(1,Int(min(memory/16,Int64(512*1024*1024))/maximum)))
        store=GraphExpertStore(reader:reader,slots:slots,readWorkers:boundedReaders);store.cancellation={cancellation?.isCancelled ?? false}
        let modules=Dictionary(uniqueKeysWithValues:model.namedModules())
        var replacements=[(String,Module)]()
        for projection in plan.projections {
            guard let module=modules[projection.path],module is SwitchLinear,let weight=module.parameters().flattened().first(where:{$0.0 == "weight"})?.1,
                  weight.shape == [projection.experts,projection.output,projection.input] else {throw ModelLibraryError.invalid("The compiled projection differs from the bundled backend graph.")}
            replacements.append((projection.path,PagedGraphLinear(store:store,projection:projection)))
        }
        // MLX recurses into module arrays only when every visited slot is a
        // dictionary. Preserve untouched dense layers as empty updates instead
        // of .none holes, without trying to replace immutable norm modules.
        func filled(_ item:NestedItem<String,Module>)->NestedItem<String,Module> {
            switch item {
            case .dictionary(let values):return .dictionary(values.mapValues(filled))
            case .array(let values):
                let recursive=values.contains{if case .dictionary=$0 {return true};return false}
                return .array(values.map{value in if recursive,case .none=value {return .dictionary([:])};return filled(value)})
            default:return item
            }
        }
        let update:ModuleChildren = .unflattened(replacements)
        try model.update(modules:ModuleChildren(values:Dictionary(uniqueKeysWithValues:update.keys.map{($0,filled(update[$0]!))})),verify:.none)
        var weights=[String:MLXArray]()
        for tensor in reader.manifest.tensors where tensor.layer == nil && !(plan.archivedTensors ?? []).contains(tensor.name) {let (_,data)=try reader.residentTensor(tensor.name);weights[tensor.name]=MLXArray(data,tensor.shape,dtype:try ExpertStore.dtype(tensor.dtype))}
        weights=model.sanitize(weights:weights)
        let base=try JSONDecoder().decode(BaseConfiguration.self,from:runtime)
        quantize(model:model,filter:{name,_ -> (groupSize:Int,bits:Int,mode:QuantizationMode)? in
            guard weights[name+".scales"] != nil else{return nil}
            return (base.perLayerQuantization?.quantization(layer:name) ?? (base.perLayerQuantization == nil ? base.quantization : nil))?.asTuple
        })
        try model.update(parameters:.unflattened(weights),verify:.all)
        eval(model.parameters());try model.prepare();cache=try model.newCache(parameters:nil)
    }
    func reset() throws {cache=try model.newCache(parameters:nil);tokens=[];checkpoint=[:];trimmed=[];checkpointPosition=0;currentLogits=nil;checkpointLogits=nil;store.failure=nil;store.clearLeases()}
    func setReference(_ enabled:Bool) {store.reference=enabled}
    func setGatheredExecution(_ enabled:Bool) {}
    func saveCheckpoint() {
        trimmed=Set(cache.indices.filter{cache[$0].isTrimmable})
        checkpoint=Dictionary(uniqueKeysWithValues:cache.enumerated().filter{!trimmed.contains($0.offset)}.map{($0.offset,$0.element.copy())})
        checkpointPosition=tokens.count;checkpointLogits=currentLogits
    }
    func rewindForPrompt(_ prompt:[Int]) throws -> Int {
        let common=zip(prompt,tokens).prefix(while:{$0 == $1}).count
        if common == tokens.count {return common}
        if checkpointPosition > 0,common >= checkpointPosition {
            for (index,state) in checkpoint {cache[index]=state.copy()}
            for index in trimmed {
                let amount=cache[index].offset-checkpointPosition
                guard cache[index].isTrimmable,amount >= 0,cache[index].trim(amount) == amount else {try reset();return 0}
            }
            tokens=Array(tokens.prefix(checkpointPosition));currentLogits=checkpointLogits;return checkpointPosition
        }
        try reset();return 0
    }
    func evaluate(_ input:[Int]) throws -> MLXArray {
        guard !input.isEmpty,input.count <= 128 else {throw ModelLibraryError.invalid("Invalid graph prompt chunk.")}
        if input.count > rowLimit {
            var last:MLXArray?;for start in stride(from:0,to:input.count,by:rowLimit) {last=try evaluate(Array(input[start..<min(input.count,start+rowLimit)]))};return last!
        }
        return try autoreleasepool {
            defer{store.clearLeases()}
            let output=model(MLXArray(input.map(Int32.init)).expandedDimensions(axis:0),cache:cache),last=output[0,-1]
            eval([last]+cache.flatMap{$0.state})
            if let error=store.failure {throw error}
            tokens+=input;currentLogits=last;return last
        }
    }
}
