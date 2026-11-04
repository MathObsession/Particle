import Foundation
import MLX
import MLXNN
import MLXLMCommon
import ParticleModelLibrary

private final class GraphReadTicket:@unchecked Sendable {
    private let condition=NSCondition()
    private var result:Result<Data,Error>?
    func finish(_ result:Result<Data,Error>) {condition.lock();self.result=result;condition.broadcast();condition.unlock()}
    func wait() throws -> Data {condition.lock();defer{condition.unlock()};while result == nil {condition.wait()};return try result!.get()}
}
/// The rewrite happens at the matrix operator, preserving the enclosing graph's
/// router, activation, bias, clamp, shared branches and reductions unchanged.
final class GraphExpertStore {
    struct Entry {let arrays:[String:MLXArray]}
    struct Lease {let selected:[Int];var consumed:Set<String>;let entries:[Int:Entry];let bytes:Int64}
    let reader:ParticleV2Reader
    let plan:ParticleGraphPlan
    let slots:Int
    let readWorkers:Int
    private let readQueue=DispatchQueue(label:"app.particle.graph-demand-reads",qos:.userInitiated,attributes:.concurrent)
    private var pending=[String:GraphReadTicket]()
    private var accesses=0
    var cancellation:@Sendable ()->Bool = {false}
    var failure:Error?
    var reference=false
    var readBytes:Int64=0
    var hits=0
    private var entries=[Int:[Int:Entry]](),frequency=[Int:[Int:Int]](),leases=[Int:Lease]()
    init(reader:ParticleV2Reader,slots:Int,readWorkers:Int) {self.reader=reader;plan=reader.manifest.model.graph!;self.slots=slots;self.readWorkers=readWorkers}
    private func prefetch(bank:Int,selected:[Int]) {
        for expert in selected where pending.count < readWorkers && entries[bank]?[expert] == nil {
            let key="\(bank):\(expert)";if pending[key] != nil {continue}
            let ticket=GraphReadTicket(),reader=reader,cancelled=cancellation;pending[key]=ticket
            readQueue.async{ticket.finish(Result{try reader.expert(layer:bank,expert:expert,cancelled:cancelled)})}
        }
    }
    private func load(bank:Int,expert:Int) throws -> Entry {
        if cancellation() {throw CancellationError()}
        accesses += 1
        if accesses.isMultiple(of:256) {for bank in frequency.keys {for expert in frequency[bank]!.keys {frequency[bank]![expert]=max(1,frequency[bank]![expert]!/2)}}}
        if let cached=entries[bank]?[expert] {frequency[bank,default:[:]][expert,default:0] += 1;hits += 1;return cached}
        let data=try pending.removeValue(forKey:"\(bank):\(expert)")?.wait() ?? reader.expert(layer:bank,expert:expert,cancelled:cancellation);readBytes += Int64(data.count)
        var arrays=[String:MLXArray]()
        for record in reader.manifest.tensors where record.layer == bank {
            let bytes=data.subdata(in:Int(record.offset)..<Int(record.offset+record.bytes))
            arrays[record.name]=MLXArray(bytes,Array(record.shape.dropFirst()),dtype:try ExpertStore.dtype(record.dtype))
        }
        let entry=Entry(arrays:arrays)
        if entries[bank,default:[:]].count >= slots,let victim=entries[bank]?.keys.min(by:{(frequency[bank]?[$0] ?? 0) < (frequency[bank]?[$1] ?? 0)}) {entries[bank]?[victim]=nil;frequency[bank]?[victim]=nil}
        entries[bank,default:[:]][expert]=entry;frequency[bank,default:[:]][expert]=1
        return entry
    }
    func project(_ x:MLXArray,indices:MLXArray,projection p:ParticleGraphPlan.Projection,sorted:Bool) throws -> MLXArray {
        if cancellation() {throw CancellationError()}
        let routes=indices.asType(.int32).asArray(Int32.self).map(Int.init),selected=Array(Set(routes)).sorted()
        guard !selected.isEmpty,selected.allSatisfy({(0..<p.experts).contains($0)}) else {throw ModelLibraryError.invalid("The graph router selected an out-of-range expert.")}
        let stride=reader.manifest.tensors.first(where:{$0.layer == p.bank})!.expertStride!
        let bytes=Int64(selected.count)*stride
        let budget=min(Int64(2_147_483_648),Int64(ProcessInfo.processInfo.physicalMemory)/4)
        guard bytes*2 <= budget else {throw ModelLibraryError.invalid("The graph's selected expert set exceeds its staging budget. Reduce the prompt chunk or active experts.")}
        if leases[p.bank]?.selected != selected || leases[p.bank]?.consumed.contains(p.path) == true {leases[p.bank]=nil}
        if leases[p.bank] == nil {
            // Every operation is evaluated before returning; evicting a staging
            // lease can cost another read but cannot invalidate a GPU consumer.
            while leases.values.reduce(Int64(0),{$0+$1.bytes})+bytes > budget/2,let victim=leases.keys.sorted().first {leases[victim]=nil}
            var fetched=[Int:Entry]();prefetch(bank:p.bank,selected:selected)
            for (index,expert) in selected.enumerated() {fetched[expert]=try load(bank:p.bank,expert:expert);prefetch(bank:p.bank,selected:Array(selected.dropFirst(index+1)))}
            leases[p.bank]=Lease(selected:selected,consumed:[],entries:fetched,bytes:bytes)
        }
        let held=leases[p.bank]!
        let mapping=Dictionary(uniqueKeysWithValues:selected.enumerated().map{($0.element,UInt32($0.offset))})
        let compact=MLXArray(routes.map{mapping[$0]!}).reshaped(indices.shape)
        let selectedIndices=reference ? indices : compact
        func bank(_ suffix:String) throws -> MLXArray {
            let arrays=try selected.map{expert -> MLXArray in guard let array=held.entries[expert]?.arrays[p.path+suffix] else {throw ModelLibraryError.invalid("A compiled expert tensor is missing.")};return array}
            if !reference {return stacked(arrays)}
            let size=arrays[0].nbytes*Int(p.experts)
            guard size <= Int(budget/2) else {throw ModelLibraryError.invalid("This diagnostic reference bank exceeds the bounded reference budget.")}
            let full=MLXArray.zeros([p.experts]+arrays[0].shape,dtype:arrays[0].dtype)
            full[MLXArray(selected.map(Int32.init))]=stacked(arrays);return full
        }
        let weights=try bank(".weight")
        var result:MLXArray
        if let bits=p.bits,let group=p.groupSize {
            // Keep MLX's full-bank sorted-kernel threshold after compaction.
            let sortedKernel=sorted && indices.size/max(1,p.experts) >= 4
            result=try gatherQuantizedMM(x,weights,scales:bank(".scales"),biases:p.mode == "affine" ? bank(".biases") : nil,rhsIndices:selectedIndices,transpose:true,groupSize:group,bits:bits,mode:QuantizationMode(rawValue:p.mode)!,sortedIndices:sortedKernel)
        } else {result=gatherMM(x,weights.swappedAxes(-1,-2),rhsIndices:selectedIndices)}
        if p.bias {let biases=try bank(".bias");result=result+biases[selectedIndices].expandedDimensions(axis:-2)}
        eval(result)
        leases[p.bank]?.consumed.insert(p.path)
        let paths=Set(plan.projections.filter{$0.bank == p.bank}.map(\.path))
        if leases[p.bank]?.consumed == paths {leases[p.bank]=nil}
        return result
    }
    func clearLeases() {leases.removeAll(keepingCapacity:true);for ticket in pending.values {if let data=try? ticket.wait(){readBytes += Int64(data.count)}};pending.removeAll(keepingCapacity:true)}
}
final class PagedGraphLinear:SwitchLinear {
    unowned let store:GraphExpertStore
    let projection:ParticleGraphPlan.Projection
    init(store:GraphExpertStore,projection:ParticleGraphPlan.Projection) {
        self.store=store;self.projection=projection
        super.init(inputDims:projection.input,outputDims:projection.output,numExperts:projection.experts,weight:MLXArray.zeros([1,1,1]),bias:nil)
    }
    override func items()->ModuleItems {ModuleItems(values:[:])}
    override func callAsFunction(_ x:MLXArray,_ indices:MLXArray,sortedIndices:Bool=false)->MLXArray {
        do {return try store.project(x,indices:indices,projection:projection,sorted:sortedIndices)}
        catch {store.failure=error;return MLXArray.zeros(indices.shape+[x.dim(-2),projection.output],dtype:x.dtype)}
    }
}
