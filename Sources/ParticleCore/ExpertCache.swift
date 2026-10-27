import Foundation
@preconcurrency import Metal
import ParticleGGUF

final class ExpertSlot: @unchecked Sendable {
    let buffer: MTLBuffer
    let gate: GPUWeight, up: GPUWeight, down: GPUWeight
    init(gpu: ParticleGPU, gate: GGUFTensor, up: GGUFTensor, down: GGUFTensor, packedStride: Int? = nil) throws {
        let upOffset = (gate.expertBytes + 255) / 256 * 256
        let downOffset = (upOffset + up.expertBytes + 255) / 256 * 256
        let required = downOffset + down.expertBytes
        guard packedStride == nil || packedStride! >= required else { throw ParticleError.invalid("Invalid packed expert stride") }
        buffer = try gpu.bytes(packedStride ?? required)
        memset(buffer.contents(),0,buffer.length)
        self.gate = GPUWeight(buffer: buffer, offset: 0, columns: gate.shape[0], rows: gate.shape[1], type: gate.type)
        self.up = GPUWeight(buffer: buffer, offset: upOffset, columns: up.shape[0], rows: up.shape[1], type: up.type)
        self.down = GPUWeight(buffer: buffer, offset: downOffset, columns: down.shape[0], rows: down.shape[1], type: down.type)
    }
}

/// Fixed-capacity, per-layer LFU cache. Three scattered GGUF projection ranges
/// fill one Metal-visible slot. No extra packed model or Swift weight array.
/// The runner waits for every GPU consumer before calling fetch again.
final class ExpertCache: @unchecked Sendable {
    private let file: any ParticleModelSource
    private let layer: Int
    private let diskCache: ExpertDiskCache?
    private let gate: GGUFTensor, up: GGUFTensor, down: GGUFTensor
    private let slots: [ExpertSlot]
    private var execution: ParticleExecutionOptions
    private var inFlight: ExpertFetch?
    private let verificationQueue: OperationQueue
    private var ids: [Int]
    private var useCounts = [Int](repeating: 0, count: 128)
    private var lastUse: [Int]
    private var clock = 0
    private(set) var readBytes = 0
    private(set) var sourceReadBytes = 0
    private(set) var hitCount = 0
    private(set) var missCount = 0
    private(set) var readWaitSeconds = 0.0
    var residentExperts: Set<Int> { Set(ids.filter { $0 >= 0 }) }
    var capacity: Int { slots.count }
    init(file: any ParticleModelSource, layer: Int, count: Int, gpu: ParticleGPU, diskCache: ExpertDiskCache? = nil, execution: ParticleExecutionOptions = .init()) throws {
        guard (6...16).contains(count) else { throw ParticleError.invalid("Expert cache slots must be in 6...16") }
        self.file = file; self.layer = layer; self.diskCache = diskCache; self.execution = execution
        let verifiers=OperationQueue(); verifiers.maxConcurrentOperationCount=2
        verifiers.qualityOfService = .userInitiated; verifiers.name="Particle verified expert slots"
        verificationQueue=verifiers
        let g = try file.tensor("blk.\(layer).ffn_gate_exps.weight")
        let u = try file.tensor("blk.\(layer).ffn_up_exps.weight")
        let d = try file.tensor("blk.\(layer).ffn_down_exps.weight")
        guard g.shape.count == 3, g.shape[2] == 128, u.shape == g.shape, d.shape == [g.shape[1],g.shape[0],128] else { throw ParticleError.invalid("Invalid expert tensor layout") }
        gate = g; up = u; down = d
        let packedStride = try file.expertRead(layer:layer,expert:0)?.range.count
        slots = try (0..<count).map { _ in try ExpertSlot(gpu: gpu, gate: g, up: u, down: d, packedStride:packedStride) }
        ids = [Int](repeating: -1, count: count); lastUse = [Int](repeating: 0, count: count)
    }
    func configure(_ options: ParticleExecutionOptions) throws {
        guard inFlight == nil else { throw ParticleError.invalid("Cannot reconfigure pinned expert reads") }
        execution=options
    }
    /// Park RAM entries before a deliberate clear or process shutdown. During
    /// active generation the SSD tier instead holds recent RAM evictions.
    func persist() {
        guard let diskCache else { return }
        for i in slots.indices where ids[i] >= 0 {
            autoreleasepool {
                diskCache.store(layer:layer,expert:ids[i],from:slots[i].buffer.contents(),count:slots[i].buffer.length)
            }
        }
    }
    func fetch(_ experts: [Int], frequencies: [Int]? = nil) throws -> [ExpertSlot] {
        try complete(beginFetch(experts,frequencies:frequencies))
    }
    /// Reserve every selected slot before any GPU consumer starts. The owner
    /// cannot reserve another batch until this transaction has completed.
    func beginFetch(_ experts: [Int], frequencies: [Int]? = nil,
                    cancellation: ParticleCancellation? = nil) throws -> ExpertFetch {
        let frequencies = frequencies ?? [Int](repeating:1,count:experts.count)
        guard inFlight == nil, !experts.isEmpty, experts.count <= slots.count, Set(experts).count == experts.count,
              experts.allSatisfy({ (0..<128).contains($0) }), frequencies.count == experts.count,
              frequencies.allSatisfy({ (1...1024).contains($0) }) else {
            throw ParticleError.invalid("Invalid routed expert selection or outstanding fetch")
        }
        diskCache?.record(layer:layer,experts:experts,frequencies:frequencies)
        let protected = Set(experts)
        var assigned: [Int] = [], ready: [Int] = [], misses: [(expert: Int, slot: Int, evicted: Int)] = []
        for (rank,expert) in experts.enumerated() {
            clock += 1; useCounts[expert] += frequencies[rank]
            let index: Int
            if let hit = ids.firstIndex(of:expert) { index = hit; hitCount += 1; ready.append(rank) }
            else {
                let candidates = ids.indices.filter { !protected.contains(ids[$0]) }
                guard let victim = candidates.min(by: {
                    let a = ids[$0] < 0 ? -1 : useCounts[ids[$0]], b = ids[$1] < 0 ? -1 : useCounts[ids[$1]]
                    return a == b ? lastUse[$0] < lastUse[$1] : a < b
                }) else { throw ParticleError.invalid("Cannot assign expert cache slot") }
                index = victim; misses.append((expert,index,ids[index])); ids[index] = expert; missCount += 1
            }
            lastUse[index] = clock; assigned.append(index)
        }
        let requests = misses
        let fetch = ExpertFetch(slots:assigned.map { slots[$0] },ready:ready,invalidated:requests.map(\.slot),
                                logicalBytes:requests.count*(gate.expertBytes+up.expertBytes+down.expertBytes))
        inFlight = fetch
        fetch.group.enter()
        DispatchQueue.global(qos:.userInitiated).async {
            let verification=DispatchGroup()
            // No partial fetch may release a slot while verification/fallback
            // is still writing it, including errors or cancellation.
            defer { verification.wait(); fetch.group.leave() }
            let sources = SourceMisses()
            // SSD checksums and reads are bounded independently of slow-source
            // reads. All mutable cache metadata remains owned by its lock.
            let workers = min(self.execution.ssdWorkers,requests.count)
            DispatchQueue.concurrentPerform(iterations:workers) { worker in
                for i in stride(from:worker,to:requests.count,by:workers) {
                    if cancellation?.isCancelled == true { fetch.errors.record(CancellationError()); break }
                    let request = requests[i], slot = self.slots[request.slot]
                    if self.diskCache?.exchange(layer:self.layer,expert:request.expert,evicted:request.evicted,
                        into:slot.buffer.contents(),count:slot.buffer.length,protecting:experts) != true {
                        sources.append(request.expert,request.slot)
                    }
                }
            }
            if fetch.errors.error != nil { return }
            let misses = sources.snapshot
            if self.execution.coalesceReads, let original = self.file as? GGUFFile {
                do {
                    var ranges: [ParticleReadRange] = []
                    for request in misses {
                        let slot = self.slots[request.slot]
                        for (tensor,weight) in [(self.gate,slot.gate),(self.up,slot.up),(self.down,slot.down)] {
                            ranges.append(ParticleReadRange(offset:tensor.offset+request.expert*tensor.expertBytes,
                                count:tensor.expertBytes,destination:slot.buffer.contents().advanced(by:weight.offset)))
                        }
                    }
                    // Only one bounded staging allocation is live, even when
                    // multiple source workers are requested.
                    let groups = ParticleReadPlan.groups(ranges)
                    for group in groups {
                        if cancellation?.isCancelled == true { throw CancellationError() }
                        if group.ranges.count == 1 {
                            let range = group.ranges[0]
                            try original.read(into:range.destination,count:range.count,offset:range.offset)
                        } else {
                            let temp = UnsafeMutableRawPointer.allocate(byteCount:group.count,alignment:64)
                            defer { temp.deallocate() }
                            try original.read(into:temp,count:group.count,offset:group.offset)
                            for range in group.ranges { range.destination.copyMemory(from:temp.advanced(by:range.offset-group.offset),byteCount:range.count) }
                        }
                        fetch.errors.source(group.count)
                    }
                    for request in misses {
                        let slot = self.slots[request.slot]
                        self.diskCache?.admitLoaded(layer:self.layer,expert:request.expert,from:slot.buffer.contents(),count:slot.buffer.length,protecting:experts)
                    }
                } catch { fetch.errors.record(error) }
            } else {
                let workers = min(self.file is ParticleMirrorSource ? self.execution.ssdWorkers : self.execution.sourceWorkers,misses.count)
                DispatchQueue.concurrentPerform(iterations:workers) { worker in
                    for i in stride(from:worker,to:misses.count,by:workers) {
                        do {
                            if cancellation?.isCancelled == true { throw CancellationError() }
                            let request = misses[i], slot = self.slots[request.slot]
                            if self.execution.pipelinedVerification, let mirror=self.file as? ParticleMirrorSource {
                                let pending=try mirror.beginExpertRead(layer:self.layer,expert:request.expert,into:slot.buffer.contents(),capacity:slot.buffer.length)
                                verification.enter()
                                self.verificationQueue.addOperation {
                                    defer { verification.leave() }
                                    do {
                                        if case .needsVerification = pending, cancellation?.isCancelled == true { throw CancellationError() }
                                        let loaded=try mirror.finishExpertRead(pending,layer:self.layer,expert:request.expert,into:slot.buffer.contents(),capacity:slot.buffer.length)
                                        if !loaded.fromSSD { fetch.errors.source(loaded.bytes) }
                                        if cancellation?.isCancelled == true { throw CancellationError() }
                                        self.diskCache?.admitLoaded(layer:self.layer,expert:request.expert,from:slot.buffer.contents(),count:slot.buffer.length,protecting:experts)
                                    } catch { fetch.errors.record(error) }
                                }
                                continue
                            }
                            if let loaded = try self.file.loadExpert(layer:self.layer,expert:request.expert,into:slot.buffer.contents(),capacity:slot.buffer.length) {
                                if !loaded.fromSSD { fetch.errors.source(loaded.bytes) }
                                self.diskCache?.admitLoaded(layer:self.layer,expert:request.expert,from:slot.buffer.contents(),count:slot.buffer.length,protecting:experts)
                                continue
                            }
                            for (tensor,weight) in [(self.gate,slot.gate),(self.up,slot.up),(self.down,slot.down)] {
                                if cancellation?.isCancelled == true { throw CancellationError() }
                                try self.file.readTensor(tensor,into:slot.buffer.contents().advanced(by:weight.offset),count:tensor.expertBytes,
                                                         relativeOffset:request.expert*tensor.expertBytes)
                                fetch.errors.source(tensor.expertBytes)
                            }
                            self.diskCache?.admitLoaded(layer:self.layer,expert:request.expert,from:slot.buffer.contents(),count:slot.buffer.length,protecting:experts)
                        } catch { fetch.errors.record(error) }
                    }
                }
            }
        }
        return fetch
    }
    @discardableResult func complete(_ fetch: ExpertFetch) throws -> [ExpertSlot] {
        if fetch.completed {
            if let error = fetch.errors.error { throw error }; return fetch.slots
        }
        precondition(inFlight === fetch)
        let start = ProcessInfo.processInfo.systemUptime
        fetch.group.wait()
        readWaitSeconds += ProcessInfo.processInfo.systemUptime-start
        sourceReadBytes += fetch.errors.sourceBytes
        readBytes += fetch.logicalBytes
        inFlight = nil; fetch.completed = true
        if let error = fetch.errors.error {
            for slot in fetch.invalidated { ids[slot] = -1 }
            throw error
        }
        return fetch.slots
    }

}

private final class ReadErrors: @unchecked Sendable {
    private let lock = NSLock()
    private var first: (any Error)?
    private(set) var sourceBytes = 0
    func source(_ bytes: Int) { lock.lock(); defer { lock.unlock() }; sourceBytes += bytes }
    var error: (any Error)? { lock.lock(); defer { lock.unlock() }; return first }
    func record(_ error: any Error) { lock.lock(); defer { lock.unlock() }; if first == nil { first = error } }
}

final class ExpertFetch: @unchecked Sendable {
    let slots: [ExpertSlot], ready: [Int], invalidated: [Int], logicalBytes: Int
    let group = DispatchGroup()
    fileprivate let errors = ReadErrors()
    fileprivate var completed = false
    init(slots: [ExpertSlot], ready: [Int], invalidated: [Int], logicalBytes: Int) {
        self.slots = slots; self.ready = ready; self.invalidated = invalidated; self.logicalBytes = logicalBytes
    }
}
private final class SourceMisses: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [(expert: Int,slot: Int)] = []
    func append(_ expert: Int,_ slot: Int) { lock.lock(); requests.append((expert,slot)); lock.unlock() }
    var snapshot: [(expert: Int,slot: Int)] { lock.lock(); defer { lock.unlock() }; return requests }
}

struct ParticleReadRange {
    let offset: Int, count: Int
    let destination: UnsafeMutableRawPointer
}
struct ParticleReadGroup {
    let offset: Int, count: Int
    let ranges: [ParticleReadRange]
}
enum ParticleReadPlan {
    static func groups(_ ranges: [ParticleReadRange]) -> [ParticleReadGroup] {
        var result: [ParticleReadGroup] = [], pending: [ParticleReadRange] = []
        var start = 0, end = 0, useful = 0
        func flush() {
            if !pending.isEmpty { result.append(ParticleReadGroup(offset:start,count:end-start,ranges:pending)) }
            pending.removeAll(keepingCapacity:true); useful = 0
        }
        for range in ranges.sorted(by:{ $0.offset < $1.offset }) {
            let span = range.offset+range.count-start
            if !pending.isEmpty && (range.offset < end || span > 8*1024*1024 || Double(span) > Double(useful+range.count)*1.10) { flush() }
            if pending.isEmpty { start = range.offset }
            pending.append(range); end = range.offset+range.count; useful += range.count
        }
        flush(); return result
    }
}
