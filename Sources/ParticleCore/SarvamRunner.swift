import Foundation
@preconcurrency import Metal
import Darwin
import ParticleGGUF

public final class ParticleCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    public init() {}
    public func reset() { lock.lock(); value = false; lock.unlock() }
    public func cancel() { lock.lock(); value = true; lock.unlock() }
    /// Build the DispatchSource callback outside any caller's actor context.
    /// An inline closure in the CLI's MainActor run method traps on SIGINT
    /// when Dispatch invokes it on the global queue under Swift 6 checks.
    nonisolated public func cancellationHandler() -> @Sendable () -> Void {
        { [self] in cancel() }
    }
    public var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
}

public struct RoutingResult: Sendable {
    public let experts: [Int]
    public let weights: [Float]
}

public enum SarvamRouting {
    /// Bias affects selection only. Mixture weights use unbiased sigmoid
    /// scores, normalized over the selected experts, then multiplied by 2.5.
    /// Six is the native graph; smaller explicit budgets are approximate.
    public static func select(logits: [Float], bias: [Float], activeExperts: Int = 6) throws -> RoutingResult {
        guard (2...6).contains(activeExperts), logits.count == 128, bias.count == 128,
              logits.allSatisfy(\.isFinite), bias.allSatisfy(\.isFinite) else {
            throw ParticleError.invalid("Router produced invalid scores")
        }
        let scores = logits.map { x -> Float in
            x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))
        }
        let ids = (0..<128).sorted {
            let a = scores[$0] + bias[$0], b = scores[$1] + bias[$1]
            return a == b ? $0 < $1 : a > b
        }.prefix(activeExperts)
        let denominator = ids.reduce(Float(1e-20)) { $0 + scores[$1] }
        return RoutingResult(experts: Array(ids), weights: ids.map { scores[$0] / denominator * 2.5 })
    }
}

public struct ParticleRunMetrics: Sendable {
    public let expertReadBytes: Int
    public let sourceReadBytes: Int
    public let ssdReadBytes: Int
    public let ssdWriteBytes: Int
    public let ssdHits: Int
    public let ssdStoredBytes: Int
    public let ssdCorruptions: Int
    public let ssdWarning: String?
    public let cacheHits: Int
    public let cacheMisses: Int
    public let footprintBytes: UInt64
    public let expertReadWaitSeconds: Double
    public let gpuWaitSeconds: Double
    public let attentionGPUSeconds: Double
}

/// Bounded-memory Sarvam decode and layer-major grouped prompt processing.
/// The scalar path remains available as the numerical reference.
public final class SarvamRunner {
    public let file: any ParticleModelSource
    public let configuration: SarvamConfiguration
    public let maxContext: Int
    public private(set) var position = 0
    private let gpu: ParticleGPU
    private let resident: [String: GPUWeight]
    private let embedding: EmbeddingReader
    private let embeddingRow: MTLBuffer
    private let cacheSlots: Int
    private let diskCache: ExpertDiskCache?
    private var caches: [Int: ExpertCache] = [:]
    private let keys: [MTLBuffer], values: [MTLBuffer]
    private let hidden: MTLBuffer, normalized: MTLBuffer, qkv: MTLBuffer, query: MTLBuffer
    private let attended: MTLBuffer, projected: MTLBuffer, gate: MTLBuffer
    private let decodeRouted: [MTLBuffer]
    private let decodeExpertScratch: [MTLBuffer]
    private let ffn: MTLBuffer, expertOut: MTLBuffer, router: MTLBuffer, logits: MTLBuffer
    private let routeIDs: MTLBuffer, routeWeights: MTLBuffer
    private var promptScratch: [PromptRow] = []
    private var recoverableCancellation = false
    private var execution: ParticleExecutionOptions
    private var failed = false
    private var hasLogits = false
    public init(file: any ParticleModelSource, maxContext: Int = 2048, expertCacheSlots: Int = 12, ssdCacheDirectory: URL? = nil, ssdCacheBytes: Int = 4 * 1024 * 1024 * 1024, ssdCachePolicy: ParticleSSDCachePolicy = .victim, execution: ParticleExecutionOptions = .init()) throws {
        guard (32...8192).contains(maxContext), (6...16).contains(expertCacheSlots) else {
            throw ParticleError.invalid("This memory-bounded CLI requires context 32...8192 and expert slots 6...16")
        }
        guard (2...6).contains(execution.activeExperts), [1,2,4].contains(execution.sourceWorkers), [1,2,4].contains(execution.ssdWorkers), [256,512].contains(execution.splitBlock) else { throw ParticleError.invalid("Invalid I/O workers or attention block") }
        self.execution = execution
        self.file = file; configuration = try SarvamConfiguration(file: file)
        self.maxContext = maxContext; cacheSlots = expertCacheSlots
        diskCache = try ssdCacheDirectory.map { try ExpertDiskCache(file:file,directory:$0,maxBytes:ssdCacheBytes,policy:ssdCachePolicy) }
        try file.prepareForInference()
        let processor = try ParticleGPU(execution:execution); gpu = processor
        var weights: [String: GPUWeight] = [:]
        func require(_ name: String, _ shape: [Int], f32: Bool = false) throws {
            let tensor = try file.tensor(name, shape: shape)
            if f32 && tensor.type != .f32 { throw ParticleError.invalid("Expected FP32 tensor: \(name)") }
            weights[name] = try mapWeight(tensor, file: file, device: processor.device)
        }
        _ = try file.tensor("token_embd.weight",shape:[4096,262144])
        let embeddings = try EmbeddingReader(file:file)
        embedding = embeddings; embeddingRow = try processor.bytes(embeddings.rowBytes)
        try require("output.weight", [4096,262144])
        try require("output_norm.weight", [4096], f32: true)
        for l in 0..<19 {
            let p = "blk.\(l)."
            try require(p + "attn_norm.weight", [4096], f32: true)
            try require(p + "attn_q_norm.weight", [64], f32: true)
            try require(p + "attn_k_norm.weight", [64], f32: true)
            try require(p + "attn_qkv.weight", [4096,4608])
            try require(p + "attn_output.weight", [4096,4096])
            try require(p + "ffn_norm.weight", [4096], f32: true)
            if l == 0 {
                try require(p + "ffn_gate.weight", [4096,8192])
                try require(p + "ffn_up.weight", [4096,8192])
                try require(p + "ffn_down.weight", [8192,4096])
            } else {
                try require(p + "ffn_gate_inp.weight", [4096,128], f32: true)
                try require(p + "exp_probs_b.bias", [128], f32: true)
                try require(p + "ffn_gate_shexp.weight", [4096,1024])
                try require(p + "ffn_up_shexp.weight", [4096,1024])
                try require(p + "ffn_down_shexp.weight", [1024,4096])
                _ = try file.tensor(p + "ffn_gate_exps.weight", shape: [4096,1024,128])
                _ = try file.tensor(p + "ffn_up_exps.weight", shape: [4096,1024,128])
                _ = try file.tensor(p + "ffn_down_exps.weight", shape: [1024,4096,128])
            }
        }
        resident = weights
        keys = try (0..<19).map { _ in try processor.buffer(maxContext * 256) }
        values = try (0..<19).map { _ in try processor.buffer(maxContext * 256) }
        hidden = try processor.buffer(4096); normalized = try processor.buffer(4096)
        qkv = try processor.buffer(4608); query = try processor.buffer(4096)
        attended = try processor.buffer(4096); projected = try processor.buffer(4096)
        gate = try processor.buffer(8192)
        ffn = try processor.buffer(4096); expertOut = try processor.buffer(4096)
        decodeRouted = try (0..<6).map { _ in try processor.buffer(4096) }
        decodeExpertScratch = try (0..<6).map { _ in try processor.buffer(1024) }
        router = try processor.buffer(128); logits = try processor.buffer(262144)
        routeIDs = try processor.bytes(32); routeWeights = try processor.buffer(6)
    }
    deinit {
        for cache in caches.values { cache.persist() }
        diskCache?.flush()
    }
    /// Call only between completed operations on the single model owner.
    public func configureExecution(_ options: ParticleExecutionOptions) throws {
        guard !failed, (2...6).contains(options.activeExperts), [1,2,4].contains(options.sourceWorkers), [1,2,4].contains(options.ssdWorkers), [256,512].contains(options.splitBlock) else {
            throw ParticleError.invalid("Cannot configure failed runner or invalid execution limits")
        }
        guard options.activeExperts == execution.activeExperts || position == 0 else {
            throw ParticleError.invalid("Reset the conversation before changing active experts; KV belongs to the previous model profile")
        }
        try gpu.configure(options)
        for cache in caches.values { try cache.configure(options) }
        execution=options
    }
    public var executionOptions: ParticleExecutionOptions { execution }
    public func reset(clearExpertCache: Bool = false) {
        position = 0; failed = false; recoverableCancellation = false; hasLogits = false
        if clearExpertCache { autoreleasepool {
            for cache in caches.values { cache.persist() }
            caches.removeAll()
        } }
    }
    /// Only a cancelled operation may recover its last complete chunk. A GPU
    /// or I/O failure still requires reset. Partial future rows remain masked.
    public func recoverCancelledPrefix() throws {
        guard recoverableCancellation else { throw ParticleError.invalid("No recoverable cancelled prefix") }
        failed = false; recoverableCancellation = false; hasLogits = false
    }
    public func flushSSDCache() { diskCache?.flush() }
    /// Discard a changed suffix without recomputing a still-identical KV prefix.
    /// Existing rows beyond the new position are masked and overwritten on use.
    public func rewind(to position: Int) throws {
        guard !failed, position >= 0, position <= self.position else {
            throw ParticleError.invalid("Invalid KV rewind; reset after a failed turn")
        }
        self.position = position; hasLogits = false
    }
    private func weight(_ name: String) -> GPUWeight { resident[name]! }

    public func forward(token: Int, emitLogits: Bool = true, cancellation: ParticleCancellation? = nil,
                        onLayer: ((Int) -> Void)? = nil) throws {
        guard !failed, position < maxContext else { throw ParticleError.invalid("Context full or runner needs reset after a failed turn") }
        hasLogits = false
        do {
            try autoreleasepool {
                let row = try embedding.load(token:token,into:embeddingRow)
                let embedding = try gpu.command("Particle embedding")
                try gpu.embedding(embedding,row,token:0,out:hidden)
                if execution.deferredDecode { gpu.submit(embedding) } else { try gpu.finish(embedding) }
                for layer in 0..<19 {
                    if cancellation?.isCancelled == true { throw CancellationError() }
                    let p = "blk.\(layer)."
                    var cb = try gpu.command("Particle layer \(layer) attention/router")
                    try gpu.rms(cb, hidden, weight(p + "attn_norm.weight"), normalized)
                    try gpu.gemv(cb, weight(p + "attn_qkv.weight"), normalized, qkv)
                    try gpu.qkv(cb, projection: qkv, qNorm: weight(p + "attn_q_norm.weight"), kNorm: weight(p + "attn_k_norm.weight"),
                                query: query, keys: keys[layer], values: values[layer], position: position, capacity: maxContext)
                    cb = try gpu.profiledAttention(cb,queries:[query],keys:keys[layer],values:values[layer],outputs:[attended],base:position,capacity:maxContext)
                    try gpu.gemv(cb, weight(p + "attn_output.weight"), attended, projected)
                    try gpu.add(cb, hidden, projected)
                    try gpu.rms(cb, hidden, weight(p + "ffn_norm.weight"), normalized)
                    if layer == 0 {
                        try gpu.mlp(cb, input: normalized, gate: weight(p + "ffn_gate.weight"), up: weight(p + "ffn_up.weight"),
                                    down: weight(p + "ffn_down.weight"), gateScratch: gate, output: ffn)
                        try gpu.add(cb, hidden, ffn)
                        if execution.deferredDecode { gpu.submit(cb) } else { try gpu.finish(cb) }
                    } else {
                        try gpu.gemv(cb, weight(p + "ffn_gate_inp.weight"), normalized, router)
                        let b = weight(p + "exp_probs_b.bias")
                        if execution.gpuRouting && execution.activeExperts == 6 { try gpu.routeSix(cb,logits:router,bias:b,ids:routeIDs,weights:routeWeights) }
                        try gpu.finish(cb)
                        let routing: RoutingResult
                        if execution.gpuRouting && execution.activeExperts == 6 { routing = try gpu.routingResult(ids:routeIDs,weights:routeWeights) }
                        else {
                            let scores = Array(UnsafeBufferPointer(start:router.contents().assumingMemoryBound(to:Float.self),count:128))
                            let bias = Array(UnsafeBufferPointer(start:b.buffer.contents().advanced(by:b.offset).assumingMemoryBound(to:Float.self),count:128))
                            routing = try SarvamRouting.select(logits:scores,bias:bias,activeExperts:execution.activeExperts)
                        }
                        if caches[layer] == nil { caches[layer] = try ExpertCache(file: file, layer: layer, count: cacheSlots, gpu: gpu, diskCache:diskCache, execution:execution) }
                        let shared = try gpu.command("Particle shared expert")
                        try gpu.mlp(shared, input: normalized, gate: weight(p + "ffn_gate_shexp.weight"), up: weight(p + "ffn_up_shexp.weight"),
                                    down: weight(p + "ffn_down_shexp.weight"), gateScratch: gate, output: ffn)
                        gpu.submit(shared)
                        let cache = caches[layer]!
                        let fetch = try cache.beginFetch(routing.experts,cancellation:cancellation)
                        defer { _ = try? cache.complete(fetch) }
                        func encode(_ indices: [Int], into supplied: MTLCommandBuffer? = nil) throws {
                            guard !indices.isEmpty else { return }
                            let cb = try supplied ?? gpu.command("Particle ready/fetched decode experts")
                            if execution.groupedExperts {
                                let slots = indices.map { fetch.slots[$0] }
                                try gpu.expertMLPs(cb,input:normalized,gates:slots.map(\.gate),ups:slots.map(\.up),downs:slots.map(\.down),
                                    scratch:indices.map { decodeExpertScratch[$0] },outputs:indices.map { decodeRouted[$0] })
                            } else {
                                for rank in indices {
                                    let slot = fetch.slots[rank]
                                    try gpu.mlp(cb,input:normalized,gate:slot.gate,up:slot.up,down:slot.down,
                                        gateScratch:gate,output:decodeRouted[rank])
                                }
                            }
                            if supplied == nil {
                                if execution.deferredDecode { gpu.submit(cb) } else { try gpu.finish(cb) }
                            }
                        }
                        if !execution.overlapExpertReads { try cache.complete(fetch) }
                        try encode(fetch.ready)
                        try cache.complete(fetch)
                        let routed = try gpu.command("Particle ordered decode residual")
                        let missing = routing.experts.indices.filter { !fetch.ready.contains($0) }
                        if execution.deferredDecode { try encode(missing,into:routed) } else { try encode(missing) }
                        for rank in routing.experts.indices { try gpu.add(routed,ffn,decodeRouted[rank],scale:routing.weights[rank]) }
                        try gpu.add(routed,hidden,ffn)
                        if execution.deferredDecode { gpu.submit(routed) }
                        else { try gpu.finish(routed) }
                        // The following layer's routing wait (or the final
                        // head/drain) validates all queued expert consumers.
                    }
                    onLayer?(layer + 1)
                }
                if emitLogits {
                    let cb = try gpu.command("Particle final norm and untied output head")
                    try gpu.rms(cb, hidden, weight("output_norm.weight"), normalized)
                    try gpu.gemv(cb, weight("output.weight"), normalized, logits)
                    try gpu.finish(cb)
                    hasLogits = true
                }
                try gpu.drain()
                position += 1
            }
        } catch {
            // Even cancellation or an encode error must finish consumers before
            // a reset may recycle their buffers on the CPU.
            try? gpu.drain()
            failed = true; recoverableCancellation = error is CancellationError; throw error
        }
    }
    public func greedyToken() throws -> Int {
        guard !failed, hasLogits else { throw ParticleError.invalid("No valid output logits") }
        return try gpu.greedyToken(logits)
    }
    public func currentLogits() throws -> [Float] {
        guard !failed, hasLogits else { throw ParticleError.invalid("No valid output logits") }
        let result = Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float.self), count: 262144))
        guard result.allSatisfy(\.isFinite) else { throw ParticleError.invalid("Non-finite model logits; generation stopped") }
        return result
    }
    public var metrics: ParticleRunMetrics {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return ParticleRunMetrics(expertReadBytes: caches.values.reduce(0) { $0 + $1.readBytes },
                                  sourceReadBytes: caches.values.reduce(0) { $0 + $1.sourceReadBytes },
                                  ssdReadBytes: (diskCache?.readBytes ?? 0) + ((file as? ParticleMirrorSource)?.readBytes ?? 0), ssdWriteBytes: diskCache?.writeBytes ?? 0,
                                  ssdHits: diskCache?.hits ?? (file as? ParticleMirrorSource)?.hits ?? 0, ssdStoredBytes: diskCache?.storedBytes ?? (file as? ParticleMirrorSource)?.storedBytes ?? 0,
                                  ssdCorruptions: diskCache?.corruptions ?? (file as? ParticleMirrorSource)?.corruptions ?? 0, ssdWarning: diskCache?.failure ?? (file as? ParticleMirrorSource)?.warning,
                                  cacheHits: caches.values.reduce(0) { $0 + $1.hitCount },
                                  cacheMisses: caches.values.reduce(0) { $0 + $1.missCount },
                                  footprintBytes: result == KERN_SUCCESS ? info.phys_footprint : 0,
                                  expertReadWaitSeconds: caches.values.reduce(0) { $0 + $1.readWaitSeconds },
                                  gpuWaitSeconds: gpu.waitSeconds,attentionGPUSeconds:gpu.attentionGPUSeconds)
    }
}

private struct PromptRow {
    let hidden: MTLBuffer, normalized: MTLBuffer, ffn: MTLBuffer, router: MTLBuffer
    let embedding: MTLBuffer
    let qkv: MTLBuffer, query: MTLBuffer, attended: MTLBuffer, projected: MTLBuffer
    let gate: MTLBuffer
    let routed: [MTLBuffer]
    init(gpu: ParticleGPU, embeddingBytes: Int) throws {
        hidden = try gpu.buffer(4096); normalized = try gpu.buffer(4096)
        ffn = try gpu.buffer(4096); router = try gpu.buffer(128)
        routed = try (0..<6).map { _ in try gpu.buffer(4096) }
        embedding = try gpu.bytes(embeddingBytes)
        qkv = try gpu.buffer(4608); query = try gpu.buffer(4096)
        attended = try gpu.buffer(4096); projected = try gpu.buffer(4096)
        gate = try gpu.buffer(8192)
    }
}

extension SarvamRunner {
    /// A small row tile changes execution order, not math: attention is still
    /// causal and each row's routed outputs are added in original top-k order.
    /// At most 256 rows of extra activation/embedding scratch (under 62 MiB
    /// for the installed Q4_K embedding table).
    public func prefill(tokens: [Int], chunkSize: Int = 128, emitLogits: Bool = true, cancellation: ParticleCancellation? = nil,
                        onLayer: ((Int,Int,Int) -> Void)? = nil,
                        onChunkComplete: ((Int) -> Void)? = nil) throws {
        guard !failed, !tokens.isEmpty, (1...256).contains(chunkSize),
              tokens.count <= maxContext-position, tokens.allSatisfy({ (0..<configuration.vocabulary).contains($0) }) else {
            throw ParticleError.invalid("Invalid prompt, prompt chunk size, or available context")
        }
        if chunkSize == 1 {
            for (i,token) in tokens.enumerated() {
                try forward(token:token,emitLogits:emitLogits && i == tokens.count-1,cancellation:cancellation,
                            onLayer:{ onLayer?(i,1,$0) })
                onChunkComplete?(i+1)
            }
            return
        }
        hasLogits = false
        do {
            // Reuse row allocations across all chunks in this prompt.
            let required = min(chunkSize,tokens.count)
            while promptScratch.count < required { promptScratch.append(try PromptRow(gpu:gpu,embeddingBytes:embedding.rowBytes)) }
            let scratch = promptScratch
            for start in stride(from:0,to:tokens.count,by:chunkSize) {
                try autoreleasepool {
                    if cancellation?.isCancelled == true { throw CancellationError() }
                    let count = min(chunkSize,tokens.count-start), rows = Array(scratch.prefix(count)), base = position
                    let embeddings = try gpu.command("Particle prompt embeddings")
                    for i in 0..<count {
                        let row = try embedding.load(token:tokens[start+i],into:rows[i].embedding)
                        try gpu.embedding(embeddings,row,token:0,out:rows[i].hidden)
                    }
                    try gpu.finish(embeddings)
                    for layer in 0..<19 {
                        try autoreleasepool {
                            if cancellation?.isCancelled == true { throw CancellationError() }
                            let p = "blk.\(layer)."
                            var cb = try gpu.command("Particle prompt attention/router")
                            for row in rows { try gpu.rms(cb,row.hidden,weight(p+"attn_norm.weight"),row.normalized) }
                            try gpu.batchGemv(cb,weight(p+"attn_qkv.weight"),inputs:rows.map(\.normalized),outputs:rows.map(\.qkv))
                            // Write/read positions in causal order. Projection work is
                            // parallel across rows, while each query masks its future.
                            for (i,row) in rows.enumerated() {
                                try gpu.qkv(cb,projection:row.qkv,qNorm:weight(p+"attn_q_norm.weight"),kNorm:weight(p+"attn_k_norm.weight"),
                                            query:row.query,keys:keys[layer],values:values[layer],position:base+i,capacity:maxContext)
                            }
                            cb = try gpu.profiledAttention(cb,queries:rows.map(\.query),keys:keys[layer],values:values[layer],
                                outputs:rows.map(\.attended),base:base,capacity:maxContext)
                            try gpu.batchGemv(cb,weight(p+"attn_output.weight"),inputs:rows.map(\.attended),outputs:rows.map(\.projected))
                            for row in rows {
                                try gpu.add(cb,row.hidden,row.projected)
                                try gpu.rms(cb,row.hidden,weight(p+"ffn_norm.weight"),row.normalized)
                            }
                            if layer == 0 {
                                try gpu.batchMLP(cb,inputs:rows.map(\.normalized),gate:weight(p+"ffn_gate.weight"),up:weight(p+"ffn_up.weight"),
                                                 down:weight(p+"ffn_down.weight"),gateScratch:rows.map(\.gate),outputs:rows.map(\.ffn))
                                for row in rows { try gpu.add(cb,row.hidden,row.ffn) }
                            } else {
                                try gpu.batchGemv(cb,weight(p+"ffn_gate_inp.weight"),inputs:rows.map(\.normalized),outputs:rows.map(\.router))
                            }
                            try gpu.finish(cb)
                            if layer > 0 {
                                let b = weight(p+"exp_probs_b.bias")
                                let bias = Array(UnsafeBufferPointer(start:b.buffer.contents().advanced(by:b.offset).assumingMemoryBound(to:Float.self),count:128))
                                let routes = try rows.map { row in
                                    try SarvamRouting.select(logits:Array(UnsafeBufferPointer(start:row.router.contents().assumingMemoryBound(to:Float.self),count:128)),bias:bias,activeExperts:execution.activeExperts)
                                }
                                if caches[layer] == nil { caches[layer] = try ExpertCache(file:file,layer:layer,count:cacheSlots,gpu:gpu,diskCache:diskCache,execution:execution) }
                                let cache = caches[layer]!
                                let plan = try ExpertBatchPlan(routes:routes,cached:cache.residentExperts,capacity:cache.capacity)
                                let shared = try gpu.command("Particle prompt shared experts")
                                try gpu.batchMLP(shared,inputs:rows.map(\.normalized),gate:weight(p+"ffn_gate_shexp.weight"),up:weight(p+"ffn_up_shexp.weight"),
                                                 down:weight(p+"ffn_down_shexp.weight"),gateScratch:rows.map(\.gate),outputs:rows.map(\.ffn))
                                shared.commit()
                                defer { shared.waitUntilCompleted() }
                                for batch in plan.batches {
                                    if cancellation?.isCancelled == true { throw CancellationError() }
                                    let fetch = try cache.beginFetch(batch,frequencies:batch.map { plan.uses[$0]!.count },cancellation:cancellation)
                                    defer { _ = try? cache.complete(fetch) }
                                    func encode(_ indices: [Int], _ label: String) throws {
                                        try autoreleasepool {
                                            guard !indices.isEmpty else { return }
                                            if execution.hybrid {
                                                for index in indices {
                                                    let slot = fetch.slots[index], uses = plan.uses[batch[index]]!
                                                    try gpu.hybridMLP(inputs:uses.map { rows[$0.row].normalized },gate:slot.gate,up:slot.up,down:slot.down,
                                                        gateScratch:uses.map { rows[$0.row].gate },outputs:uses.map { rows[$0.row].routed[$0.rank] },cancellation:cancellation)
                                                }
                                                return
                                            }
                                            let routed = try gpu.command(label)
                                            for index in indices {
                                                let slot = fetch.slots[index], uses = plan.uses[batch[index]]!
                                                try gpu.batchMLP(routed,inputs:uses.map { rows[$0.row].normalized },gate:slot.gate,up:slot.up,down:slot.down,
                                                    gateScratch:uses.map { rows[$0.row].gate },outputs:uses.map { rows[$0.row].routed[$0.rank] })
                                            }
                                            try gpu.finish(routed)
                                        }
                                    }
                                    if !execution.overlapExpertReads { try cache.complete(fetch) }
                                    try encode(fetch.ready,"Particle ready prompt experts")
                                    try cache.complete(fetch)
                                    try encode(batch.indices.filter { !fetch.ready.contains($0) },"Particle fetched prompt experts")
                                }
                                guard shared.status == .completed, shared.error == nil else { throw ParticleError.invalid("Shared prompt expert failed") }
                                let combine = try gpu.command("Particle ordered prompt residuals")
                                for (i,row) in rows.enumerated() {
                                    for rank in routes[i].experts.indices { try gpu.add(combine,row.ffn,row.routed[rank],scale:routes[i].weights[rank]) }
                                    try gpu.add(combine,row.hidden,row.ffn)
                                }
                                try gpu.finish(combine)
                            }
                            onLayer?(start,count,layer+1)
                        }
                    }
                    if emitLogits && start+count == tokens.count {
                        let head = try gpu.command("Particle prompt output head")
                        try gpu.rms(head,rows[count-1].hidden,weight("output_norm.weight"),normalized)
                        try gpu.gemv(head,weight("output.weight"),normalized,logits)
                        try gpu.finish(head)
                        hasLogits = true
                    }
                    position += count
                    onChunkComplete?(start+count)
                }
            }
        } catch { failed = true; recoverableCancellation = error is CancellationError; hasLogits = false; throw error }
    }
}
