import Foundation
@preconcurrency import Metal
import Darwin
import ParticleGGUF
import TurboFieldfare

struct GPUWeight {
    let buffer: MTLBuffer
    let offset: Int
    let columns: Int
    let rows: Int
    let type: GGMLType
    var rowBytes: Int { columns / type.blockElements * type.blockBytes }
}

/// Uses the upstream device factory (including its Apple GPU watchdog setup),
/// while keeping Sarvam's math and GGUF representations independent of Gemma.
final class ParticleGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private(set) var waitSeconds = 0.0
    private var pendingCommands: [MTLCommandBuffer] = []
    private let batchArguments: MTLArgumentEncoder
    private let expertArguments: MTLArgumentEncoder
    private let pipelines: [String: MTLComputePipelineState]
    private(set) var execution: ParticleExecutionOptions
    private var splitScratch: MTLBuffer?
    private var greedyScratch: MTLBuffer?, greedyResult: MTLBuffer?
    private(set) var attentionGPUSeconds = 0.0
    private(set) var attentionCalls = 0
    init(execution: ParticleExecutionOptions = .init()) throws {
        self.execution = execution
        guard let device = MetalContext.makeSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let url = Bundle.module.url(forResource: "particle", withExtension: "metal", subdirectory: "Metal") else {
            throw ParticleError.invalid("Metal device, command queue, or Particle shaders are unavailable")
        }
        self.device = device; self.queue = queue
        let options = MTLCompileOptions(); options.languageVersion = .version4_0; options.mathMode = .safe
        guard let expertURL = Bundle.module.url(forResource:"particle_experts",withExtension:"metal",subdirectory:"Metal") else { throw ParticleError.invalid("Missing grouped expert shader") }
        guard let ffnURL = Bundle.module.url(forResource:"particle_ffn_vector",withExtension:"metal",subdirectory:"Metal") else { throw ParticleError.invalid("Missing native FFN block shader") }
        guard let cooperativeURL=Bundle.module.url(forResource:"particle_cooperative",withExtension:"metal",subdirectory:"Metal") else { throw ParticleError.invalid("Missing cooperative quantization shader") }
        let source = try String(contentsOf:url,encoding:.utf8)+String(contentsOf:expertURL,encoding:.utf8)+String(contentsOf:ffnURL,encoding:.utf8)+String(contentsOf:cooperativeURL,encoding:.utf8)
        let library = try device.makeLibrary(source:source,options:options)
        var ps: [String: MTLComputePipelineState] = [:]
        for name in ["cooperative_vector_gemv","cooperative_batch_vector_gemv","cooperative_vector_gate_up","cooperative_batch_vector_gate_up","cooperative_expert_gate_up_vector","cooperative_expert_down_vector","gemv", "route_six", "expert_gate_up", "expert_down", "expert_gate_up_vector", "expert_down_vector", "vector_gate_up", "batch_vector_gate_up", "vector_gemv", "batch_vector_gemv", "argmax", "argmax_merge", "batch_gemv", "batch_gate_up", "gate_up", "embedding", "rms", "qkv", "attention", "grouped_attention", "split_attention", "merge_attention", "add", "swiglu"] {
            guard let function = library.makeFunction(name: "particle_" + name) else {
                throw ParticleError.invalid("Missing shader: \(name)")
            }
            ps[name] = try device.makeComputePipelineState(function: function)
        }
        if execution.projection == .staged || [.tensor,.fast].contains(execution.attention) {
            var optionalSource = try String(contentsOf:url,encoding:.utf8)
            for name in ["particle_staged","particle_tensor_attention"] {
                guard let path = Bundle.module.url(forResource:name,withExtension:"metal",subdirectory:"Metal") else {
                    throw ParticleError.invalid("Missing optional matrix shader: \(name)")
                }
                optionalSource += try String(contentsOf:path,encoding:.utf8)
            }
            let optionalLibrary = try device.makeLibrary(source:optionalSource,options:options)
            for name in ["staged_gemv","staged_gate_up","tensor_attention"] {
                guard let function = optionalLibrary.makeFunction(name:"particle_"+name) else {
                    throw ParticleError.invalid("Optional matrix kernel unavailable: \(name)")
                }
                ps[name] = try device.makeComputePipelineState(function:function)
            }
        }
        pipelines = ps
        batchArguments = library.makeFunction(name:"particle_batch_gemv")!.makeArgumentEncoder(bufferIndex:0)
        expertArguments = library.makeFunction(name:"particle_expert_gate_up")!.makeArgumentEncoder(bufferIndex:0)
    }
    func configure(_ options: ParticleExecutionOptions) throws {
        guard options.projection != .staged || pipelines["staged_gemv"] != nil,
              ![.tensor,.fast].contains(options.attention) || pipelines["tensor_attention"] != nil else {
            throw ParticleError.invalid("Initialize the runner with the requested experimental matrix pipeline")
        }
        execution=options
    }
    func buffer(_ floats: Int) throws -> MTLBuffer { try bytes(floats * 4) }
    func bytes(_ n: Int) throws -> MTLBuffer {
        guard n > 0, n <= device.maxBufferLength, let b = device.makeBuffer(length: n, options: .storageModeShared) else {
            throw ParticleError.invalid("Metal allocation failed (\(n) bytes)")
        }
        return b
    }
    func command(_ label: String = "Particle") throws -> MTLCommandBuffer {
        guard let cb = queue.makeCommandBuffer() else { throw ParticleError.invalid("Cannot create GPU command buffer") }
        cb.label = label; return cb
    }
    /// The single command queue preserves GPU dependencies. Keep every
    /// submission alive until the final dependency completes, then check all
    /// errors rather than hiding a failed earlier command behind a later one.
    func submit(_ cb: MTLCommandBuffer) {
        pendingCommands.append(cb); cb.commit()
    }
    func drain() throws {
        guard let last = pendingCommands.last else { return }
        let start = ProcessInfo.processInfo.systemUptime
        last.waitUntilCompleted()
        waitSeconds += ProcessInfo.processInfo.systemUptime - start
        let completed = pendingCommands; pendingCommands.removeAll(keepingCapacity:true)
        for cb in completed {
            guard cb.status == .completed, cb.error == nil else {
                throw ParticleError.invalid("GPU command failed: \(cb.error?.localizedDescription ?? String(describing: cb.status))")
            }
        }
    }
    func finish(_ cb: MTLCommandBuffer) throws {
        submit(cb); try drain()
    }
    private func encoder(_ cb: MTLCommandBuffer, _ name: String) throws -> MTLComputeCommandEncoder {
        guard let e = cb.makeComputeCommandEncoder(), let p = pipelines[name] else {
            throw ParticleError.invalid("Cannot encode \(name)")
        }
        e.setComputePipelineState(p); return e
    }
    private func value<T>(_ e: MTLComputeCommandEncoder, _ value: T, _ index: Int) {
        var v = value; withUnsafePointer(to: &v) { e.setBytes($0, length: MemoryLayout<T>.stride, index: index) }
    }
    func routeSix(_ cb: MTLCommandBuffer, logits: MTLBuffer, bias: GPUWeight, ids: MTLBuffer, weights: MTLBuffer) throws {
        guard bias.type == .f32, bias.columns*bias.rows == 128, logits.length >= 512, ids.length >= 32, weights.length >= 24 else {
            throw ParticleError.invalid("Invalid GPU router buffers")
        }
        let e = try encoder(cb,"route_six")
        e.setBuffer(logits,offset:0,index:0); e.setBuffer(bias.buffer,offset:bias.offset,index:1)
        e.setBuffer(ids,offset:0,index:2); e.setBuffer(weights,offset:0,index:3)
        e.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1)); e.endEncoding()
    }
    func routingResult(ids: MTLBuffer, weights: MTLBuffer) throws -> RoutingResult {
        let p = ids.contents().assumingMemoryBound(to:UInt32.self), w = weights.contents().assumingMemoryBound(to:Float.self)
        let experts = (0..<6).map { Int(p[$0]) }, values = Array(UnsafeBufferPointer(start:w,count:6))
        guard p[6] == 0, Set(experts).count == 6, experts.allSatisfy({ (0..<128).contains($0) }), values.allSatisfy(\.isFinite) else {
            throw ParticleError.invalid("GPU router produced invalid scores")
        }
        return RoutingResult(experts:experts,weights:values)
    }
    /// Independent experts share two dispatches. Every rank has its own
    /// scratch/output and keeps the reference lane accumulation order.
    func expertMLPs(_ cb: MTLCommandBuffer, input: MTLBuffer, gates: [GPUWeight], ups: [GPUWeight], downs: [GPUWeight],
                    scratch: [MTLBuffer], outputs: [MTLBuffer]) throws {
        let count = gates.count
        guard (1...6).contains(count), ups.count == count, downs.count == count, scratch.count == count, outputs.count == count else { throw ParticleError.invalid("Invalid grouped expert count") }
        let g=gates[0], u=ups[0], d=downs[0]
        guard g.columns == u.columns, g.rows == u.rows, d.columns == g.rows, d.rows == g.columns, input.length >= g.columns*4 else { throw ParticleError.invalid("Invalid grouped expert shape") }
        let arguments = try bytes(expertArguments.encodedLength)
        expertArguments.setArgumentBuffer(arguments,offset:0)
        for rank in 0..<count {
            let gate=gates[rank], up=ups[rank], down=downs[rank]
            guard gate.columns == g.columns, gate.rows == g.rows, gate.type == g.type,
                  up.columns == u.columns, up.rows == u.rows, up.type == u.type,
                  down.columns == d.columns, down.rows == d.rows, down.type == d.type,
                  scratch[rank].length >= g.rows*4, outputs[rank].length >= d.rows*4 else { throw ParticleError.invalid("Incompatible grouped expert bindings") }
            expertArguments.setBuffer(gate.buffer,offset:gate.offset,index:rank)
            expertArguments.setBuffer(up.buffer,offset:up.offset,index:6+rank)
            expertArguments.setBuffer(down.buffer,offset:down.offset,index:12+rank)
            expertArguments.setBuffer(scratch[rank],offset:0,index:18+rank)
            expertArguments.setBuffer(outputs[rank],offset:0,index:24+rank)
        }
        let parameters: [UInt32] = [UInt32(g.columns),UInt32(g.rows),g.type.rawValue,u.type.rawValue,d.type.rawValue,UInt32(g.rowBytes),UInt32(u.rowBytes),UInt32(d.rowBytes),UInt32(count)]
        let vector = execution.ffnVector && g.columns%256 == 0 && g.rows%256 == 0 && [g.type,u.type,d.type].allSatisfy { [.q4K,.q5K,.q6K].contains($0) }
        let gateEncoder = try encoder(cb,vector ? (execution.cooperativeQuant && g.type != .q6K && u.type != .q6K ? "cooperative_expert_gate_up_vector" : "expert_gate_up_vector") : "expert_gate_up")
        gateEncoder.setBuffer(arguments,offset:0,index:0); gateEncoder.setBuffer(input,offset:0,index:1)
        parameters.withUnsafeBytes { gateEncoder.setBytes($0.baseAddress!,length:$0.count,index:2) }
        for rank in 0..<count {
            gateEncoder.useResource(gates[rank].buffer,usage:.read); gateEncoder.useResource(ups[rank].buffer,usage:.read)
            gateEncoder.useResource(scratch[rank],usage:.write)
        }
        gateEncoder.dispatchThreadgroups(MTLSize(width:(g.rows+3)/4,height:count,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1)); gateEncoder.endEncoding()
        let downEncoder = try encoder(cb,vector ? (execution.cooperativeQuant && d.type != .q6K ? "cooperative_expert_down_vector" : "expert_down_vector") : "expert_down")
        downEncoder.setBuffer(arguments,offset:0,index:0)
        parameters.withUnsafeBytes { downEncoder.setBytes($0.baseAddress!,length:$0.count,index:1) }
        for rank in 0..<count {
            downEncoder.useResource(downs[rank].buffer,usage:.read); downEncoder.useResource(scratch[rank],usage:.read)
            downEncoder.useResource(outputs[rank],usage:.write)
        }
        downEncoder.dispatchThreadgroups(MTLSize(width:(d.rows+3)/4,height:count,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1)); downEncoder.endEncoding()
    }
    func gemv(_ cb: MTLCommandBuffer, _ w: GPUWeight, _ x: MTLBuffer, _ out: MTLBuffer) throws {
        let vector = [.vector,.staged].contains(execution.projection) && w.columns % 256 == 0 && [.q4K,.q5K,.q6K].contains(w.type)
        let e = try encoder(cb, vector ? (execution.cooperativeQuant && w.type != .q6K ? "cooperative_vector_gemv" : "vector_gemv") : "gemv")
        e.setBuffer(w.buffer, offset: w.offset, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(out, offset: 0, index: 2)
        value(e, SIMD4<UInt32>(UInt32(w.columns), UInt32(w.rows), w.type.rawValue, UInt32(w.rowBytes)), 3)
        e.dispatchThreadgroups(MTLSize(width: (w.rows + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.endEncoding()
    }
    /// Four independent token rows share each decoded weight. Each row keeps
    /// the scalar path's lane accumulation and reduction order (FP32 throughout).
    func batchGemv(_ cb: MTLCommandBuffer, _ w: GPUWeight, inputs: [MTLBuffer], outputs: [MTLBuffer]) throws {
        guard !inputs.isEmpty, inputs.count <= 256, inputs.count == outputs.count else {
            throw ParticleError.invalid("Invalid batched projection rows")
        }
        if inputs.count == 1 { try gemv(cb,w,inputs[0],outputs[0]); return }
        let arguments = try bytes(batchArguments.encodedLength)
        batchArguments.setArgumentBuffer(arguments,offset:0)
        for i in inputs.indices {
            batchArguments.setBuffer(inputs[i],offset:0,index:i)
            batchArguments.setBuffer(outputs[i],offset:0,index:256+i)
        }
        let staged = execution.projection == .staged && inputs.count >= 16 && w.columns % 64 == 0 && [.q4K,.q5K,.q6K].contains(w.type)
        let vector = [.vector,.staged].contains(execution.projection) && w.columns % 256 == 0 && [.q4K,.q5K,.q6K].contains(w.type)
        let e = try encoder(cb,staged ? "staged_gemv" : vector ? (execution.cooperativeQuant && w.type != .q6K ? "cooperative_batch_vector_gemv" : "batch_vector_gemv") : "batch_gemv")
        e.setBuffer(arguments,offset:0,index:0); e.setBuffer(w.buffer,offset:w.offset,index:1)
        value(e,SIMD4<UInt32>(UInt32(w.columns),UInt32(w.rows),w.type.rawValue,UInt32(w.rowBytes)),2)
        value(e,UInt32(inputs.count),3)
        for b in inputs { e.useResource(b,usage:.read) }
        for b in outputs { e.useResource(b,usage:.write) }
        e.dispatchThreadgroups(MTLSize(width:staged ? (w.rows+31)/32 : (w.rows+3)/4,height:staged ? (inputs.count+31)/32 : (inputs.count+3)/4,depth:1),
                               threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding()
    }
    func batchMLP(_ cb: MTLCommandBuffer, inputs: [MTLBuffer], gate: GPUWeight, up: GPUWeight, down: GPUWeight,
                  gateScratch: [MTLBuffer], outputs: [MTLBuffer]) throws {
        try batchGateUp(cb,inputs:inputs,gate:gate,up:up,outputs:gateScratch)
        try batchGemv(cb,down,inputs:gateScratch,outputs:outputs)
    }
    func batchGateUp(_ cb: MTLCommandBuffer, inputs: [MTLBuffer], gate: GPUWeight, up: GPUWeight, outputs: [MTLBuffer]) throws {
        guard !inputs.isEmpty, inputs.count <= 256, inputs.count == outputs.count,
              gate.columns == up.columns, gate.rows == up.rows, gate.type == up.type else {
            throw ParticleError.invalid("Invalid batched gate/up rows or weights")
        }
        if inputs.count == 1 { try gateUp(cb,input:inputs[0],gate:gate,up:up,output:outputs[0]); return }
        let arguments = try bytes(batchArguments.encodedLength)
        batchArguments.setArgumentBuffer(arguments,offset:0)
        for i in inputs.indices {
            batchArguments.setBuffer(inputs[i],offset:0,index:i)
            batchArguments.setBuffer(outputs[i],offset:0,index:256+i)
        }
        let staged = execution.projection == .staged && inputs.count >= 16 && gate.columns % 64 == 0 && [.q4K,.q5K,.q6K].contains(gate.type)
        let vector = execution.ffnVector && gate.columns%256 == 0 && [.q4K,.q5K,.q6K].contains(gate.type)
        let e = try encoder(cb,staged ? "staged_gate_up" : vector ? (execution.cooperativeQuant && gate.type != .q6K ? "cooperative_batch_vector_gate_up" : "batch_vector_gate_up") : "batch_gate_up")
        e.setBuffer(arguments,offset:0,index:0)
        e.setBuffer(gate.buffer,offset:gate.offset,index:1); e.setBuffer(up.buffer,offset:up.offset,index:2)
        value(e,SIMD4<UInt32>(UInt32(gate.columns),UInt32(gate.rows),gate.type.rawValue,UInt32(gate.rowBytes)),3)
        value(e,UInt32(inputs.count),4)
        for b in inputs { e.useResource(b,usage:.read) }
        for b in outputs { e.useResource(b,usage:.write) }
        e.dispatchThreadgroups(MTLSize(width:staged ? (gate.rows+31)/32 : (gate.rows+3)/4,height:staged ? (inputs.count+31)/32 : (inputs.count+3)/4,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding()
    }
    private func gateUp(_ cb: MTLCommandBuffer, input: MTLBuffer, gate: GPUWeight, up: GPUWeight, output: MTLBuffer) throws {
        guard gate.columns == up.columns, gate.rows == up.rows, gate.type == up.type else {
            throw ParticleError.invalid("Incompatible gate/up fusion")
        }
        let vector = execution.ffnVector && gate.columns%256 == 0 && [.q4K,.q5K,.q6K].contains(gate.type)
        let e = try encoder(cb,vector ? (execution.cooperativeQuant && gate.type != .q6K ? "cooperative_vector_gate_up" : "vector_gate_up") : "gate_up")
        e.setBuffer(gate.buffer,offset:gate.offset,index:0); e.setBuffer(up.buffer,offset:up.offset,index:1)
        e.setBuffer(input,offset:0,index:2); e.setBuffer(output,offset:0,index:3)
        value(e,SIMD4<UInt32>(UInt32(gate.columns),UInt32(gate.rows),gate.type.rawValue,UInt32(gate.rowBytes)),4)
        e.dispatchThreadgroups(MTLSize(width:(gate.rows+3)/4,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding()
    }
    func embedding(_ cb: MTLCommandBuffer, _ w: GPUWeight, token: Int, out: MTLBuffer) throws {
        guard (0..<w.rows).contains(token) else { throw ParticleError.invalid("Token ID outside vocabulary") }
        let e = try encoder(cb, "embedding")
        e.setBuffer(w.buffer, offset: w.offset, index: 0); e.setBuffer(out, offset: 0, index: 1)
        value(e, SIMD4<UInt32>(UInt32(w.columns), UInt32(token), w.type.rawValue, UInt32(w.rowBytes)), 2)
        e.dispatchThreads(MTLSize(width: w.columns, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    }
    func rms(_ cb: MTLCommandBuffer, _ x: MTLBuffer, _ w: GPUWeight, _ out: MTLBuffer, n: Int = 4096) throws {
        let e = try encoder(cb, "rms")
        e.setBuffer(x, offset: 0, index: 0); e.setBuffer(w.buffer, offset: w.offset, index: 1); e.setBuffer(out, offset: 0, index: 2)
        value(e, UInt32(n), 3)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    }
    func qkv(_ cb: MTLCommandBuffer, projection: MTLBuffer, qNorm: GPUWeight, kNorm: GPUWeight,
             query: MTLBuffer, keys: MTLBuffer, values: MTLBuffer, position: Int, capacity: Int) throws {
        let e = try encoder(cb, "qkv")
        e.setBuffer(projection, offset: 0, index: 0); e.setBuffer(qNorm.buffer, offset: qNorm.offset, index: 1)
        e.setBuffer(kNorm.buffer, offset: kNorm.offset, index: 2); e.setBuffer(query, offset: 0, index: 3)
        e.setBuffer(keys, offset: 0, index: 4); e.setBuffer(values, offset: 0, index: 5)
        value(e, SIMD2<UInt32>(UInt32(position), UInt32(capacity)), 6)
        e.dispatchThreadgroups(MTLSize(width: 72, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)); e.endEncoding()
    }
    func attention(_ cb: MTLCommandBuffer, query: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
                   out: MTLBuffer, position: Int, capacity: Int) throws {
        if execution.attention == .tiled || execution.attention == .split {
            try batchAttention(cb,queries:[query],keys:keys,values:values,outputs:[out],base:position,capacity:capacity)
            return
        }
        try referenceAttention(cb,query:query,keys:keys,values:values,out:out,position:position,capacity:capacity)
    }
    private func referenceAttention(_ cb: MTLCommandBuffer, query: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
                                    out: MTLBuffer, position: Int, capacity: Int) throws {
        let e = try encoder(cb, "attention")
        for (i,b) in [query,keys,values,out].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        value(e, SIMD2<UInt32>(UInt32(position), UInt32(capacity)), 4)
        e.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)); e.endEncoding()
    }
    /// Profiling inserts an attention-only command boundary. GPU timestamps
    /// exclude expert I/O; the extra synchronization is explicitly opt-in.
    func profiledAttention(_ cb: MTLCommandBuffer, queries: [MTLBuffer], keys: MTLBuffer, values: MTLBuffer,
                           outputs: [MTLBuffer], base: Int, capacity: Int) throws -> MTLCommandBuffer {
        guard execution.profileAttention else {
            try batchAttention(cb,queries:queries,keys:keys,values:values,outputs:outputs,base:base,capacity:capacity)
            return cb
        }
        submit(cb)
        let measured = try command("Particle attention profiling")
        try batchAttention(measured,queries:queries,keys:keys,values:values,outputs:outputs,base:base,capacity:capacity)
        try finish(measured)
        attentionGPUSeconds += max(0,measured.gpuEndTime-measured.gpuStartTime); attentionCalls += 1
        return try command("Particle post-attention projections")
    }
    func batchAttention(_ cb: MTLCommandBuffer, queries: [MTLBuffer], keys: MTLBuffer, values: MTLBuffer,
                        outputs: [MTLBuffer], base: Int, capacity: Int,
                        mode override: ParticleAttentionMode? = nil) throws {
        guard !queries.isEmpty, queries.count <= 256, queries.count == outputs.count,
              base >= 0, base+queries.count <= capacity else { throw ParticleError.invalid("Invalid batched attention") }
        if queries.count > 16 { splitScratch = nil }
        let requested = override ?? execution.attention
        let mode: ParticleAttentionMode
        if requested == .fast {
            let context=base+queries.count
            if queries.count<=16 && context>execution.splitBlock { mode = .split }
            else if queries.count>=32 && context>=512 { mode = .tensor }
            else if queries.count>=2 && context>=128 { mode = .tiled }
            else { mode = .reference }
        } else { mode=requested }
        // Automatic is promoted separately after whole-model parity and speed
        // gates. Explicit tiled/split modes remain available for attribution.
        if mode == .reference || mode == .automatic {
            for i in queries.indices { try referenceAttention(cb,query:queries[i],keys:keys,values:values,out:outputs[i],position:base+i,capacity:capacity) }
            return
        }
        let args = try bytes(batchArguments.encodedLength)
        batchArguments.setArgumentBuffer(args,offset:0)
        for i in queries.indices {
            batchArguments.setBuffer(queries[i],offset:0,index:i)
            batchArguments.setBuffer(outputs[i],offset:0,index:256+i)
        }
        if mode == .tensor {
            let e = try encoder(cb,"tensor_attention")
            for (i,b) in [args,keys,values].enumerated() { e.setBuffer(b,offset:0,index:i) }
            value(e,SIMD4<UInt32>(UInt32(base),UInt32(queries.count),0,0),3)
            for b in queries { e.useResource(b,usage:.read) }
            for b in outputs { e.useResource(b,usage:.write) }
            e.dispatchThreadgroups(MTLSize(width:4,height:(queries.count+3)/4,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1)); e.endEncoding()
        } else if mode == .split && queries.count <= 16 && base+queries.count > execution.splitBlock {
            let blocks = (base+queries.count+execution.splitBlock-1)/execution.splitBlock
            let length = queries.count*64*blocks*66*4
            if splitScratch == nil || splitScratch!.length < length { splitScratch = try bytes(length) }
            let parameters = SIMD4<UInt32>(UInt32(base),UInt32(queries.count),UInt32(execution.splitBlock),UInt32(blocks))
            let e = try encoder(cb,"split_attention")
            for (i,b) in [args,keys,values,splitScratch!].enumerated() { e.setBuffer(b,offset:0,index:i) }
            value(e,parameters,4)
            for b in queries { e.useResource(b,usage:.read) }
            e.dispatchThreadgroups(MTLSize(width:64,height:queries.count,depth:blocks),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1)); e.endEncoding()
            let merge = try encoder(cb,"merge_attention")
            merge.setBuffer(args,offset:0,index:0); merge.setBuffer(splitScratch!,offset:0,index:1); value(merge,parameters,2)
            for b in outputs { merge.useResource(b,usage:.write) }
            merge.dispatchThreadgroups(MTLSize(width:64,height:queries.count,depth:1),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1)); merge.endEncoding()
        } else {
            let e = try encoder(cb,"grouped_attention")
            for (i,b) in [args,keys,values].enumerated() { e.setBuffer(b,offset:0,index:i) }
            value(e,SIMD4<UInt32>(UInt32(base),UInt32(queries.count),0,0),3)
            for b in queries { e.useResource(b,usage:.read) }
            for b in outputs { e.useResource(b,usage:.write) }
            e.dispatchThreadgroups(MTLSize(width:16,height:(queries.count+1)/2,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1)); e.endEncoding()
        }
    }
    func greedyToken(_ logits: MTLBuffer) throws -> Int {
        guard logits.length >= 262144*4 else { throw ParticleError.invalid("Invalid greedy vocabulary buffer") }
        if greedyScratch == nil { greedyScratch = try bytes(1024*12); greedyResult = try bytes(8) }
        let cb = try command("Particle compact greedy sample")
        let e = try encoder(cb,"argmax")
        e.setBuffer(logits,offset:0,index:0); e.setBuffer(greedyScratch!,offset:0,index:1)
        e.dispatchThreadgroups(MTLSize(width:1024,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1)); e.endEncoding()
        let merge = try encoder(cb,"argmax_merge")
        merge.setBuffer(greedyScratch!,offset:0,index:0); merge.setBuffer(greedyResult!,offset:0,index:1)
        merge.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1)); merge.endEncoding()
        try finish(cb)
        let result = greedyResult!.contents().assumingMemoryBound(to:UInt32.self)
        guard result[1] == 0, result[0] < 262144 else { throw ParticleError.invalid("Non-finite model logits; generation stopped") }
        return Int(result[0])
    }
    func add(_ cb: MTLCommandBuffer, _ x: MTLBuffer, _ y: MTLBuffer, n: Int = 4096, scale: Float = 1) throws {
        let e = try encoder(cb, "add")
        e.setBuffer(x, offset: 0, index: 0); e.setBuffer(y, offset: 0, index: 1)
        // Explicit bytes avoid platform-dependent padding between uint and float.
        value(e, SIMD2<UInt32>(UInt32(n), scale.bitPattern), 2)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    }
    func swiglu(_ cb: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer, n: Int) throws {
        let e = try encoder(cb, "swiglu")
        e.setBuffer(gate, offset: 0, index: 0); e.setBuffer(up, offset: 0, index: 1); value(e, UInt32(n), 2)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    }
    func mlp(_ cb: MTLCommandBuffer, input: MTLBuffer, gate: GPUWeight, up: GPUWeight, down: GPUWeight,
             gateScratch: MTLBuffer, output: MTLBuffer) throws {
        try gateUp(cb,input:input,gate:gate,up:up,output:gateScratch)
        try gemv(cb, down, gateScratch, output)
    }
}

func mapWeight(_ tensor: GGUFTensor, file: any ParticleModelSource, device: MTLDevice) throws -> GPUWeight {
    let location = try file.location(for:tensor,relativeOffset:0)
    let page = Int(getpagesize()), start = location.offset / page * page, delta = location.offset - start
    let length = (delta + tensor.byteCount + page - 1) / page * page
    guard length <= device.maxBufferLength,
          let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, location.descriptor, off_t(start)), p != MAP_FAILED else {
        throw ParticleError.invalid("Cannot map resident tensor: \(tensor.name)")
    }
    nonisolated(unsafe) let captured = p
    // Match TurboFieldfare's resident mapping policy; expert data is
    // streamed separately and embedding rows use bounded pread.
    _ = posix_madvise(p,length,POSIX_MADV_RANDOM)
    guard let buffer = device.makeBuffer(bytesNoCopy: p, length: length, options: .storageModeShared,
                                         deallocator: { _,_ in munmap(captured, length) }) else {
        munmap(p, length); throw ParticleError.invalid("Cannot wrap mapped weights: \(tensor.name)")
    }
    buffer.label = tensor.name
    return GPUWeight(buffer: buffer, offset: delta, columns: tensor.shape[0], rows: tensor.shape.count > 1 ? tensor.shape[1] : 1, type: tensor.type)
}
