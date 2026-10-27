import Foundation
import Accelerate
@preconcurrency import Metal
import ParticleGGUF

/// Independent row tasks share pinned native weights. CPU expansion is bounded
/// to 32 output rows (at most 1 MiB per worker for the dense FFN), never a model.
enum ParticleCPU {
    static func weight(_ p: UnsafeRawPointer, _ i: Int, _ type: GGMLType) -> Float {
        if type == .f32 { return p.loadUnaligned(fromByteOffset:i*4,as:Float.self) }
        func byte(_ n: Int) -> Int { Int(p.load(fromByteOffset:n,as:UInt8.self)) }
        func half(_ n: Int) -> Float { Float(Float16(bitPattern:UInt16(byte(n)) | UInt16(byte(n+1))<<8)) }
        if type == .f16 { return half(i*2) }
        let k=i%256, block=i/256*type.blockBytes
        if type == .q6K {
            let h=k/128, j=k%128, q=j/32, l=j%32
            let packed=byte(block+h*64+(q%2)*32+l)
            let lo=q<2 ? packed&15 : packed>>4
            let v=(lo | ((byte(block+128+h*32+l)>>(2*q))&3)<<4)-32
            let scale=Int(Int8(bitPattern:UInt8(byte(block+192+h*8+q*2+l/16))))
            return half(block+208)*Float(scale)*Float(v)
        }
        let g=k/32, l=k%32
        let sc=g<4 ? byte(block+4+g)&63 : (byte(block+8+g)&15) | (byte(block+g)>>6)<<4
        let mn=g<4 ? byte(block+8+g)&63 : (byte(block+8+g)>>4) | (byte(block+4+g)>>6)<<4
        let packed=byte(block+(type == .q4K ? 16 : 48)+(k/64)*32+l)
        var v=(packed>>((g%2)*4))&15
        if type == .q5K { v |= ((byte(block+16+l)>>g)&1)<<4 }
        return half(block)*Float(sc)*Float(v)-half(block+2)*Float(mn)
    }
    static func linear(_ w: GPUWeight, _ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>) {
        let columns=w.columns, rows=w.rows
        var tile=[Float](repeating:0,count:32*columns)
        for first in stride(from:0,to:rows,by:32) {
            let count=min(32,rows-first)
            for row in 0..<count {
                let p=w.buffer.contents().advanced(by:w.offset+(first+row)*w.rowBytes)
                for col in 0..<columns { tile[row*columns+col]=weight(p,col,w.type) }
            }
            tile.withUnsafeBufferPointer { p in
                cblas_sgemv(CblasRowMajor,CblasNoTrans,Int32(count),Int32(columns),1,p.baseAddress!,Int32(columns),x,1,0,out.advanced(by:first),1)
            }
        }
    }
    static func mlp(input: MTLBuffer, gate: GPUWeight, up: GPUWeight, down: GPUWeight, output: MTLBuffer) {
        var g=[Float](repeating:0,count:gate.rows), u=g
        let x=input.contents().assumingMemoryBound(to:Float.self)
        g.withUnsafeMutableBufferPointer { linear(gate,x,$0.baseAddress!) }
        u.withUnsafeMutableBufferPointer { linear(up,x,$0.baseAddress!) }
        for i in g.indices { g[i]=(g[i]/(1+exp(-g[i])))*u[i] }
        g.withUnsafeBufferPointer { linear(down,$0.baseAddress!,output.contents().assumingMemoryBound(to:Float.self)) }
    }
}

/// Both consumers claim from the same queue. GPU claims four ready rows at a
/// time; a CPU finishing early may claim another row. No task is evaluated twice.
final class ParticleTileQueue: @unchecked Sendable {
    private let lock=NSLock()
    private var next=0, cpuRows=0, gpuRows=0
    let count: Int
    init(count: Int) { self.count=count }
    func claim(_ maximum: Int, cpu: Bool? = nil) -> Range<Int>? {
        lock.lock(); defer { lock.unlock() }
        guard next<count else { return nil }
        let start=next; next=min(count,next+maximum)
        if cpu==true { cpuRows += next-start }; if cpu==false { gpuRows += next-start }
        return start..<next
    }
    var counts: (cpu: Int,gpu: Int) { lock.lock(); defer { lock.unlock() }; return (cpuRows,gpuRows) }
}
struct ParticleHybridWorkMetrics { let cpuRows: Int, gpuRows: Int }
private final class HybridWork: @unchecked Sendable {
    let inputs: [MTLBuffer], outputs: [MTLBuffer]
    let gate: GPUWeight, up: GPUWeight, down: GPUWeight
    let queue: ParticleTileQueue
    let cancellation: ParticleCancellation?
    init(inputs: [MTLBuffer], outputs: [MTLBuffer], gate: GPUWeight, up: GPUWeight, down: GPUWeight, cancellation: ParticleCancellation?) {
        self.inputs=inputs; self.outputs=outputs; self.gate=gate; self.up=up; self.down=down
        self.cancellation=cancellation; queue=ParticleTileQueue(count:inputs.count)
    }
    func runCPU() {
        while cancellation?.isCancelled != true, let rows=queue.claim(1,cpu:true) {
            for row in rows { ParticleCPU.mlp(input:inputs[row],gate:gate,up:up,down:down,output:outputs[row]) }
        }
    }
}
extension ParticleGPU {
    /// Inputs must already be complete and weights remain pinned until return.
    /// This path is explicitly experimental: Accelerate changes reduction order.
    @discardableResult func hybridMLP(inputs: [MTLBuffer], gate: GPUWeight, up: GPUWeight, down: GPUWeight,
                   gateScratch: [MTLBuffer], outputs: [MTLBuffer], cancellation: ParticleCancellation?) throws -> ParticleHybridWorkMetrics {
        let work=HybridWork(inputs:inputs,outputs:outputs,gate:gate,up:up,down:down,cancellation:cancellation)
        let group=DispatchGroup()
        for _ in 0..<2 { group.enter(); DispatchQueue.global(qos:.userInitiated).async { defer { group.leave() }; work.runCPU() } }
        defer { group.wait() }
        while cancellation?.isCancelled != true, let range=work.queue.claim(4,cpu:false) {
            let cb=try command("Particle hybrid GPU row tiles")
            try batchMLP(cb,inputs:range.map { inputs[$0] },gate:gate,up:up,down:down,
                gateScratch:range.map { gateScratch[$0] },outputs:range.map { outputs[$0] })
            try finish(cb)
        }
        group.wait()
        if cancellation?.isCancelled == true { throw CancellationError() }
        let counts=work.queue.counts
        return ParticleHybridWorkMetrics(cpuRows:counts.cpu,gpuRows:counts.gpu)
    }
}
