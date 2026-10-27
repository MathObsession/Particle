import Foundation
import ParticleGGUF

public struct ParticleAttentionTiming: Codable {
    public let context: Int, suffix: Int, splitBlock: Int
    public let referenceGPUSeconds: Double, tiledGPUSeconds: Double, splitGPUSeconds: Double
    public let tiledSpeedup: Double, splitSpeedup: Double
    public let tensorGPUSeconds: Double, tensorSpeedup: Double
    public let tensorMaxError: Float
    public let tiledBitwiseMismatches: Int
    public let splitMaxError: Float
}
extension ParticleValidation {
    /// Resident synthetic K/V isolates attention from storage. Context includes
    /// the new suffix; benchmarks do not fabricate a full-model long-context run.
    public static func benchmarkAttention() throws -> [ParticleAttentionTiming] {
        var result: [ParticleAttentionTiming]=[]
        for block in [256,512] {
            var options=ParticleExecutionOptions(); options.splitBlock=block; options.attention = .tensor
            let gpu=try ParticleGPU(execution:options), capacity=8192
            let keys=try gpu.buffer(capacity*256), values=try gpu.buffer(capacity*256)
            for (buffer,scale) in [(keys,Float(0.013)),(values,Float(0.017))] {
                let p=buffer.contents().assumingMemoryBound(to:Float.self)
                for i in 0..<capacity*256 { p[i]=sin(Float(i)*scale)*0.2 }
            }
            for context in [32,128,512,2048,4096,8192] { for count in [1,16,64,128] where count<=context {
                try autoreleasepool {
                    let queries=try (0..<count).map { _ in try gpu.buffer(4096) }
                    let a=try (0..<count).map { _ in try gpu.buffer(4096) }
                    let b=try (0..<count).map { _ in try gpu.buffer(4096) }
                    let c=try (0..<count).map { _ in try gpu.buffer(4096) }
                    let d=try (0..<count).map { _ in try gpu.buffer(4096) }
                    for (row,buffer) in queries.enumerated() {
                        let p=buffer.contents().assumingMemoryBound(to:Float.self)
                        for i in 0..<4096 { p[i]=cos(Float(i+row*19)*0.021) }
                    }
                    var timings: [ParticleAttentionMode:[Double]]=[:]
                    let modes: [ParticleAttentionMode]=[.reference,.tiled,.split,.tensor]
                    for round in 0..<5 {
                        for index in 0..<4 {
                            let mode=modes[(index+round)%4], outputs=mode == .reference ? a : mode == .tiled ? b : mode == .split ? c : d
                            let cb=try gpu.command("Particle attention candidate benchmark")
                            try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:outputs,base:context-count,capacity:capacity,mode:mode)
                            try gpu.finish(cb)
                            if round>0 { timings[mode,default:[]].append(cb.gpuEndTime-cb.gpuStartTime) }
                        }
                    }
                    var mismatches=0, splitError: Float=0, tensorError: Float=0
                    for row in 0..<count {
                        let x=a[row].contents().assumingMemoryBound(to:Float.self), y=b[row].contents().assumingMemoryBound(to:Float.self), z=c[row].contents().assumingMemoryBound(to:Float.self)
                        let t=d[row].contents().assumingMemoryBound(to:Float.self)
                        for i in 0..<4096 { tensorError=max(tensorError,abs(x[i]-t[i])); if x[i].bitPattern != y[i].bitPattern { mismatches += 1 }; splitError=max(splitError,abs(x[i]-z[i])) }
                    }
                    func median(_ mode: ParticleAttentionMode) -> Double { let t=timings[mode]!.sorted(); return (t[1]+t[2])*0.5 }
                    let ref=median(.reference), tiled=median(.tiled), split=median(.split), tensor=median(.tensor)
                    result.append(ParticleAttentionTiming(context:context,suffix:count,splitBlock:block,
                        referenceGPUSeconds:ref,tiledGPUSeconds:tiled,splitGPUSeconds:split,tiledSpeedup:ref/tiled,splitSpeedup:ref/split,
                        tensorGPUSeconds:tensor,tensorSpeedup:ref/tensor,tensorMaxError:tensorError,tiledBitwiseMismatches:mismatches,splitMaxError:splitError))
                }
            } }
        }
        return result
    }
}
