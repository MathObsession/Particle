import Foundation
import ParticleGGUF

public struct ParticleKernelTiming: Codable {
    public let quantization: String
    public let implementation: String
    public let referenceImplementation: String
    public let columns: Int
    public let outputRows: Int
    public let tokens: Int
    public let scalarGPUSeconds: Double
    public let batchedGPUSeconds: Double
    public let speedup: Double
    public let referenceScalarGPUSeconds: Double, referenceBatchedGPUSeconds: Double
    public let candidateScalarSpeedup: Double, candidateBatchedSpeedup: Double
}
extension ParticleValidation {
    /// Synthetic, resident weights isolate GPU compute. These numbers exclude
    /// storage and do not predict end-to-end model throughput.
    public static func benchmarkKernels(projection: ParticleProjectionMode = .automatic, cooperative: Bool = false) throws -> [ParticleKernelTiming] {
        var execution=ParticleExecutionOptions(); execution.projection=projection; execution.cooperativeQuant=cooperative
        var control=ParticleExecutionOptions(); if cooperative { control.projection = .vector }
        let gpu = try ParticleGPU(execution:execution), reference = try ParticleGPU(execution:control), n = 4096, m = 1024
        var reports: [ParticleKernelTiming] = []
        for type in [GGMLType.q4K,.q5K,.q6K] {
            let buffer = try gpu.bytes(n*m/256*type.blockBytes+256)
            let p = buffer.contents().advanced(by:256).assumingMemoryBound(to:UInt8.self)
            for i in 0..<n*m/256*type.blockBytes { p[i] = UInt8(truncatingIfNeeded:i*17+13) }
            for block in 0..<n*m/256 {
                let b = block*type.blockBytes, off = type == .q6K ? b+208 : b
                p[off]=0x19; p[off+1]=0x14
                if type != .q6K { p[b+2]=0x19; p[b+3]=0x10 }
            }
            let w = GPUWeight(buffer:buffer,offset:256,columns:n,rows:m,type:type)
            for count in [1,4,16,128] {
                try autoreleasepool {
                    let inputs = try (0..<count).map { _ in try gpu.buffer(n) }
                    let outputs = try (0..<count).map { _ in try gpu.buffer(m) }
                    for (row,b) in inputs.enumerated() {
                        let x = b.contents().assumingMemoryBound(to:Float.self)
                        for i in 0..<n { x[i] = sin(Float(i+row)*0.03)*0.01 }
                    }
                    var timings=[[Double]](repeating:[],count:4)
                    for round in 0..<5 {
                        for offset in 0..<4 {
                            let mode=(round+offset)%4, selected=mode<2 ? reference : gpu, batch=mode%2==1
                            let cb = try selected.command("Particle isolated projection comparison")
                            if batch { try selected.batchGemv(cb,w,inputs:inputs,outputs:outputs) }
                            else { for i in 0..<count { try selected.gemv(cb,w,inputs[i],outputs[i]) } }
                            try selected.finish(cb)
                            if round>0 { timings[mode].append(cb.gpuEndTime-cb.gpuStartTime) }
                        }
                    }
                    func median(_ mode: Int) -> Double { let t=timings[mode].sorted(); return (t[1]+t[2])*0.5 }
                    let refScalar=median(0), refBatch=median(1), scalar=median(2), batch=median(3)
                    reports.append(ParticleKernelTiming(quantization:String(describing:type),implementation:cooperative ? (type == .q6K ? "Q6 native-vector fallback" : "SIMD-shared metadata FP32") : projection == .staged && count >= 16 ? "staged" : [.vector,.staged].contains(projection) ? "block metadata FP32" : "four-row FP32",referenceImplementation:cooperative ? "existing native vector FP32" : "scalar FP32",columns:n,outputRows:m,tokens:count,
                        scalarGPUSeconds:scalar,batchedGPUSeconds:batch,speedup:scalar/max(batch,1e-9),
                        referenceScalarGPUSeconds:refScalar,referenceBatchedGPUSeconds:refBatch,
                        candidateScalarSpeedup:refScalar/max(scalar,1e-9),candidateBatchedSpeedup:refBatch/max(batch,1e-9)))
                }
            }
        }
        return reports
    }
}
