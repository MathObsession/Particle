import Foundation
@preconcurrency import Metal
import ParticleGGUF

extension ParticleValidation {
    /// Model-free production-shape checks. Scalar reference kernels and the
    /// independent CPU unpack checks in kernels(execution:) provide separate
    /// oracles for the cooperative decoder.
    public static func cooperativeKernels() throws -> [String] {
        var options=ParticleExecutionOptions(); options.projection = .vector; options.ffnVector=true; options.cooperativeQuant=true
        let gpu=try ParticleGPU(execution:options), reference=try ParticleGPU()
        var state: UInt64=0x631259
        func weight(_ type: GGMLType, _ n: Int, _ m: Int) throws -> GPUWeight {
            let size=n*m/256*type.blockBytes, buffer=try gpu.bytes(size+256)
            let bytes=buffer.contents().advanced(by:256).assumingMemoryBound(to:UInt8.self)
            for i in 0..<size { state=state &* 6364136223846793005 &+ 1; bytes[i]=UInt8(truncatingIfNeeded:state>>32) }
            for block in 0..<n*m/256 {
                let offset=block*type.blockBytes
                if type == .q6K { bytes[offset+208]=0x19; bytes[offset+209]=0x14 }
                else { bytes[offset]=0x19; bytes[offset+1]=0x14; bytes[offset+2]=0x19; bytes[offset+3]=0x10 }
            }
            return GPUWeight(buffer:buffer,offset:256,columns:n,rows:m,type:type)
        }
        func equal(_ actual: MTLBuffer, _ expected: MTLBuffer, count: Int, label: String) throws {
            let a=actual.contents().assumingMemoryBound(to:Float.self), b=expected.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<count where a[i].bitPattern != b[i].bitPattern {
                throw ParticleError.invalid("Cooperative native metadata changed FP32 bits: \(label), row \(i)")
            }
        }
        var reports=[String]()
        for type in [GGMLType.q4K,.q5K,.q6K] {
            try autoreleasepool {
                let gate=try weight(type,4096,1024), up=try weight(type,4096,1024), down=try weight(type,1024,4096)
                for count in [1,3,17] {
                    let input=try (0..<count).map { _ in try gpu.buffer(4096) }
                    let output=try (0..<count).map { _ in try gpu.buffer(1024) }, expected=try (0..<count).map { _ in try gpu.buffer(1024) }
                    for (row,b) in input.enumerated() { for i in 0..<4096 { b.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i+row*11)*0.03)*0.001 } }
                    for gateUp in [false,true] {
                        let cb=try gpu.command()
                        if gateUp { try gpu.batchGateUp(cb,inputs:input,gate:gate,up:up,outputs:output) }
                        else { try gpu.batchGemv(cb,gate,inputs:input,outputs:output) }
                        try gpu.finish(cb)
                        let control=try reference.command()
                        if gateUp { try reference.batchGateUp(control,inputs:input,gate:gate,up:up,outputs:expected) }
                        else { try reference.batchGemv(control,gate,inputs:input,outputs:expected) }
                        try reference.finish(control)
                        for row in 0..<count { try equal(output[row],expected[row],count:1024,label:"\(type) batch \(count), gateUp \(gateUp)") }
                    }
                }
                let input=try gpu.buffer(4096), scratch=try (0..<4).map { _ in try gpu.buffer(1024) }
                let output=try (0..<4).map { _ in try gpu.buffer(4096) }, expected=try (0..<4).map { _ in try gpu.buffer(4096) }
                for i in 0..<4096 { input.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i)*0.03)*0.001 }
                let upperType: GGMLType=type == .q4K ? .q5K : type == .q5K ? .q6K : .q4K
                let mixedUp=try weight(upperType,4096,1024)
                for count in [3,4] {
                    let cb=try gpu.command()
                    try gpu.expertMLPs(cb,input:input,gates:[GPUWeight](repeating:gate,count:count),ups:[GPUWeight](repeating:mixedUp,count:count),downs:[GPUWeight](repeating:down,count:count),scratch:Array(scratch.prefix(count)),outputs:Array(output.prefix(count)))
                    try gpu.finish(cb)
                    let control=try reference.command()
                    try reference.expertMLPs(control,input:input,gates:[GPUWeight](repeating:gate,count:count),ups:[GPUWeight](repeating:mixedUp,count:count),downs:[GPUWeight](repeating:down,count:count),scratch:Array(scratch.prefix(count)),outputs:Array(expected.prefix(count)))
                    try reference.finish(control)
                    for row in 0..<count { try equal(output[row],expected[row],count:4096,label:"\(type) grouped \(count), mixed up \(upperType)") }
                }
                reports.append("\(type): Q4/Q5 cooperative metadata and Q6 fallback bitwise match scalar projections, fused gate/up, and 3/4 mixed-type experts at production dimensions")
            }
        }
        return reports
    }
}
