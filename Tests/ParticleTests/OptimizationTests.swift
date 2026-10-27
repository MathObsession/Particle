import Foundation
import Testing
import ParticleGGUF
@testable import ParticleCore

@Suite struct OptimizationTests {
    @Test func prefixReuseHonorsReformattedHistoryAndUnforwardedTail() throws {
        let plan = try ParticlePromptReuse(committed:[6,107,25,19,107,100,101],target:[6,107,25,19,107,28,29,18],context:32,outputAllowance:2)
        #expect(plan.reused == 5); #expect(plan.suffix == [28,29,18])
        let capped = try ParticlePromptReuse(committed:[1,2,3],target:[1,2,3,4,5],context:32,outputAllowance:2)
        #expect(capped.reused == 3); #expect(capped.suffix == [4,5])
        let repeated = try ParticlePromptReuse(committed:[1,2,3,4],target:[1,2,3],context:32,outputAllowance:2)
        #expect(repeated.reused == 2); #expect(repeated.suffix == [3])
        #expect(throws:ParticleError.self) { try ParticlePromptReuse(committed:[],target:[1,2,3],context:4,outputAllowance:2) }
    }
    @Test func boundedSamplerKeepsExactRankingAndTies() {
        let logits = (0..<262144).map { Float(($0*7919)%1000) }
        for k in [1,6,50,512] {
            let expected = Array(logits.indices.sorted { logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1] }.prefix(k))
            #expect(ParticleSampler.topCandidates(logits,count:k) == expected)
        }
    }
    @Test func batchedKQuantProjectionsAndFusion() throws {
        let gpu = try ParticleGPU()
        // Production dimensions, nonzero offsets, every native format and
        // incomplete four-row tiles. Compare every output, not sampled rows.
        for type in [GGMLType.q4K,.q5K,.q6K,.f32] {
            let n = 4096, m = 35
            var bytes = [UInt8](repeating:0,count:n*m/type.blockElements*type.blockBytes)
            var state: UInt64 = 123
            for i in bytes.indices { state = state &* 6364136223846793005 &+ 1; bytes[i] = UInt8(truncatingIfNeeded:state>>32) }
            if type == .f32 {
                for i in 0..<n*m {
                    let bits = sin(Float(i)*0.017).bitPattern
                    for j in 0..<4 { bytes[4*i+j] = UInt8(truncatingIfNeeded:bits>>(8*j)) }
                }
            } else {
                for block in 0..<n*m/256 {
                    let p = block*type.blockBytes
                    let offset = type == .q6K ? p+208 : p
                    bytes[offset]=0x19; bytes[offset+1]=0x14
                    if type != .q6K { bytes[p+2]=0x19; bytes[p+3]=0x10 }
                }
            }
            let b = try gpu.bytes(bytes.count+256)
            bytes.withUnsafeBytes { b.contents().advanced(by:256).copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
            let w = GPUWeight(buffer:b,offset:256,columns:n,rows:m,type:type)
            let reference = QuantReference.decode(bytes,type:type,count:n*m)
            var other = bytes
            if type == .f32 { for i in 0..<n*m { other[4*i+3] ^= 0x80 } }
            else { for block in 0..<n*m/256 { other[block*type.blockBytes+(type == .q4K ? 16 : type == .q5K ? 48 : 0)] ^= 0x35 } }
            let ub = try gpu.bytes(other.count+256)
            other.withUnsafeBytes { ub.contents().advanced(by:256).copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
            let uw = GPUWeight(buffer:ub,offset:256,columns:n,rows:m,type:type)
            for count in [1,3,4,5,17,128] {
                let inputs = try (0..<count).map { _ in try gpu.buffer(n) }
                let outputs = try (0..<count).map { _ in try gpu.buffer(m) }
                let scalars = try (0..<count).map { _ in try gpu.buffer(m) }
                for row in 0..<count {
                    let p = inputs[row].contents().assumingMemoryBound(to:Float.self)
                    for i in 0..<n { p[i]=sin(Float(i+row*13)*0.03)*0.01 }
                }
                let cb = try gpu.command()
                try gpu.batchGemv(cb,w,inputs:inputs,outputs:outputs)
                for row in 0..<count { try gpu.gemv(cb,w,inputs[row],scalars[row]) }
                try gpu.finish(cb)
                for row in 0..<count {
                    let a = outputs[row].contents().assumingMemoryBound(to:Float.self)
                    let b = scalars[row].contents().assumingMemoryBound(to:Float.self)
                    let x = inputs[row].contents().assumingMemoryBound(to:Float.self)
                    for j in 0..<m {
                        #expect(a[j].bitPattern == b[j].bitPattern)
                        if row == 0 {
                            let expected = Float((0..<n).reduce(0.0) { $0 + Double(reference[j*n+$1])*Double(x[$1]) })
                            #expect(abs(a[j]-expected) < 0.002+abs(expected)*0.00003)
                        }
                    }
                }
                let fusedOutputs = try (0..<count).map { _ in try gpu.buffer(m) }
                let cb2 = try gpu.command()
                try gpu.batchGateUp(cb2,inputs:inputs,gate:w,up:uw,outputs:fusedOutputs)
                for row in 0..<count {
                    try gpu.gemv(cb2,uw,inputs[row],scalars[row])
                    try gpu.swiglu(cb2,gate:outputs[row],up:scalars[row],n:m)
                }
                try gpu.finish(cb2)
                for row in 0..<count { for j in 0..<m {
                    #expect(outputs[row].contents().assumingMemoryBound(to:Float.self)[j].bitPattern == fusedOutputs[row].contents().assumingMemoryBound(to:Float.self)[j].bitPattern)
                } }
            }
            // Fused gate/up + SwiGLU must match the original sequence exactly.
            let x = try gpu.buffer(n), g = try gpu.buffer(m), u = try gpu.buffer(m)
            let fused = try gpu.buffer(m), out = try gpu.buffer(m)
            let cb = try gpu.command()
            let xp=x.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<n { xp[i]=cos(Float(i)*0.02)*0.001 }
            try gpu.gemv(cb,w,x,g); try gpu.gemv(cb,uw,x,u); try gpu.swiglu(cb,gate:g,up:u,n:m)
            // Down is an identity-like FP32 fixture with shape [m,m].
            let identity = try gpu.buffer(m*m), ip = identity.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<m*m { ip[i] = i/m == i%m ? 1 : 0 }
            let down = GPUWeight(buffer:identity,offset:0,columns:m,rows:m,type:.f32)
            try gpu.mlp(cb,input:x,gate:w,up:uw,down:down,gateScratch:fused,output:out)
            try gpu.finish(cb)
            for i in 0..<m { #expect(g.contents().assumingMemoryBound(to:Float.self)[i].bitPattern == fused.contents().assumingMemoryBound(to:Float.self)[i].bitPattern) }
        }
    }
    @Test func gpuTopSixMatchesStableReferenceAndRejectsNonFinite() throws {
        let gpu = try ParticleGPU(), logits = try gpu.buffer(128), bias = try gpu.buffer(128)
        let ids = try gpu.bytes(32), weights = try gpu.buffer(6)
        let lp = logits.contents().assumingMemoryBound(to:Float.self), bp = bias.contents().assumingMemoryBound(to:Float.self)
        let b = GPUWeight(buffer:bias,offset:0,columns:128,rows:1,type:.f32)
        for fixture in 0..<40 {
            for i in 0..<128 {
                lp[i] = fixture == 0 ? 0 : fixture == 1 ? -1000 : sin(Float(i*7919+fixture*31))*12
                bp[i] = fixture < 2 ? 0 : cos(Float(i*17+fixture))*0.25
            }
            let expected = try SarvamRouting.select(logits:Array(UnsafeBufferPointer(start:lp,count:128)),bias:Array(UnsafeBufferPointer(start:bp,count:128)))
            let cb = try gpu.command(); try gpu.routeSix(cb,logits:logits,bias:b,ids:ids,weights:weights); try gpu.finish(cb)
            let result = try gpu.routingResult(ids:ids,weights:weights)
            #expect(result.experts == expected.experts)
            for (x,y) in zip(result.weights,expected.weights) { #expect(abs(x-y) < 0.000001) }
        }
        for invalid in [Float.nan,Float.infinity,-Float.infinity] {
            lp[127] = invalid
            let cb = try gpu.command(); try gpu.routeSix(cb,logits:logits,bias:b,ids:ids,weights:weights); try gpu.finish(cb)
            #expect(throws:ParticleError.self) { try gpu.routingResult(ids:ids,weights:weights) }
        }
    }

    @Test func groupedExpertDispatchMatchesEachNativeProjection() throws {
        let gpu = try ParticleGPU()
        for (n,hidden) in [(512,256),(4096,1024)] {
            var state: UInt64 = 761
            func weight(_ type: GGMLType, columns: Int, rows: Int) throws -> GPUWeight {
                let buffer = try gpu.bytes(columns*rows/type.blockElements*type.blockBytes+256)
                let p = buffer.contents().advanced(by:256).assumingMemoryBound(to:UInt8.self)
                for i in 0..<columns*rows/type.blockElements*type.blockBytes { state = state &* 6364136223846793005 &+ 1; p[i] = UInt8(truncatingIfNeeded:state>>32) }
                for block in 0..<columns*rows/256 {
                    let base = block*type.blockBytes, scale = type == .q6K ? base+208 : base
                    p[scale]=0x19; p[scale+1]=0x14
                    if type != .q6K { p[base+2]=0x19; p[base+3]=0x10 }
                }
                return GPUWeight(buffer:buffer,offset:256,columns:columns,rows:rows,type:type)
            }
            let input = try gpu.buffer(n)
            for i in 0..<n { input.contents().assumingMemoryBound(to:Float.self)[i] = sin(Float(i)*0.03)*0.001 }
            for (gt,ut,dt) in [(GGMLType.q4K,GGMLType.q4K,GGMLType.q6K),(.q5K,.q5K,.q4K),(.q6K,.q6K,.q5K),(.q4K,.q5K,.q6K)] {
                try autoreleasepool {
                    let gates = try (0..<6).map { _ in try weight(gt,columns:n,rows:hidden) }
                    let ups = try (0..<6).map { _ in try weight(ut,columns:n,rows:hidden) }
                    let downs = try (0..<6).map { _ in try weight(dt,columns:hidden,rows:n) }
                    let scratch = try (0..<6).map { _ in try gpu.buffer(hidden) }, outputs = try (0..<6).map { _ in try gpu.buffer(n) }
                    let expected = try (0..<6).map { _ in try gpu.buffer(n) }
                    let g = try gpu.buffer(hidden), u = try gpu.buffer(hidden)
                    let scalar = try gpu.command()
                    for rank in 0..<6 {
                        try gpu.gemv(scalar,gates[rank],input,g); try gpu.gemv(scalar,ups[rank],input,u)
                        try gpu.swiglu(scalar,gate:g,up:u,n:hidden); try gpu.gemv(scalar,downs[rank],g,expected[rank])
                    }
                    try gpu.finish(scalar)
                    for vectorized in [false,true] {
                        var options = ParticleExecutionOptions(); options.ffnVector = vectorized
                        try gpu.configure(options)
                        for ranks in [[0],[1,4],[0,2,5],[1,2,3,5],[0,1,2,3,4,5]] {
                            let cb = try gpu.command()
                            try gpu.expertMLPs(cb,input:input,gates:ranks.map { gates[$0] },ups:ranks.map { ups[$0] },downs:ranks.map { downs[$0] },scratch:ranks.map { scratch[$0] },outputs:ranks.map { outputs[$0] })
                            try gpu.finish(cb)
                            for rank in ranks { for row in 0..<n {
                                #expect(outputs[rank].contents().assumingMemoryBound(to:Float.self)[row].bitPattern == expected[rank].contents().assumingMemoryBound(to:Float.self)[row].bitPattern)
                            } }
                        }
                    }
                }
            }
        }
    }

}
