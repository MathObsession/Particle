import Foundation
import Testing
@preconcurrency import Metal
import ParticleGGUF
@testable import ParticleCore

@Suite(.serialized) struct ProjectionOptimizationTests {
    private func weight(_ gpu: ParticleGPU, type: GGMLType, columns: Int, rows: Int) throws -> (GPUWeight,[UInt8]) {
        var bytes=[UInt8](repeating:0,count:columns*rows/type.blockElements*type.blockBytes)
        for i in bytes.indices { bytes[i]=UInt8(truncatingIfNeeded:i*17+13) }
        if type == .f32 {
            for i in 0..<columns*rows {
                let bits=(sin(Float(i)*0.017)*0.01).bitPattern
                for j in 0..<4 { bytes[i*4+j]=UInt8(truncatingIfNeeded:bits>>(j*8)) }
            }
        } else {
            for block in 0..<columns*rows/256 {
                let base=block*type.blockBytes, off=type == .q6K ? base+208 : base
                bytes[off]=0x19; bytes[off+1]=0x14
                if type != .q6K { bytes[base+2]=0x19; bytes[base+3]=0x10 }
            }
        }
        let buffer=try gpu.bytes(bytes.count+256)
        bytes.withUnsafeBytes { buffer.contents().advanced(by:256).copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
        return (GPUWeight(buffer:buffer,offset:256,columns:columns,rows:rows,type:type),bytes)
    }
    @Test func stagedNativeKQuantsAndFusedActivationUseBoundedTiles() throws {
        var options=ParticleExecutionOptions(); options.projection = .staged
        let gpu=try ParticleGPU(execution:options), referenceGPU=try ParticleGPU()
        for type in [GGMLType.q4K,.q5K,.q6K] {
            let (w,packed)=try weight(gpu,type:type,columns:256,rows:35)
            let decoded=QuantReference.decode(packed,type:type,count:256*35)
            var other=packed
            for block in 0..<256*35/256 { other[block*type.blockBytes+(type == .q4K ? 16 : type == .q5K ? 48 : 0)] ^= 0x35 }
            let upper=try gpu.bytes(other.count+512)
            other.withUnsafeBytes { upper.contents().advanced(by:512).copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
            let up=GPUWeight(buffer:upper,offset:512,columns:256,rows:35,type:type)
            let upDecoded=QuantReference.decode(other,type:type,count:256*35)
            for count in [1,17,65,256] {
                let inputs=try (0..<count).map { _ in try gpu.buffer(256) }
                let out=try (0..<count).map { _ in try gpu.buffer(35) }
                let ref=try (0..<count).map { _ in try gpu.buffer(35) }
                let fused=try (0..<count).map { _ in try gpu.buffer(35) }
                for (row,b) in inputs.enumerated() { for i in 0..<256 { b.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i+row)*0.03)*0.01 } }
                let cb=try gpu.command()
                try gpu.batchGemv(cb,w,inputs:inputs,outputs:out)
                try gpu.batchGateUp(cb,inputs:inputs,gate:w,up:up,outputs:fused)
                try gpu.finish(cb)
                let control=try referenceGPU.command()
                try referenceGPU.batchGemv(control,w,inputs:inputs,outputs:ref)
                try referenceGPU.finish(control)
                for row in 0..<count { for i in 0..<35 {
                    let a=out[row].contents().assumingMemoryBound(to:Float.self)[i], b=ref[row].contents().assumingMemoryBound(to:Float.self)[i]
                    let f=fused[row].contents().assumingMemoryBound(to:Float.self)[i]
                    let x=inputs[row].contents().assumingMemoryBound(to:Float.self)
                    let expected=Float((0..<256).reduce(0.0) { partial,col in
                        partial+(count>=16 ? Double(Float16(decoded[i*256+col]))*Double(Float16(x[col])) : Double(decoded[i*256+col])*Double(x[col]))
                    })
                    let upExpected=Float((0..<256).reduce(0.0) { partial,col in
                        partial+(count>=16 ? Double(Float16(upDecoded[i*256+col]))*Double(Float16(x[col])) : Double(upDecoded[i*256+col])*Double(x[col]))
                    })
                    #expect(a.isFinite && abs(a-expected)<0.00002+abs(expected)*0.0001)
                    if count==1 { #expect(a.bitPattern==b.bitPattern) }
                    // Half staging is a different numerical path, never an
                    // assertion of FP32 parity. The model gate decides promotion.
                    #expect(b.isFinite)
                    #expect(f.isFinite && abs(f-(expected/(1+exp(-expected)))*upExpected)<0.00002+abs(f)*0.002)
                } }
            }
        }
    }
    @Test func vectorMetadataReusePreservesEveryFP32Bit() throws {
        var options=ParticleExecutionOptions(); options.projection = .vector
        let gpu=try ParticleGPU(execution:options), reference=try ParticleGPU()
        for type in [GGMLType.q4K,.q5K,.q6K] {
            let (w,_)=try weight(gpu,type:type,columns:4096,rows:35)
            for count in [1,3,17,128,256] {
                let inputs=try (0..<count).map { _ in try gpu.buffer(4096) }
                let out=try (0..<count).map { _ in try gpu.buffer(35) }, ref=try (0..<count).map { _ in try gpu.buffer(35) }
                for (row,b) in inputs.enumerated() { for i in 0..<4096 { b.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i+row)*0.03)*0.01 } }
                let a=try gpu.command(); try gpu.batchGemv(a,w,inputs:inputs,outputs:out); try gpu.finish(a)
                let b=try reference.command(); try reference.batchGemv(b,w,inputs:inputs,outputs:ref); try reference.finish(b)
                var mismatches=0
                for row in 0..<count { for i in 0..<35 {
                    if out[row].contents().assumingMemoryBound(to:Float.self)[i].bitPattern != ref[row].contents().assumingMemoryBound(to:Float.self)[i].bitPattern { mismatches += 1 }
                } }
                #expect(mismatches==0,"Native block reuse must be exact: \(type), rows \(count)")
            }
        }
    }
    @Test func compactGreedySampleKeepsTiesAndRejectsNonFiniteValues() throws {
        let gpu=try ParticleGPU(), logits=try gpu.buffer(262144), p=logits.contents().assumingMemoryBound(to:Float.self)
        for i in 0..<262144 { p[i]=Float(i%13)-20 }
        p[891]=10; p[1555]=10
        #expect(try gpu.greedyToken(logits)==891)
        p[1]=Float.nan; #expect(throws:ParticleError.self) { try gpu.greedyToken(logits) }
        p[1]=0; p[262143] = .infinity; #expect(throws:ParticleError.self) { try gpu.greedyToken(logits) }
        for i in 0..<262144 { p[i] = -1 }
        #expect(try gpu.greedyToken(logits)==0)
    }
    @Test func cpuNativeUnpackingAndHybridMLPMatchReference() throws {
        let gpu=try ParticleGPU()
        for type in [GGMLType.q4K,.q5K,.q6K,.f32] {
            let (w,bytes)=try weight(gpu,type:type,columns:256,rows:256)
            let decoded=QuantReference.decode(bytes,type:type,count:256*256)
            let pointer=w.buffer.contents().advanced(by:w.offset)
            for i in decoded.indices { #expect(ParticleCPU.weight(pointer,i,type)==decoded[i]) }
            let (down,_)=try weight(gpu,type:type,columns:256,rows:33)
            let inputs=try (0..<9).map { _ in try gpu.buffer(256) }, scratch=try (0..<9).map { _ in try gpu.buffer(256) }
            let out=try (0..<9).map { _ in try gpu.buffer(33) }, ref=try (0..<9).map { _ in try gpu.buffer(33) }
            for (row,b) in inputs.enumerated() { for i in 0..<256 { b.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i+row)*0.03)*0.01 } }
            let cb=try gpu.command()
            try gpu.batchMLP(cb,inputs:inputs,gate:w,up:w,down:down,gateScratch:scratch,outputs:ref)
            try gpu.finish(cb)
            ParticleCPU.mlp(input:inputs[0],gate:w,up:w,down:down,output:out[0])
            for i in 0..<33 {
                let a=out[0].contents().assumingMemoryBound(to:Float.self)[i], b=ref[0].contents().assumingMemoryBound(to:Float.self)[i]
                #expect(abs(a-b)<0.00001+abs(b)*0.0001)
            }
            try gpu.hybridMLP(inputs:inputs,gate:w,up:w,down:down,gateScratch:scratch,outputs:out,cancellation:nil)
            for row in 0..<9 { for i in 0..<33 {
                let a=out[row].contents().assumingMemoryBound(to:Float.self)[i], b=ref[row].contents().assumingMemoryBound(to:Float.self)[i]
                #expect(a.isFinite && abs(a-b)<0.00001+abs(b)*0.0001)
            } }
        }
    }
    @Test func nativeFFNBlockReusePreservesAllGateAndUpBits() throws {
        var options = ParticleExecutionOptions(); options.ffnVector = true
        let gpu = try ParticleGPU(execution:options), reference = try ParticleGPU()
        for type in [GGMLType.q4K,.q5K,.q6K,.f32] {
            let (gate,_) = try weight(gpu,type:type,columns:4096,rows:35)
            let (up,_) = try weight(gpu,type:type,columns:4096,rows:35)
            if type != .f32 {
                let bytes = up.buffer.contents().advanced(by:up.offset).assumingMemoryBound(to:UInt8.self)
                for block in 0..<4096*35/256 { bytes[block*type.blockBytes+(type == .q4K ? 16 : type == .q5K ? 48 : 0)] ^= 0x37 }
            }
            for count in [1,3,17,128,256] {
                try autoreleasepool {
                    let inputs = try (0..<count).map { _ in try gpu.buffer(4096) }
                    let outputs = try (0..<count).map { _ in try gpu.buffer(35) }
                    let expected = try (0..<count).map { _ in try gpu.buffer(35) }
                    let upper = try (0..<count).map { _ in try gpu.buffer(35) }
                    for (row,input) in inputs.enumerated() { for column in 0..<4096 {
                        input.contents().assumingMemoryBound(to:Float.self)[column] = sin(Float(column+row*13)*0.03)*0.01
                    } }
                    let cb = try gpu.command()
                    try gpu.batchGateUp(cb,inputs:inputs,gate:gate,up:up,outputs:outputs); try gpu.finish(cb)
                    let control = try reference.command()
                    try reference.batchGemv(control,gate,inputs:inputs,outputs:expected)
                    try reference.batchGemv(control,up,inputs:inputs,outputs:upper)
                    for row in 0..<count { try reference.swiglu(control,gate:expected[row],up:upper[row],n:35) }
                    try reference.finish(control)
                    var mismatches = 0
                    for row in 0..<count { for column in 0..<35 {
                        if outputs[row].contents().assumingMemoryBound(to:Float.self)[column].bitPattern != expected[row].contents().assumingMemoryBound(to:Float.self)[column].bitPattern { mismatches += 1 }
                    } }
                    #expect(mismatches == 0,"FFN block metadata reuse must preserve FP32 order: \(type), \(count) rows")
                }
            }
        }
    }

}
