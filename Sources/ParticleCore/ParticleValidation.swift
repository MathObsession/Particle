import Foundation
@preconcurrency import Metal
import ParticleGGUF

/// Independent bulk CPU unpacker used only for bounded validation fixtures.
/// Block semantics follow GGML's MIT-licensed dequantize_row_q{4,5,6}_K.
enum QuantReference {
    static func decode(_ bytes: [UInt8], type: GGMLType, count: Int) -> [Float] {
        func half(_ p: Int) -> Float { Float(Float16(bitPattern: UInt16(bytes[p]) | (UInt16(bytes[p+1]) << 8))) }
        if type == .f32 { return (0..<count).map { i in
            let p = i * 4
            return Float(bitPattern: UInt32(bytes[p]) | UInt32(bytes[p+1]) << 8 | UInt32(bytes[p+2]) << 16 | UInt32(bytes[p+3]) << 24)
        } }
        if type == .f16 { return (0..<count).map { half($0 * 2) } }
        var out = [Float](repeating: 0, count: count)
        for block in 0..<(count / 256) {
            let p = block * type.blockBytes, base = block * 256
            if type == .q6K {
                let d = half(p + 208)
                for h in 0..<2 { for lane in 0..<32 {
                    let l0 = bytes[p+h*64+lane], l1 = bytes[p+h*64+32+lane], high = bytes[p+128+h*32+lane]
                    let quants = [Int(l0 & 15) | Int(high & 3) << 4,
                                  Int(l1 & 15) | Int((high >> 2) & 3) << 4,
                                  Int(l0 >> 4) | Int((high >> 4) & 3) << 4,
                                  Int(l1 >> 4) | Int((high >> 6) & 3) << 4]
                    for q in 0..<4 {
                        let scale = Int8(bitPattern: bytes[p+192+h*8+lane/16+q*2])
                        out[base+h*128+q*32+lane] = d * Float(scale) * Float(quants[q]-32)
                    }
                } }
            } else {
                func scales(_ g: Int) -> (Float, Float) {
                    let s = Array(bytes[(p+4)..<(p+16)])
                    if g < 4 { return (Float(s[g] & 63), Float(s[g+4] & 63)) }
                    return (Float((s[g+4] & 15) | ((s[g-4] >> 6) << 4)), Float((s[g+4] >> 4) | ((s[g] >> 6) << 4)))
                }
                let d = half(p), min = half(p+2)
                for chunk in 0..<4 {
                    for halfGroup in 0..<2 {
                        let group = chunk * 2 + halfGroup, (scale,minimum) = scales(chunk*2+halfGroup)
                        for lane in 0..<32 {
                            let packed = bytes[p+(type == .q4K ? 16 : 48)+chunk*32+lane]
                            var q = Int(halfGroup == 0 ? packed & 15 : packed >> 4)
                            if type == .q5K, bytes[p+16+lane] & (1 << group) != 0 { q += 16 }
                            out[base+chunk*64+halfGroup*32+lane] = d * scale * Float(q) - min * minimum
                        }
                    }
                }
            }
        }
        return out
    }
}

public enum ParticleValidation {
    public static func kernels(execution: ParticleExecutionOptions = .init()) throws -> [String] {
        let gpu = try ParticleGPU(execution:execution)
        var reports: [String] = []
        let columns = 512, rows = 7
        let x = try gpu.buffer(columns)
        let input = (0..<columns).map { sin(Float($0) * 0.13) }
        input.withUnsafeBytes { x.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        for type in GGMLType.allCases {
            var bytes = [UInt8](repeating: 0, count: columns * rows / type.blockElements * type.blockBytes)
            var state: UInt64 = 0x12345678
            for i in bytes.indices { state = state &* 6364136223846793005 &+ 1; bytes[i] = UInt8(truncatingIfNeeded: state >> 32) }
            func putHalf(_ offset: Int, _ value: Float16) {
                bytes[offset] = UInt8(truncatingIfNeeded: value.bitPattern); bytes[offset+1] = UInt8(truncatingIfNeeded: value.bitPattern >> 8)
            }
            if type == .f32 {
                for i in 0..<(columns*rows) {
                    let v = sin(Float(i)*0.27).bitPattern
                    for j in 0..<4 { bytes[i*4+j] = UInt8(truncatingIfNeeded: v >> (j*8)) }
                }
            } else if type == .f16 {
                for i in 0..<(columns*rows) { putHalf(i*2, Float16(sin(Float(i)*0.27))) }
            } else {
                for block in 0..<(columns*rows/256) {
                    let p = block * type.blockBytes
                    if type == .q6K { putHalf(p+208, 0.001) }
                    else { putHalf(p, 0.01); putHalf(p+2, 0.005) }
                }
            }
            let buffer = try gpu.bytes(bytes.count + 64)
            bytes.withUnsafeBytes { buffer.contents().advanced(by: 64).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            let weight = GPUWeight(buffer: buffer, offset: 64, columns: columns, rows: rows, type: type)
            let out = try gpu.buffer(rows), cb = try gpu.command("Particle quantization parity")
            try gpu.gemv(cb, weight, x, out); try gpu.finish(cb)
            let ref = QuantReference.decode(bytes, type: type, count: columns*rows)
            let result = out.contents().assumingMemoryBound(to: Float.self)
            var error: Float = 0
            for row in 0..<rows {
                let expected = Float((0..<columns).reduce(0.0) { $0 + Double(ref[row*columns+$1]) * Double(input[$1]) })
                let delta = abs(result[row]-expected); error = max(error,delta)
                guard result[row].isFinite, delta < 0.002 + abs(expected) * 0.00003 else {
                    throw ParticleError.invalid("\(type) GPU/CPU mismatch at row \(row): \(result[row]) vs \(expected)")
                }
            }
            reports.append("\(type): GEMV matches independent CPU unpack (max abs error \(error))")
            let embedding = try gpu.buffer(columns), ec = try gpu.command("Particle embedding row parity")
            try gpu.embedding(ec,weight,token:3,out:embedding); try gpu.finish(ec)
            let elements = embedding.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<columns where abs(elements[i]-ref[3*columns+i]) > 0.00001 {
                throw ParticleError.invalid("\(type) embedding row mismatch at \(i)")
            }
        }
        let d = 4096, normalized = try gpu.buffer(d), source = try gpu.buffer(d), gain = try gpu.buffer(d)
        let a = source.contents().assumingMemoryBound(to: Float.self), w = gain.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<d { a[i] = sin(Float(i)*0.031); w[i] = 0.5 + Float(i%31)*0.02 }
        let cb = try gpu.command("Particle RMSNorm parity")
        try gpu.rms(cb, source, GPUWeight(buffer: gain, offset: 0, columns: d, rows: 1, type: .f32), normalized)
        try gpu.finish(cb)
        let inv = 1 / sqrt(Float((0..<d).reduce(0.0) { $0 + Double(a[$1])*Double(a[$1]) }) / Float(d) + 1e-6)
        let result = normalized.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<d where abs(result[i] - a[i]*inv*w[i]) > 0.00002 {
            throw ParticleError.invalid("RMSNorm GPU/CPU mismatch at \(i)")
        }
        reports.append("RMSNorm: matches FP64-accumulating CPU reference")
        let raw = try gpu.buffer(4608), qGain = try gpu.buffer(64), kGain = try gpu.buffer(64)
        let rotated = try gpu.buffer(4096), kCache = try gpu.buffer(8*256), vCache = try gpu.buffer(8*256)
        let rawPointer = raw.contents().assumingMemoryBound(to:Float.self)
        let qw = qGain.contents().assumingMemoryBound(to:Float.self), kw = kGain.contents().assumingMemoryBound(to:Float.self)
        for i in 0..<4608 { rawPointer[i] = sin(Float(i)*0.073) }
        for i in 0..<64 { qw[i] = 0.7+Float(i)*0.01; kw[i] = 0.6+Float(i)*0.02 }
        let qc = try gpu.command("Particle QKV normalization and RoPE parity")
        try gpu.qkv(qc,projection:raw,qNorm:GPUWeight(buffer:qGain,offset:0,columns:64,rows:1,type:.f32),
                    kNorm:GPUWeight(buffer:kGain,offset:0,columns:64,rows:1,type:.f32),query:rotated,
                    keys:kCache,values:vCache,position:7,capacity:8)
        try gpu.finish(qc)
        for h in 0..<68 {
            let inverse = 1/sqrt((0..<64).reduce(0.0) { $0 + pow(Double(rawPointer[h*64+$1]),2) }/64+1e-6)
            let gain = h < 64 ? qw : kw
            let actual = h < 64 ? rotated.contents().assumingMemoryBound(to:Float.self).advanced(by:h*64)
                : kCache.contents().assumingMemoryBound(to:Float.self).advanced(by:7*256+(h-64)*64)
            for i in 0..<64 {
                let angle = 7*pow(8000000.0,-Double(i%32)/32)
                let x = Double(rawPointer[h*64+i])*inverse*Double(gain[i])
                let partner = (i+32)%64
                let y = Double(rawPointer[h*64+partner])*inverse*Double(gain[partner])
                let expected = x*cos(angle)+(i<32 ? -1.0 : 1.0)*y*sin(angle)
                guard abs(Double(actual[i])-expected) < 0.00003 else { throw ParticleError.invalid("Q/K normalization or RoPE mismatch") }
            }
        }
        let storedV = vCache.contents().assumingMemoryBound(to:Float.self).advanced(by:7*256)
        for i in 0..<256 where storedV[i] != rawPointer[4352+i] { throw ParticleError.invalid("V cache must contain the unmodified V projection") }
        reports.append("Q/K normalization + NeoX RoPE: matches CPU reference; V remains separate and unmodified")
        let keys = try gpu.buffer(3*256), values = try gpu.buffer(3*256), query = try gpu.buffer(4096), out = try gpu.buffer(4096)
        let k = keys.contents().assumingMemoryBound(to: Float.self), v = values.contents().assumingMemoryBound(to: Float.self)
        let q = query.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(3*256) { k[i] = sin(Float(i)*0.017); v[i] = cos(Float(i)*0.029) }
        for i in 0..<4096 { q[i] = cos(Float(i)*0.023) }
        let ac = try gpu.command("Particle full GQA attention parity")
        try gpu.attention(ac, query: query, keys: keys, values: values, out: out, position: 2, capacity: 3); try gpu.finish(ac)
        let output = out.contents().assumingMemoryBound(to: Float.self)
        for h in 0..<64 {
            let scores = (0..<3).map { t in (0..<64).reduce(0.0) { $0 + Double(q[h*64+$1]) * Double(k[t*256+h/16*64+$1]) } / 8 }
            let expScores = scores.map { exp($0 - scores.max()!) }, sum = expScores.reduce(0,+)
            for i in 0..<64 {
                let expected = (0..<3).reduce(0.0) { $0 + expScores[$1] / sum * Double(v[$1*256+h/16*64+i]) }
                guard abs(Double(output[h*64+i]) - expected) < 0.00002 else { throw ParticleError.invalid("Attention GPU/CPU mismatch") }
            }
        }
        reports.append("Full causal GQA attention: matches FP64 CPU reference")
        return reports
    }
}
