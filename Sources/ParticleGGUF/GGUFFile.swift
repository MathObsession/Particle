import Foundation
import Darwin

public enum ParticleError: Error, CustomStringConvertible, Sendable {
    case invalid(String)
    public var description: String { switch self { case .invalid(let message): return message } }
}

public enum GGUFValue: Sendable {
    case number(Double), string(String), bool(Bool), strings([String]), numbers([Double])
    public var text: String? { if case .string(let s) = self { return s }; return nil }
    public var strings: [String]? { if case .strings(let s) = self { return s }; return nil }
    public var number: Double? { if case .number(let n) = self { return n }; return nil }
}

public enum GGMLType: UInt32, Sendable, CaseIterable {
    case f32 = 0, f16 = 1, q4K = 12, q5K = 13, q6K = 14
    public var blockElements: Int { self == .f32 || self == .f16 ? 1 : 256 }
    public var blockBytes: Int {
        switch self { case .f32: 4; case .f16: 2; case .q4K: 144; case .q5K: 176; case .q6K: 210 }
    }
}

public struct GGUFTensor: Sendable {
    public let name: String
    /// GGUF dimensions: fastest-varying input dimension first.
    public let shape: [Int]
    public let type: GGMLType
    public let offset: Int
    public let byteCount: Int
    public init(name: String, shape: [Int], type: GGMLType, offset: Int, byteCount: Int) {
        self.name = name; self.shape = shape; self.type = type; self.offset = offset; self.byteCount = byteCount
    }
    public var rowBytes: Int { shape[0] / type.blockElements * type.blockBytes }
    public var expertBytes: Int { rowBytes * shape[1] }
}

/// Metadata-only parse followed by bounded pread access. Never maps the whole
/// model, nor materializes a weight tensor in a Swift collection.
public final class GGUFFile: @unchecked Sendable {
    public let path: String
    public let descriptor: Int32
    public let size: Int
    public let metadata: [String: GGUFValue]
    public let tensors: [String: GGUFTensor]
    public let dataOffset: Int

    public init(path: String) throws {
        let resolved = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        let fd = open(resolved, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw ParticleError.invalid("Cannot open model: \(String(cString: strerror(errno)))") }
        var succeeded = false
        defer { if !succeeded { close(fd) } }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_mode & S_IFMT == S_IFREG,
              st.st_size >= 24, st.st_size <= Int.max else {
            throw ParticleError.invalid("Model must be an addressable regular GGUF file")
        }
        let fileSize = Int(st.st_size)
        let prefixSize = min(fileSize, 64 * 1024 * 1024)
        guard let pointer = mmap(nil, prefixSize, PROT_READ, MAP_PRIVATE, fd, 0), pointer != MAP_FAILED else {
            throw ParticleError.invalid("Cannot map GGUF metadata")
        }
        defer { munmap(pointer, prefixSize) }
        var reader = HeaderReader(bytes: UnsafeRawBufferPointer(start: pointer, count: prefixSize))
        guard try reader.u32() == 0x46554747, try reader.u32() == 3 else {
            throw ParticleError.invalid("Particle requires a GGUF v3 model")
        }
        let count = try reader.count(limit: 100_000)
        let metaCount = try reader.count(limit: 10_000)
        var meta: [String: GGUFValue] = [:]
        for _ in 0..<metaCount {
            let key = try reader.string()
            guard meta[key] == nil else { throw ParticleError.invalid("Duplicate GGUF metadata key: \(key)") }
            meta[key] = try reader.value(type: reader.u32())
        }
        if let split = meta["split.count"]?.number, split > 1 {
            throw ParticleError.invalid("This CLI requires a merged GGUF; the installed single-file model is supported")
        }
        var entries: [(String, [Int], GGMLType, Int)] = []
        for _ in 0..<count {
            let name = try reader.string()
            let rank = Int(try reader.u32())
            guard (1...4).contains(rank) else { throw ParticleError.invalid("Invalid tensor rank: \(name)") }
            var shape: [Int] = []
            for _ in 0..<rank { shape.append(try reader.count(limit: 1 << 32)) }
            guard shape.allSatisfy({ $0 > 0 }), let type = GGMLType(rawValue: try reader.u32()) else {
                throw ParticleError.invalid("Unsupported quantization or shape for \(name)")
            }
            entries.append((name, shape, type, try reader.count(limit: fileSize)))
        }
        let alignmentValue = meta["general.alignment"]?.number ?? 32
        guard alignmentValue >= 1, alignmentValue <= 65536, alignmentValue.rounded() == alignmentValue else {
            throw ParticleError.invalid("Invalid GGUF alignment")
        }
        let alignment = Int(alignmentValue)
        guard alignment.nonzeroBitCount == 1 else { throw ParticleError.invalid("GGUF alignment must be a power of two") }
        let base = (reader.position + alignment - 1) / alignment * alignment
        guard base <= fileSize else { throw ParticleError.invalid("Truncated GGUF data alignment") }
        var table: [String: GGUFTensor] = [:]
        var previousEnd = base
        for (name, shape, type, relative) in entries.sorted(by: { $0.3 < $1.3 }) {
            var elements = 1
            for n in shape {
                let (product, overflow) = elements.multipliedReportingOverflow(by: n)
                guard !overflow else { throw ParticleError.invalid("Overflow in tensor \(name)") }
                elements = product
            }
            guard shape[0] % type.blockElements == 0 else { throw ParticleError.invalid("Invalid quantization block shape: \(name)") }
            let (bytes, overflow) = (elements / type.blockElements).multipliedReportingOverflow(by: type.blockBytes)
            guard !overflow, relative <= fileSize - base, bytes <= fileSize - base - relative,
                  relative % alignment == 0, base + relative >= previousEnd, table[name] == nil else {
                throw ParticleError.invalid("Truncated, overlapping, or invalid GGUF tensor: \(name)")
            }
            table[name] = GGUFTensor(name: name, shape: shape, type: type, offset: base + relative, byteCount: bytes)
            previousEnd = base + relative + bytes
        }
        self.path = resolved; descriptor = fd; size = fileSize
        metadata = meta; tensors = table; dataOffset = base
        succeeded = true
    }

    deinit { close(descriptor) }

    public func tensor(_ name: String, shape: [Int]? = nil) throws -> GGUFTensor {
        guard let tensor = tensors[name], shape == nil || tensor.shape == shape else {
            throw ParticleError.invalid("Missing tensor or unexpected shape: \(name), expected \(shape ?? [])")
        }
        return tensor
    }

    public func read(into pointer: UnsafeMutableRawPointer, count: Int, offset: Int) throws {
        guard offset >= 0, count >= 0, offset <= size, count <= size - offset else {
            throw ParticleError.invalid("Read exceeds GGUF bounds")
        }
        var done = 0
        while done < count {
            let n = pread(descriptor, pointer.advanced(by: done), count - done, off_t(offset + done))
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { throw ParticleError.invalid("GGUF read failed or card disconnected: \(String(cString: strerror(errno)))") }
            done += n
        }
    }
}

private struct HeaderReader {
    let bytes: UnsafeRawBufferPointer
    var position = 0
    mutating func take(_ n: Int) throws -> UnsafeRawBufferPointer {
        guard n >= 0, position <= bytes.count, n <= bytes.count - position else {
            throw ParticleError.invalid("Truncated GGUF metadata or metadata exceeds 64 MiB limit")
        }
        defer { position += n }
        return UnsafeRawBufferPointer(rebasing: bytes[position..<(position + n)])
    }
    mutating func u32() throws -> UInt32 { try take(4).loadUnaligned(as: UInt32.self).littleEndian }
    mutating func u64() throws -> UInt64 { try take(8).loadUnaligned(as: UInt64.self).littleEndian }
    mutating func count(limit: Int) throws -> Int {
        let n = try u64()
        guard n <= UInt64(limit) else { throw ParticleError.invalid("GGUF count exceeds limit") }
        return Int(n)
    }
    mutating func string() throws -> String {
        let n = try count(limit: 16 * 1024 * 1024)
        guard let s = String(bytes: try take(n), encoding: .utf8) else { throw ParticleError.invalid("Invalid UTF-8 in GGUF metadata") }
        return s
    }
    mutating func scalar(_ type: UInt32) throws -> Double {
        switch type {
        case 0: return Double(try take(1)[0])
        case 1: return Double(Int8(bitPattern: try take(1)[0]))
        case 2: return Double(try take(2).loadUnaligned(as: UInt16.self).littleEndian)
        case 3: return Double(Int16(bitPattern: try take(2).loadUnaligned(as: UInt16.self).littleEndian))
        case 4: return Double(try u32())
        case 5: return Double(Int32(bitPattern: try u32()))
        case 6: return Double(Float(bitPattern: try u32()))
        case 10: return Double(try u64())
        case 11: return Double(Int64(bitPattern: try u64()))
        case 12: return Double(bitPattern: try u64())
        default: throw ParticleError.invalid("Unsupported GGUF metadata scalar type \(type)")
        }
    }
    mutating func value(type: UInt32) throws -> GGUFValue {
        switch type {
        case 7:
            let n = try take(1)[0]
            guard n <= 1 else { throw ParticleError.invalid("Invalid GGUF boolean") }
            return .bool(n != 0)
        case 8: return .string(try string())
        case 9:
            let subtype = try u32(), n = try count(limit: 2_000_000)
            if subtype == 8 {
                var strings: [String] = []; strings.reserveCapacity(n)
                for _ in 0..<n { strings.append(try string()) }
                return .strings(strings)
            }
            var numbers: [Double] = []; numbers.reserveCapacity(n)
            for _ in 0..<n { numbers.append(try scalar(subtype)) }
            return .numbers(numbers)
        default: return .number(try scalar(type))
        }
    }
}

public struct SarvamConfiguration: Sendable {
    public let hidden = 4096, layers = 19, heads = 64, kvHeads = 4, headDimension = 64
    public let experts = 128, topK = 6, expertWidth = 1024, denseWidth = 8192, vocabulary = 262144
    public let epsilon: Float = 1e-6, theta: Float = 8_000_000, routingScale: Float = 2.5
    public init(file: any ParticleModelSource) throws {
        guard file.metadata["general.architecture"]?.text == "sarvam_moe" else {
            throw ParticleError.invalid("This CLI supports the sarvam_moe GGUF architecture")
        }
        let required: [String: Double] = ["block_count": 19, "embedding_length": 4096, "attention.head_count": 64,
            "attention.head_count_kv": 4, "attention.key_length": 64, "attention.value_length": 64,
            "expert_count": 128, "expert_used_count": 6, "leading_dense_block_count": 1,
            "feed_forward_length": 8192, "expert_feed_forward_length": 1024,
            "expert_shared_feed_forward_length": 1024, "expert_shared_count": 1,
            "expert_weights_scale": 2.5, "expert_gating_func": 2, "expert_group_count": 1,
            "expert_group_used_count": 1, "rope.dimension_count": 64, "rope.freq_base": 8_000_000,
            "vocab_size": 262144, "context_length": 131072]
        for (key, expected) in required where file.metadata["sarvam_moe." + key]?.number != expected {
            throw ParticleError.invalid("Unsupported Sarvam architecture field \(key); expected \(expected)")
        }
        guard case .bool(true)? = file.metadata["sarvam_moe.expert_weights_norm"],
              let eps = file.metadata["sarvam_moe.attention.layer_norm_rms_epsilon"]?.number,
              abs(eps - 1e-6) < 1e-10 else { throw ParticleError.invalid("Unsupported Sarvam normalization") }
    }
}
