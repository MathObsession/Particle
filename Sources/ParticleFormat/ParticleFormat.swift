import Foundation

public enum ParticleFormat {
    public static let magic = "PARTICLE"
    public static let major = 1
    public static let minor = 0
    public static let alignment = 16_384
    public static let indexLimit = 16 * 1024 * 1024
    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
    public static func digestIsValid(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    public static func validatePath(_ path: String) throws {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.contains("\0"), parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
              !path.contains("\\") else { throw ParticleFormatError.invalid("Unsafe pack path: \(path)") }
    }
    public static func aligned(_ value: Int, to alignment: Int = alignment) throws -> Int {
        guard value >= 0, alignment > 0, alignment.nonzeroBitCount == 1,
              value <= Int.max - alignment + 1 else { throw ParticleFormatError.invalid("Alignment overflow") }
        return (value + alignment - 1) / alignment * alignment
    }
}

public enum ParticleFormatError: Error, CustomStringConvertible, Sendable {
    case invalid(String)
    public var description: String { switch self { case .invalid(let text): text } }
}

public enum ParticleScalar: Codable, Equatable, Sendable {
    case number(Double), string(String), bool(Bool)
    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else { let v = try c.decode(Double.self); guard v.isFinite else { throw ParticleFormatError.invalid("Non-finite metadata") }; self = .number(v) }
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self { case .number(let v): try c.encode(v); case .string(let v): try c.encode(v); case .bool(let v): try c.encode(v) }
    }
}

public struct ParticleFileRecord: Codable, Equatable, Sendable {
    public var bytes: Int
    public var sha256: String
    public init(bytes: Int, sha256: String) { self.bytes = bytes; self.sha256 = sha256 }
}
public struct ParticleTensorRecord: Codable, Equatable, Sendable {
    public var name: String, file: String
    public var shape: [Int]
    public var encoding: UInt32
    public var offset: Int, bytes: Int
    public var layer: Int?
    public init(name: String, file: String, shape: [Int], encoding: UInt32, offset: Int, bytes: Int, layer: Int? = nil) {
        self.name = name; self.file = file; self.shape = shape; self.encoding = encoding
        self.offset = offset; self.bytes = bytes; self.layer = layer
    }
}
public struct ParticleLayerRecord: Codable, Equatable, Sendable {
    public var layer: Int, stride: Int
    public var file: String
    public var hashes: [String]
    public init(layer: Int, stride: Int, file: String, hashes: [String] = []) {
        self.layer = layer; self.stride = stride; self.file = file; self.hashes = hashes
    }
}
public struct ParticleExpertLayout: Codable, Equatable, Sendable {
    public var expertsPerLayer: Int
    public var layers: [ParticleLayerRecord]
    public init(expertsPerLayer: Int = 128, layers: [ParticleLayerRecord]) { self.expertsPerLayer = expertsPerLayer; self.layers = layers }
}
public struct ParticleManifest: Codable, Equatable, Sendable {
    public var magic: String = ParticleFormat.magic
    public var versionMajor: Int = ParticleFormat.major
    public var versionMinor: Int = ParticleFormat.minor
    public var architecture: String = "sarvam_moe"
    public var profile: String = "native"
    public var approximate: Bool = false
    public var sourceSHA256: String
    public var sourceBytes: Int
    public var configuration: [String: ParticleScalar]
    public var files: [String: ParticleFileRecord]
    public init(sourceSHA256: String, sourceBytes: Int, configuration: [String: ParticleScalar], files: [String: ParticleFileRecord]) {
        self.sourceSHA256 = sourceSHA256; self.sourceBytes = sourceBytes; self.configuration = configuration; self.files = files
    }
    public func validate() throws {
        guard magic == ParticleFormat.magic, versionMajor == 1, versionMinor == 0,
              architecture == "sarvam_moe", profile == "native", !approximate,
              sourceBytes >= 24, ParticleFormat.digestIsValid(sourceSHA256), !files.isEmpty,
              files.count < 100, configuration.count < 10_000 else { throw ParticleFormatError.invalid("Unsupported or incomplete .particle manifest") }
        for required in ["weights.json", "packed_experts/layout.json", "model_weights.bin", "embeddings.bin", "vocabulary.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/config.json"] {
            guard files[required] != nil else { throw ParticleFormatError.invalid("Missing pack file: \(required)") }
        }
        var names = Set<String>()
        for (path, file) in files {
            try ParticleFormat.validatePath(path)
            let key = path.precomposedStringWithCanonicalMapping.lowercased()
            guard names.insert(key).inserted, path != "manifest.json", path != "verified-install.json",
                  file.bytes >= 0, ParticleFormat.digestIsValid(file.sha256) else { throw ParticleFormatError.invalid("Invalid file record: \(path)") }
        }
    }
}
public struct ParticleInstallReceipt: Codable, Equatable, Sendable {
    public var manifestSHA256: String
    public var complete: Bool = true
    public init(manifestSHA256: String) { self.manifestSHA256 = manifestSHA256 }
}
