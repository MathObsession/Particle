import Foundation
import CryptoKit

public enum ModelArchitecture: String, Codable, Sendable { case sarvam = "sarvam_moe", qwen35MoE = "qwen3_5_moe", qwen3MoE = "qwen3_moe", mlxGraph = "mlx_graph_moe" }
public enum ModelArtifact: String, Codable, Sendable { case particle, gguf, mlx }
public enum AdapterQualification: String, Codable, Sendable { case qualified, experimental }

public struct ModelProvenance: Codable, Equatable, Sendable {
    public let repository: String?
    public let revision: String?
    public let license: String?
    public init(repository: String? = nil, revision: String? = nil, license: String? = nil) {
        self.repository = repository; self.revision = revision; self.license = license
    }
}

public struct ModelQuantization: Codable, Equatable, Sendable {
    public let scheme: String
    public let bits: Int?
    public let groupSize: Int?
    public let overrides: [String: Int]
    public init(scheme: String, bits: Int? = nil, groupSize: Int? = nil, overrides: [String: Int] = [:]) {
        self.scheme = scheme; self.bits = bits; self.groupSize = groupSize; self.overrides = overrides
    }
    public var label: String { bits.map { "\($0)-bit · \(scheme)" } ?? scheme }
}

/// Identity follows a pinned source artifact, never a display name or mutable branch.
public struct ModelDescriptor: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let architecture: ModelArchitecture
    public let artifact: ModelArtifact
    public let provenance: ModelProvenance
    public let quantization: ModelQuantization
    public let layers: Int
    public let experts: Int
    public let activeExperts: Int
    public let contextLimit: Int
    public let sourceBytes: Int64
    public let kvBytesPerToken: Int64
    public let supportsTools: Bool
    public let graph:ParticleGraphPlan?
    public let supportsReasoning:Bool?
    public init(id: String, name: String, architecture: ModelArchitecture, artifact: ModelArtifact,
                provenance: ModelProvenance, quantization: ModelQuantization, layers: Int, experts: Int,
                activeExperts: Int, contextLimit: Int, sourceBytes: Int64, kvBytesPerToken: Int64, supportsTools: Bool,graph:ParticleGraphPlan? = nil,supportsReasoning:Bool? = nil) {
        self.id = id; self.name = name; self.architecture = architecture; self.artifact = artifact
        self.provenance = provenance; self.quantization = quantization; self.layers = layers
        self.experts = experts; self.activeExperts = activeExperts; self.contextLimit = contextLimit
        self.sourceBytes = sourceBytes; self.kvBytesPerToken = kvBytesPerToken; self.supportsTools = supportsTools;self.graph=graph;self.supportsReasoning=supportsReasoning
    }
    public static func identity(repository: String, revision: String, artifact: String) -> String {
        SHA256.hash(data: Data((repository+"\n"+revision+"\n"+artifact).utf8)).map { String(format: "%02x", $0) }.joined()
    }
    public var expertRange:ClosedRange<Int> {(graph?.routingFields.isEmpty == true ? activeExperts : architecture == .sarvam ? min(2,activeExperts) : 1)...activeExperts}
    public func validate() throws {
        guard id.range(of:"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\\z",options:.regularExpression) != nil, !name.isEmpty, name.utf8.count <= 512,
              (1...512).contains(layers), (2...4096).contains(experts), (1...experts).contains(activeExperts),
              (128...4_194_304).contains(contextLimit), sourceBytes >= 0,sourceBytes <= Int64.max-2_147_483_648, kvBytesPerToken > 0 else {
            throw ModelLibraryError.invalid("Invalid model dimensions or identity.")
        }
        if architecture == .mlxGraph {guard let graph else {throw ModelLibraryError.invalid("A graph model requires its compiled execution plan.")};try graph.validate()} else if graph != nil {throw ModelLibraryError.invalid("A graph plan must use the graph runtime.")}
        if let repo = provenance.repository { try HubRepository.validate(repo) }
        if let revision = provenance.revision { guard HubRepository.validRevision(revision) else { throw ModelLibraryError.invalid("A model must use an immutable source revision.") } }
    }
    public static let sarvam = ModelDescriptor(id: "particle.sarvam.default", name: "Churo Chat", architecture: .sarvam,
        artifact: .particle, provenance: .init(repository: "MathObsession/sarvam-30b-particle", revision: "40febdc7327c6aa572af4cc01c9829c0ae68b2e1", license: "apache-2.0"),
        quantization: .init(scheme: "Q4_K_M"), layers: 19, experts: 128, activeExperts: 6,
        contextLimit: 131_072, sourceBytes: 19_605_181_515, kvBytesPerToken: 38_912, supportsTools: true)
    public static let sarvamManifestSHA256 = "1aa97aae017f8ef868e0ba0d263a5e4f8a08b0fb8547f4741260315d5880f0e6"
}

public struct CompatibilityReport: Codable, Equatable, Sendable {
    public let descriptor: ModelDescriptor?
    public let conversionSupported: Bool
    public let runtimeAvailable: Bool
    public let qualification: AdapterQualification
    public let reason: String
    public var installable: Bool { descriptor != nil && conversionSupported && runtimeAvailable && qualification == .qualified }
    public init(descriptor: ModelDescriptor?, conversionSupported: Bool, runtimeAvailable: Bool,
                qualification: AdapterQualification = .experimental, reason: String) {
        self.descriptor = descriptor; self.conversionSupported = conversionSupported
        self.runtimeAvailable = runtimeAvailable; self.qualification = qualification; self.reason = reason
    }
    public static func unsupported(_ reason: String) -> Self {
        .init(descriptor: nil, conversionSupported: false, runtimeAvailable: false, reason: reason)
    }
}

public enum ModelLibraryError: LocalizedError, Sendable {
    case invalid(String), incompatible(String), network(Int), rateLimited(Date), diskSpace(Int64), checksum(String), memoryPressure
    public var errorDescription: String? {
        switch self {
        case .invalid(let text), .incompatible(let text): text
        case .network(let code): "Hugging Face returned HTTP \(code). Please retry."
        case .rateLimited(let date): "Hugging Face is limiting requests. Retry after \(date.formatted(date: .omitted, time: .shortened))."
        case .diskSpace(let bytes): "This installation needs \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)) of free space."
        case .checksum(let file): "Verification failed for \(file). The model was not installed."
        case .memoryPressure: "Conversion paused because the Mac needs memory. Close other applications, then resume; verified progress is preserved."
        }
    }
}

public enum HubRepository {
    public static func validate(_ repository: String) throws {
        guard repository.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$", options: .regularExpression) != nil,
              !repository.contains("..") else { throw ModelLibraryError.invalid("Use a Hugging Face repository such as owner/model-name.") }
    }
    public static func parse(_ input: String) throws -> String {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = URL(string: text), url.scheme != nil {
            guard url.scheme == "https", url.host == "huggingface.co", url.user == nil, url.password == nil,
                  url.query == nil, url.fragment == nil, url.port == nil else { throw ModelLibraryError.invalid("Use a public huggingface.co model URL.") }
            let repo = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")); try validate(repo); return repo
        }
        try validate(text); return text
    }
    public static func validRevision(_ value: String) -> Bool { value.count == 40 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
}
