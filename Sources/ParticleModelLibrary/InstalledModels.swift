import Foundation
import ParticleFormat
import ParticleDesktopProtocol

public struct ModelRuntimeProfile: Codable, Equatable, Sendable {
    public static func memoryIssue(_ model:ModelDescriptor,physicalMemory:UInt64=ProcessInfo.processInfo.physicalMemory)->String? {
        guard let graph=model.graph else{return nil}
        let profile=Self(model:model,physicalMemory:physicalMemory)
        let required=graph.residentBytes+graph.expertBytesPerSlot*Int64(profile.slots)+512*1024*1024
        guard required>=Int64(physicalMemory)*3/4 else{return nil}
        return "This checkpoint’s resident weights and minimum expert cache exceed this Mac’s memory allowance. Choose a smaller quantization or use a Mac with more RAM."
    }
    public var slots: Int
    public var readWorkers: Int
    public var prefillChunk: Int
    public var context: Int
    public var activeExperts: Int
    public var expertBytesPerSlot: Int64?
    public init(model: ModelDescriptor, physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        let kvBudget = Int64(physicalMemory/6)
        slots = model.architecture == .sarvam ? 12 : 16
        readWorkers = model.architecture == .sarvam ? 4 : 2; prefillChunk = 128; activeExperts = model.activeExperts
        let ceiling = Int(min(Int64(model.contextLimit), kvBudget/model.kvBytesPerToken))
        context = [4096,8192,16384,32768,65536,131072,262144].last(where: {$0 <= ceiling}) ?? 4096
        if model.architecture == .sarvam {context=ParticleContextPolicy.recommended(physicalMemory:physicalMemory)}
        if let graph=model.graph {
            expertBytesPerSlot=graph.expertBytesPerSlot
            slots=max(1,min(32,Int(Int64(physicalMemory/6)/graph.expertBytesPerSlot),graph.banks.values.max() ?? 32))
            let remaining=max(0,Int64(physicalMemory)*2/3-graph.residentBytes-Int64(slots)*graph.expertBytesPerSlot-512*1024*1024)
            let limit=min(kvBudget,remaining)/model.kvBytesPerToken
            context=[4096,8192,16384,32768,65536,131072,262144].last(where:{$0 <= limit && $0 <= model.contextLimit}) ?? 4096
        }
    }
}

public struct InstalledModel: Codable, Equatable, Identifiable, Sendable {
    public var id: String { descriptor.id }
    public let descriptor: ModelDescriptor
    public let path: String
    public let managed: Bool
    public let installedAt: Date
    public var profile: ModelRuntimeProfile
    public var calibration: ModelCalibrationRecord?
    public init(descriptor: ModelDescriptor, path: String, managed: Bool) {
        self.descriptor = descriptor; self.path = path; self.managed = managed
        installedAt = Date(); profile = .init(model: descriptor)
    }
}

/// Stored separately from workspace databases so all workspaces share weights.
public struct InstalledModelRegistry: Codable, Sendable {
    public var version = 1
    public private(set) var models: [InstalledModel] = []
    public init() {}
    public mutating func register(_ model: InstalledModel) throws {
        try model.descriptor.validate()
        guard model.path.hasPrefix("/"), !model.path.contains("\0"), models.count < 1024 || models.contains(where: {$0.id == model.id}) else { throw ModelLibraryError.invalid("Invalid installation path or registry capacity.") }
        if let index = models.firstIndex(where: {$0.id == model.id}) { models[index] = model }
        else { models.append(model) }
    }
    public mutating func forget(_ id: String) { models.removeAll { $0.id == id } }
    public func save(to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions:0o700])
        let data=try ParticleFormat.encode(self)
        guard data.count <= 32*1024*1024 else {throw ModelLibraryError.invalid("The model registry reached its size limit. Existing records were preserved.")}
        try data.write(to: url, options: [.atomic, .completeFileProtectionUnlessOpen])
        try FileManager.default.setAttributes([.posixPermissions:0o600], ofItemAtPath: url.path)
    }
    public static func load(from url: URL) throws -> Self {
        guard FileManager.default.fileExists(atPath: url.path) else { return Self() }
        let value = try JSONDecoder().decode(Self.self, from: ArchitectureRegistry.boundedData(url, limit: 32*1024*1024))
        guard value.version == 1, value.models.count <= 1024, Set(value.models.map(\.id)).count == value.models.count else { throw ModelLibraryError.invalid("The installed-model registry is invalid. Existing model files were preserved.") }
        for model in value.models { try model.descriptor.validate() }
        return value
    }
    public static func removableURL(_ model: InstalledModel, managedRoot: URL) throws -> URL {
        let root = managedRoot.standardizedFileURL.resolvingSymlinksInPath()
        let source = URL(fileURLWithPath: model.path).standardizedFileURL
        guard model.managed, source.pathExtension == "particle", source.deletingLastPathComponent() == root,
              source.resolvingSymlinksInPath() == source else { throw ModelLibraryError.invalid("Only Particle-managed model folders can be deleted. Imported originals are preserved.") }
        return source
    }
}

public enum ModelInstallStage: String, Codable, Sendable { case queued, inspecting, downloading, converting, verifying, paused, ready, failed, cancelled }
public struct ModelInstallJob: Codable, Identifiable, Sendable {
    public let id: UUID
    public let repository: String
    public let revision: String
    public let destination: String
    public let localSource: String?
    public var stage: ModelInstallStage
    public var completedBytes: Int64
    public var totalBytes: Int64
    public var detail: String
    public init(repository: String, revision: String, destination: String) throws {
        try HubRepository.validate(repository)
        guard HubRepository.validRevision(revision) else { throw ModelLibraryError.invalid("Installations require a pinned revision.") }
        id = UUID(); self.repository = repository; self.revision = revision; self.destination = destination
        localSource=nil
        stage = .queued; completedBytes = 0; totalBytes = 0; detail = "Waiting to install"
    }
    public init(localSource:URL,identity:String,destination:URL) throws {
        guard localSource.path.hasPrefix("/"),!identity.isEmpty,identity.utf8.count <= 256 else {throw ModelLibraryError.invalid("Invalid local model source.")}
        id=UUID();repository="Local · "+localSource.lastPathComponent
        // A local revision is its inspected source identity, not a Hub commit.
        revision=identity;self.destination=destination.path;self.localSource=localSource.path
        stage = .queued;completedBytes=0;totalBytes=0;detail="Waiting to inspect local model"
    }
}
