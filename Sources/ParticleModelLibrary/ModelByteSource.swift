import Foundation
import Darwin
import ParticleGGUF

public protocol ModelByteSource: Sendable {
    var identity: String { get }
    var files: [HubFile] { get }
    func read(_ path: String, range: Range<Int64>) async throws -> Data
    func verifyIdentity() throws
}
extension ModelByteSource {public func verifyIdentity() throws {}}
public struct LocalModelByteSource: ModelByteSource {
    public let root: URL
    public let identity: String
    public let files: [HubFile]
    private let signatures: [String:String]
    public init(root: URL, paths: [String]) throws {
        self.root = root.resolvingSymlinksInPath()
        var records = [HubFile](); var identities = [String]();var signatures = [String:String]()
        for path in paths.sorted() {
            let fd = try Self.open(root: self.root, path: path); defer { close(fd) }
            var st = stat(); guard fstat(fd,&st) == 0, st.st_mode & S_IFMT == S_IFREG else { throw ModelLibraryError.invalid("Source must be a regular model file.") }
            records.append(.init(path: path, bytes: st.st_size, sha256: nil))
            let signature = Self.signature(st);signatures[path] = signature;identities.append(path+"|"+signature)
        }
        files = records; identity = "local:"+ModelDescriptor.identity(repository: self.root.path, revision: identities.joined(separator:"\n"), artifact: "mlx")
        self.signatures = signatures
    }
    public func read(_ path: String, range: Range<Int64>) async throws -> Data {
        try Task.checkCancellation()
        guard let file = files.first(where: {$0.path == path}), range.lowerBound >= 0, range.upperBound <= file.bytes,
              range.count <= 16*1024*1024 else { throw ModelLibraryError.invalid("Source read exceeds its bounded range.") }
        let fd = try Self.open(root: root, path: path); defer { close(fd) }
        var before = stat();guard fstat(fd,&before) == 0,Self.signature(before) == signatures[path] else {throw ModelLibraryError.invalid("The local source changed during conversion.")}
        var data = Data(count: range.count)
        try data.withUnsafeMutableBytes { bytes in
            try ParticleFileRange(descriptor: fd, offset: Int(range.lowerBound), count: range.count).read(into: bytes.baseAddress!)
        }
        var after = stat();guard fstat(fd,&after) == 0,Self.signature(after) == signatures[path] else {throw ModelLibraryError.invalid("The local source changed during a read.")}
        return data
    }
    public func verifyIdentity() throws {
        let current = try Self(root:root,paths:files.map(\.path))
        guard current.identity == identity else {throw ModelLibraryError.invalid("The local checkpoint changed. No pack was published.")}
    }
    private static func signature(_ st: stat) -> String {"\(st.st_dev)|\(st.st_ino)|\(st.st_size)|\(st.st_mtimespec.tv_sec)|\(st.st_mtimespec.tv_nsec)|\(st.st_ctimespec.tv_sec)|\(st.st_ctimespec.tv_nsec)"}
    private static func open(root: URL, path: String) throws -> Int32 {
        let fd = Darwin.open(root.path, O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd >= 0 else { throw ModelLibraryError.invalid("The source folder is unavailable.") }; defer { close(fd) }
        return try ParticleIO.openRelative(root: fd, path: path)
    }
}
public struct RemoteModelByteSource: ModelByteSource {
    public let card: HubModelCard
    public let http: PublicHubHTTP
    public var identity: String { card.repository+"@"+card.revision }
    public var files: [HubFile] { card.files }
    public init(card: HubModelCard, http: PublicHubHTTP = PublicHubHTTP()) throws {
        try HubRepository.validate(card.repository)
        guard HubRepository.validRevision(card.revision), !card.gated else { throw ModelLibraryError.incompatible("A public, pinned model revision is required.") }
        self.card = card; self.http = http
    }
    public func read(_ path: String, range: Range<Int64>) async throws -> Data {
        guard let file = files.first(where: {$0.path == path}), range.lowerBound >= 0, range.upperBound <= file.bytes,
              range.count > 0, range.count <= 16*1024*1024 else { throw ModelLibraryError.invalid("Remote read exceeds its source range.") }
        return try await http.get(HuggingFaceClient.resolve(card,path:path),limit:range.count,range:range).0
    }
}
