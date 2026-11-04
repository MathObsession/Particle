import Foundation

public struct HubFile: Codable, Equatable, Sendable {
    public let path: String
    public let bytes: Int64
    public let sha256: String?
    public let gitBlobSHA1: String?
    public init(path:String,bytes:Int64,sha256:String?,gitBlobSHA1:String? = nil) {
        self.path=path;self.bytes=bytes;self.sha256=sha256;self.gitBlobSHA1=gitBlobSHA1
    }
}
public struct HubModelCard: Codable, Identifiable, Equatable, Sendable {
    public var id: String { repository }
    public let repository: String
    public let revision: String
    public let downloads: Int
    public let likes: Int
    public let license: String?
    public let gated: Bool
    public let tags: [String]
    public let files: [HubFile]
    public init(repository: String, revision: String, downloads: Int = 0, likes: Int = 0,
                license: String? = nil, gated: Bool = false, tags: [String] = [], files: [HubFile] = []) {
        self.repository = repository; self.revision = revision; self.downloads = downloads; self.likes = likes
        self.license = license; self.gated = gated; self.tags = tags; self.files = files
    }
}
public struct HubModelPage: Sendable {
    public let models: [HubModelCard]
    public let next: URL?
}

/// Anonymous HTTPS only. A response cannot allocate more than its caller's bound.
public actor PublicHubHTTP {
    private let router: PublicHubRequestRouter
    private let session: URLSession
    public init() {
        router = PublicHubRequestRouter()
        let config = URLSessionConfiguration.ephemeral
        config.httpCookieStorage = nil;config.urlCredentialStorage = nil
        config.timeoutIntervalForRequest = 60;config.timeoutIntervalForResource = 300
        let queue = OperationQueue();queue.maxConcurrentOperationCount = 1
        session = URLSession(configuration:config,delegate:router,delegateQueue:queue)
    }
    public nonisolated static func allowed(_ url: URL) -> Bool {
        guard url.scheme == "https", url.user == nil, url.password == nil, url.port == nil,
              let host = url.host?.lowercased() else { return false }
        return host == "huggingface.co" || host.hasSuffix(".huggingface.co") || host.hasSuffix(".hf.co")
    }
    public func get(_ url: URL, limit: Int, range: Range<Int64>? = nil) async throws -> (Data, HTTPURLResponse) {
        guard Self.allowed(url), limit > 0, limit <= 64*1024*1024 else { throw ModelLibraryError.invalid("Invalid public Hub request.") }
        var request = URLRequest(url: url); request.setValue("Particle/0.4 model-library", forHTTPHeaderField: "User-Agent")
        if let range {
            guard range.lowerBound >= 0, range.upperBound > range.lowerBound, range.count <= limit else { throw ModelLibraryError.invalid("Invalid download range.") }
            request.setValue("bytes=\(range.lowerBound)-\(range.upperBound-1)", forHTTPHeaderField: "Range")
        }
        return try await BoundedHubRequest(request: request, limit: limit, range: range).start(session:session,router:router)
    }
}

public actor HuggingFaceClient {
    public let http: PublicHubHTTP
    public init(http: PublicHubHTTP = PublicHubHTTP()) { self.http = http }
    public func search(_ text: String, next: URL? = nil) async throws -> HubModelPage {
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [.init(name: "search", value: String(text.prefix(200))), .init(name: "limit", value: "20"),
            .init(name:"filter",value:"qwen3_5_moe"),.init(name: "full", value: "true"), .init(name: "sort", value: "downloads"), .init(name: "direction", value: "-1")]
        if let next { guard next.scheme == "https", next.host == "huggingface.co", next.path == "/api/models", next.user == nil, next.password == nil else { throw ModelLibraryError.invalid("Invalid catalog page.") } }
        let (data, response) = try await http.get(next ?? components.url!, limit: 4*1024*1024)
        guard let rows = try JSONSerialization.jsonObject(with: data) as? [[String: Any]], rows.count <= 100 else { throw ModelLibraryError.invalid("Invalid model catalog response.") }
        let models = rows.compactMap { try? Self.decodeCard($0) }
        let links = response.value(forHTTPHeaderField: "Link") ?? ""
        let nextLink = links.components(separatedBy: ",").first(where: {$0.contains("rel=\"next\"")}).flatMap { component -> URL? in
            guard let start = component.firstIndex(of: "<"), let end = component.firstIndex(of: ">"), start < end else { return nil }
            return URL(string: String(component[component.index(after:start)..<end]))
        }
        return HubModelPage(models: models, next: nextLink)
    }
    public func model(_ repository: String, revision: String? = nil) async throws -> HubModelCard {
        try HubRepository.validate(repository)
        if let revision {guard HubRepository.validRevision(revision) else {throw ModelLibraryError.invalid("A pinned model revision is required.")}}
        let endpoint = "https://huggingface.co/api/models/\(repository)"+(revision.map{"/revision/"+$0} ?? "")+"?blobs=true"
        let (data, _) = try await http.get(URL(string:endpoint)!, limit: 8*1024*1024)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw ModelLibraryError.invalid("Invalid model metadata.") }
        let card = try Self.decodeCard(object)
        guard revision == nil || card.revision == revision else {throw ModelLibraryError.invalid("The Hub returned a different model revision.")}
        return card
    }
    public nonisolated static func resolve(_ card: HubModelCard, path: String) throws -> URL {
        try HubRepository.validate(card.repository)
        guard HubRepository.validRevision(card.revision) else { throw ModelLibraryError.invalid("Missing immutable model revision.") }
        try ParticlePath.validate(path)
        return URL(string: "https://huggingface.co/\(card.repository)/resolve/\(card.revision)/")!.appendingPathComponent(path)
    }
    public func metadata(_ card: HubModelCard, path: String, limit: Int) async throws -> Data {
        try await http.get(Self.resolve(card, path: path), limit: limit).0
    }
    public func inspect(_ card: HubModelCard) async throws -> CompatibilityReport {
        guard !card.gated else { return .unsupported("Only public, ungated models are available in this release.") }
        guard card.files.contains(where: {$0.path == "config.json"}), card.files.contains(where: {$0.path == "model.safetensors.index.json"}) else {
            return .unsupported("This repository needs an indexed, supported MLX checkpoint or a verified Particle release.")
        }
        let config = try await metadata(card, path: "config.json", limit: 1_048_576)
        guard let object = try JSONSerialization.jsonObject(with: config) as? [String: Any], object["model_type"] as? String == "qwen3_5_moe" else {
            return .unsupported("No qualified Particle streaming adapter matches this model family.")
        }
        if let reason=try ArchitectureRegistry.configurationIncompatibility(config) {return .unsupported(reason)}
        let indexData = try await metadata(card, path: "model.safetensors.index.json", limit: 16*1024*1024)
        guard let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any], let map = index["weight_map"] as? [String: String],
              !map.isEmpty, map.count <= 100_000, Set(map.values).count <= 128 else { throw ModelLibraryError.invalid("Invalid checkpoint shard index.") }
        var tensors = [String: SafetensorRecord](); var bytes: Int64 = 0
        for path in Set(map.values).sorted() {
            guard let file = card.files.first(where: {$0.path == path}), file.bytes > 8, file.sha256 != nil else { throw ModelLibraryError.invalid("Missing shard size or source checksum.") }
            let url = try Self.resolve(card, path: path)
            let prefix = try await http.get(url, limit: 8, range: 0..<8).0
            let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8*$1.offset) }
            guard length > 1, length <= SafetensorsHeader.maximumHeaderBytes, length <= UInt64(file.bytes-8) else { throw ModelLibraryError.invalid("Invalid remote tensor header length.") }
            let data = try await http.get(url, limit: Int(length), range: 8..<(8+Int64(length))).0
            let header = try SafetensorsHeader(prefix: prefix+data, fileBytes: file.bytes)
            for (name, record) in header.tensors {
                guard map[name] == path, tensors[name] == nil else { throw ModelLibraryError.invalid("Tensor index does not match remote shards.") }
                tensors[name] = record
            }
            let (total, overflow) = bytes.addingReportingOverflow(file.bytes); guard !overflow else { throw ModelLibraryError.invalid("Model size overflow.") }; bytes = total
        }
        guard Set(tensors.keys) == Set(map.keys) else { throw ModelLibraryError.invalid("Remote tensor graph is incomplete.") }
        let template = try await metadata(card, path: "chat_template.jinja", limit: 262_144)
        let tokenizer = try await metadata(card,path:"tokenizer.json",limit:64*1024*1024)
        let tokenizerConfiguration = try await metadata(card,path:"tokenizer_config.json",limit:1_048_576)
        try ArchitectureRegistry.validateChronosTokenizer(tokenizer:tokenizer,configuration:tokenizerConfiguration)
        return try ArchitectureRegistry.inspectMLX(configuration: config, tensors: tensors, template: String(decoding: template, as: UTF8.self),
            repository: card.repository, revision: card.revision, sourceBytes: bytes, license: card.license)
    }
    private nonisolated static func decodeCard(_ object: [String: Any]) throws -> HubModelCard {
        guard let repo = (object["id"] ?? object["modelId"]) as? String, let revision = object["sha"] as? String, HubRepository.validRevision(revision) else { throw ModelLibraryError.invalid("Model metadata is not pinned.") }
        try HubRepository.validate(repo)
        let files = (object["siblings"] as? [[String: Any]] ?? []).compactMap { entry -> HubFile? in
            guard let path = entry["rfilename"] as? String, (try? ParticlePath.validate(path)) != nil else { return nil }
            let lfs = entry["lfs"] as? [String: Any]
            return HubFile(path: path, bytes: (lfs?["size"] as? Int64) ?? (entry["size"] as? Int64) ?? 0, sha256: lfs?["sha256"] as? String,
                gitBlobSHA1:lfs == nil ? entry["blobId"] as? String : nil)
        }
        return .init(repository: repo, revision: revision, downloads: object["downloads"] as? Int ?? 0, likes: object["likes"] as? Int ?? 0,
            license: (object["cardData"] as? [String: Any])?["license"] as? String,
            gated: (object["gated"] as? Bool == true) || (object["gated"] as? String != nil), tags: object["tags"] as? [String] ?? [], files: files)
    }
}

private enum ParticlePath {
    static func validate(_ path: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("%"), !path.contains(":"), !path.contains("\0"),
              path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw ModelLibraryError.invalid("Unsafe repository file path.") }
    }
}
