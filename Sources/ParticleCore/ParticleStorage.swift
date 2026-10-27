import Foundation
import Darwin
import ParticleGGUF
import ParticleFormat

public enum ParticleStorageProfile: String, Sendable { case legacy, sdOnly = "sd-only", sdHybrid = "sd-hybrid", ssd }
public enum ParticleMirrorReadHint: String, Sendable { case standard = "default", noCache = "nocache" }
public enum ParticleSSDCachePolicy: String, Codable, Sendable { case victim, readThrough = "read-through", mirror }

/// A cache-wide lease prevents two policies from spending the same disk budget.
final class ParticleCacheDirectoryLease: @unchecked Sendable {
    let directory: URL
    private let descriptor: Int32
    init(directory: URL) throws {
        self.directory = directory.standardizedFileURL
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let fd = Darwin.open(directory.appendingPathComponent(".tier.lock").path,O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0o600)
        guard fd >= 0 else { throw ParticleError.invalid("Cannot open cache-wide lease") }
        guard flock(fd,LOCK_EX|LOCK_NB) == 0 else { close(fd); throw ParticleError.invalid("SSD tier is already in use") }
        descriptor = fd
    }
    /// Migrate away from old content-identity directories without leaving
    /// their payload outside the new single-tier budget. Caller owns lease.
    static func removeObsoleteNamespaces(in directory: URL) throws {
        let fm = FileManager.default
        for url in try fm.contentsOfDirectory(at:directory,includingPropertiesForKeys:nil) {
            let name = url.lastPathComponent
            guard ParticleFormat.digestIsValid(name) else { continue }
            let root = Darwin.open(url.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
            guard root >= 0 else { throw ParticleError.invalid("Invalid legacy cache namespace") }
            defer { close(root) }
            let data = try ParticleIO.data(root:root,path:"index.json",limit:2_000_000)
            guard let index = try JSONSerialization.jsonObject(with:data) as? [String:Any],
                  (index["version"] as? Int) == 1, (index["identity"] as? String) == name,
                  index["entries"] is [String:Any] else { throw ParticleError.invalid("Unrecognized legacy cache namespace; cannot establish budget") }
            try fm.removeItem(at:url)
        }
    }
    deinit { flock(descriptor,LOCK_UN); close(descriptor) }
}
private struct MirrorReceipt: Codable {
    let identity: String
    let files: [String:ParticleFileRecord]
}
public enum ParticleMirror {
    public static let maximumBytes = 18 * 1024 * 1024 * 1024
    static func payload(_ pack: ParticlePackFile) -> [String:ParticleFileRecord] {
        pack.manifest.files.filter { $0.key == "model_weights.bin" || ($0.key.hasPrefix("packed_experts/layer_") && $0.key.hasSuffix(".bin")) }
    }
    static func filename(_ name: String) -> String { URL(fileURLWithPath:name).lastPathComponent }
    static func validateContents(folder: URL, files: [String:ParticleFileRecord]) throws {
        let allowed = Set(files.keys.map(filename)).union(["mirror-receipt.json"])
        let actual = try FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:nil)
        guard actual.allSatisfy({ allowed.contains($0.lastPathComponent) }) else { throw ParticleError.invalid("Unexpected file in owned SSD mirror; cannot establish payload budget") }
    }
    public static func requiredBytes(for pack: ParticlePackFile) -> Int { payload(pack).values.reduce(0) { $0+$1.bytes } }
    /// Explicit preparation replaces the old partial cache with one complete
    /// immutable tier; it never creates both full mirror and victim pools.
    public static func prepare(pack: ParticlePackFile, directory: URL, maxBytes: Int,
                               cancellation: ParticleCancellation? = nil,
                               progress: ((String,Int,Int) -> Void)? = nil) throws {
        let files = payload(pack), required = requiredBytes(for:pack)
        guard maxBytes > 0, maxBytes <= maximumBytes, required <= maxBytes else { throw ParticleError.invalid("Complete mirror needs \(required) payload bytes; configured limit is \(maxBytes), maximum 18432 MiB") }
        let lease = try ParticleCacheDirectoryLease(directory:directory)
        defer { withExtendedLifetime(lease) {} }
        let fm = FileManager.default, final = directory.appendingPathComponent("mirror-v1"), partial = directory.appendingPathComponent("mirror-v1.partial")
        // Only application-owned namespaces are evicted by this explicit
        // policy change. A cache-wide lock prevents a live reader losing data.
        try ParticleCacheDirectoryLease.removeObsoleteNamespaces(in:directory)
        let legacy = directory.appendingPathComponent("kquant-v1")
        if fm.fileExists(atPath:legacy.path) { try fm.removeItem(at:legacy) }
        if fm.fileExists(atPath:final.path) {
            if let existing = try? openReceipt(folder:final), existing.identity == pack.sourceIdentity, existing.files == files {
                try validateContents(folder:final,files:files)
                let root = Darwin.open(final.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
                guard root >= 0 else { throw ParticleError.invalid("Invalid mirror directory") }
                defer { close(root) }
                var valid = true
                for (name,expected) in files {
                    let fd = try ParticleIO.openRelative(root:root,path:filename(name)); defer { close(fd) }
                    var st = stat()
                    if fstat(fd,&st) != 0 || st.st_size != expected.bytes { valid = false; break }
                    if try ParticleIO.hashFile(descriptor:fd,size:expected.bytes) != expected.sha256 { valid = false; break }
                }
                if valid { return }
            }
            // The complete old tier is removed before reserving a replacement.
            try fm.removeItem(at:final)
        }
        try fm.createDirectory(at:partial,withIntermediateDirectories:true)
        let root = Darwin.open(partial.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard root >= 0 else { throw ParticleError.invalid("Invalid partial mirror directory") }; defer { close(root) }
        let allowed = Set(files.keys.map(filename)).union(["mirror-receipt.json"])
        for url in try fm.contentsOfDirectory(at:partial,includingPropertiesForKeys:nil) {
            guard allowed.contains(url.lastPathComponent) else { throw ParticleError.invalid("Unexpected file in owned partial mirror") }
        }
        let attrs = try fm.attributesOfFileSystem(forPath:directory.path)
        let free = (attrs[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
        var existing = 0
        // Shrink old partial files before growing any replacement. Even a
        // differently sized interrupted pack cannot transiently overspend.
        for (name,expected) in files {
            let short = filename(name), url = partial.appendingPathComponent(short)
            if !fm.fileExists(atPath:url.path) { continue }
            let fd = try ParticleIO.openRelative(root:root,path:short,flags:O_RDWR)
            defer { close(fd) }
            var st = stat()
            guard fstat(fd,&st) == 0, st.st_size >= 0 else { throw ParticleError.invalid("Cannot inspect partial mirror payload") }
            if st.st_size > expected.bytes {
                guard ftruncate(fd,off_t(expected.bytes)) == 0 else { throw ParticleError.invalid("Cannot trim partial mirror payload") }
            }
            existing += min(Int(st.st_size),expected.bytes)
        }
        guard free + Int64(existing) >= Int64(required)+1024*1024*1024 else { throw ParticleError.invalid("Mirror needs \(required) bytes plus 1 GiB free headroom") }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount:4*1024*1024,alignment:16_384); defer { buffer.deallocate() }
        var completed = 0
        for (name,expected) in files.sorted(by: { $0.key < $1.key }) {
            if cancellation?.isCancelled == true { throw CancellationError() }
            let short = filename(name), url = partial.appendingPathComponent(short)
            if fm.fileExists(atPath:url.path) {
                let fd = try ParticleIO.openRelative(root:root,path:short)
                var st = stat()
                let matches = fstat(fd,&st) == 0 && st.st_size == expected.bytes
                let valid = try matches && ParticleIO.hashFile(descriptor:fd,size:expected.bytes) == expected.sha256
                close(fd)
                if valid { completed += expected.bytes; progress?(name,completed,required); continue }
            }
            let output = openat(root,short,O_WRONLY|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0o600)
            guard output >= 0 else { throw ParticleError.invalid("Cannot create mirror payload") }; defer { close(output) }
            guard ftruncate(output,off_t(expected.bytes)) == 0 else { throw ParticleError.invalid("Cannot reserve mirror payload") }
            let source: Int32
            if name == "model_weights.bin" {
                guard let t = pack.tensors.values.first(where: { $0.shape.count < 3 && $0.name != "token_embd.weight" }) else { throw ParticleError.invalid("No common weights to mirror") }
                source = try pack.location(for:t,relativeOffset:0).descriptor
            } else {
                guard let layer = pack.layout.layers.first(where: { $0.file == name }) else { throw ParticleError.invalid("Unknown mirror layer") }
                source = try pack.expertRead(layer:layer.layer,expert:0)!.range.descriptor
            }
            var offset = 0
            while offset < expected.bytes {
                if cancellation?.isCancelled == true { throw CancellationError() }
                let count = min(4*1024*1024,expected.bytes-offset)
                try ParticleFileRange(descriptor:source,offset:offset,count:count).read(into:buffer)
                var written = 0
                while written < count {
                    let n = pwrite(output,buffer.advanced(by:written),count-written,off_t(offset+written))
                    if n < 0 && errno == EINTR { continue }
                    guard n > 0 else { throw ParticleError.invalid("Mirror write failed") }; written += n
                }
                offset += count
                progress?(name,completed+offset,required)
            }
            guard fsync(output) == 0 else { throw ParticleError.invalid("Cannot sync mirror payload") }
            let check = try ParticleIO.openRelative(root:root,path:short); defer { close(check) }
            guard try ParticleIO.hashFile(descriptor:check,size:expected.bytes) == expected.sha256 else { throw ParticleError.invalid("Mirror copy checksum mismatch") }
            completed += expected.bytes
        }
        let receipt = try ParticleFormat.encode(MirrorReceipt(identity:pack.sourceIdentity,files:files))
        try receipt.write(to:partial.appendingPathComponent("mirror-receipt.json"),options:.atomic)
        let receiptFD = try ParticleIO.openRelative(root:root,path:"mirror-receipt.json")
        let synced = fsync(receiptFD); close(receiptFD)
        guard synced == 0, fsync(root) == 0 else { throw ParticleError.invalid("Cannot commit mirror receipt") }
        try fm.moveItem(at:partial,to:final)
        let parent = Darwin.open(directory.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard parent >= 0 else { throw ParticleError.invalid("Cannot sync mirror parent") }; defer { close(parent) }
        guard fsync(parent) == 0 else { throw ParticleError.invalid("Cannot sync mirror publication") }
    }
    private static func openReceipt(folder: URL) throws -> MirrorReceipt {
        let root = Darwin.open(folder.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard root >= 0 else { throw ParticleError.invalid("Mirror not prepared; run prepare-cache first") }; defer { close(root) }
        return try JSONDecoder().decode(MirrorReceipt.self,from:ParticleIO.data(root:root,path:"mirror-receipt.json",limit:1024*1024))
    }
    public static func open(pack: ParticlePackFile, directory: URL, maxBytes: Int, expertReadHint: ParticleMirrorReadHint = .standard) throws -> ParticleMirrorSource {
        try ParticleMirrorSource(pack:pack,directory:directory,maxBytes:maxBytes,expertReadHint:expertReadHint)
    }
}

public final class ParticleMirrorSource: ParticleModelSource, @unchecked Sendable {
    private let pack: ParticlePackFile, lease: ParticleCacheDirectoryLease
    private let files: [String:Int32]
    private let lock = NSLock()
    private var bytesRead = 0, successfulReads = 0, failedReads = 0
    private var degraded: String?
    public var path: String { pack.path }
    public var size: Int { pack.size }
    public var metadata: [String:GGUFValue] { pack.metadata }
    public var tensors: [String:GGUFTensor] { pack.tensors }
    public var sourceIdentity: String { pack.sourceIdentity }
    public var tokenizerDirectory: String { pack.tokenizerDirectory }
    public var formatName: String { "particle+ssd-mirror" }
    public var readBytes: Int { lock.lock(); defer { lock.unlock() }; return bytesRead }
    public var hits: Int { lock.lock(); defer { lock.unlock() }; return successfulReads }
    public var corruptions: Int { lock.lock(); defer { lock.unlock() }; return failedReads }
    public var storedBytes: Int { ParticleMirror.requiredBytes(for:pack) }
    public var warning: String? { lock.lock(); defer { lock.unlock() }; return degraded }
    init(pack: ParticlePackFile, directory: URL, maxBytes: Int, expertReadHint: ParticleMirrorReadHint = .standard) throws {
        let acquired = try ParticleCacheDirectoryLease(directory:directory)
        let folder = directory.appendingPathComponent("mirror-v1")
        let root = Darwin.open(folder.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard root >= 0 else { throw ParticleError.invalid("SSD mirror missing; run prepare-cache first") }; defer { close(root) }
        let receipt = try JSONDecoder().decode(MirrorReceipt.self,from:ParticleIO.data(root:root,path:"mirror-receipt.json",limit:1024*1024))
        let expected = ParticleMirror.payload(pack)
        try ParticleMirror.validateContents(folder:folder,files:expected)
        guard maxBytes > 0, maxBytes <= ParticleMirror.maximumBytes,
              receipt.identity == pack.sourceIdentity, receipt.files == expected,
              ParticleMirror.requiredBytes(for:pack) <= maxBytes else { throw ParticleError.invalid("Mirror source identity or capacity mismatch") }
        var opened: [String:Int32] = [:], success = false
        defer { if !success { for fd in opened.values { close(fd) } } }
        for (name,record) in expected {
            let fd = try ParticleIO.openRelative(root:root,path:ParticleMirror.filename(name)); opened[name] = fd
            var st = stat()
            guard fstat(fd,&st) == 0, st.st_size == record.bytes else { throw ParticleError.invalid("Mirror payload size mismatch") }
            // Demand-loaded expert bytes already live in fixed Metal slots.
            // This candidate avoids retaining a second copy in the file cache;
            // common mappings deliberately keep their normal caching policy.
            if expertReadHint == .noCache && name.hasPrefix("packed_experts/") {
                guard fcntl(fd,F_NOCACHE,1) == 0 else { throw ParticleError.invalid("Cannot apply expert-only F_NOCACHE hint") }
            }
        }
        self.pack = pack; lease = acquired; files = opened; success = true
    }
    deinit { for fd in files.values { close(fd) } }
    public func location(for tensor: GGUFTensor, relativeOffset: Int) throws -> ParticleFileRange {
        let original = try pack.location(for:tensor,relativeOffset:relativeOffset)
        if tensor.name != "token_embd.weight", tensor.shape.count < 3, let fd = files["model_weights.bin"] {
            return ParticleFileRange(descriptor:fd,offset:original.offset,count:original.count)
        }
        return original
    }
    public func expertRead(layer: Int, expert: Int) throws -> ParticleExpertRead? {
        guard let original = try pack.expertRead(layer:layer,expert:expert),
              let record = pack.layout.layers.first(where: { $0.layer == layer }), let fd = files[record.file] else { throw ParticleError.invalid("Unknown mirror expert") }
        return ParticleExpertRead(range:ParticleFileRange(descriptor:fd,offset:original.range.offset,count:original.range.count),sha256:original.sha256,fromSSD:true)
    }
    enum PendingExpertRead: Sendable {
        case needsVerification(ParticleExpertRead)
        case completed(ParticleExpertLoad)
    }
    /// Read into a reserved slot; the caller must retain that reservation until
    /// finishExpertRead has verified these exact bytes and every GPU consumer
    /// has completed. Read failures use the existing verified SD fallback.
    func beginExpertRead(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int) throws -> PendingExpertRead {
        let read = try expertRead(layer:layer,expert:expert)!
        guard read.range.count == capacity else { throw ParticleError.invalid("Mirror expert buffer size mismatch") }
        lock.lock(); bytesRead += capacity; lock.unlock()
        do { try read.range.read(into:pointer); return .needsVerification(read) }
        catch { return .completed(try recoverExpertRead(layer:layer,expert:expert,into:pointer,capacity:capacity,error:error)) }
    }
    func finishExpertRead(_ pending: PendingExpertRead, layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int) throws -> ParticleExpertLoad {
        switch pending {
        case .completed(let loaded):
            guard loaded.bytes == capacity else { throw ParticleError.invalid("Completed expert capacity mismatch") }; return loaded
        case .needsVerification(let read):
            guard read.range.count == capacity else { throw ParticleError.invalid("Unverified expert capacity mismatch") }
            if ParticleIO.hash(UnsafeRawBufferPointer(start:pointer,count:capacity)) == read.sha256 {
                lock.lock(); successfulReads += 1; lock.unlock()
                return ParticleExpertLoad(bytes:capacity,fromSSD:true)
            }
            return try recoverExpertRead(layer:layer,expert:expert,into:pointer,capacity:capacity,
                error:ParticleError.invalid("Mirror expert checksum mismatch"))
        }
    }
    private func recoverExpertRead(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int, error: any Error) throws -> ParticleExpertLoad {
        lock.lock(); failedReads += 1; degraded = "SSD mirror degraded; fetching failed expert from SD: \(error)"; lock.unlock()
        guard let loaded=try pack.loadExpert(layer:layer,expert:expert,into:pointer,capacity:capacity) else {
            throw ParticleError.invalid("SD fallback did not provide an expert")
        }
        return loaded
    }
    public func loadExpert(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int) throws -> ParticleExpertLoad? {
        let pending=try beginExpertRead(layer:layer,expert:expert,into:pointer,capacity:capacity)
        return try finishExpertRead(pending,layer:layer,expert:expert,into:pointer,capacity:capacity)
    }
    public func prepareForInference() throws {
        guard let common = files["model_weights.bin"], let expected = pack.manifest.files["model_weights.bin"],
              try ParticleIO.hashFile(descriptor:common,size:expected.bytes) == expected.sha256 else { throw ParticleError.invalid("Mirror common-weight checksum mismatch; re-run prepare-cache") }
        let embedding = try pack.tensor("token_embd.weight"), range = try pack.location(for:embedding,relativeOffset:0)
        guard let expected = pack.manifest.files["embeddings.bin"], try ParticleIO.hashFile(descriptor:range.descriptor,size:expected.bytes) == expected.sha256 else { throw ParticleError.invalid("SD embedding checksum mismatch") }
    }
}
