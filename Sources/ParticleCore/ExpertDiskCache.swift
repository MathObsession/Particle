import Foundation
import CryptoKit
import Darwin
import ParticleGGUF

/// A bounded, lossless second cache tier. Only experts actually requested by
/// the router are copied; the original GGUF stays on its source volume.
/// Entries contain the exact native K-quant bytes in Metal slot order.
final class ExpertDiskCache: @unchecked Sendable {
    struct Entry: Codable {
        var bytes: Int
        var digest: String
        var frequency: Int
        var age: UInt64
    }
    private struct Index: Codable {
        let version: Int
        let identity: String
        let entries: [String:Entry]
    }
    private let lock = NSRecursiveLock()
    private let policy: ParticleSSDCachePolicy
    private let directoryLease: ParticleCacheDirectoryLease
    private var protectedKeys = Set<String>()
    private let folder: URL, identity: String, limit: Int
    private let lease: Int32
    private var reading = Set<String>(), writing = Set<String>()
    private var reservedBytes = 0
    // At most two saved victims (under 16 MiB for Sarvam), independently
    // checked and exchanged. The metadata lock is never held for bulk I/O.
    private let exchangePermits = DispatchSemaphore(value:2)
    private var entries: [String:Entry] = [:]
    private var frequencies: [String:Int] = [:]
    private var clock: UInt64 = 0
    private(set) var hits = 0
    private(set) var readBytes = 0
    private(set) var writeBytes = 0
    private(set) var corruptions = 0
    private(set) var failure: String?
    var storedBytes: Int { lock.lock(); defer { lock.unlock() }; return entries.values.reduce(0) { $0+$1.bytes } }

    init(file: any ParticleModelSource, directory: URL, maxBytes: Int, policy: ParticleSSDCachePolicy = .victim) throws {
        guard maxBytes > 0, maxBytes <= ParticleMirror.maximumBytes, policy != .mirror else {
            throw ParticleError.invalid("SSD expert cache must be bounded to at most 18432 MiB; mirror uses its dedicated source")
        }
        let acquired = try ParticleCacheDirectoryLease(directory:directory)
        guard !FileManager.default.fileExists(atPath:directory.appendingPathComponent("mirror-v1").path),
              !FileManager.default.fileExists(atPath:directory.appendingPathComponent("mirror-v1.partial").path) else {
            throw ParticleError.invalid("A complete or partial mirror owns this SSD tier; select mirror policy or --no-ssd-cache")
        }
        self.directoryLease = acquired; self.policy = policy
        let stamp = file.sourceIdentity
        identity = Self.hash(Data(stamp.utf8)); limit = maxBytes
        folder = directory.appendingPathComponent("kquant-v1",isDirectory:true)
        // Keep one payload budget across source revisions. Preserve entries
        // from this development pass's identity-directory layout on first use.
        let previous = directory.appendingPathComponent(identity,isDirectory:true)
        if !FileManager.default.fileExists(atPath:folder.path),
           FileManager.default.fileExists(atPath:previous.appendingPathComponent("index.json").path) {
            try FileManager.default.moveItem(at:previous,to:folder)
        }
        try ParticleCacheDirectoryLease.removeObsoleteNamespaces(in:directory)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let cacheLease = open(folder.appendingPathComponent("cache.lock").path,O_RDWR|O_CREAT|O_NOFOLLOW,0o600)
        guard cacheLease >= 0 else { throw ParticleError.invalid("Cannot open SSD expert cache lock") }
        guard flock(cacheLease,LOCK_EX|LOCK_NB) == 0 else {
            close(cacheLease); throw ParticleError.invalid("SSD expert cache is in use by another process")
        }
        // Failed initializers never own a descriptor or flush another owner's
        // index. Store the lease only after exclusive ownership succeeds.
        lease=cacheLease
        let indexURL = folder.appendingPathComponent("index.json")
        let indexData: Data? = {
            guard let handle = try? FileHandle(forReadingFrom:indexURL) else { return nil }
            defer { try? handle.close() }
            return try? handle.read(upToCount:2_000_001)
        }()
        if let data = indexData, data.count <= 2_000_000,
           let index = try? JSONDecoder().decode(Index.self,from:data), index.version == 1, index.identity == identity {
            for (key,entry) in index.entries {
                let parts = key.split(separator:"-").compactMap { Int($0) }
                guard parts.count == 2, (1...18).contains(parts[0]), (0..<128).contains(parts[1]),
                      key == "\(parts[0])-\(parts[1])", entry.frequency > 0, entry.frequency <= Int.max/2,
                      entry.bytes > 0, entry.bytes <= 32 * 1024 * 1024,
                      entry.digest.count == 64, entry.digest.allSatisfy({ $0.isHexDigit }) else { continue }
                guard let size = try? FileManager.default.attributesOfItem(atPath:folder.appendingPathComponent(key+".expert").path)[.size] as? NSNumber,
                      size.intValue == entry.bytes else { continue }
                entries[key] = entry; frequencies[key] = entry.frequency; clock = max(clock,entry.age)
            }
        }
        // Clean only owned entry names in this model-specific cache directory.
        // Temporary/orphan files can result from a crash before index commit.
        let owned = Set(entries.keys.map { $0+".expert" })
        for url in (try? FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:nil)) ?? [] {
            if (url.pathExtension == "expert" && !owned.contains(url.lastPathComponent)) || url.lastPathComponent.hasPrefix(".entry-") {
                try? FileManager.default.removeItem(at:url)
            }
        }
        trim(to:maxBytes)
    }
    deinit { flush(); flock(lease,LOCK_UN); close(lease) }
    private static func hash(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    private static func hash(_ pointer: UnsafeRawPointer, count: Int) -> String {
        var digest = SHA256()
        digest.update(bufferPointer:UnsafeRawBufferPointer(start:pointer,count:count))
        return digest.finalize().map { String(format:"%02x",$0) }.joined()
    }
    func admitLoaded(layer: Int, expert: Int, from pointer: UnsafeMutableRawPointer, count: Int, protecting experts: [Int]) {
        guard policy == .readThrough else { return }
        store(layer:layer,expert:expert,from:pointer,count:count,protecting:Set(experts.map { key(layer,$0) }))
    }
    private func key(_ layer: Int, _ expert: Int) -> String { "\(layer)-\(expert)" }
    func record(layer: Int, experts: [Int], frequencies counts: [Int]) {
        lock.lock(); defer { lock.unlock() }
        for (i,expert) in experts.enumerated() {
            let k = key(layer,expert)
            frequencies[k,default:0] = min(Int.max/2,frequencies[k,default:0]+counts[i])
            if var entry = entries[k] {
                clock &+= 1; entry.frequency = frequencies[k]!; entry.age = clock; entries[k] = entry
            }
        }
    }
    /// Pin under the metadata lock; validate bytes and SHA outside it. The
    /// entry cannot be evicted or replaced until this read is committed.
    func load(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, count: Int) -> Bool {
        autoreleasepool {
            let k = key(layer,expert)
            lock.lock()
            guard let entry = entries[k], entry.bytes == count, !reading.contains(k) else { lock.unlock(); return false }
            reading.insert(k); lock.unlock()
            let fd = open(folder.appendingPathComponent(k+".expert").path,O_RDONLY|O_NOFOLLOW)
            var valid = fd >= 0
            if fd >= 0 {
                _ = fcntl(fd,F_NOCACHE,1)
                defer { close(fd) }
                var info = stat()
                valid = fstat(fd,&info) == 0 && info.st_size == count && (info.st_mode & S_IFMT) == S_IFREG
                var done = 0
                while valid && done < count {
                    let n = pread(fd,pointer.advanced(by:done),min(count-done,8*1024*1024),off_t(done))
                    if n < 0 && errno == EINTR { continue }
                    if n <= 0 { valid = false; break }; done += n
                }
            }
            if valid { valid = Self.hash(pointer,count:count) == entry.digest }
            lock.lock(); defer { reading.remove(k); lock.unlock() }
            if policy == .victim || !valid {
                do {
                    try FileManager.default.removeItem(at:folder.appendingPathComponent(k+".expert"))
                    entries.removeValue(forKey:k); protectedKeys.remove(k)
                } catch {
                    failure = "SSD cache writes disabled after entry removal failed: \(error.localizedDescription)"
                }
            } else {
                clock &+= 1; entries[k]?.age = clock; protectedKeys.insert(k)
                while protectedKeys.reduce(0, { $0+(entries[$1]?.bytes ?? 0) }) > limit/5*4 {
                    guard let oldest = protectedKeys.min(by: { (entries[$0]?.age ?? 0) < (entries[$1]?.age ?? 0) }) else { break }
                    protectedKeys.remove(oldest)
                }
            }
            if valid { hits += 1; readBytes += count }
            else { corruptions += 1 }
            return valid
        }
    }
    func store(layer: Int, expert: Int, from pointer: UnsafeMutableRawPointer, count: Int, protecting: Set<String> = []) {
        autoreleasepool {
            let k = key(layer,expert)
            lock.lock()
            guard failure == nil, count <= limit, entries[k] == nil, !writing.contains(k),
                  trim(to:limit-reservedBytes-count,protecting:protecting.union(reading)) else { lock.unlock(); return }
            writing.insert(k); reservedBytes += count
            lock.unlock()
            let url = folder.appendingPathComponent(k+".expert")
            let temp = folder.appendingPathComponent(".entry-"+UUID().uuidString)
            do {
                let fd = open(temp.path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0o600)
                guard fd >= 0 else { throw ParticleError.invalid("Cannot create SSD expert entry") }
                do {
                    defer { close(fd) }; _ = fcntl(fd,F_NOCACHE,1)
                    var done = 0
                    while done < count {
                        let n = pwrite(fd,pointer.advanced(by:done),min(count-done,8*1024*1024),off_t(done))
                        if n < 0 && errno == EINTR { continue }
                        guard n > 0 else { throw ParticleError.invalid("Cannot write SSD expert entry") }; done += n
                    }
                }
                let digest = Self.hash(pointer,count:count)
                lock.lock(); defer { lock.unlock() }
                do {
                    try FileManager.default.moveItem(at:temp,to:url)
                    clock &+= 1
                    entries[k] = Entry(bytes:count,digest:digest,frequency:frequencies[k,default:1],age:clock)
                    writeBytes += count
                    reservedBytes -= count; writing.remove(k)
                } catch {
                    // Keep the reservation if removal fails: physical payload
                    // remains bounded even after a filesystem error.
                    if (try? FileManager.default.removeItem(at:temp)) != nil || !FileManager.default.fileExists(atPath:temp.path) { reservedBytes -= count }
                    writing.remove(k); failure = "SSD cache writes disabled: \(error.localizedDescription)"
                }
            } catch {
                lock.lock(); defer { lock.unlock() }
                if (try? FileManager.default.removeItem(at:temp)) != nil || !FileManager.default.fileExists(atPath:temp.path) { reservedBytes -= count }
                writing.remove(k); failure = "SSD cache writes disabled: \(error.localizedDescription)"
            }
        }
    }
    func exchange(layer: Int, expert: Int, evicted: Int, into pointer: UnsafeMutableRawPointer,
                  count: Int, protecting experts: [Int]) -> Bool {
        exchangePermits.wait(); defer { exchangePermits.signal() }
        if policy == .readThrough { return load(layer:layer,expert:expert,into:pointer,count:count) }
        return autoreleasepool {
            let protected = Set(experts.map { key(layer,$0) })
            lock.lock(); let available = entries[key(layer,expert)]?.bytes == count; lock.unlock()
            if available {
                let victim = evicted >= 0 ? UnsafeMutableRawPointer.allocate(byteCount:count,alignment:64) : nil
                defer { victim?.deallocate() }
                victim?.copyMemory(from:pointer,byteCount:count)
                let found = load(layer:layer,expert:expert,into:pointer,count:count)
                if let victim { store(layer:layer,expert:evicted,from:victim,count:count,protecting:protected) }
                return found
            }
            if evicted >= 0 { store(layer:layer,expert:evicted,from:pointer,count:count,protecting:protected) }
            return false
        }
    }
    private func victim(protecting: Set<String>) -> String? {
        let eligible = entries.keys.filter { !protecting.contains($0) && !reading.contains($0) && !writing.contains($0) }
        let probation = eligible.filter { !protectedKeys.contains($0) }
        return (policy == .readThrough && !probation.isEmpty ? probation : eligible).min {
            let a = entries[$0]!, b = entries[$1]!
            return a.age == b.age ? $0 < $1 : a.age < b.age
        }
    }
    @discardableResult private func trim(to bytes: Int, protecting: Set<String> = []) -> Bool {
        var total = entries.values.reduce(0) { $0+$1.bytes }
        while total > bytes {
            guard let k = victim(protecting:protecting), let entry = entries[k] else { return false }
            let url = folder.appendingPathComponent(k+".expert")
            do {
                if FileManager.default.fileExists(atPath:url.path) { try FileManager.default.removeItem(at:url) }
            } catch {
                failure = "SSD cache writes disabled: cannot evict entry (\(error.localizedDescription))"; return false
            }
            total -= entry.bytes; entries.removeValue(forKey:k); protectedKeys.remove(k)
        }
        return true
    }
    func flush() {
        lock.lock(); defer { lock.unlock() }
        do { try JSONEncoder().encode(Index(version:1,identity:identity,entries:entries)).write(to:folder.appendingPathComponent("index.json"),options:.atomic) }
        catch { failure = "SSD cache index could not be saved: \(error.localizedDescription)" }
    }
}
