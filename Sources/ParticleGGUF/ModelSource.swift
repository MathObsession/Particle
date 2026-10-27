import Foundation
import Darwin
import CryptoKit

public struct ParticleFileRange: Sendable {
    public let descriptor: Int32
    public let offset: Int
    public let count: Int
    public init(descriptor: Int32, offset: Int, count: Int) { self.descriptor = descriptor; self.offset = offset; self.count = count }
    public func read(into pointer: UnsafeMutableRawPointer, count: Int? = nil) throws {
        let length = count ?? self.count
        guard length >= 0, length <= self.count else { throw ParticleError.invalid("Read exceeds tensor range") }
        var done = 0
        while done < length {
            let n = pread(descriptor, pointer.advanced(by: done), length - done, off_t(offset + done))
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { throw ParticleError.invalid("Model read failed or storage disconnected: \(String(cString: strerror(errno)))") }
            done += n
        }
    }
}
public struct ParticleExpertRead: Sendable {
    public let range: ParticleFileRange
    public let sha256: String
    public let fromSSD: Bool
    public init(range: ParticleFileRange, sha256: String, fromSSD: Bool = false) { self.range = range; self.sha256 = sha256; self.fromSSD = fromSSD }
}
public struct ParticleExpertLoad: Sendable {
    public let bytes: Int
    public let fromSSD: Bool
    public init(bytes: Int, fromSSD: Bool) { self.bytes = bytes; self.fromSSD = fromSSD }
}

public protocol ParticleModelSource: AnyObject, Sendable {
    var path: String { get }
    var size: Int { get }
    var metadata: [String: GGUFValue] { get }
    var tensors: [String: GGUFTensor] { get }
    var sourceIdentity: String { get }
    var tokenizerDirectory: String { get }
    var formatName: String { get }
    func location(for tensor: GGUFTensor, relativeOffset: Int) throws -> ParticleFileRange
    func expertRead(layer: Int, expert: Int) throws -> ParticleExpertRead?
    func loadExpert(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int) throws -> ParticleExpertLoad?
    func prepareForInference() throws
}
extension ParticleModelSource {
    public func tensor(_ name: String, shape: [Int]? = nil) throws -> GGUFTensor {
        guard let value = tensors[name], shape == nil || value.shape == shape else { throw ParticleError.invalid("Missing tensor or unexpected shape: \(name), expected \(shape ?? [])") }
        return value
    }
    public func readTensor(_ tensor: GGUFTensor, into pointer: UnsafeMutableRawPointer, count: Int, relativeOffset: Int = 0) throws {
        guard relativeOffset >= 0, count >= 0, relativeOffset <= tensor.byteCount, count <= tensor.byteCount - relativeOffset else { throw ParticleError.invalid("Read exceeds tensor bounds") }
        try location(for: tensor, relativeOffset: relativeOffset).read(into: pointer, count: count)
    }
    public func expertRead(layer: Int, expert: Int) throws -> ParticleExpertRead? { nil }
    public func loadExpert(layer: Int, expert: Int, into pointer: UnsafeMutableRawPointer, capacity: Int) throws -> ParticleExpertLoad? {
        guard let packed = try expertRead(layer:layer,expert:expert) else { return nil }
        guard packed.range.count == capacity else { throw ParticleError.invalid("Packed expert stride differs from RAM slot") }
        try packed.range.read(into:pointer)
        guard ParticleIO.hash(UnsafeRawBufferPointer(start:pointer,count:capacity)) == packed.sha256 else { throw ParticleError.invalid("Packed expert checksum mismatch") }
        return ParticleExpertLoad(bytes:capacity,fromSSD:packed.fromSSD)
    }
    public func prepareForInference() throws {}
}
extension GGUFFile: ParticleModelSource {
    public var formatName: String { "gguf" }
    public var tokenizerDirectory: String { URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("tokenizer").path }
    public var sourceIdentity: String {
        var st = stat()
        guard fstat(descriptor, &st) == 0 else { return "unavailable|\(path)|\(size)" }
        return "particle-kquant-expert-v1|\(path)|\(st.st_dev)|\(st.st_ino)|\(st.st_size)|\(st.st_mtimespec.tv_sec)|\(st.st_mtimespec.tv_nsec)|\(st.st_ctimespec.tv_sec)|\(st.st_ctimespec.tv_nsec)"
    }
    public func location(for tensor: GGUFTensor, relativeOffset: Int = 0) throws -> ParticleFileRange {
        guard tensors[tensor.name]?.offset == tensor.offset, relativeOffset >= 0, relativeOffset <= tensor.byteCount else { throw ParticleError.invalid("Invalid tensor location") }
        return ParticleFileRange(descriptor: descriptor, offset: tensor.offset + relativeOffset, count: tensor.byteCount - relativeOffset)
    }
}
public enum ParticleModels {
    public static func open(path: String) throws -> any ParticleModelSource {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else { throw ParticleError.invalid("Model path is missing: \(path)") }
        return isDirectory.boolValue ? try ParticlePackFile(path: path) : try GGUFFile(path: path)
    }
}
public enum ParticleIO {
    public static func hash(_ bytes: UnsafeRawBufferPointer) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    public static func hash(_ data: Data) -> String { data.withUnsafeBytes { hash($0) } }
    public static func hashFile(descriptor: Int32, size: Int, chunk: Int = 4 * 1024 * 1024) throws -> String {
        let p = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16_384); defer { p.deallocate() }
        var hash = SHA256(), offset = 0
        while offset < size {
            let n = min(chunk, size-offset)
            try ParticleFileRange(descriptor: descriptor, offset: offset, count: n).read(into: p)
            hash.update(bufferPointer: UnsafeRawBufferPointer(start: p, count: n)); offset += n
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    /// Every component is opened relative to an already-owned directory FD.
    public static func openRelative(root: Int32, path: String, flags: Int32 = O_RDONLY) throws -> Int32 {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.isEmpty, components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") && !$0.contains("\\") }) else { throw ParticleError.invalid("Unsafe pack path") }
        var parent = dup(root)
        guard parent >= 0 else { throw ParticleError.invalid("Cannot duplicate pack directory") }
        defer { close(parent) }
        for component in components.dropLast() {
            let next = openat(parent, String(component), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            guard next >= 0 else { throw ParticleError.invalid("Invalid pack directory: \(path)") }
            close(parent); parent = next
        }
        let fd = openat(parent, String(components.last!), flags | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw ParticleError.invalid("Cannot open pack file: \(path)") }
        var st = stat()
        guard fstat(fd,&st) == 0, (st.st_mode & S_IFMT) == S_IFREG else { close(fd); throw ParticleError.invalid("Pack entry is not a regular file: \(path)") }
        return fd
    }
    public static func data(root: Int32, path: String, limit: Int) throws -> Data {
        let fd = try openRelative(root: root, path: path); defer { close(fd) }
        var st = stat()
        guard fstat(fd,&st) == 0, st.st_size >= 0, st.st_size <= limit else { throw ParticleError.invalid("Pack index exceeds limit: \(path)") }
        var data = Data(count: Int(st.st_size))
        try data.withUnsafeMutableBytes { p in if let base = p.baseAddress { try ParticleFileRange(descriptor:fd,offset:0,count:p.count).read(into:base) } }
        return data
    }
}
