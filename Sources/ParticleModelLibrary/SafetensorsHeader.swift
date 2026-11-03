import Foundation
import CoreFoundation

public struct SafetensorRecord: Codable, Equatable, Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let offset: Int64
    public let bytes: Int64
    public init(name: String, dtype: String, shape: [Int], offset: Int64, bytes: Int64) {
        self.name = name; self.dtype = dtype; self.shape = shape; self.offset = offset; self.bytes = bytes
    }
}

/// Parses only the bounded header. All offsets refer to the original file.
public struct SafetensorsHeader: Sendable {
    public static let maximumHeaderBytes = 16 * 1024 * 1024
    public let tensors: [String: SafetensorRecord]
    public let dataOffset: Int64
    public init(prefix: Data, fileBytes: Int64) throws {
        guard prefix.count >= 8 else { throw ModelLibraryError.invalid("Truncated safetensors header.") }
        let length = prefix.prefix(8).enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        guard length > 1, length <= Self.maximumHeaderBytes, fileBytes >= 8,
              length <= UInt64(fileBytes-8), prefix.count >= 8+Int(length) else { throw ModelLibraryError.invalid("Invalid safetensors header length.") }
        let start = Int64(length)+8
        guard let object = try JSONSerialization.jsonObject(with: prefix.subdata(in: 8..<(8+Int(length)))) as? [String: Any], object.count <= 100_000 else {
            throw ModelLibraryError.invalid("Invalid safetensors index.")
        }
        let widths: [String: Int64] = ["F32":4,"F16":2,"BF16":2,"U32":4,"I32":4,"I64":8,"U8":1,"I8":1,"BOOL":1]
        var records = [String: SafetensorRecord]()
        for (name, value) in object where name != "__metadata__" {
            guard name.utf8.count <= 1024, let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                  let width = widths[dtype], let shapeNumbers = entry["shape"] as? [NSNumber], shapeNumbers.count <= 8,
                  let offsets = entry["data_offsets"] as? [NSNumber], offsets.count == 2 else { throw ModelLibraryError.invalid("Unsupported tensor header: \(name).") }
            func integer(_ value: NSNumber) throws -> Int64 {
                let n = value.int64Value
                guard CFGetTypeID(value) != CFBooleanGetTypeID(), n >= 0, value.doubleValue == Double(n) else { throw ModelLibraryError.invalid("Invalid tensor dimension or offset.") }
                return n
            }
            let shape = try shapeNumbers.map { Int(try integer($0)) }
            var elements: Int64 = 1
            for n in shape { let (next, overflow) = elements.multipliedReportingOverflow(by: Int64(n)); guard !overflow else { throw ModelLibraryError.invalid("Tensor shape overflow.") }; elements = next }
            let lo = try integer(offsets[0]), hi = try integer(offsets[1])
            let (expected, overflow) = elements.multipliedReportingOverflow(by: width)
            guard !overflow, hi >= lo, hi <= fileBytes-start, hi-lo == expected else { throw ModelLibraryError.invalid("Tensor bounds do not match its shape: \(name).") }
            records[name] = .init(name: name, dtype: dtype, shape: shape, offset: start+lo, bytes: hi-lo)
        }
        var end = start
        for tensor in records.values.sorted(by: { $0.offset < $1.offset }) where tensor.bytes > 0 {
            guard tensor.offset == end else { throw ModelLibraryError.invalid("Overlapping or missing safetensors ranges.") }
            end = tensor.offset+tensor.bytes
        }
        guard end == fileBytes, !records.isEmpty else { throw ModelLibraryError.invalid("Incomplete safetensors payload.") }
        tensors = records; dataOffset = start
    }
    public static func read(_ url: URL) throws -> Self {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true, let size = values.fileSize else { throw ModelLibraryError.invalid("Choose a regular safetensors file.") }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let initial = try handle.read(upToCount: 8) ?? Data()
        guard initial.count == 8 else { throw ModelLibraryError.invalid("Truncated safetensors file.") }
        let length = initial.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
        guard length <= Self.maximumHeaderBytes else { throw ModelLibraryError.invalid("Safetensors header exceeds the safety limit.") }
        return try Self(prefix: initial+(handle.read(upToCount: Int(length)) ?? Data()), fileBytes: Int64(size))
    }
}
