import Foundation
import ParticleFormat

public struct ParticleV2Tensor: Codable, Equatable, Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let file: String
    public let offset: Int64
    /// Bytes per expert for routed tensors; full bytes for resident tensors.
    public let bytes: Int64
    public let layer: Int?
    public let expertStride: Int64?
    public init(name: String, dtype: String, shape: [Int], file: String, offset: Int64, bytes: Int64,
                layer: Int? = nil, expertStride: Int64? = nil) {
        self.name = name; self.dtype = dtype; self.shape = shape; self.file = file
        self.offset = offset; self.bytes = bytes; self.layer = layer; self.expertStride = expertStride
    }
}
public struct ParticleV2File: Codable, Equatable, Sendable {
    public let bytes: Int64
    public let sha256: String
    public init(bytes: Int64, sha256: String) { self.bytes = bytes; self.sha256 = sha256 }
}
public struct ParticleV2Manifest: Codable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let model: ModelDescriptor
    public let files: [String: ParticleV2File]
    public let tensors: [ParticleV2Tensor]
    public let sourceFiles: [String: ParticleV2File]
    public let expertHashes: [String: [String]]
    public init(model: ModelDescriptor, files: [String: ParticleV2File], tensors: [ParticleV2Tensor], sourceFiles: [String: ParticleV2File], expertHashes: [String: [String]]) {
        magic = ParticleFormat.magic; versionMajor = 2; self.model = model
        self.files = files; self.tensors = tensors; self.sourceFiles = sourceFiles
        self.expertHashes = expertHashes
    }
    public func validate() throws {
        try model.validate()
        guard magic == ParticleFormat.magic, versionMajor == 2, model.architecture == .qwen35MoE,
              !files.isEmpty, files.count <= 512, !sourceFiles.isEmpty, sourceFiles.count <= 512,
              !tensors.isEmpty, tensors.count <= 100_000, Set(tensors.map(\.name)).count == tensors.count else {
            throw ModelLibraryError.invalid("Unsupported or incomplete Particle v2 manifest.")
        }
        for required in ["config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja", "common.bin"] {
            guard files[required] != nil else { throw ModelLibraryError.invalid("Pack file is missing: \(required).") }
        }
        for (path, file) in files.merging(sourceFiles, uniquingKeysWith: { a, _ in a }) {
            try ParticleFormat.validatePath(path)
            guard path != "manifest.json", path != "verified-install.json", path != "install-progress.json",
                  file.bytes >= 0, ParticleFormat.digestIsValid(file.sha256) else { throw ModelLibraryError.invalid("Invalid pack file record.") }
        }
        var names = Set<String>()
        for path in files.keys {
            guard names.insert(path.precomposedStringWithCanonicalMapping.lowercased()).inserted else { throw ModelLibraryError.invalid("Conflicting pack filenames.") }
        }
        for tensor in tensors {
            guard let file = files[tensor.file], tensor.offset >= 0, tensor.bytes > 0, tensor.offset <= file.bytes,
                  tensor.bytes <= file.bytes-tensor.offset, !tensor.shape.isEmpty,tensor.shape.count <= 8,
                  tensor.name.utf8.count <= 1024,tensor.shape.allSatisfy({$0 > 0 && $0 <= Int(Int32.max)}) else { throw ModelLibraryError.invalid("Tensor exceeds its payload: \(tensor.name).") }
            let widths:[String:Int64] = ["U32":4,"F32":4,"F16":2,"BF16":2,"I32":4,"I64":8,"U8":1,"I8":1,"BOOL":1]
            guard let width = widths[tensor.dtype] else {throw ModelLibraryError.invalid("Unsupported tensor dtype.")}
            var expected = width
            for dimension in (tensor.layer == nil ? tensor.shape : Array(tensor.shape.dropFirst())) {
                let (next,overflow) = expected.multipliedReportingOverflow(by:Int64(dimension));guard !overflow else {throw ModelLibraryError.invalid("Tensor shape overflow.")};expected = next
            }
            guard expected == tensor.bytes else {throw ModelLibraryError.invalid("Tensor bytes do not match its shape.")}
            if let layer = tensor.layer {
                guard (0..<model.layers).contains(layer), tensor.shape.first == model.experts,
                      let stride = tensor.expertStride, stride >= tensor.bytes,stride <= 32*1024*1024, file.bytes % stride == 0,
                      file.bytes/stride == Int64(model.experts), tensor.offset < stride,
                      tensor.bytes <= stride-tensor.offset else { throw ModelLibraryError.invalid("Invalid expert tensor layout.") }
            } else if tensor.expertStride != nil { throw ModelLibraryError.invalid("Resident tensor has an expert stride.") }
        }
        let expertFiles = Set(tensors.filter {$0.layer != nil}.map(\.file))
        let layers=Dictionary(grouping:tensors.filter{$0.layer != nil},by:{$0.layer!})
        guard layers.count == model.layers,expertFiles.count == model.layers,
              Set(tensors.filter{$0.layer == nil}.map(\.file)).isDisjoint(with:expertFiles) else {
            throw ModelLibraryError.invalid("Resident and routed tensor ownership is inconsistent.")
        }
        for records in layers.values {
            guard Set(records.map(\.file)).count == 1,Set(records.compactMap(\.expertStride)).count == 1 else {
                throw ModelLibraryError.invalid("Each layer must have one expert file and stride.")
            }
        }
        for records in Dictionary(grouping:tensors,by:\.file).values {
            var end:Int64=0
            for tensor in records.sorted(by:{$0.offset < $1.offset}) {
                guard tensor.offset >= end else {throw ModelLibraryError.invalid("Overlapping tensor payload ranges.")}
                end=tensor.offset+tensor.bytes
            }
        }
        guard Set(expertHashes.keys) == expertFiles,expertHashes.values.allSatisfy({$0.count == model.experts && $0.allSatisfy(ParticleFormat.digestIsValid)}) else {
            throw ModelLibraryError.invalid("Missing per-expert verification hashes.")
        }
    }
}
