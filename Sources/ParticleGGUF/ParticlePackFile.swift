import Foundation
import Darwin
import CryptoKit
import ParticleFormat

/// A verified index and owned file descriptors. Weight payloads are read or
/// mapped on demand, never collected into a Swift array.
public final class ParticlePackFile: ParticleModelSource, @unchecked Sendable {
    public let path: String, size: Int
    public let metadata: [String: GGUFValue]
    public let tensors: [String: GGUFTensor]
    public let manifest: ParticleManifest
    public let layout: ParticleExpertLayout
    public let sourceIdentity: String
    public var tokenizerDirectory: String { URL(fileURLWithPath:path).appendingPathComponent("tokenizer").path }
    public var formatName: String { "particle" }
    private let root: Int32
    private let descriptors: [String: Int32]
    private let records: [String: ParticleTensorRecord]
    private let layers: [Int: ParticleLayerRecord]
    public init(path: String) throws {
        let resolved = URL(fileURLWithPath:path).resolvingSymlinksInPath().path
        let directory = open(resolved,O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard directory >= 0 else { throw ParticleError.invalid("Cannot open .particle directory") }
        var owned: [String: Int32] = [:], success = false
        defer { if !success { for fd in owned.values { close(fd) }; close(directory) } }
        let manifestData = try ParticleIO.data(root:directory,path:"manifest.json",limit:ParticleFormat.indexLimit)
        let decoded = try JSONDecoder().decode(ParticleManifest.self,from:manifestData)
        try decoded.validate()
        let receipt = try JSONDecoder().decode(ParticleInstallReceipt.self,from:ParticleIO.data(root:directory,path:"verified-install.json",limit:4096))
        let manifestDigest = ParticleIO.hash(manifestData)
        guard receipt.complete, receipt.manifestSHA256 == manifestDigest else { throw ParticleError.invalid("Incomplete or modified .particle installation") }
        var total = 0
        for (name,expected) in decoded.files {
            let fd = try ParticleIO.openRelative(root:directory,path:name)
            owned[name] = fd
            var st = stat()
            guard fstat(fd,&st) == 0, st.st_size == expected.bytes else { throw ParticleError.invalid("Pack file size changed: \(name)") }
            let (next,overflow) = total.addingReportingOverflow(expected.bytes)
            guard !overflow else { throw ParticleError.invalid("Pack size overflow") }; total = next
        }
        func index(_ name: String, limit: Int = ParticleFormat.indexLimit) throws -> Data {
            let bytes = try ParticleIO.data(root:directory,path:name,limit:limit)
            guard let expected = decoded.files[name], bytes.count == expected.bytes, ParticleIO.hash(bytes) == expected.sha256 else { throw ParticleError.invalid("Pack index checksum mismatch: \(name)") }
            return bytes
        }
        let tensorRecords = try JSONDecoder().decode([ParticleTensorRecord].self,from:index("weights.json"))
        let expertLayout = try JSONDecoder().decode(ParticleExpertLayout.self,from:index("packed_experts/layout.json"))
        guard tensorRecords.count <= 100_000, expertLayout.expertsPerLayer == 128, expertLayout.layers.count <= 18 else { throw ParticleError.invalid("Invalid pack index dimensions") }
        var layerTable: [Int: ParticleLayerRecord] = [:]
        for layer in expertLayout.layers {
            guard (1...18).contains(layer.layer), layerTable[layer.layer] == nil,
                  layer.file == String(format:"packed_experts/layer_%02d.bin",layer.layer),
                  layer.stride > 0, layer.stride <= 32 * 1024 * 1024,
                  layer.stride % ParticleFormat.alignment == 0,
                  layer.hashes.count == 128, layer.hashes.allSatisfy(ParticleFormat.digestIsValid),
                  decoded.files[layer.file]?.bytes == layer.stride * 128 else { throw ParticleError.invalid("Invalid expert layer layout") }
            layerTable[layer.layer] = layer
        }
        var table: [String: GGUFTensor] = [:], recordTable: [String: ParticleTensorRecord] = [:]
        var intervals: [String: [(Int,Int)]] = [:]
        for record in tensorRecords {
            guard !record.name.isEmpty, table[record.name] == nil, (1...4).contains(record.shape.count),
                  record.shape.allSatisfy({ $0 > 0 && $0 <= 1 << 32 }),
                  let type = GGMLType(rawValue:record.encoding), record.shape[0] % type.blockElements == 0,
                  record.offset >= 0, record.bytes > 0 else { throw ParticleError.invalid("Invalid tensor record: \(record.name)") }
            var elements = 1
            for dimension in record.shape {
                let (next,overflow) = elements.multipliedReportingOverflow(by:dimension)
                guard !overflow else { throw ParticleError.invalid("Tensor shape overflow") }; elements = next
            }
            let (bytes,overflow) = (elements / type.blockElements).multipliedReportingOverflow(by:type.blockBytes)
            guard !overflow, bytes == record.bytes else { throw ParticleError.invalid("Tensor size or quantization mismatch") }
            let count: Int
            if let layerID = record.layer {
                guard let layer = layerTable[layerID], record.file == layer.file,
                      record.shape.count == 3, record.shape[2] == 128,
                      ["blk.\(layerID).ffn_gate_exps.weight","blk.\(layerID).ffn_up_exps.weight","blk.\(layerID).ffn_down_exps.weight"].contains(record.name),
                      record.offset % 256 == 0 else { throw ParticleError.invalid("Invalid packed expert tensor") }
                count = bytes / 128
                guard record.offset <= layer.stride, count <= layer.stride-record.offset else { throw ParticleError.invalid("Expert tensor exceeds blob") }
            } else {
                guard record.file == (record.name == "token_embd.weight" ? "embeddings.bin" : "model_weights.bin"),
                      !record.name.contains("_exps.weight"), record.offset % ParticleFormat.alignment == 0,
                      let file = decoded.files[record.file], record.offset <= file.bytes, bytes <= file.bytes-record.offset else { throw ParticleError.invalid("Invalid common tensor range") }
                count = bytes
            }
            intervals[record.file,default:[]].append((record.offset,record.offset+count))
            table[record.name] = GGUFTensor(name:record.name,shape:record.shape,type:type,offset:record.offset,byteCount:record.bytes)
            recordTable[record.name] = record
        }
        for ranges in intervals.values {
            var end = 0
            for range in ranges.sorted(by: { $0.0 < $1.0 }) {
                guard range.0 >= end else { throw ParticleError.invalid("Overlapping pack tensors") }; end = range.1
            }
        }
        for layer in layerTable.values {
            let g = table["blk.\(layer.layer).ffn_gate_exps.weight"], u = table["blk.\(layer.layer).ffn_up_exps.weight"], d = table["blk.\(layer.layer).ffn_down_exps.weight"]
            guard let g, let u, let d, g.shape == u.shape, d.shape == [g.shape[1],g.shape[0],128],
                  g.offset == 0, u.offset == (try ParticleFormat.aligned(g.expertBytes,to:256)),
                  d.offset == (try ParticleFormat.aligned(u.offset+u.expertBytes,to:256)),
                  layer.stride == (try ParticleFormat.aligned(d.offset+d.expertBytes)) else { throw ParticleError.invalid("Expert components do not match slot layout") }
        }
        var meta: [String: GGUFValue] = ["general.architecture":.string(decoded.architecture)]
        for (name,value) in decoded.configuration {
            switch value { case .number(let n): meta[name] = .number(n); case .string(let s): meta[name] = .string(s); case .bool(let b): meta[name] = .bool(b) }
        }
        let vocabulary = try JSONDecoder().decode([String].self,from:index("vocabulary.json",limit:64*1024*1024))
        guard vocabulary.count <= 2_000_000 else { throw ParticleError.invalid("Pack vocabulary exceeds limit") }
        meta["tokenizer.ggml.tokens"] = .strings(vocabulary)
        // Tokenizer is small enough to verify here. Common weights are verified
        // at model admission, so --inspect remains metadata-only.
        for (name,record) in decoded.files where name.hasPrefix("tokenizer/") {
            guard record.bytes <= 64*1024*1024, let fd = owned[name], try ParticleIO.hashFile(descriptor:fd,size:record.bytes) == record.sha256 else { throw ParticleError.invalid("Tokenizer checksum mismatch: \(name)") }
        }
        self.path = resolved; size = total; metadata = meta; tensors = table
        manifest = decoded; layout = expertLayout; root = directory; descriptors = owned
        records = recordTable; layers = layerTable; sourceIdentity = "particle-native-v1|\(manifestDigest)"
        success = true
    }
    deinit { for fd in descriptors.values { close(fd) }; close(root) }
    public func location(for tensor: GGUFTensor, relativeOffset: Int = 0) throws -> ParticleFileRange {
        guard let r = records[tensor.name], tensor.shape == r.shape, tensor.type.rawValue == r.encoding,
              tensor.byteCount == r.bytes, tensor.offset == r.offset, relativeOffset >= 0, relativeOffset <= r.bytes,
              let fd = descriptors[r.file] else { throw ParticleError.invalid("Invalid pack tensor location") }
        if let layerID = r.layer, let layer = layers[layerID] {
            guard relativeOffset < r.bytes else { return ParticleFileRange(descriptor:fd,offset:0,count:0) }
            let bytes = r.bytes/128, expert = relativeOffset/bytes, within = relativeOffset%bytes
            return ParticleFileRange(descriptor:fd,offset:expert*layer.stride+r.offset+within,count:bytes-within)
        }
        return ParticleFileRange(descriptor:fd,offset:r.offset+relativeOffset,count:r.bytes-relativeOffset)
    }
    public func expertRead(layer: Int, expert: Int) throws -> ParticleExpertRead? {
        guard let record = layers[layer], (0..<128).contains(expert), let fd = descriptors[record.file] else { throw ParticleError.invalid("Unknown packed expert") }
        return ParticleExpertRead(range:ParticleFileRange(descriptor:fd,offset:expert*record.stride,count:record.stride),sha256:record.hashes[expert])
    }
    public func prepareForInference() throws {
        for name in ["model_weights.bin","embeddings.bin"] {
            guard let fd = descriptors[name], let expected = manifest.files[name],
                  try ParticleIO.hashFile(descriptor:fd,size:expected.bytes) == expected.sha256 else { throw ParticleError.invalid("Common-weight checksum mismatch: \(name)") }
        }
    }
    public func verifyAll(onFile: ((String) -> Void)? = nil) throws {
        // Verify file and per-expert digests during the same bounded read.
        // Reading every expert file twice needlessly doubles slow-SD traffic.
        let chunk = 4*1024*1024
        let p = UnsafeMutableRawPointer.allocate(byteCount:chunk,alignment:ParticleFormat.alignment)
        defer { p.deallocate() }
        let expertFiles = Dictionary(uniqueKeysWithValues:layout.layers.map { ($0.file,$0) })
        for name in manifest.files.keys.sorted() {
            let record = manifest.files[name]!
            guard let fd = descriptors[name] else { throw ParticleError.invalid("Missing pack descriptor") }
            if let layer = expertFiles[name] {
                var fileHash = SHA256()
                for expert in 0..<128 {
                    var local = 0, expertHash = SHA256()
                    while local < layer.stride {
                        let count = min(chunk,layer.stride-local)
                        try ParticleFileRange(descriptor:fd,offset:expert*layer.stride+local,count:count).read(into:p)
                        let bytes = UnsafeRawBufferPointer(start:p,count:count)
                        fileHash.update(bufferPointer:bytes); expertHash.update(bufferPointer:bytes); local += count
                    }
                    guard expertHash.finalize().map({ String(format:"%02x",$0) }).joined() == layer.hashes[expert] else {
                        throw ParticleError.invalid("Packed expert checksum mismatch: layer \(layer.layer), expert \(expert)")
                    }
                }
                guard fileHash.finalize().map({ String(format:"%02x",$0) }).joined() == record.sha256 else { throw ParticleError.invalid("Pack checksum mismatch: \(name)") }
            } else {
                guard try ParticleIO.hashFile(descriptor:fd,size:record.bytes) == record.sha256 else { throw ParticleError.invalid("Pack checksum mismatch: \(name)") }
            }
            onFile?(name)
        }
    }
}
