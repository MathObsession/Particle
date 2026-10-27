import Foundation
import Darwin
import CryptoKit
import ParticleFormat
import ParticleGGUF

public struct ParticleCopyRange: Codable, Equatable, Sendable {
    public let source: Int, destination: Int, bytes: Int
    public let file: String
}
public struct ParticlePackPlan: Codable, Sendable {
    public let sourceBytes: Int
    public let tensors: [ParticleTensorRecord]
    public let layout: ParticleExpertLayout
    public let fileSizes: [String: Int]
    public let ranges: [ParticleCopyRange]
    public var payloadBytes: Int { fileSizes.values.reduce(0,+) }
}
private struct CopyChunk: Codable { let offset: Int, bytes: Int, sha256: String }
private struct Checkpoint: Codable {
    let sourceIdentity: String, planSHA256: String
    var chunks: [CopyChunk]
    var position: Int { chunks.last.map { $0.offset+$0.bytes } ?? 0 }
}
public enum ParticlePacking {
    public static func plan(file: GGUFFile, validateArchitecture: Bool = true) throws -> ParticlePackPlan {
        if validateArchitecture { _ = try SarvamConfiguration(file:file); try SarvamConfiguration.validateTensorGraph(file:file) }
        var records: [ParticleTensorRecord] = [], ranges: [ParticleCopyRange] = []
        var sizes = ["model_weights.bin":0,"embeddings.bin":0]
        var layouts: [ParticleLayerRecord] = []
        let ordered = file.tensors.values.sorted { $0.offset < $1.offset }
        for tensor in ordered where !tensor.name.contains("_exps.weight") {
            let name = tensor.name == "token_embd.weight" ? "embeddings.bin" : "model_weights.bin"
            let offset = try ParticleFormat.aligned(sizes[name]!)
            guard tensor.byteCount <= Int.max-offset else { throw ParticleError.invalid("Pack payload overflow") }
            sizes[name] = try ParticleFormat.aligned(offset+tensor.byteCount)
            records.append(ParticleTensorRecord(name:tensor.name,file:name,shape:tensor.shape,encoding:tensor.type.rawValue,offset:offset,bytes:tensor.byteCount))
            ranges.append(ParticleCopyRange(source:tensor.offset,destination:offset,bytes:tensor.byteCount,file:name))
        }
        let expertTensors = ordered.filter { $0.name.contains("_exps.weight") }
        let layerIDs = Set(expertTensors.compactMap { t -> Int? in
            let p = t.name.split(separator:"."); return p.count == 4 && p[0] == "blk" ? Int(p[1]) : nil
        })
        guard expertTensors.count == layerIDs.count*3, layerIDs.allSatisfy({ (1...18).contains($0) }) else { throw ParticleError.invalid("Invalid expert tensor names") }
        for layer in layerIDs.sorted() {
            let g = try file.tensor("blk.\(layer).ffn_gate_exps.weight"), u = try file.tensor("blk.\(layer).ffn_up_exps.weight"), d = try file.tensor("blk.\(layer).ffn_down_exps.weight")
            guard g.shape.count == 3, g.shape[2] == 128, u.shape == g.shape, d.shape == [g.shape[1],g.shape[0],128] else { throw ParticleError.invalid("Invalid expert matrix dimensions") }
            let up = try ParticleFormat.aligned(g.expertBytes,to:256)
            let down = try ParticleFormat.aligned(up+u.expertBytes,to:256)
            let stride = try ParticleFormat.aligned(down+d.expertBytes)
            guard stride <= 32*1024*1024 else { throw ParticleError.invalid("Expert exceeds bounded blob size") }
            let name = String(format:"packed_experts/layer_%02d.bin",layer)
            sizes[name] = stride*128
            layouts.append(ParticleLayerRecord(layer:layer,stride:stride,file:name))
            for (t,offset) in [(g,0),(u,up),(d,down)] {
                records.append(ParticleTensorRecord(name:t.name,file:name,shape:t.shape,encoding:t.type.rawValue,offset:offset,bytes:t.byteCount,layer:layer))
                for expert in 0..<128 { ranges.append(ParticleCopyRange(source:t.offset+expert*t.expertBytes,destination:expert*stride+offset,bytes:t.expertBytes,file:name)) }
            }
        }
        return ParticlePackPlan(sourceBytes:file.size,tensors:records.sorted { $0.name < $1.name },layout:ParticleExpertLayout(layers:layouts),fileSizes:sizes,ranges:ranges.sorted { $0.source < $1.source })
    }
    private static func write(_ pointer: UnsafeRawPointer, count: Int, descriptor: Int32, offset: Int) throws {
        var done = 0
        while done < count {
            let n = pwrite(descriptor,pointer.advanced(by:done),count-done,off_t(offset+done))
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { throw ParticleError.invalid("Pack write failed; check free space and mounted storage") }; done += n
        }
    }
    private static func durableJSON<T: Encodable>(_ value: T, at url: URL) throws {
        let data = try ParticleFormat.encode(value)
        try data.write(to:url,options:.atomic)
        let fd = open(url.path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW)
        guard fd >= 0 else { throw ParticleError.invalid("Cannot sync pack checkpoint") }
        defer { close(fd) }
        guard fsync(fd) == 0 else { throw ParticleError.invalid("Cannot make pack checkpoint durable") }
        let parent = open(url.deletingLastPathComponent().path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard parent >= 0 else { throw ParticleError.invalid("Cannot sync pack directory") }
        defer { close(parent) }
        guard fsync(parent) == 0 else { throw ParticleError.invalid("Cannot sync pack directory") }
    }
    /// Source-order reads, bounded pwrite scattering, durable checkpoints and
    /// bit-preserving native packing. Progress may throw to cancel safely.
    public static func pack(file: GGUFFile, tokenizer: URL, output: URL, resume: Bool = false,
                            validateArchitecture: Bool = true, chunkBytes: Int = 4*1024*1024,
                            checkpointBytes: Int = 64*1024*1024,
                            progress: ((Int,Int) throws -> Void)? = nil,
                            status: ((String) -> Void)? = nil) throws {
        guard chunkBytes > 0, chunkBytes <= 4*1024*1024, checkpointBytes >= chunkBytes else { throw ParticleError.invalid("Invalid bounded-copy configuration") }
        let plan = try plan(file:file,validateArchitecture:validateArchitecture)
        let fm = FileManager.default
        let target = output.standardizedFileURL
        guard target.pathExtension == "particle", !fm.fileExists(atPath:target.path) else { throw ParticleError.invalid("Output must be a new .particle directory") }
        let partial = URL(fileURLWithPath:target.path+".partial",isDirectory:true)
        let existed = fm.fileExists(atPath:partial.path)
        guard !existed || resume else { throw ParticleError.invalid("Partial pack exists; pass --resume") }
        if !existed {
            let attrs = try fm.attributesOfFileSystem(forPath:target.deletingLastPathComponent().path)
            let free = (attrs[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
            guard free >= Int64(plan.payloadBytes)+2*1024*1024*1024 else { throw ParticleError.invalid("Pack needs \(plan.payloadBytes) bytes plus 2 GiB free headroom") }
            try fm.createDirectory(at:partial,withIntermediateDirectories:false)
            try fm.createDirectory(at:partial.appendingPathComponent("packed_experts"),withIntermediateDirectories:false)
            try fm.createDirectory(at:partial.appendingPathComponent("tokenizer"),withIntermediateDirectories:false)
        }
        let root = open(partial.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard root >= 0 else { throw ParticleError.invalid("Invalid partial pack directory") }
        defer { close(root) }
        let lock = openat(root,"pack.lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0o600)
        guard lock >= 0 else { throw ParticleError.invalid("Cannot open pack lease") }
        defer { close(lock) }
        guard flock(lock,LOCK_EX|LOCK_NB) == 0 else { throw ParticleError.invalid("Pack conversion is already running") }
        defer { flock(lock,LOCK_UN) }
        let sourceIdentity = file.sourceIdentity
        let planHash = ParticleIO.hash(try ParticleFormat.encode(plan))
        let journalURL = partial.appendingPathComponent("install-progress.json")
        var checkpoint: Checkpoint
        if existed {
            checkpoint = try JSONDecoder().decode(Checkpoint.self,from:ParticleIO.data(root:root,path:"install-progress.json",limit:16*1024*1024))
            guard checkpoint.sourceIdentity == sourceIdentity, checkpoint.planSHA256 == planHash else { throw ParticleError.invalid("Source model or copy plan changed; partial pack cannot resume") }
            var end = 0
            for c in checkpoint.chunks {
                guard c.offset == end, c.bytes > 0, c.bytes <= chunkBytes, c.offset <= file.size, c.bytes <= file.size-c.offset,
                      ParticleFormat.digestIsValid(c.sha256) else { throw ParticleError.invalid("Invalid copy checkpoint") }; end += c.bytes
            }
        } else {
            checkpoint = Checkpoint(sourceIdentity:sourceIdentity,planSHA256:planHash,chunks:[])
            try durableJSON(checkpoint,at:journalURL)
        }
        var outputs: [String:Int32] = [:]
        defer { for fd in outputs.values { close(fd) } }
        for (name,bytes) in plan.fileSizes {
            let fd: Int32
            if existed { fd = try ParticleIO.openRelative(root:root,path:name,flags:O_RDWR) }
            else {
                // Parent directories were just created and are re-opened with
                // no-follow semantics before any payload file is created.
                let parentName = name.contains("/") ? "packed_experts" : nil
                let parent = parentName.map { openat(root,$0,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW) } ?? dup(root)
                guard parent >= 0 else { throw ParticleError.invalid("Cannot open output directory") }
                fd = openat(parent,URL(fileURLWithPath:name).lastPathComponent,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0o600)
                close(parent)
            }
            guard fd >= 0 else { throw ParticleError.invalid("Cannot create pack payload") }; outputs[name] = fd
            if !existed {
                var allocation = fstore_t()
                allocation.fst_flags = UInt32(F_ALLOCATECONTIG); allocation.fst_posmode = F_PEOFPOSMODE
                allocation.fst_offset = 0; allocation.fst_length = off_t(bytes)
                if bytes > 0 && fcntl(fd,F_PREALLOCATE,&allocation) != 0 {
                    allocation.fst_flags = UInt32(F_ALLOCATEALL)
                    if fcntl(fd,F_PREALLOCATE,&allocation) != 0 { status?("Preallocation unavailable for \(name); using ordinary allocation") }
                }
                guard ftruncate(fd,off_t(bytes)) == 0 else { throw ParticleError.invalid("Cannot size pack payload") }
            } else {
                var st = stat(); guard fstat(fd,&st) == 0, st.st_size == bytes else { throw ParticleError.invalid("Partial payload size changed") }
            }
        }
        // Make newly created directory entries durable before a checkpoint
        // can claim that their payload ranges exist after power loss.
        for name in ["packed_experts", "tokenizer"] {
            let fd = openat(root,name,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
            guard fd >= 0 else { throw ParticleError.invalid("Cannot open payload directory") }
            let result = fsync(fd); close(fd)
            guard result == 0 else { throw ParticleError.invalid("Cannot sync payload directory") }
        }
        guard fsync(root) == 0 else { throw ParticleError.invalid("Cannot sync partial pack directory") }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount:chunkBytes,alignment:ParticleFormat.alignment)
        let verify = UnsafeMutableRawPointer.allocate(byteCount:chunkBytes,alignment:ParticleFormat.alignment)
        defer { buffer.deallocate(); verify.deallocate() }
        var hash = SHA256(), nextRange = 0
        func overlaps(_ start: Int, _ count: Int, _ action: (ParticleCopyRange,Int,Int) throws -> Void) throws {
            while nextRange < plan.ranges.count && plan.ranges[nextRange].source+plan.ranges[nextRange].bytes <= start { nextRange += 1 }
            var index = nextRange
            while index < plan.ranges.count && plan.ranges[index].source < start+count {
                let r = plan.ranges[index], a = max(start,r.source), b = min(start+count,r.source+r.bytes)
                if b > a { try action(r,a,b-a) }; index += 1
            }
        }
        // Restore the streaming source digest and verify every durable copy
        // against the matching source range before trusting resume progress.
        for c in checkpoint.chunks {
            try file.read(into:buffer,count:c.bytes,offset:c.offset)
            guard ParticleIO.hash(UnsafeRawBufferPointer(start:buffer,count:c.bytes)) == c.sha256 else { throw ParticleError.invalid("Source bytes changed since checkpoint") }
            hash.update(bufferPointer:UnsafeRawBufferPointer(start:buffer,count:c.bytes))
            try overlaps(c.offset,c.bytes) { r,a,n in
                try ParticleFileRange(descriptor:outputs[r.file]!,offset:r.destination+a-r.source,count:n).read(into:verify)
                guard memcmp(verify,buffer.advanced(by:a-c.offset),n) == 0 else { throw ParticleError.invalid("Committed partial payload is corrupt: \(r.file)") }
            }
        }
        var offset = checkpoint.position, lastCommit = offset
        var pending: [CopyChunk] = []
        status?("Copying \(file.size) source bytes into \(plan.payloadBytes) packed bytes; resume at \(offset)")
        while offset < file.size {
            let count = min(chunkBytes,file.size-offset)
            try file.read(into:buffer,count:count,offset:offset)
            hash.update(bufferPointer:UnsafeRawBufferPointer(start:buffer,count:count))
            try overlaps(offset,count) { r,a,n in try write(buffer.advanced(by:a-offset),count:n,descriptor:outputs[r.file]!,offset:r.destination+a-r.source) }
            pending.append(CopyChunk(offset:offset,bytes:count,sha256:ParticleIO.hash(UnsafeRawBufferPointer(start:buffer,count:count))))
            offset += count
            if offset-lastCommit >= checkpointBytes || offset == file.size {
                for fd in outputs.values { guard fsync(fd) == 0 else { throw ParticleError.invalid("Cannot sync partial model payload") } }
                checkpoint.chunks.append(contentsOf:pending); pending.removeAll(keepingCapacity:true)
                try durableJSON(checkpoint,at:journalURL); lastCommit = offset
                try progress?(offset,file.size)
            }
        }
        guard file.sourceIdentity == sourceIdentity else { throw ParticleError.invalid("Source model changed during conversion") }
        let sourceDigest = hash.finalize().map { String(format:"%02x",$0) }.joined()
        var config: [String:ParticleScalar] = [:]
        for (key,value) in file.metadata where key == "general.architecture" || key.hasPrefix("sarvam_moe.") {
            switch value { case .number(let n): config[key] = .number(n); case .string(let s): config[key] = .string(s); case .bool(let b): config[key] = .bool(b); default: break }
        }
        let tokenRoot = open(tokenizer.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard tokenRoot >= 0 else { throw ParticleError.invalid("Tokenizer directory is missing") }; defer { close(tokenRoot) }
        let tokenizerNames = ["tokenizer.json","tokenizer_config.json","config.json","special_tokens_map.json","chat_template.jinja","chat_template.json","generation_config.json"]
        var files: [String:ParticleFileRecord] = [:]
        for (i,name) in tokenizerNames.enumerated() {
            let candidate = tokenizer.appendingPathComponent(name)
            if i >= 3 && !fm.fileExists(atPath:candidate.path) { continue }
            let data = try ParticleIO.data(root:tokenRoot,path:name,limit:64*1024*1024)
            // Sidecars are bounded independently of the 8 MiB payload scratch.
            try data.write(to:partial.appendingPathComponent("tokenizer/"+name),options:.atomic)
            files["tokenizer/"+name] = ParticleFileRecord(bytes:data.count,sha256:ParticleIO.hash(data))
        }
        let vocabulary = file.metadata["tokenizer.ggml.tokens"]?.strings ?? []
        let vocabularyData = try ParticleFormat.encode(vocabulary)
        guard vocabularyData.count <= 64*1024*1024 else { throw ParticleError.invalid("Vocabulary exceeds limit") }
        try vocabularyData.write(to:partial.appendingPathComponent("vocabulary.json"),options:.atomic)
        files["vocabulary.json"] = ParticleFileRecord(bytes:vocabularyData.count,sha256:ParticleIO.hash(vocabularyData))
        status?("Verifying packed payload and expert checksums")
        var layout = plan.layout
        for (name,bytes) in plan.fileSizes.sorted(by: { $0.key < $1.key }) {
            let fd = outputs[name]!
            if let index = layout.layers.firstIndex(where: { $0.file == name }) {
                let stride = layout.layers[index].stride
                // Never allocate a complete tensor or a model: one expert is
                // hashed in bounded source-buffer-sized pieces.
                var fileHash = SHA256(), hashes: [String] = []
                for expert in 0..<128 {
                    var expertHash = SHA256(), local = 0
                    while local < stride {
                        let n = min(chunkBytes,stride-local)
                        try ParticleFileRange(descriptor:fd,offset:expert*stride+local,count:n).read(into:buffer)
                        let p = UnsafeRawBufferPointer(start:buffer,count:n)
                        expertHash.update(bufferPointer:p); fileHash.update(bufferPointer:p); local += n
                    }
                    hashes.append(expertHash.finalize().map { String(format:"%02x",$0) }.joined())
                }
                layout.layers[index].hashes = hashes
                files[name] = ParticleFileRecord(bytes:bytes,sha256:fileHash.finalize().map { String(format:"%02x",$0) }.joined())
            } else { files[name] = ParticleFileRecord(bytes:bytes,sha256:try ParticleIO.hashFile(descriptor:fd,size:bytes,chunk:chunkBytes)) }
        }
        for (name,data) in [("weights.json",try ParticleFormat.encode(plan.tensors)),("packed_experts/layout.json",try ParticleFormat.encode(layout))] {
            guard data.count <= ParticleFormat.indexLimit else { throw ParticleError.invalid("Pack index exceeds limit") }
            try data.write(to:partial.appendingPathComponent(name),options:.atomic)
            files[name] = ParticleFileRecord(bytes:data.count,sha256:ParticleIO.hash(data))
        }
        let manifest = ParticleManifest(sourceSHA256:sourceDigest,sourceBytes:file.size,configuration:config,files:files)
        try manifest.validate()
        try durableJSON(manifest,at:partial.appendingPathComponent("manifest.json"))
        try durableJSON(ParticleInstallReceipt(manifestSHA256:ParticleIO.hash(try ParticleFormat.encode(manifest))),at:partial.appendingPathComponent("verified-install.json"))
        // Sync every indexed sidecar/index before publication, not just the
        // weight files and receipt. The directory rename is the commit point.
        for name in files.keys {
            let fd = try ParticleIO.openRelative(root:root,path:name)
            let result = fsync(fd); close(fd)
            guard result == 0 else { throw ParticleError.invalid("Cannot sync pack file: \(name)") }
        }
        for name in ["packed_experts", "tokenizer"] {
            let fd = openat(root,name,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
            guard fd >= 0 else { throw ParticleError.invalid("Cannot open final index directory") }
            let result = fsync(fd); close(fd)
            guard result == 0 else { throw ParticleError.invalid("Cannot sync final index directory") }
        }
        let verified = try ParticlePackFile(path:partial.path)
        guard verified.tensors.count == file.tensors.count else { throw ParticleError.invalid("Pack lost source tensors") }
        try fm.moveItem(at:partial,to:target)
        let parent = open(target.deletingLastPathComponent().path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard parent >= 0 else { throw ParticleError.invalid("Cannot sync completed pack parent") }
        defer { close(parent) }
        guard fsync(parent) == 0 else { throw ParticleError.invalid("Cannot sync completed pack publication") }
        status?("Verified native .particle pack: \(target.path)")
    }
}
