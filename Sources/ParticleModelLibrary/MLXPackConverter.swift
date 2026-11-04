import Foundation
import CryptoKit
import Darwin
import ParticleFormat
import ParticleGGUF

public struct MLXPackCopyRange: Codable, Equatable, Sendable {
    public let source: String
    public let offset: Int64
    public let bytes: Int64
    public let file: String
    public let destination: Int64
}
public struct MLXPackPlan: Codable, Sendable {
    public let model: ModelDescriptor
    public let sourceFiles: [HubFile]
    public let payloadSizes: [String: Int64]
    public let tensors: [ParticleV2Tensor]
    public let ranges: [MLXPackCopyRange]
    public var generatedFiles:[String:Data]? = nil
    public var payloadBytes: Int64 { payloadSizes.values.reduce(0,+) }
}
private struct MLXCopyCheckpoint: Codable {
    struct Chunk: Codable { let source: String; let offset: Int64; let bytes: Int; let sha256: String }
    let identity: String
    let planSHA256: String
    var chunks: [Chunk]
    var sourceGaps:Bool? = nil
}
private struct MLXSourceGap:Sendable {let offset:Int64;let bytes:Int64;let stored:Int64}

public enum MLXPackConverter {
    private static func hash(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    public static func metadata(_ source: any ModelByteSource, path: String, limit: Int) async throws -> Data {
        guard let file = source.files.first(where: {$0.path == path}), file.bytes > 0, file.bytes <= limit else { throw ModelLibraryError.invalid("Missing or oversized metadata: \(path).") }
        var data = Data(); var offset: Int64 = 0
        while offset < file.bytes {
            let end = min(file.bytes,offset+4*1024*1024)
            data.append(try await source.read(path,range:offset..<end)); offset = end
        }
        return data
    }
    public static func plan(source: any ModelByteSource, provenance: ModelProvenance = .init()) async throws -> MLXPackPlan {
        let config = try await metadata(source,path:"config.json",limit:1_048_576)
        if let reason=try ArchitectureRegistry.configurationIncompatibility(config) {throw ModelLibraryError.incompatible(reason)}
        let map:[String:String]
        if source.files.contains(where:{$0.path == "model.safetensors.index.json"}) {
            let index=try await metadata(source,path:"model.safetensors.index.json",limit:16*1024*1024)
            guard let object=try JSONSerialization.jsonObject(with:index) as? [String:Any],let value=object["weight_map"] as? [String:String],!value.isEmpty,value.count <= 100_000,Set(value.values).count <= 128 else {throw ModelLibraryError.invalid("Invalid checkpoint shard index.")}
            map=value
        } else {
            let candidates=source.files.filter{$0.path == "model.safetensors" || ($0.path.hasSuffix(".safetensors") && !$0.path.contains("/"))}
            guard candidates.count == 1,let file=candidates.first else {throw ModelLibraryError.incompatible("A single safetensors file or an indexed checkpoint is required.")}
            let prefix=try await source.read(file.path,range:0..<8)
            let length=prefix.enumerated().reduce(UInt64(0)){$0 | UInt64($1.element) << (8*$1.offset)}
            guard length > 1,length <= SafetensorsHeader.maximumHeaderBytes,length <= UInt64(max(0,file.bytes-8)) else {throw ModelLibraryError.invalid("Invalid single-file tensor header.")}
            let header=try await source.read(file.path,range:8..<(8+Int64(length)))
            let parsed=try SafetensorsHeader(prefix:prefix+header,fileBytes:file.bytes)
            map=Dictionary(uniqueKeysWithValues:parsed.tensors.keys.map{($0,file.path)})
        }
        var records = [String:SafetensorRecord](); var shardFiles = [HubFile]()
        for path in Set(map.values).sorted() {
            try ParticleFormat.validatePath(path)
            guard let file = source.files.first(where: {$0.path == path}), file.bytes > 8, path.hasSuffix(".safetensors") else { throw ModelLibraryError.invalid("Missing safetensors shard.") }
            if provenance.repository != nil,file.sha256 == nil {throw ModelLibraryError.invalid("The remote shard has no source checksum.")}
            let prefix = try await source.read(path,range:0..<8)
            let n = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8*$1.offset) }
            guard n > 1, n <= SafetensorsHeader.maximumHeaderBytes, n <= UInt64(file.bytes-8) else { throw ModelLibraryError.invalid("Invalid safetensors header size.") }
            var header = prefix; var position: Int64 = 8
            while position < Int64(n)+8 {
                let end = min(Int64(n)+8,position+4*1024*1024)
                header.append(try await source.read(path,range:position..<end)); position = end
            }
            let parsed = try SafetensorsHeader(prefix:header,fileBytes:file.bytes)
            for (name, record) in parsed.tensors {
                guard map[name] == path, records[name] == nil else { throw ModelLibraryError.invalid("Shard index mismatch.") }; records[name] = record
            }
            shardFiles.append(file)
        }
        guard Set(records.keys) == Set(map.keys) else { throw ModelLibraryError.invalid("Missing checkpoint tensors.") }
        let template:Data
        var generated=[String:Data]()
        if source.files.contains(where:{$0.path == "chat_template.jinja"}) {template=try await metadata(source,path:"chat_template.jinja",limit:262_144)}
        else {
            let data=try await metadata(source,path:"tokenizer_config.json",limit:1_048_576)
            guard let config=try JSONSerialization.jsonObject(with:data) as? [String:Any],let text=config["chat_template"] as? String,!text.isEmpty,text.utf8.count <= 262_144 else {throw ModelLibraryError.incompatible("No supported chat template was found in the checkpoint.")}
            template=Data(text.utf8);generated["tokenizer/chat_template.jinja"]=template
        }
        try await ArchitectureRegistry.validateChronosTokenizer(tokenizer:metadata(source,path:"tokenizer.json",limit:64*1024*1024),configuration:metadata(source,path:"tokenizer_config.json",limit:1_048_576))
        let total = try shardFiles.reduce(Int64(0)) { value,file in
            let (sum,overflow) = value.addingReportingOverflow(file.bytes); guard !overflow else { throw ModelLibraryError.invalid("Checkpoint size overflow.") }; return sum
        }
        let report = try ArchitectureRegistry.inspectMLX(configuration:config,tensors:records,template:String(decoding:template,as:UTF8.self),
            repository:provenance.repository,revision:provenance.revision,sourceBytes:total,license:provenance.license,sourceIdentity:source.identity)
        guard report.conversionSupported,let model = report.descriptor else { throw ModelLibraryError.incompatible(report.reason) }
        var ranges = [MLXPackCopyRange](); var tensors = [ParticleV2Tensor](); var sizes:[String:Int64] = ["common.bin":0]
        func align(_ n:Int64,to:Int = 256) throws -> Int64 { Int64(try ParticleFormat.aligned(Int(n),to:to)) }
        func routedLayer(_ name:String) -> Int? {MoEModelSchema.routedLayer(name)}
        for record in records.values.sorted(by: {$0.name < $1.name}) where routedLayer(record.name) == nil {
            let offset = try align(sizes["common.bin"]!)
            sizes["common.bin"] = offset+record.bytes
            tensors.append(.init(name:record.name,dtype:record.dtype,shape:record.shape,file:"common.bin",offset:offset,bytes:record.bytes))
            ranges.append(.init(source:map[record.name]!,offset:record.offset,bytes:record.bytes,file:"common.bin",destination:offset))
        }
        for layer in 0..<model.layers {
            let group = records.values.filter { routedLayer($0.name) == layer }.sorted { $0.name < $1.name }
            guard group.count == 9 else { throw ModelLibraryError.invalid("Each layer requires three complete quantized expert projections.") }
            var stride: Int64 = 0; var offsets = [String:Int64]()
            for record in group {
                guard record.shape.first == model.experts, record.bytes % Int64(model.experts) == 0 else { throw ModelLibraryError.invalid("Expert tensor cannot be split losslessly.") }
                stride = try align(stride); offsets[record.name] = stride; stride += record.bytes/Int64(model.experts)
            }
            stride = try align(stride,to:16_384)
            guard stride <= 32*1024*1024 else { throw ModelLibraryError.invalid("Expert blob exceeds the bounded runtime capacity.") }
            let path = String(format:"experts/layer_%03d.bin",layer)
            sizes[path] = stride*Int64(model.experts)
            for record in group {
                let bytes = record.bytes/Int64(model.experts), offset = offsets[record.name]!
                tensors.append(.init(name:record.name,dtype:record.dtype,shape:record.shape,file:path,offset:offset,bytes:bytes,layer:layer,expertStride:stride))
                for expert in 0..<model.experts {
                    ranges.append(.init(source:map[record.name]!,offset:record.offset+Int64(expert)*bytes,bytes:bytes,file:path,destination:Int64(expert)*stride+offset))
                }
            }
        }
        var sources = shardFiles
        let copied = ["config.json":"config.json", "model.safetensors.index.json":"source-index.json", "tokenizer.json":"tokenizer/tokenizer.json", "tokenizer_config.json":"tokenizer/tokenizer_config.json",
                      "chat_template.jinja":"tokenizer/chat_template.jinja", "generation_config.json":"tokenizer/generation_config.json",
                      "README.md":"MODEL_CARD.md", "LICENSE":"MODEL_LICENSE", "LICENSE.txt":"MODEL_LICENSE.txt"]
        for (name,destination) in copied.sorted(by: {$0.key < $1.key}) {
            guard let file = source.files.first(where: {$0.path == name}) else {
                guard !["config.json","tokenizer.json","tokenizer_config.json"].contains(name) else { throw ModelLibraryError.invalid("Missing model metadata: \(name).") }; continue
            }
            guard file.bytes > 0,file.bytes <= 64*1024*1024 else { throw ModelLibraryError.invalid("Model metadata exceeds its bound.") }
            sources.append(file); sizes[destination] = file.bytes
            ranges.append(.init(source:name,offset:0,bytes:file.bytes,file:destination,destination:0))
        }
        for (path,data) in generated {sizes[path]=Int64(data.count)}
        return MLXPackPlan(model:model,sourceFiles:sources.sorted {$0.path < $1.path},payloadSizes:sizes,tensors:tensors.sorted {$0.name < $1.name},
                           ranges:ranges.sorted { ($0.source,$0.offset) < ($1.source,$1.offset) },generatedFiles:generated.isEmpty ? nil : generated)
    }

    /// Full-source sequential verification, bounded scatter writes and durable resumability.
    public static func convert(source: any ModelByteSource, plan: MLXPackPlan, output: URL, resume: Bool, nativeManifest:Data? = nil,
                               progress: @Sendable (Int64,Int64,String) -> Void = {_,_,_ in}) async throws {
        let existingManifest=try nativeManifest.map{try JSONDecoder().decode(ParticleV2Manifest.self,from:$0)}
        if let existingManifest {
            try existingManifest.validate()
            guard existingManifest.model == plan.model,existingManifest.tensors == plan.tensors,
                  existingManifest.files.mapValues(\.bytes) == plan.payloadSizes else {throw ModelLibraryError.invalid("The native download layout changed.")}
        }
        let lease = try ModelOperationLease();defer{withExtendedLifetime(lease){}}
        let memoryPressure = ModelMemoryPressure()
        try source.verifyIdentity()
        let fm = FileManager.default
        guard output.pathExtension == "particle" else {throw ModelLibraryError.invalid("Choose a .particle destination.")}
        if fm.fileExists(atPath:output.path) {
            guard resume else {throw ModelLibraryError.invalid("Choose a new .particle destination.")}
            let reader = try ParticleV2Reader(path:output)
            guard reader.manifest.model == plan.model,reader.manifest.tensors == plan.tensors,
                  (existingManifest.map{$0 == reader.manifest} ?? (Set(reader.manifest.sourceFiles.keys) == Set(plan.sourceFiles.map(\.path)))) else {throw ModelLibraryError.invalid("The existing model is a different conversion. Choose a new destination.")}
            if let nativeManifest {
                guard try ArchitectureRegistry.boundedData(output.appendingPathComponent("manifest.json"),limit:16*1024*1024) == nativeManifest else {throw ModelLibraryError.invalid("The completed destination has a different native manifest.")}
            }
            try reader.verifyAll();try source.verifyIdentity()
            progress(plan.model.sourceBytes,plan.model.sourceBytes,"Verified existing completed installation")
            return
        }
        let partial = output.appendingPathExtension("partial")
        let existed = fm.fileExists(atPath:partial.path)
        guard !existed || resume else { throw ModelLibraryError.invalid("A partial conversion exists; resume it or choose another destination.") }
        let free = try output.deletingLastPathComponent().resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage ?? 0
        guard existed || free >= plan.payloadBytes+1_073_741_824 else { throw ModelLibraryError.diskSpace(plan.payloadBytes+1_073_741_824) }
        if !existed {
            try fm.createDirectory(at:partial,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
            for dir in ["experts","tokenizer"] { try fm.createDirectory(at:partial.appendingPathComponent(dir),withIntermediateDirectories:false) }
        }
        let root = open(partial.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard root >= 0 else { throw ModelLibraryError.invalid("Unsafe partial model folder.") }; defer {close(root)}
        let lock = openat(root,"pack.lock",O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard lock >= 0 else { throw ModelLibraryError.invalid("Cannot open conversion lease.") }; defer {close(lock)}
        guard flock(lock,LOCK_EX|LOCK_NB) == 0 else { throw ModelLibraryError.invalid("This model is already being converted.") }; defer {flock(lock,LOCK_UN)}
        let planHash = hash(try ParticleFormat.encode(plan)+(nativeManifest ?? Data())), checkpointURL = partial.appendingPathComponent("install-progress.json")
        var checkpoint = MLXCopyCheckpoint(identity:source.identity,planSHA256:planHash,chunks:[])
        if existed {
            checkpoint = try JSONDecoder().decode(MLXCopyCheckpoint.self,from:ArchitectureRegistry.boundedData(checkpointURL,limit:16*1024*1024))
            guard checkpoint.identity == source.identity,checkpoint.planSHA256 == planHash else { throw ModelLibraryError.invalid("The source or conversion layout changed. This partial pack cannot be resumed.") }
        }
        if !existed {checkpoint.sourceGaps=true}
        let gapsEnabled=checkpoint.sourceGaps == true
        var gaps=[String:[MLXSourceGap]](),gapBytes:Int64=0
        if gapsEnabled {
            for file in plan.sourceFiles {
                var position:Int64=0
                func appendGap(until end:Int64) throws {
                    guard end >= position else {throw ModelLibraryError.invalid("Overlapping source ranges.")}
                    if end > position {
                        let bytes=end-position
                        guard bytes <= 256*1024*1024-gapBytes else {throw ModelLibraryError.invalid("Source padding exceeds the bounded resume archive.")}
                        gaps[file.path,default:[]].append(.init(offset:position,bytes:bytes,stored:gapBytes));gapBytes += bytes
                    }
                }
                for range in plan.ranges.filter({$0.source == file.path}).sorted(by:{$0.offset < $1.offset}) {
                    guard range.offset >= 0,range.offset <= file.bytes,range.bytes >= 0,range.bytes <= file.bytes-range.offset else {throw ModelLibraryError.invalid("Invalid source range.")}
                    try appendGap(until:range.offset);position=range.offset+range.bytes
                }
                try appendGap(until:file.bytes)
            }
        }
        let gapsName="_ingest_gaps.bin"
        var gapFD:Int32 = -1
        if gapsEnabled {
            if existed {gapFD=try ParticleIO.openRelative(root:root,path:gapsName,flags:O_RDWR)}
            else {gapFD=openat(root,gapsName,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)}
            guard gapFD >= 0 else {throw ModelLibraryError.invalid("Cannot open the source resume archive.")}
        }
        defer{if gapFD >= 0{close(gapFD)}}
        if gapsEnabled {
            if !existed {guard ftruncate(gapFD,gapBytes) == 0 else {throw ModelLibraryError.invalid("Cannot allocate source resume metadata.")}}
            var st=stat();guard fstat(gapFD,&st) == 0,st.st_size == gapBytes else {throw ModelLibraryError.invalid("Source resume metadata changed.")}
        }
        func write(_ fd:Int32,_ offset:Int64,_ data:Data) throws {
            try data.withUnsafeBytes {buffer in
                var written=0
                while written < data.count {
                    let count=pwrite(fd,buffer.baseAddress!.advanced(by:written),data.count-written,offset+Int64(written))
                    if count < 0 && errno == EINTR {continue}
                    guard count > 0 else {throw ModelLibraryError.invalid("Model write failed; check free space.")};written += count
                }
            }
        }
        var outputs = [String:Int32](); defer { for fd in outputs.values {close(fd)} }
        for (path,size) in plan.payloadSizes {
            let fd: Int32
            if existed {fd = try ParticleIO.openRelative(root:root,path:path,flags:O_RDWR)}
            else {
                let parts = path.split(separator:"/")
                let parent = parts.count == 1 ? dup(root) : openat(root,String(parts[0]),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
                guard parent >= 0,parts.count <= 2 else {throw ModelLibraryError.invalid("Unsafe output folder.")};defer{close(parent)}
                fd = openat(parent,String(parts.last!),O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
                guard fd >= 0 else {throw ModelLibraryError.invalid("Cannot create model payload.")}
            }
            outputs[path] = fd
            if !existed { guard ftruncate(fd,size) == 0 else { throw ModelLibraryError.invalid("Could not allocate pack payload.") } }
            var st = stat(); guard fstat(fd,&st) == 0,st.st_size == size else {throw ModelLibraryError.invalid("Partial pack size changed.")}
        }
        if existed {
            let allocated = outputs.values.reduce(Int64(0)){sum,fd in var st = stat();guard fstat(fd,&st) == 0 else{return sum};return sum+st.st_blocks*512}
            let available = try partial.resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage ?? 0
            let required = max(0,plan.payloadBytes-allocated)+1_073_741_824
            guard available >= required else {throw ModelLibraryError.diskSpace(required)}
        }
        func commit() throws {
            if gapsEnabled {guard fsync(gapFD) == 0 else {throw ModelLibraryError.invalid("Cannot synchronize source resume metadata.")}}
            for fd in outputs.values {guard fsync(fd) == 0 else {throw ModelLibraryError.invalid("Could not synchronize model payload.")}}
            try ParticleFormat.encode(checkpoint).write(to:checkpointURL,options:.atomic)
            let fd = try ParticleIO.openRelative(root:root,path:"install-progress.json");defer{close(fd)}
            guard fsync(fd) == 0,fsync(root) == 0 else {throw ModelLibraryError.invalid("Could not synchronize conversion progress.")}
        }
        if !existed {
            for directory in ["experts","tokenizer"] {
                let fd = openat(root,directory,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);guard fd >= 0 else {throw ModelLibraryError.invalid("Cannot synchronize output directory.")};defer{close(fd)}
                guard fsync(fd) == 0 else {throw ModelLibraryError.invalid("Cannot synchronize output directory.")}
            }
            try commit()
        }
        var committedIndex = 0; var done: Int64 = 0; var sinceCommit: Int64 = 0
        let total = plan.sourceFiles.reduce(Int64(0)) {$0+$1.bytes}
        var verifiedSources = [String:ParticleV2File]()
        for file in plan.sourceFiles {
            var digest = SHA256(); var offset: Int64 = 0
            var gitDigest = Insecure.SHA1();gitDigest.update(data:Data("blob \(file.bytes)\0".utf8))
            let fileRanges = plan.ranges.filter {$0.source == file.path}
            while offset < file.bytes {
                try Task.checkCancellation()
                try memoryPressure.check()
                let committed = committedIndex < checkpoint.chunks.count
                // Respect older 4 MiB checkpoints while new transfers use bounded 16 MiB reads.
                let count = committed ? checkpoint.chunks[committedIndex].bytes : Int(min(Int64(16*1024*1024),file.bytes-offset))
                guard count > 0,count <= 16*1024*1024,Int64(count) <= file.bytes-offset else {throw ModelLibraryError.invalid("Invalid conversion checkpoint range.")}
                let data:Data
                if committed && gapsEnabled {
                    // Reconstruct the original source chunk from verified local
                    // payload ranges and saved headers/padding, without network
                    // or SD-card prefix reads. The original full-file digest
                    // is still computed and verified below.
                    var restored=Data(count:count)
                    try restored.withUnsafeMutableBytes {buffer in
                        for range in fileRanges where range.offset < offset+Int64(count) && range.offset+range.bytes > offset {
                            let start=max(range.offset,offset),end=min(range.offset+range.bytes,offset+Int64(count))
                            try ParticleFileRange(descriptor:outputs[range.file]!,offset:Int(range.destination+start-range.offset),count:Int(end-start)).read(into:buffer.baseAddress!.advanced(by:Int(start-offset)))
                        }
                        for gap in gaps[file.path] ?? [] where gap.offset < offset+Int64(count) && gap.offset+gap.bytes > offset {
                            let start=max(gap.offset,offset),end=min(gap.offset+gap.bytes,offset+Int64(count))
                            try ParticleFileRange(descriptor:gapFD,offset:Int(gap.stored+start-gap.offset),count:Int(end-start)).read(into:buffer.baseAddress!.advanced(by:Int(start-offset)))
                        }
                    }
                    data=restored
                } else {data=try await source.read(file.path,range:offset..<(offset+Int64(count)))}
                guard data.count == count else {throw ModelLibraryError.invalid("Truncated model read.")}
                let digestChunk = hash(data)
                if committed {
                    let c = checkpoint.chunks[committedIndex]
                    guard c.source == file.path,c.offset == offset,c.bytes == count,c.sha256 == digestChunk else {throw ModelLibraryError.checksum(file.path)}
                }
                for range in fileRanges where !(committed && gapsEnabled) && range.offset < offset+Int64(count) && range.offset+range.bytes > offset {
                    let start = max(range.offset,offset),end = min(range.offset+range.bytes,offset+Int64(count))
                    let piece = data.subdata(in:Int(start-offset)..<Int(end-offset)),destination = range.destination+start-range.offset
                    let fd = outputs[range.file]!
                    if committed {
                        var existing = Data(count:piece.count)
                        try existing.withUnsafeMutableBytes {try ParticleFileRange(descriptor:fd,offset:Int(destination),count:piece.count).read(into:$0.baseAddress!)}
                        guard existing == piece else {throw ModelLibraryError.checksum(range.file)}
                    } else {
                        try write(fd,destination,piece)
                    }
                }
                if !committed && gapsEnabled {
                    for gap in gaps[file.path] ?? [] where gap.offset < offset+Int64(count) && gap.offset+gap.bytes > offset {
                        let start=max(gap.offset,offset),end=min(gap.offset+gap.bytes,offset+Int64(count))
                        try write(gapFD,gap.stored+start-gap.offset,data.subdata(in:Int(start-offset)..<Int(end-offset)))
                    }
                }
                if !committed {checkpoint.chunks.append(.init(source:file.path,offset:offset,bytes:count,sha256:digestChunk))}
                committedIndex += 1;digest.update(data:data);offset += Int64(count);done += Int64(count);sinceCommit += Int64(count)
                if file.gitBlobSHA1 != nil {gitDigest.update(data:data)}
                progress(done,total,file.path)
                if sinceCommit >= 64*1024*1024 {try commit();sinceCommit = 0}
            }
            let sha = digest.finalize().map{String(format:"%02x",$0)}.joined()
            if let expected = file.sha256 {guard expected == sha else {throw ModelLibraryError.checksum(file.path)}}
            if let expected = file.gitBlobSHA1 {guard expected == gitDigest.finalize().map({String(format:"%02x",$0)}).joined() else {throw ModelLibraryError.checksum(file.path)}}
            verifiedSources[file.path] = .init(bytes:file.bytes,sha256:sha);try commit()
        }
        guard committedIndex == checkpoint.chunks.count else {throw ModelLibraryError.invalid("Unexpected conversion checkpoint suffix.")}
        for (path,data) in plan.generatedFiles ?? [:] {
            guard path == "tokenizer/chat_template.jinja",data.count <= 262_144,Int64(data.count) == plan.payloadSizes[path],let fd=outputs[path] else {throw ModelLibraryError.invalid("Invalid derived chat template.")}
            try write(fd,0,data);guard fsync(fd) == 0 else {throw ModelLibraryError.invalid("Could not synchronize the derived template.")}
        }
        var files = [String:ParticleV2File]();var expertHashes = [String:[String]]()
        for path in plan.payloadSizes.keys.sorted() {
            try Task.checkCancellation();progress(total,total,"Verifying "+path)
            let fd = outputs[path]!,size = plan.payloadSizes[path]!
            if let stride = plan.tensors.first(where: {$0.file == path && $0.layer != nil})?.expertStride {
                var digest = SHA256();var hashes = [String]()
                for expert in 0..<plan.model.experts {
                    try Task.checkCancellation()
                    var data = Data(count:Int(stride))
                    try data.withUnsafeMutableBytes {try ParticleFileRange(descriptor:fd,offset:expert*Int(stride),count:Int(stride)).read(into:$0.baseAddress!)}
                    digest.update(data:data);hashes.append(hash(data))
                }
                files[path] = .init(bytes:size,sha256:digest.finalize().map{String(format:"%02x",$0)}.joined());expertHashes[path] = hashes
            } else {files[path] = .init(bytes:size,sha256:try ParticleIO.hashFile(descriptor:fd,size:Int(size)))}
        }
        try source.verifyIdentity()
        let manifest = ParticleV2Manifest(model:plan.model,files:files,tensors:plan.tensors,sourceFiles:verifiedSources,expertHashes:expertHashes)
        try manifest.validate()
        let data:Data
        if let existingManifest,let nativeManifest {
            guard existingManifest.files == files,existingManifest.expertHashes == expertHashes else {throw ModelLibraryError.checksum("Native pack payload or expert hashes")}
            data=nativeManifest
        } else {data=try ParticleFormat.encode(manifest)}
        try data.write(to:partial.appendingPathComponent("manifest.json"),options:.atomic)
        try ParticleFormat.encode(ParticleInstallReceipt(manifestSHA256:hash(data))).write(to:partial.appendingPathComponent("verified-install.json"),options:.atomic)
        for name in ["manifest.json","verified-install.json"] {let fd = try ParticleIO.openRelative(root:root,path:name);defer{close(fd)};guard fsync(fd) == 0 else {throw ModelLibraryError.invalid("Cannot finalize model receipt.")}}
        // Keep the durable checkpoint until publication. If interrupted before
        // rename, the complete partial pack is still resumable.
        guard fsync(root) == 0 else {throw ModelLibraryError.invalid("Cannot finalize model directory.")}
        try fm.moveItem(at:partial,to:output)
        let parent = open(output.deletingLastPathComponent().path,O_RDONLY|O_DIRECTORY|O_CLOEXEC)
        guard parent >= 0 else {throw ModelLibraryError.invalid("Cannot finalize the model installation directory.")};defer{close(parent)}
        guard fsync(parent) == 0 else {throw ModelLibraryError.invalid("Cannot synchronize the published model installation.")}
        try fm.removeItem(at:output.appendingPathComponent("install-progress.json"))
        if gapsEnabled {try fm.removeItem(at:output.appendingPathComponent(gapsName))}
        guard fsync(root) == 0 else {throw ModelLibraryError.invalid("Cannot synchronize the completed installation.")}
    }
}
