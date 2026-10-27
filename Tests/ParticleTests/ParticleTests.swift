import Foundation
import Testing
import ParticleGGUF
@testable import ParticleCore

private struct FixtureTensor {
    let name: String, shape: [Int], type: GGMLType, data: Data
}
private func integer<T: FixedWidthInteger>(_ value: T, into data: inout Data) {
    var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf:$0) }
}
private func string(_ value: String, into data: inout Data) {
    integer(UInt64(value.utf8.count),into:&data); data.append(contentsOf:value.utf8)
}
private func fixture(_ tensors: [FixtureTensor]) throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ParticleTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
    var header = Data(), payload = Data()
    integer(UInt32(0x46554747),into:&header); integer(UInt32(3),into:&header)
    integer(UInt64(tensors.count),into:&header); integer(UInt64(0),into:&header)
    for tensor in tensors {
        while payload.count % 32 != 0 { payload.append(0) }
        string(tensor.name,into:&header); integer(UInt32(tensor.shape.count),into:&header)
        for n in tensor.shape { integer(UInt64(n),into:&header) }
        integer(tensor.type.rawValue,into:&header); integer(UInt64(payload.count),into:&header)
        payload.append(tensor.data)
    }
    while header.count % 32 != 0 { header.append(0) }
    header.append(payload)
    let url = folder.appendingPathComponent("fixture.gguf"); try header.write(to:url); return url
}
private func clean(_ url: URL) { try? FileManager.default.removeItem(at:url.deletingLastPathComponent()) }

@Suite struct ParticleTests {
    @Test func ggufBoundsAndTruncation() throws {
        let url = try fixture([FixtureTensor(name:"test.weight",shape:[256,2],type:.q4K,data:Data(repeating:17,count:288))])
        defer { clean(url) }
        let file = try GGUFFile(path:url.path)
        #expect(file.tensors.count == 1)
        #expect(try file.tensor("test.weight").byteCount == 288)
        #expect(throws:ParticleError.self) { try file.tensor("test.weight",shape:[256,3]) }
        let handle = try FileHandle(forWritingTo:url); try handle.truncate(atOffset:UInt64(file.size-1)); try handle.close()
        #expect(throws:ParticleError.self) { try GGUFFile(path:url.path) }
    }
    @Test func ggufRejectsBadMagicAndDuplicateTensors() throws {
        let url = try fixture([FixtureTensor(name:"a",shape:[256],type:.q4K,data:Data(count:144)),FixtureTensor(name:"a",shape:[256],type:.q4K,data:Data(count:144))])
        defer { clean(url) }
        #expect(throws:ParticleError.self) { try GGUFFile(path:url.path) }
        var data = try Data(contentsOf:url); data[0] = 0; try data.write(to:url)
        #expect(throws:ParticleError.self) { try GGUFFile(path:url.path) }
    }
    @Test func quantReferenceKnownBlocks() {
        for type in [GGMLType.q4K,.q5K] {
            var bytes = [UInt8](repeating:0,count:type.blockBytes)
            bytes[1] = 0x3C; bytes[3] = 0x3C // d = dmin = 1
            for i in 0..<4 { bytes[4+i] = 1; bytes[8+i] = 2; bytes[12+i] = 0x21 }
            if type == .q5K { for i in 16..<48 { bytes[i] = 255 } }
            for i in (type == .q4K ? 16 : 48)..<type.blockBytes { bytes[i] = 255 }
            let result = QuantReference.decode(bytes,type:type,count:256)
            #expect(result.allSatisfy { $0 == (type == .q4K ? 13 : 29) })
        }
        var bytes = [UInt8](repeating:0,count:210); bytes[209] = 0x3C
        for i in 192..<208 { bytes[i] = 1 }
        #expect(QuantReference.decode(bytes,type:.q6K,count:256).allSatisfy { $0 == -32 })
    }
    @Test func kernelParity() throws {
        let reports = try ParticleValidation.kernels()
        #expect(reports.count >= 7)
    }
    @Test func routerBiasOnlyAffectsSelection() throws {
        let logits = (0..<128).map { -Float($0)/16 }
        var bias = [Float](repeating:0,count:128); bias[127] = 100
        let result = try SarvamRouting.select(logits:logits,bias:bias)
        #expect(result.experts.first == 127)
        #expect(abs(result.weights.reduce(0,+)-2.5) < 1e-5)
        #expect(result.weights[0] < result.weights[1])
        var bad = logits; bad[0] = .nan
        #expect(throws:ParticleError.self) { try SarvamRouting.select(logits:bad,bias:bias) }
        let extreme = try SarvamRouting.select(logits:[Float](repeating:1000,count:128),bias:[Float](repeating:0,count:128))
        #expect(extreme.experts == [0,1,2,3,4,5])
        #expect(extreme.weights.allSatisfy { $0.isFinite })
    }
    @Test func samplingAndStreaming() throws {
        var a = ParticleSampler(seed:42), b = ParticleSampler(seed:42)
        let logits: [Float] = [-2,3,1,0]
        #expect(try a.sample(logits,temperature:0,topK:4,topP:1) == 1)
        for _ in 0..<20 {
            #expect(try a.sample(logits,temperature:0.8,topK:3,topP:0.95) == b.sample(logits,temperature:0.8,topK:3,topP:0.95))
        }
        var decoder = SarvamTextDecoder()
        #expect(decoder.append("<0xE2>") == "")
        #expect(decoder.append("<0x82>") == "")
        #expect(decoder.append("<0xAC>") == "")
        #expect(decoder.append("▁ok") == "€ ok")
        #expect(decoder.append("<0xFF>") == "")
        #expect(decoder.finish() == "\u{FFFD}")
        #expect(Array(decoder.append("▁्").utf8) == Array(" ्".utf8))
    }
    @Test func expertGatherAndCacheReplacement() throws {
        let expertBytes = 256*256/256*144
        var payload = Data()
        for expert in 0..<128 { payload.append(Data(repeating:UInt8(expert),count:expertBytes)) }
        let url = try fixture([
            FixtureTensor(name:"blk.1.ffn_gate_exps.weight",shape:[256,256,128],type:.q4K,data:payload),
            FixtureTensor(name:"blk.1.ffn_up_exps.weight",shape:[256,256,128],type:.q4K,data:payload),
            FixtureTensor(name:"blk.1.ffn_down_exps.weight",shape:[256,256,128],type:.q4K,data:payload)])
        defer { clean(url) }
        let file = try GGUFFile(path:url.path), gpu = try ParticleGPU(), cache = try ExpertCache(file:file,layer:1,count:6,gpu:gpu)
        for (ids,hits,misses) in [([0,1,2,3,4,5],0,6),([0,1,2,6,7,8],3,9),([8,7,6,2,1,0],9,9)] {
            let slots = try cache.fetch(ids)
            #expect(cache.hitCount == hits); #expect(cache.missCount == misses)
            for (i,slot) in slots.enumerated() {
                for w in [slot.gate,slot.up,slot.down] {
                    let p = slot.buffer.contents().advanced(by:w.offset).assumingMemoryBound(to:UInt8.self)
                    #expect(p[0] == UInt8(ids[i])); #expect(p[expertBytes-1] == UInt8(ids[i]))
                }
            }
        }
    }
    @Test func diskExpertCacheRoundTripBudgetAndCorruption() throws {
        let bytes = 256*256/256*144
        var payload = Data()
        for expert in 0..<128 { payload.append(Data(repeating:UInt8(expert),count:bytes)) }
        let url = try fixture(["gate","up","down"].map { FixtureTensor(name:"blk.1.ffn_\($0)_exps.weight",shape:[256,256,128],type:.q4K,data:payload) })
        defer { clean(url) }
        let file = try GGUFFile(path:url.path), folder = url.deletingLastPathComponent().appendingPathComponent("ssd")
        let gpu = try ParticleGPU(), budget = bytes*3*12
        func populate() throws {
            let disk = try ExpertDiskCache(file:file,directory:folder,maxBytes:budget)
            #expect(throws:ParticleError.self) { try ExpertDiskCache(file:file,directory:folder,maxBytes:budget) }
            let cache = try ExpertCache(file:file,layer:1,count:6,gpu:gpu,diskCache:disk)
            _ = try cache.fetch([0,1,2,3,4,5]); #expect(disk.storedBytes == 0)
            _ = try cache.fetch([6,7,8,9,10,11]); #expect(disk.storedBytes == bytes*3*6)
            let slots = try cache.fetch([0,1,2,3,4,5])
            #expect(cache.sourceReadBytes == budget)
            #expect(disk.hits == 6); #expect(disk.readBytes == bytes*3*6)
            for (i,slot) in slots.enumerated() { #expect(slot.buffer.contents().load(as:UInt8.self) == UInt8(i)) }
            #expect(disk.storedBytes == bytes*3*6); cache.persist(); disk.flush()
            let index=folder.appendingPathComponent("kquant-v1/index.json")
            let committedIndex=try Data(contentsOf:index)
            #expect(throws:ParticleError.self) { try ExpertDiskCache(file:file,directory:folder,maxBytes:budget) }
            #expect(try Data(contentsOf:index)==committedIndex)
        }
        try populate()
        func reopen() throws {
            let disk = try ExpertDiskCache(file:file,directory:folder,maxBytes:budget)
            let cache = try ExpertCache(file:file,layer:1,count:6,gpu:gpu,diskCache:disk)
            _ = try cache.fetch([0,1,2,3,4,5])
            #expect(cache.sourceReadBytes == 0); #expect(disk.hits == 6)
            cache.persist()
        }
        try reopen()
        let namespace = folder.appendingPathComponent("kquant-v1")
        // Correct owned key, same length, modified content: SHA must catch it.
        let entry = namespace.appendingPathComponent("1-0.expert")
        var data = try Data(contentsOf:entry); data[0] ^= 255; try data.write(to:entry)
        do {
            let disk = try ExpertDiskCache(file:file,directory:folder,maxBytes:budget)
            let cache = try ExpertCache(file:file,layer:1,count:6,gpu:gpu,diskCache:disk)
            let slots = try cache.fetch([0,1,2,3,4,5])
            #expect(disk.corruptions == 1); #expect(cache.sourceReadBytes == bytes*3)
            #expect(slots[0].buffer.contents().load(as:UInt8.self) == 0)
            _ = try cache.fetch([12,13,14,15,16,17]); #expect(disk.storedBytes <= budget)
        }
        do {
            let disk = try ExpertDiskCache(file:file,directory:folder.appendingPathComponent("exclusive-test"),maxBytes:bytes*3*6)
            let cache = try ExpertCache(file:file,layer:1,count:6,gpu:gpu,diskCache:disk)
            _ = try cache.fetch([0,1,2,3,4,5]); _ = try cache.fetch([6,7,8,9,10,11])
            let initial = cache.sourceReadBytes
            for _ in 0..<3 {
                _ = try cache.fetch([0,1,2,3,4,5]); _ = try cache.fetch([6,7,8,9,10,11])
            }
            #expect(cache.sourceReadBytes == initial)
            #expect(disk.storedBytes <= bytes*3*6)
        }
        // A changed source must never reuse a previous cache namespace.
        let h = try FileHandle(forWritingTo:url); try h.seek(toOffset:UInt64(file.size-1)); try h.write(contentsOf:Data([99])); try h.close()
        let changed = try GGUFFile(path:url.path)
        let disk = try ExpertDiskCache(file:changed,directory:folder,maxBytes:budget)
        #expect(disk.storedBytes == 0)
    }
    @Test func asyncExpertReservationsPinReadySlotsAndCoalescingIsExact() throws {
        let bytes=256*256/256*144
        var payload=Data()
        for expert in 0..<128 { payload.append(Data(repeating:UInt8(expert),count:bytes)) }
        let url=try fixture(["gate","up","down"].map { FixtureTensor(name:"blk.1.ffn_\($0)_exps.weight",shape:[256,256,128],type:.q4K,data:payload) })
        defer { clean(url) }
        let file=try GGUFFile(path:url.path), gpu=try ParticleGPU()
        for workers in [1,2,4] { for coalesce in [false,true] {
            var options=ParticleExecutionOptions(); options.sourceWorkers=workers; options.coalesceReads=coalesce
            let cache=try ExpertCache(file:file,layer:1,count:6,gpu:gpu,execution:options)
            _=try cache.fetch([0,1,2,3,4,5])
            let transaction=try cache.beginFetch([0,1,6,7,8,9])
            #expect(transaction.ready==[0,1])
            #expect(throws:ParticleError.self) { try cache.beginFetch([0,1,2,3,4,5]) }
            for rank in transaction.ready { #expect(transaction.slots[rank].buffer.contents().load(as:UInt8.self)==UInt8(rank)) }
            let slots=try cache.complete(transaction)
            _=try cache.complete(transaction) // Idempotent drain for error paths.
            for (rank,expert) in [0,1,6,7,8,9].enumerated() {
                for w in [slots[rank].gate,slots[rank].up,slots[rank].down] {
                    let p=slots[rank].buffer.contents().advanced(by:w.offset).assumingMemoryBound(to:UInt8.self)
                    #expect(p[0]==UInt8(expert)); #expect(p[bytes-1]==UInt8(expert))
                }
            }
            #expect(cache.sourceReadBytes==bytes*3*10)
        } }
    }
    @Test func cancelledExpertReadInvalidatesOnlyReservedMisses() throws {
        let bytes=256*256/256*144
        var payload=Data()
        for expert in 0..<128 { payload.append(Data(repeating:UInt8(expert),count:bytes)) }
        let url=try fixture(["gate","up","down"].map { FixtureTensor(name:"blk.1.ffn_\($0)_exps.weight",shape:[256,256,128],type:.q4K,data:payload) })
        defer { clean(url) }
        let file=try GGUFFile(path:url.path), gpu=try ParticleGPU(), cache=try ExpertCache(file:file,layer:1,count:6,gpu:gpu)
        _=try cache.fetch([0,1,2,3,4,5])
        let cancellation=ParticleCancellation(); cancellation.cancel()
        let fetch=try cache.beginFetch([0,1,6,7,8,9],cancellation:cancellation)
        #expect(throws:CancellationError.self) { try cache.complete(fetch) }
        #expect(cache.residentExperts==[0,1])
        let retained=try cache.fetch([0,1])
        #expect(retained[0].buffer.contents().load(as:UInt8.self)==0)
        #expect(retained[1].buffer.contents().load(as:UInt8.self)==1)
        // A short source read must also release reservations, without making
        // incomplete expert bytes appear as a valid future hit.
        let writer=try FileHandle(forWritingTo:url); try writer.truncate(atOffset:0); try writer.close()
        let failed=try cache.beginFetch([0,1,6,7,8,9])
        #expect(throws:ParticleError.self) { try cache.complete(failed) }
        #expect(cache.residentExperts==[0,1])
        #expect(try cache.fetch([0,1]).count==2)
    }
    @Test func tokenizerMatchesRustGoldenFixtures() async throws {
        let folder = URL(fileURLWithPath:ProcessInfo.processInfo.environment["PARTICLE_TOKENIZER_DIR"] ?? "/Volumes/Particle/Models/sarvam-30b/tokenizer")
        guard FileManager.default.fileExists(atPath:folder.appendingPathComponent("tokenizer.json").path) else {
            print("SKIP: tokenizer parity requires the official tokenizer sidecars"); return
        }
        let model = ProcessInfo.processInfo.environment["PARTICLE_MODEL_PATH"] ?? "/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf"
        let file = FileManager.default.fileExists(atPath:model) ? try GGUFFile(path:model) : nil
        let tokenizer = try await SarvamTokenizer.load(folder:folder,vocabulary:file?.metadata["tokenizer.ggml.tokens"]?.strings)
        let url = try #require(Bundle.module.url(forResource:"tokenizer-golden",withExtension:"json",subdirectory:"Fixtures"))
        let fixtures = try #require(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [[String:Any]])
        for fixture in fixtures {
            let text = try #require(fixture["text"] as? String), ids = try #require(fixture["ids"] as? [Int])
            #expect(tokenizer.encode(text) == ids)
            #expect(try tokenizer.decode(ids) == fixture["decoded"] as? String)
        }
        let chat = try tokenizer.chat(prompt:"Hi",thinking:false)
        let history = [SarvamMessage(role:.user,content:"Hi"),SarvamMessage(role:.assistant,content:"<think>private reasoning</think>\nHello."),SarvamMessage(role:.user,content:"Next")]
        let rendered = "[@BOS@]\n<|start_of_turn|><|user|>\nHi<|nothink|><|end_of_turn|>\n<|start_of_turn|><|assistant|>\n<think></think>\nHello.<|end_of_turn|>\n<|start_of_turn|><|user|>\nNext<|nothink|><|end_of_turn|>\n<|start_of_turn|><|assistant|>\n"
        #expect(try tokenizer.chat(messages:history) == tokenizer.encode(rendered))
        // Independent Python Jinja + Rust tokenizers rendering of the pinned template.
        #expect(chat == [6,107,25,18,107,10979,28,26,107,25,19,107])
    }
    @Test func groupedExpertPlanReusesWeightsAndPreservesRanks() throws {
        let ids = [[5,1,8,2,9,3],[8,5,2,4,6,7],[7,1,5,2,3,4]]
        let routes = ids.map { RoutingResult(experts:$0,weights:[0.9,0.6,0.4,0.3,0.2,0.1]) }
        let plan = try ExpertBatchPlan(routes:routes,cached:[8,9],capacity:6)
        let reads = plan.batches.flatMap { $0 }
        #expect(Set(reads) == Set(ids.flatMap { $0 }))
        #expect(reads.count == 9) // 18 expert uses need just nine loads.
        #expect(Set(reads.prefix(2)) == [8,9])
        #expect(plan.batches.allSatisfy { $0.count <= 6 })
        var visited = Set<String>()
        for (expert,uses) in plan.uses {
            for use in uses {
                #expect(ids[use.row][use.rank] == expert)
                #expect(visited.insert("\(use.row):\(use.rank)").inserted)
            }
        }
        #expect(visited.count == 18)
        for capacity in [6,8,12,16] {
            let plan = try ExpertBatchPlan(routes:Array(repeating:routes[0],count:128),cached:[],capacity:capacity)
            #expect(plan.batches.flatMap { $0 }.count == 6)
            #expect(plan.uses.values.allSatisfy { $0.count == 128 })
        }
        #expect(throws:ParticleError.self) { try ExpertBatchPlan(routes:routes,cached:[],capacity:5) }
        let invalid = RoutingResult(experts:[0,0,1,2,3,4],weights:[Float](repeating:1,count:6))
        #expect(throws:ParticleError.self) { try ExpertBatchPlan(routes:[invalid],cached:[],capacity:6) }
    }
    @Test func embeddingRowsUseSeparateBoundedBuffers() throws {
        let columns = 4096, rowBytes = columns/256*144
        var bytes = [UInt8](repeating:0,count:3*rowBytes)
        for row in 0..<3 {
            for block in 0..<(columns/256) {
                let base = row*rowBytes+block*144
                bytes[base+1] = 0x3C // d=1, dmin=0
                for i in 4..<16 { bytes[base+i] = 1 }
                for i in 16..<144 { bytes[base+i] = UInt8((row+1)*17) }
            }
        }
        let url = try fixture([FixtureTensor(name:"token_embd.weight",shape:[columns,3],type:.q4K,data:Data(bytes))])
        defer { clean(url) }
        let reader = try EmbeddingReader(file:GGUFFile(path:url.path)), gpu = try ParticleGPU()
        let row0 = try gpu.bytes(reader.rowBytes), row2 = try gpu.bytes(reader.rowBytes)
        let a = try reader.load(token:0,into:row0), b = try reader.load(token:2,into:row2)
        #expect(row0.length == 2304); #expect(row2.length == 2304)
        let out0 = try gpu.buffer(columns), out2 = try gpu.buffer(columns), cb = try gpu.command()
        try gpu.embedding(cb,a,token:0,out:out0); try gpu.embedding(cb,b,token:0,out:out2)
        try gpu.finish(cb)
        for i in 0..<columns {
            #expect(out0.contents().assumingMemoryBound(to:Float.self)[i] == 1)
            #expect(out2.contents().assumingMemoryBound(to:Float.self)[i] == 3)
        }
        #expect(throws:ParticleError.self) { try reader.load(token:3,into:row0) }
        #expect(throws:ParticleError.self) { try reader.load(token:0,into:gpu.bytes(32)) }
    }
    @Test func variableExpertBatchesRetainFrequentEntries() throws {
        let size = 256*256/256*144
        var payload = Data()
        for i in 0..<128 { payload.append(Data(repeating:UInt8(i),count:size)) }
        let url = try fixture(["gate","up","down"].map {
            FixtureTensor(name:"blk.1.ffn_\($0)_exps.weight",shape:[256,256,128],type:.q4K,data:payload)
        })
        defer { clean(url) }
        let cache = try ExpertCache(file:GGUFFile(path:url.path),layer:1,count:6,gpu:ParticleGPU())
        _ = try cache.fetch([0,1,2,3,4,5],frequencies:[16,1,1,1,1,1])
        _ = try cache.fetch([6])
        #expect(cache.residentExperts.contains(0))
        #expect(!cache.residentExperts.contains(1))
        let slots = try cache.fetch([0,7],frequencies:[4,2])
        #expect(cache.hitCount == 1)
        #expect(cache.missCount == 8)
        #expect(cache.readBytes == 8*size*3)
        #expect(slots[1].buffer.contents().load(as:UInt8.self) == 7)
        #expect(throws:ParticleError.self) { try cache.fetch([8,9],frequencies:[1]) }
        #expect(throws:ParticleError.self) { try cache.fetch([8],frequencies:[0]) }
        #expect(throws:ParticleError.self) { try cache.fetch([0,1,2,3,4,5,6]) }
        #expect(cache.missCount == 8)
    }
}
