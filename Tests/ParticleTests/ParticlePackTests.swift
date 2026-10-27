import Foundation
import Darwin
import CryptoKit
import Testing
import ParticleFormat
import ParticleGGUF
import ParticlePackCore
@testable import ParticleCore

private func packInteger<T: FixedWidthInteger>(_ value: T, _ into: inout Data) {
    var n = value.littleEndian; withUnsafeBytes(of:&n) { into.append(contentsOf:$0) }
}
private func packString(_ value: String, _ into: inout Data) {
    packInteger(UInt64(value.utf8.count),&into); into.append(contentsOf:value.utf8)
}
private struct PackFixture {
    let folder: URL, source: URL, tokenizer: URL
    init() throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("ParticlePackTests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        tokenizer = folder.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at:tokenizer,withIntermediateDirectories:false)
        for name in ["tokenizer.json","tokenizer_config.json","config.json"] { try Data("{}".utf8).write(to:tokenizer.appendingPathComponent(name)) }
        var specs: [(String,[Int],GGMLType,Data)] = [
            ("output_norm.weight",[256],.f32,Data(repeating:23,count:256*4)),
            ("token_embd.weight",[256,16],.q4K,Data(repeating:41,count:144*16))]
        for (name,type) in [("gate",GGMLType.q4K),("up",.q5K),("down",.q6K)] {
            var payload = Data()
            for expert in 0..<128 { payload.append(Data(repeating:UInt8(expert),count:256*type.blockBytes)) }
            specs.append(("blk.1.ffn_\(name)_exps.weight",[256,256,128],type,payload))
        }
        var header = Data(), bytes = Data()
        packInteger(UInt32(0x46554747),&header); packInteger(UInt32(3),&header)
        packInteger(UInt64(specs.count),&header); packInteger(UInt64(0),&header)
        for (name,shape,type,payload) in specs {
            while bytes.count%32 != 0 { bytes.append(0) }
            packString(name,&header); packInteger(UInt32(shape.count),&header)
            for n in shape { packInteger(UInt64(n),&header) }
            packInteger(type.rawValue,&header); packInteger(UInt64(bytes.count),&header); bytes.append(payload)
        }
        while header.count%32 != 0 { header.append(0) }
        header.append(bytes); source = folder.appendingPathComponent("fixture.gguf"); try header.write(to:source)
    }
    func build(_ name: String = "fixture") throws -> URL {
        let out = folder.appendingPathComponent(name+".particle")
        try ParticlePacking.pack(file:GGUFFile(path:source.path),tokenizer:tokenizer,output:out,validateArchitecture:false)
        return out
    }
    func clean() { try? FileManager.default.removeItem(at:folder) }
}
private func reseal(_ folder: URL, _ transform: (inout ParticleManifest) throws -> Void) throws {
    let url = folder.appendingPathComponent("manifest.json")
    var manifest = try JSONDecoder().decode(ParticleManifest.self,from:Data(contentsOf:url))
    try transform(&manifest)
    let bytes = try ParticleFormat.encode(manifest); try bytes.write(to:url)
    try ParticleFormat.encode(ParticleInstallReceipt(manifestSHA256:ParticleIO.hash(bytes))).write(to:folder.appendingPathComponent("verified-install.json"))
}

@Suite struct ParticlePackTests {
    @Test func roundTripMixedNativeQuantsAndIndependentPack() throws {
        let fixture = try PackFixture(); defer { fixture.clean() }
        let original = try GGUFFile(path:fixture.source.path)
        let out = try fixture.build(), pack = try ParticlePackFile(path:out.path)
        #expect(pack.formatName == "particle")
        #expect(throws:(any Error).self) { try SarvamConfiguration.validateTensorGraph(file:pack) }
        #expect(pack.tensors.count == original.tensors.count)
        #expect(pack.layout.layers[0].stride%16_384 == 0)
        #expect(pack.manifest.sourceSHA256 == ParticleIO.hash(try Data(contentsOf:fixture.source)))
        #expect(pack.manifest.approximate == false)
        for tensor in original.tensors.values {
            let other = try pack.tensor(tensor.name)
            #expect(other.shape == tensor.shape); #expect(other.type == tensor.type)
            let count = tensor.shape.count == 3 ? tensor.expertBytes : tensor.byteCount
            for row in tensor.shape.count == 3 ? [0,1,63,127] : [0] {
                var a = Data(count:count), b = Data(count:count)
                try a.withUnsafeMutableBytes { try original.readTensor(tensor,into:$0.baseAddress!,count:count,relativeOffset:row*count) }
                try b.withUnsafeMutableBytes { try pack.readTensor(other,into:$0.baseAddress!,count:count,relativeOffset:row*count) }
                #expect(a == b)
            }
        }
        try pack.verifyAll()
        try FileManager.default.removeItem(at:fixture.source)
        let independent = try ParticleModels.open(path:out.path)
        #expect(independent.formatName == "particle")
        #expect(independent.sourceIdentity == pack.sourceIdentity)
        try independent.prepareForInference()
        #expect(throws:(any Error).self) { try fixture.build() }
    }
    @Test func cancellationResumeAndCorruptCommittedRange() throws {
        enum Interrupted: Error { case stop }
        let fixture = try PackFixture(); defer { fixture.clean() }
        let file = try GGUFFile(path:fixture.source.path), out = fixture.folder.appendingPathComponent("resume.particle")
        #expect(throws:Interrupted.self) {
            try ParticlePacking.pack(file:file,tokenizer:fixture.tokenizer,output:out,validateArchitecture:false,
                chunkBytes:128*1024,checkpointBytes:256*1024,progress: { _,_ in throw Interrupted.stop })
        }
        #expect(!FileManager.default.fileExists(atPath:out.path))
        let partial = URL(fileURLWithPath:out.path+".partial")
        #expect(throws:(any Error).self) { try ParticlePackFile(path:partial.path) }
        let common = partial.appendingPathComponent("model_weights.bin")
        let h = try FileHandle(forUpdating:common)
        let old = try h.read(upToCount:1)!
        try h.seek(toOffset:0); try h.write(contentsOf:Data([old[0]^255])); try h.close()
        #expect(throws:(any Error).self) {
            try ParticlePacking.pack(file:file,tokenizer:fixture.tokenizer,output:out,resume:true,validateArchitecture:false,chunkBytes:128*1024,checkpointBytes:256*1024)
        }
        let repair = try FileHandle(forWritingTo:common); try repair.write(contentsOf:old); try repair.close()
        try ParticlePacking.pack(file:file,tokenizer:fixture.tokenizer,output:out,resume:true,validateArchitecture:false,chunkBytes:128*1024,checkpointBytes:256*1024)
        try ParticlePackFile(path:out.path).verifyAll()
    }
    @Test func malformedIndexesAndPayloadCorruptionFailClosed() throws {
        let fixture = try PackFixture(); defer { fixture.clean() }
        let out = try fixture.build()
        let manifestURL = out.appendingPathComponent("manifest.json"), manifestData = try Data(contentsOf:manifestURL)
        let receiptURL = out.appendingPathComponent("verified-install.json"), receiptData = try Data(contentsOf:receiptURL)
        try reseal(out) { $0.versionMajor = 2 }
        #expect(throws:(any Error).self) { try ParticlePackFile(path:out.path) }
        try manifestData.write(to:manifestURL); try receiptData.write(to:receiptURL)
        let weightsURL = out.appendingPathComponent("weights.json"), weightsData = try Data(contentsOf:weightsURL)
        var records = try JSONDecoder().decode([ParticleTensorRecord].self,from:weightsData)
        records[0].shape = [Int.max,Int.max]
        let malformed = try ParticleFormat.encode(records); try malformed.write(to:weightsURL)
        try reseal(out) { $0.files["weights.json"] = ParticleFileRecord(bytes:malformed.count,sha256:ParticleIO.hash(malformed)) }
        #expect(throws:(any Error).self) { try ParticlePackFile(path:out.path) }
        try weightsData.write(to:weightsURL); try manifestData.write(to:manifestURL); try receiptData.write(to:receiptURL)
        let common = out.appendingPathComponent("model_weights.bin"), h = try FileHandle(forWritingTo:common)
        try h.write(contentsOf:Data([99])); try h.close()
        let pack = try ParticlePackFile(path:out.path)
        #expect(throws:(any Error).self) { try pack.prepareForInference() }
        #expect(throws:(any Error).self) { try pack.verifyAll() }
    }
    @Test func internalSymlinksAndTraversalAreRejected() throws {
        for path in ["../bad","/absolute","packed_experts/../bad","a//b","a\\b"] {
            #expect(throws:(any Error).self) { try ParticleFormat.validatePath(path) }
        }
        let fixture = try PackFixture(); defer { fixture.clean() }
        let out = try fixture.build(), layer = out.appendingPathComponent("packed_experts/layer_01.bin")
        try FileManager.default.removeItem(at:layer)
        try FileManager.default.createSymbolicLink(at:layer,withDestinationURL:fixture.source)
        #expect(throws:(any Error).self) { try ParticlePackFile(path:out.path) }
    }
    @Test func packedExpertCacheUsesOneChecksummedRead() throws {
        let fixture = try PackFixture(); defer { fixture.clean() }
        let out = try fixture.build(), pack = try ParticlePackFile(path:out.path), gpu = try ParticleGPU()
        let cache = try ExpertCache(file:pack,layer:1,count:6,gpu:gpu)
        let slots = try cache.fetch([0,1,2,3,4,5])
        #expect(cache.sourceReadBytes == pack.layout.layers[0].stride*6)
        for (expert,slot) in slots.enumerated() {
            for weight in [slot.gate,slot.up,slot.down] { #expect(slot.buffer.contents().advanced(by:weight.offset).load(as:UInt8.self) == UInt8(expert)) }
        }
        let layer = out.appendingPathComponent("packed_experts/layer_01.bin"), h = try FileHandle(forWritingTo:layer)
        try h.seek(toOffset:UInt64(10*pack.layout.layers[0].stride)); try h.write(contentsOf:Data([255])); try h.close()
        #expect(throws:(any Error).self) { try cache.fetch([6,7,8,9,10,11]) }
    }
    @Test func readThroughRetainsHitsAndHonorsOneGlobalBudget() throws {
        let fixture = try PackFixture(); defer { fixture.clean() }
        let pack = try ParticlePackFile(path:fixture.build().path)
        let folder = fixture.folder.appendingPathComponent("tier"), stride = pack.layout.layers[0].stride
        let obsolete = folder.appendingPathComponent(String(repeating:"a",count:64))
        try FileManager.default.createDirectory(at:obsolete,withIntermediateDirectories:true)
        try JSONSerialization.data(withJSONObject:["version":1,"identity":obsolete.lastPathComponent,"entries":[:]]).write(to:obsolete.appendingPathComponent("index.json"))
        try Data(repeating:9,count:stride).write(to:obsolete.appendingPathComponent("1-0.expert"))
        let p = UnsafeMutableRawPointer.allocate(byteCount:stride,alignment:16_384); defer { p.deallocate() }
        do {
            let tier = try ExpertDiskCache(file:pack,directory:folder,maxBytes:stride*2,policy:.readThrough)
            #expect(!FileManager.default.fileExists(atPath:obsolete.path))
            #expect(throws:(any Error).self) { try ExpertDiskCache(file:pack,directory:folder,maxBytes:stride*2) }
            for expert in [0,1] {
                _ = try pack.loadExpert(layer:1,expert:expert,into:p,capacity:stride)
                tier.admitLoaded(layer:1,expert:expert,from:p,count:stride,protecting:[])
            }
            let written = tier.writeBytes
            #expect(tier.load(layer:1,expert:0,into:p,count:stride))
            #expect(tier.load(layer:1,expert:0,into:p,count:stride))
            #expect(tier.writeBytes == written)
            #expect(tier.storedBytes == stride*2)
            _ = try pack.loadExpert(layer:1,expert:2,into:p,capacity:stride)
            tier.admitLoaded(layer:1,expert:2,from:p,count:stride,protecting:[])
            #expect(tier.storedBytes <= stride*2)
            #expect(tier.load(layer:1,expert:0,into:p,count:stride))
            tier.flush()
        }
        do {
            let reopened = try ExpertDiskCache(file:pack,directory:folder,maxBytes:stride*2,policy:.readThrough)
            #expect(reopened.load(layer:1,expert:0,into:p,count:stride))
            #expect(reopened.writeBytes == 0)
            #expect(reopened.storedBytes <= stride*2)
        }
    }
    @Test func completeMirrorIsResumableVerifiedAndExclusive() throws {
        let fixture = try PackFixture(); defer { fixture.clean() }
        let pack = try ParticlePackFile(path:fixture.build().path), folder = fixture.folder.appendingPathComponent("mirror")
        let required = ParticleMirror.requiredBytes(for:pack), cancellation = ParticleCancellation()
        #expect(throws:(any Error).self) { try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required-1) }
        #expect(throws:CancellationError.self) {
            try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required,cancellation:cancellation) { _,_,_ in cancellation.cancel() }
        }
        #expect(!FileManager.default.fileExists(atPath:folder.appendingPathComponent("mirror-v1").path))
        let oversized = folder.appendingPathComponent("mirror-v1.partial/layer_01.bin")
        let fd = open(oversized.path,O_WRONLY|O_CREAT|O_TRUNC,0o600)
        #expect(fd >= 0)
        #expect(ftruncate(fd,off_t(required*2)) == 0); close(fd)
        cancellation.reset()
        try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required,cancellation:cancellation)
        // Re-preparation verifies an existing publication without duplicating it.
        try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required)
        do {
            let source = try ParticleMirror.open(pack:pack,directory:folder,maxBytes:required)
            try source.prepareForInference()
            #expect(throws:(any Error).self) { try ParticleMirror.open(pack:pack,directory:folder,maxBytes:required) }
            #expect(throws:(any Error).self) { try ExpertDiskCache(file:pack,directory:folder,maxBytes:required) }
            let stride = pack.layout.layers[0].stride
            let p = UnsafeMutableRawPointer.allocate(byteCount:stride,alignment:16_384); defer { p.deallocate() }
            let first = try source.loadExpert(layer:1,expert:0,into:p,capacity:stride)
            #expect(first?.fromSSD == true)
            #expect(source.readBytes == stride)
            let h = try FileHandle(forWritingTo:folder.appendingPathComponent("mirror-v1/layer_01.bin"))
            try h.write(contentsOf:Data([255])); try h.close()
            let fallback = try source.loadExpert(layer:1,expert:0,into:p,capacity:stride)
            #expect(fallback?.fromSSD == false)
            #expect(source.warning != nil)
            #expect(ParticleIO.hash(UnsafeRawBufferPointer(start:p,count:stride)) == pack.layout.layers[0].hashes[0])
        }
        // No reader owns the tier: explicit preparation repairs the corruption.
        try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required)
        let restored = try ParticleMirror.open(pack:pack,directory:folder,maxBytes:required)
        try restored.prepareForInference()
        #expect(restored.warning == nil)
    }

    @Test func pipelinedMirrorVerificationKeepsSlotsPinnedAndRecoversFailures() throws {
        let fixture=try PackFixture(); defer { fixture.clean() }
        let pack=try ParticlePackFile(path:fixture.build().path), folder=fixture.folder.appendingPathComponent("pipeline-mirror")
        let required=ParticleMirror.requiredBytes(for:pack)
        try ParticleMirror.prepare(pack:pack,directory:folder,maxBytes:required)
        let source=try ParticleMirror.open(pack:pack,directory:folder,maxBytes:required,expertReadHint:.noCache)
        var options=ParticleExecutionOptions(); options.pipelinedVerification=true
        let gpu=try ParticleGPU(execution:options), cache=try ExpertCache(file:source,layer:1,count:6,gpu:gpu,execution:options)
        let stride=pack.layout.layers[0].stride
        func check(_ experts: [Int], _ slots: [ExpertSlot]) {
            for (i,expert) in experts.enumerated() {
                #expect(ParticleIO.hash(UnsafeRawBufferPointer(start:slots[i].buffer.contents(),count:stride)) == pack.layout.layers[0].hashes[expert])
            }
        }
        check([0,1,2,3,4,5],try cache.fetch([0,1,2,3,4,5]))
        #expect(source.hits==6)
        let handle=try FileHandle(forUpdating:folder.appendingPathComponent("mirror-v1/layer_01.bin"))
        try handle.seek(toOffset:UInt64(8*stride)); let old=try #require(try handle.read(upToCount:1)?.first)
        try handle.seek(toOffset:UInt64(8*stride)); try handle.write(contentsOf:Data([old^1])); try handle.close()
        check([6,7,8],try cache.fetch([6,7,8]))
        #expect(source.corruptions==1); #expect(cache.sourceReadBytes==stride)
        let cancellation=ParticleCancellation(); cancellation.cancel()
        let cancelled=try cache.beginFetch([20,21,22],cancellation:cancellation)
        #expect(throws:CancellationError.self) { try cache.complete(cancelled) }
        cancellation.reset()
        check([20,21,22],try cache.fetch([20,21,22]))
        // A successful read is still untrusted until the later verifier runs.
        let pointer=UnsafeMutableRawPointer.allocate(byteCount:stride,alignment:16_384); defer { pointer.deallocate() }
        let pending=try source.beginExpertRead(layer:1,expert:30,into:pointer,capacity:stride)
        pointer.assumingMemoryBound(to:UInt8.self)[0] ^= 1
        let recovered=try source.finishExpertRead(pending,layer:1,expert:30,into:pointer,capacity:stride)
        #expect(!recovered.fromSSD)
        #expect(ParticleIO.hash(UnsafeRawBufferPointer(start:pointer,count:stride)) == pack.layout.layers[0].hashes[30])
    }

    @Test func queuedCommandsPreserveDependenciesAndDrainOwnership() throws {
        let gpu = try ParticleGPU(), a = try gpu.buffer(4096), b = try gpu.buffer(4096)
        for i in 0..<4096 { a.contents().assumingMemoryBound(to:Float.self)[i] = Float(i); b.contents().assumingMemoryBound(to:Float.self)[i] = 1 }
        let first = try gpu.command("queued add one"), second = try gpu.command("queued add two")
        try gpu.add(first,a,b); gpu.submit(first)
        try gpu.add(second,a,b); try gpu.finish(second)
        #expect(first.status == .completed && second.status == .completed)
        for i in 0..<4096 { #expect(a.contents().assumingMemoryBound(to:Float.self)[i] == Float(i)+2) }
        try gpu.drain() // An already drained queue is safe.
    }

}
