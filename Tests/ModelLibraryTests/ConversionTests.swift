import Foundation
import Testing
import CryptoKit
@testable import ParticleModelLibrary
import ParticleOneCore

private struct GeneratedSource: ModelByteSource {
    let identity = "fixture-v1"
    let length: Int64
    let failAt: Int64?
    var files: [HubFile] { [.init(path:"checkpoint.bin",bytes:length,sha256:nil)] }
    func read(_ path: String, range: Range<Int64>) async throws -> Data {
        if let failAt,range.lowerBound >= failAt {throw CancellationError()}
        var data = Data(count:range.count)
        for index in 0..<min(256,data.count) where range.lowerBound+Int64(index) < 256 {data[index] = UInt8(truncatingIfNeeded:range.lowerBound+Int64(index))}
        return data
    }
}

@Suite("Lossless bounded conversion") struct ConversionTests {
    private func plan(_ source: GeneratedSource) -> MLXPackPlan {
        let model = ModelDescriptor(id:"fixture",name:"Fixture",architecture:.qwen35MoE,artifact:.mlx,
            provenance:.init(),quantization:.init(scheme:"fixture"),layers:1,experts:2,activeExperts:1,
            contextLimit:128,sourceBytes:source.length,kvBytesPerToken:8,supportsTools:false)
        let files:[String:Int64] = ["common.bin":16,"experts/layer_000.bin":32,"config.json":4,
            "tokenizer/tokenizer.json":4,"tokenizer/tokenizer_config.json":4,"tokenizer/chat_template.jinja":4]
        var position:Int64 = 0
        let ranges = files.sorted{$0.key < $1.key}.map {path,bytes in
            defer{position += bytes}
            return MLXPackCopyRange(source:"checkpoint.bin",offset:position,bytes:bytes,file:path,destination:0)
        }
        return .init(model:model,sourceFiles:source.files,payloadSizes:files,
            tensors:[.init(name:"common.w",dtype:"F32",shape:[4],file:"common.bin",offset:0,bytes:16),
                     .init(name:"expert.w",dtype:"F32",shape:[2,4],file:"experts/layer_000.bin",offset:0,bytes:16,layer:0,expertStride:16)],ranges:ranges)
    }
    @Test func exactBytesAndTamperedExpert() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("particle-converter-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let output = root.appendingPathComponent("fixture.particle"),source = GeneratedSource(length:256,failAt:nil)
        let layout = plan(source)
        try await MLXPackConverter.convert(source:source,plan:layout,output:output,resume:false)
        let reader = try ParticleV2Reader(path:output);try reader.verifyAll()
        // Whole-file checksums alone cannot validate the per-expert manifest.
        let manifestURL=output.appendingPathComponent("manifest.json"),receiptURL=output.appendingPathComponent("verified-install.json")
        let originalManifest=try Data(contentsOf:manifestURL),originalReceipt=try Data(contentsOf:receiptURL)
        var altered=try #require(JSONSerialization.jsonObject(with:originalManifest) as? [String:Any])
        var hashes=try #require(altered["expertHashes"] as? [String:[String]])
        hashes["experts/layer_000.bin"]?[0]=String(repeating:"a",count:64);altered["expertHashes"]=hashes
        let tampered=try JSONSerialization.data(withJSONObject:altered,options:.sortedKeys)
        try tampered.write(to:manifestURL)
        var receipt=try #require(JSONSerialization.jsonObject(with:originalReceipt) as? [String:Any])
        receipt["manifestSHA256"]=SHA256.hash(data:tampered).map{String(format:"%02x",$0)}.joined()
        try JSONSerialization.data(withJSONObject:receipt).write(to:receiptURL)
        #expect(throws:(any Error).self){try ParticleV2Reader(path:output).verifyAll()}
        try originalManifest.write(to:manifestURL);try originalReceipt.write(to:receiptURL)
        // A crash after publication must resume as a verified no-op.
        try await MLXPackConverter.convert(source:source,plan:layout,output:output,resume:true)
        for range in layout.ranges {
            let actual = try Data(contentsOf:output.appendingPathComponent(range.file))
            #expect(actual == source.readFixture(range.offset..<range.offset+range.bytes))
        }
        #expect(try reader.expert(layer:0,expert:0).count == 16)
        let handle = try FileHandle(forWritingTo:output.appendingPathComponent("experts/layer_000.bin"))
        try handle.seek(toOffset:0);try handle.write(contentsOf:Data([255]));try handle.close()
        #expect(throws:(any Error).self){try reader.expert(layer:0,expert:0)}
    }
    @Test func interruptionResumesAndVerifiesCommittedBytes() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("particle-converter-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let output = root.appendingPathComponent("fixture.particle"),length:Int64 = 68*1024*1024
        let interrupted = GeneratedSource(length:length,failAt:64*1024*1024),layout = plan(interrupted)
        do {try await MLXPackConverter.convert(source:interrupted,plan:layout,output:output,resume:false);Issue.record("Expected interruption")}
        catch is CancellationError {} catch {throw error}
        #expect(!FileManager.default.fileExists(atPath:output.path))
        let partial = output.appendingPathExtension("partial")
        #expect(FileManager.default.fileExists(atPath:partial.appendingPathComponent("install-progress.json").path))
        let handle = try FileHandle(forWritingTo:partial.appendingPathComponent("common.bin"));try handle.write(contentsOf:Data([255]));try handle.close()
        let resumed = GeneratedSource(length:length,failAt:nil)
        do {try await MLXPackConverter.convert(source:resumed,plan:layout,output:output,resume:true);Issue.record("Corrupted committed bytes must fail")}
        catch let error as ModelLibraryError {if case .checksum = error {} else {throw error}}
        let range = try #require(layout.ranges.first(where:{$0.file == "common.bin"}))
        let repair = try FileHandle(forWritingTo:partial.appendingPathComponent("common.bin"));try repair.write(contentsOf:sourceBytes(range.offset..<range.offset+range.bytes));try repair.close()
        try await MLXPackConverter.convert(source:resumed,plan:layout,output:output,resume:true)
        try ParticleV2Reader(path:output).verifyAll()
        #expect(!FileManager.default.fileExists(atPath:partial.path))
    }
    @Test func existingConversationsDecodeWithoutModelBindings() throws {
        var workspace = SavedWorkspace();var conversation = Conversation();conversation.messages = [.init(role:"user",content:"Keep me")]
        workspace.conversations = [conversation]
        var json = try #require(JSONSerialization.jsonObject(with:JSONEncoder().encode(workspace)) as? [String:Any])
        var settings = try #require(json["settings"] as? [String:Any]);settings.removeValue(forKey:"modelID");json["settings"] = settings
        let restored = try JSONDecoder().decode(SavedWorkspace.self,from:JSONSerialization.data(withJSONObject:json))
        #expect(restored.settings.modelID == ModelDescriptor.sarvam.id)
        #expect(restored.conversations[0].messages == conversation.messages)
    }
    private func sourceBytes(_ range: Range<Int64>) -> Data {Data(range.map{UInt8(truncatingIfNeeded:$0)})}
}
private extension GeneratedSource {
    func readFixture(_ range: Range<Int64>) -> Data {Data(range.map{UInt8(truncatingIfNeeded:$0)})}
}
