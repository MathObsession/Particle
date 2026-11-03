import Foundation
import Testing
import CryptoKit
@testable import ParticleModelLibrary
import ParticleOneCore

@Suite("Model library compatibility and persistence") struct ModelLibraryTests {
    @Test func publicRepositoryValidation() throws {
        #expect(try HubRepository.parse("https://huggingface.co/owner/model") == "owner/model")
        for input in ["../model","owner/../model","https://example.org/owner/model","https://huggingface.co/owner/model?token=secret","https://user@huggingface.co/owner/model","owner/model/resolve/main"] {
            #expect(throws: (any Error).self) { try HubRepository.parse(input) }
        }
        #expect(!PublicHubHTTP.allowed(URL(string:"http://huggingface.co/x")!))
        #expect(!PublicHubHTTP.allowed(URL(string:"https://huggingface.co.attacker.example/x")!))
        #expect(PublicHubHTTP.allowed(URL(string:"https://cas-bridge.xethub.hf.co/x")!))
    }
    @Test func checkpointIdentityIsRevisionSpecific() {
        let a = ModelDescriptor.identity(repository:"owner/model",revision:String(repeating:"a",count:40),artifact:"q4")
        let b = ModelDescriptor.identity(repository:"owner/model",revision:String(repeating:"b",count:40),artifact:"q4")
        #expect(a != b); #expect(a.count == 64)
    }
    @Test func respectsBothRetryAfterForms() {
        let now=Date(timeIntervalSince1970:0)
        #expect(BoundedHubRequest.retryDate("7200",now:now).timeIntervalSince(now) == 7200)
        #expect(BoundedHubRequest.retryDate("Thu, 01 Jan 1970 02:00:00 GMT",now:now).timeIntervalSince(now) == 7200)
        #expect(BoundedHubRequest.retryDate("nan",now:now).timeIntervalSince(now) == 300)
    }
    @Test func qualificationCannotComeFromModelMetadata() {
        let adapter = ArchitectureRegistry.adapter(for:.qwen35MoE)
        #expect(adapter.qualification == .experimental)
        #expect(!adapter.supportsTools)
        #expect(ArchitectureRegistry.report(.sarvam).installable)
    }
    @Test func recommendationsAreReadyToInstallRatherThanUnqualifiedNames() {
        let pending=ModelDescriptor(id:"chronos.fixture",name:"Chronos",architecture:.qwen35MoE,artifact:.mlx,
            provenance:.init(repository:ModelRecommendations.chronosRepository),quantization:.init(scheme:"affine",bits:3),
            layers:40,experts:256,activeExperts:8,contextLimit:262144,sourceBytes:15_000_000_000,kvBytesPerToken:40960,supportsTools:false)
        #expect(ModelRecommendations.reason(for:pending) != nil)
        #expect(ModelRecommendations.eligible([pending,.sarvam]) == [.sarvam])
        #expect(ModelRecommendations.reason(for:.sarvam)?.contains("Start here") == true)
    }
    @Test func validatesSafetensorsBoundsAndTypes() throws {
        func header(_ json:String, payload:Int) -> Data {
            var n = UInt64(json.utf8.count).littleEndian
            return withUnsafeBytes(of:&n){Data($0)}+Data(json.utf8)+Data(repeating:0,count:payload)
        }
        let valid = header("{\"w\":{\"dtype\":\"U32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}",payload:24)
        let parsed = try SafetensorsHeader(prefix:valid,fileBytes:Int64(valid.count))
        #expect(parsed.tensors["w"]?.shape == [2,3]);#expect(parsed.tensors["w"]?.bytes == 24)
        for json in ["{\"w\":{\"dtype\":\"U32\",\"shape\":[2,3],\"data_offsets\":[0,25]}}",
                     "{\"w\":{\"dtype\":\"U32\",\"shape\":[true],\"data_offsets\":[0,4]}}",
                     "{\"w\":{\"dtype\":\"U32\",\"shape\":[2,3],\"data_offsets\":[1,25]}}"] {
            let invalid = header(json,payload:25)
            #expect(throws: (any Error).self) { try SafetensorsHeader(prefix:invalid,fileBytes:Int64(invalid.count)) }
        }
        #expect(throws: (any Error).self) { try SafetensorsHeader(prefix:Data(repeating:255,count:8),fileBytes:16) }
    }
    @Test func registryRoundTripAndRemovalBoundary() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("particle-library-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let path = root.appendingPathComponent("Models",isDirectory:true)
        try FileManager.default.createDirectory(at:path,withIntermediateDirectories:true)
        var registry = InstalledModelRegistry()
        let model = InstalledModel(descriptor:.sarvam,path:path.appendingPathComponent("default.particle").path,managed:true)
        try registry.register(model);try registry.save(to:root.appendingPathComponent("installed.json"))
        let restored = try InstalledModelRegistry.load(from:root.appendingPathComponent("installed.json"))
        #expect(restored.models == [model])
        #expect(try InstalledModelRegistry.removableURL(model,managedRoot:path).path == model.path)
        let imported = InstalledModel(descriptor:.sarvam,path:model.path,managed:false)
        #expect(throws: (any Error).self) {try InstalledModelRegistry.removableURL(imported,managedRoot:path)}
        #expect(throws: (any Error).self) {try InstalledModelRegistry.removableURL(model,managedRoot:root)}
        registry.forget(model.id);#expect(registry.models.isEmpty)
    }
    @Test func corruptedRegistryIsNotSilentlyReset() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data("broken".utf8).write(to:file);defer{try? FileManager.default.removeItem(at:file)}
        #expect(throws: (any Error).self) {try InstalledModelRegistry.load(from:file)}
        #expect(try Data(contentsOf:file) == Data("broken".utf8))
    }
    @Test func preservesNativeExpertSelectionAndArchitectureMemory() {
        let profile = ModelRuntimeProfile(model:.sarvam,physicalMemory:8*1024*1024*1024)
        #expect(profile.activeExperts == 6);#expect(profile.prefillChunk == 128)
        #expect(profile.context <= ModelDescriptor.sarvam.contextLimit)
    }
    @Test func safetensorsShardLengthsExceedTwoGiBWithoutLoadingPayload() throws {
        let json = "{\"large\":{\"dtype\":\"U32\",\"shape\":[1073741824],\"data_offsets\":[0,4294967296]}}"
        var length = UInt64(json.utf8.count).littleEndian
        let header = withUnsafeBytes(of:&length){Data($0)}+Data(json.utf8)
        let parsed = try SafetensorsHeader(prefix:header,fileBytes:Int64(header.count)+4_294_967_296)
        #expect(parsed.tensors["large"]?.bytes == 4_294_967_296)
    }
}
