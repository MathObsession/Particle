import Foundation
import Testing
@testable import ParticleModelLibrary

private struct MetadataOnlyPackSource:ModelByteSource {
    let base:LocalModelByteSource
    var identity:String {base.identity}
    var files:[HubFile] {base.files}
    func verifyIdentity() throws {try base.verifyIdentity()}
    func read(_ path:String,range:Range<Int64>) async throws -> Data {
        guard ["manifest.json","config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/chat_template.jinja"].contains(path) else {
            throw ModelLibraryError.invalid("Compatibility inspection attempted to read model weights.")
        }
        return try await base.read(path,range:range)
    }
}

@Suite("Native pack metadata integration") struct NativePackMetadataTests {
    @Test(.enabled(if:ProcessInfo.processInfo.environment["PARTICLE_METADATA_PACK"] != nil,"Optional installed-pack metadata check; no trained weights are evaluated."))
    func inspectsAnExistingPackWithoutReadingItsWeights() async throws {
        let path=try #require(ProcessInfo.processInfo.environment["PARTICLE_METADATA_PACK"])
        let root=URL(fileURLWithPath:path),reader=try ParticleV2Reader(path:root)
        let source=try MetadataOnlyPackSource(base:LocalModelByteSource(root:root,paths:Array(reader.manifest.files.keys)+["manifest.json"]))
        let plan=try await NativePackDownload.plan(source:source)
        #expect(plan.report.conversionSupported)
        #expect(plan.layout.model == reader.manifest.model)
        #expect(plan.layout.ranges.count == reader.manifest.files.count)
        #expect(plan.layout.payloadSizes == reader.manifest.files.mapValues(\.bytes))
        #expect(try ArchitectureRegistry.boundedData(root.appendingPathComponent("manifest.json"),limit:16*1024*1024) == plan.manifest)
    }
}
