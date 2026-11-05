import Foundation
import Testing
@testable import ParticleModelLibrary

@Suite("Data-only PyTorch ingestion") struct TorchTensorArchiveTests {
    @Test func extractsExactContiguousBFloatAndFloatRanges() async throws {
        let path=try #require(Bundle.module.url(forResource:"torch-tensor-state",withExtension:"bin",subdirectory:"Fixtures"))
        let source=try LocalModelByteSource(root:path.deletingLastPathComponent(),paths:[path.lastPathComponent])
        let records=try await TorchTensorArchive.records(source:source,file:source.files[0])
        let weight=try #require(records["weight"]),norm=try #require(records["norm"])
        #expect(weight.dtype == "BF16" && weight.shape == [2,3,4] && weight.bytes == 48)
        #expect(norm.dtype == "F32" && norm.shape == [4] && norm.bytes == 16)
        let data=try await source.read(path.lastPathComponent,range:norm.offset..<norm.offset+norm.bytes)
        let expected:[Float]=[0,1,2,3]
        let expectedData=expected.withUnsafeBufferPointer{Data(buffer:$0)}
        #expect(data == expectedData)
    }
    @Test func rejectsNonArchiveInputWithoutExecutingAnything() async throws {
        let path=try #require(Bundle.module.url(forResource:"chronos-headers",withExtension:"json",subdirectory:"Fixtures"))
        let source=try LocalModelByteSource(root:path.deletingLastPathComponent(),paths:[path.lastPathComponent])
        await #expect(throws:(any Error).self) {try await TorchTensorArchive.records(source:source,file:source.files[0])}
    }
}
