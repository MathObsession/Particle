import Foundation
import CryptoKit

/// Copies a published v2 pack verbatim. The original source provenance and
/// manifest bytes survive; only the verified local installation receipt is new.
public enum NativePackDownload {
    public struct Plan: Sendable {
        public let layout:MLXPackPlan
        public let manifest:Data
        public let report:CompatibilityReport
    }
    public static func plan(source:any ModelByteSource) async throws -> Plan {
        let raw=try await MLXPackConverter.metadata(source,path:"manifest.json",limit:16*1024*1024)
        guard let record=source.files.first(where:{$0.path == "manifest.json"}) else {throw ModelLibraryError.invalid("Missing native manifest.")}
        if let expected=record.sha256 {
            guard SHA256.hash(data:raw).map({String(format:"%02x",$0)}).joined() == expected else {throw ModelLibraryError.checksum("manifest.json")}
        }
        if let expected=record.gitBlobSHA1 {
            var digest=Insecure.SHA1();digest.update(data:Data("blob \(record.bytes)\0".utf8));digest.update(data:raw)
            guard digest.finalize().map({String(format:"%02x",$0)}).joined() == expected else {throw ModelLibraryError.checksum("manifest.json")}
        }
        let manifest=try JSONDecoder().decode(ParticleV2Manifest.self,from:raw)
        try manifest.validate()
        func metadata(_ name:String,_ limit:Int) async throws -> Data {
            guard let file=manifest.files[name] else {throw ModelLibraryError.invalid("Missing native pack metadata.")}
            let data=try await MLXPackConverter.metadata(source,path:name,limit:limit)
            guard Int64(data.count) == file.bytes,SHA256.hash(data:data).map({String(format:"%02x",$0)}).joined() == file.sha256 else {throw ModelLibraryError.checksum(name)}
            return data
        }
        let configuration=try await metadata("config.json",1_048_576)
        if let reason=try ArchitectureRegistry.configurationIncompatibility(configuration) {throw ModelLibraryError.incompatible(reason)}
        let tokenizer=try await metadata("tokenizer/tokenizer.json",64*1024*1024)
        let tokenizerConfiguration=try await metadata("tokenizer/tokenizer_config.json",1_048_576)
        let template=try await metadata("tokenizer/chat_template.jinja",262_144)
        let report=try ArchitectureRegistry.inspectV2Metadata(manifest,configuration:configuration,tokenizer:tokenizer,tokenizerConfiguration:tokenizerConfiguration,template:template)
        guard report.conversionSupported else {throw ModelLibraryError.incompatible(report.reason)}
        return .init(layout:try copyPlan(manifest,sourceFiles:source.files),manifest:raw,report:report)
    }
    static func copyPlan(_ manifest:ParticleV2Manifest,sourceFiles:[HubFile]) throws -> MLXPackPlan {
        try manifest.validate()
        var files=[HubFile](),ranges=[MLXPackCopyRange]()
        for (name,file) in manifest.files.sorted(by:{$0.key < $1.key}) {
            guard let original=sourceFiles.first(where:{$0.path == name}),original.bytes == file.bytes,
                  original.sha256 == nil || original.sha256 == file.sha256 else {throw ModelLibraryError.invalid("The published pack inventory does not match its manifest: \(name).")}
            let components=name.split(separator:"/")
            guard components.count <= 2,components.count == 1 || ["experts","tokenizer"].contains(String(components[0])) else {throw ModelLibraryError.invalid("Unsupported native pack folder.")}
            files.append(.init(path:name,bytes:file.bytes,sha256:file.sha256,gitBlobSHA1:original.gitBlobSHA1))
            ranges.append(.init(source:name,offset:0,bytes:file.bytes,file:name,destination:0))
        }
        return .init(model:manifest.model,sourceFiles:files,payloadSizes:manifest.files.mapValues(\.bytes),tensors:manifest.tensors,ranges:ranges)
    }
}
