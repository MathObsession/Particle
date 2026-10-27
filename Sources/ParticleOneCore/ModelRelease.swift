import Foundation
import CryptoKit

public struct ModelCatalog: Codable, Sendable {
    public let repo: String
    public let revision: String
    public let distributionSHA256: String
    public let modelManifestSHA256: String
    public let folder: String
    public let totalBytes: Int64
    public func validate() throws {
        guard repo=="MathObsession/sarvam-30b-particle", revision.count==40,revision.allSatisfy(\.isHexDigit),distributionSHA256.count==64,modelManifestSHA256.count==64,
              folder=="sarvam-30b-q4_k_m.particle",totalBytes>0 else { throw ReleaseError.invalidManifest }
    }
}
public struct ModelRelease: Codable, Sendable {
    public struct Entry: Codable, Sendable {
        public let path: String
        public let bytes: Int64
        public let sha256: String
        public init(path: String,bytes: Int64,sha256: String) { self.path=path;self.bytes=bytes;self.sha256=sha256 }
    }
    public let version: Int
    public let format: String
    public let folder: String
    public let architecture: String
    public let profile: String
    public let approximate: Bool
    public let manifestSHA256: String
    public let totalBytes: Int64
    public let files: [Entry]
    public func validate(catalog: ModelCatalog) throws {
        try catalog.validate()
        guard version==1,format=="particle",folder==catalog.folder,architecture=="sarvam_moe",profile=="native",!approximate,
              manifestSHA256==catalog.modelManifestSHA256,totalBytes==catalog.totalBytes,files.count<=128,
              Set(files.map(\.path)).count==files.count,files.contains(where:{$0.path=="manifest.json"}),
              files.contains(where:{$0.path=="weights.json"}),files.contains(where:{$0.path=="tokenizer/tokenizer.json"}) else { throw ReleaseError.invalidManifest }
        var total:Int64=0
        for file in files {
            guard Self.safePath(file.path),file.bytes>0,file.bytes<=2_147_483_648,file.sha256.count==64,file.sha256.allSatisfy(\.isHexDigit),file.bytes<=Int64.max-total else { throw ReleaseError.invalidManifest }
            total+=file.bytes
        }
        guard total==totalBytes else { throw ReleaseError.invalidManifest }
    }
    public static func safePath(_ path: String) -> Bool {
        guard !path.hasPrefix("/"),!path.contains("\\"),!path.contains(":"),!path.contains("\0"),!path.contains("%") else { return false }
        let parts=path.split(separator:"/",omittingEmptySubsequences:false)
        return !parts.isEmpty && parts.allSatisfy{!$0.isEmpty && $0 != "." && $0 != ".." && !$0.hasPrefix(".")} && !["verified-install.json","install-progress.json","pack.lock"].contains(path)
    }
    public static func digest(_ data: Data) -> String { SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined() }
    public static func digestFile(_ url: URL, cancelled: @Sendable () -> Bool = {false}) throws -> String {
        let handle=try FileHandle(forReadingFrom:url);defer{try? handle.close()};var sha=SHA256()
        while let data=try handle.read(upToCount:4*1024*1024),!data.isEmpty { if cancelled(){throw CancellationError()};sha.update(data:data) }
        return sha.finalize().map{String(format:"%02x",$0)}.joined()
    }
}
public enum ReleaseError: LocalizedError {
    case invalidManifest, network(Int), storage, checksum(String), notPublished
    public var errorDescription: String? {
        switch self {
        case .invalidManifest:return "The model release could not be verified. Nothing has been installed."
        case .network(let status):return "The download server returned \(status). Please retry when connected."
        case .storage:return "There is not enough free space for this model. Choose a disk with at least 22 GB free."
        case .checksum(let file):return "Verification failed for \(file). Retry to download that file again."
        case .notPublished:return "The model release is not published yet. You can choose an existing .particle model."
        }
    }
}
