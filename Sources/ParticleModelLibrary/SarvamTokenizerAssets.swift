import Foundation
import CryptoKit

/// Native replacement for the developer-only tokenizer fetch script. Downloads
/// public metadata only; no weights or repository code are loaded.
public enum SarvamTokenizerAssets {
    public static let revision = "071ae95e933605ca1104a6b4524a36a98488efa4"
    public static let hashes = [
        "config.json":"db5a70c723fc5bd1ca123ab94d1c0a99d914a20c25eefcf041d137906e2341e9",
        "tokenizer.json":"a574ceaaff7c7a8f091179c53fd17ae33567089c099d4ff37d4cb3bc1a87e80e",
        "tokenizer_config.json":"3cdb41a3bcbc8a7e14dbcf8987ccad0cbfe8f82091f2a2c82020f1d0e219cdd5",
        "chat_template.jinja":"07a8ca6f96384a6878c9aa6d4e0616ae57f5937fa8d0ff014855704fa26a0ef4",
        "generation_config.json":"39add3da02bdcab42533aaeb36493e2041a5d18b5e3ada6baddc3daf8105e3cc"
    ]
    private static func digest(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    public static func verified(_ folder:URL)->Bool {
        hashes.allSatisfy {name,expected in
            guard let data = try? ArchitectureRegistry.boundedData(folder.appendingPathComponent(name),limit:50*1024*1024) else{return false}
            return digest(data) == expected
        }
    }
    public static func prepare(at folder:URL,existing:[URL] = []) async throws -> URL {
        for candidate in existing+[folder] {if verified(candidate){return candidate}}
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        guard folder.resolvingSymlinksInPath() == folder.standardizedFileURL else {throw ModelLibraryError.invalid("Tokenizer storage must be a local folder without links.")}
        let card = HubModelCard(repository:"sarvamai/sarvam-30b",revision:revision),client = HuggingFaceClient()
        for (name,expected) in hashes.sorted(by:{$0.key<$1.key}) {
            try Task.checkCancellation()
            let destination = folder.appendingPathComponent(name)
            if let data = try? ArchitectureRegistry.boundedData(destination,limit:50*1024*1024),digest(data) == expected {continue}
            let data = try await client.metadata(card,path:name,limit:50*1024*1024)
            guard digest(data) == expected else {throw ModelLibraryError.checksum(name)}
            try data.write(to:destination,options:.atomic)
        }
        return folder
    }
}
