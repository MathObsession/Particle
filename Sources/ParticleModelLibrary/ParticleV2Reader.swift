import Foundation
import CryptoKit
import Darwin
import ParticleFormat
import ParticleGGUF

public final class ParticleV2Reader: @unchecked Sendable {
    public let manifest: ParticleV2Manifest
    public let path: URL
    public let tensorIndex: [String:ParticleV2Tensor]
    private let root: Int32
    public init(path: URL) throws {
        self.path = path
        let fd = open(path.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd >= 0 else {throw ModelLibraryError.invalid("The Particle model folder is unavailable.")}
        var success = false;defer{if !success {close(fd)}}
        let data = try ParticleIO.data(root:fd,path:"manifest.json",limit:16*1024*1024)
        let manifest = try JSONDecoder().decode(ParticleV2Manifest.self,from:data);try manifest.validate()
        let receipt = try JSONDecoder().decode(ParticleInstallReceipt.self,from:ParticleIO.data(root:fd,path:"verified-install.json",limit:65_536))
        guard receipt.complete,receipt.manifestSHA256 == ParticleIO.hash(data) else {throw ModelLibraryError.invalid("The model receipt does not match its manifest.")}
        for (name,file) in manifest.files {
            let entry = try ParticleIO.openRelative(root:fd,path:name);defer{close(entry)}
            var st = stat();guard fstat(entry,&st) == 0,st.st_size == file.bytes else {throw ModelLibraryError.invalid("Model payload size mismatch: \(name).")}
        }
        self.manifest = manifest;tensorIndex = Dictionary(uniqueKeysWithValues:manifest.tensors.map{($0.name,$0)});root = fd;success = true
    }
    deinit {close(root)}
    public func verifyAll(progress: (String)->Void = {_ in}) throws {
        for (path,file) in manifest.files.sorted(by: {$0.key < $1.key}) {
            let fd = try ParticleIO.openRelative(root:root,path:path);defer{close(fd)}
            if let expected=manifest.expertHashes[path],let stride=manifest.tensors.first(where:{$0.file == path && $0.layer != nil})?.expertStride {
                var digest=SHA256()
                for expert in 0..<manifest.model.experts {
                    try Task.checkCancellation()
                    var data=Data(count:Int(stride))
                    try data.withUnsafeMutableBytes{try ParticleFileRange(descriptor:fd,offset:expert*Int(stride),count:Int(stride)).read(into:$0.baseAddress!)}
                    guard ParticleIO.hash(data) == expected[expert] else {throw ModelLibraryError.checksum("\(path), expert \(expert)")}
                    digest.update(data:data)
                }
                guard digest.finalize().map({String(format:"%02x",$0)}).joined() == file.sha256 else {throw ModelLibraryError.checksum(path)}
            } else {guard try ParticleIO.hashFile(descriptor:fd,size:Int(file.bytes)) == file.sha256 else {throw ModelLibraryError.checksum(path)}}
            progress(path)
        }
    }
    public func residentTensor(_ name: String) throws -> (ParticleV2Tensor,Data) {
        guard let tensor = tensorIndex[name],tensor.layer == nil,
              tensor.bytes <= 512*1024*1024 else {throw ModelLibraryError.invalid("Missing or oversized resident tensor: \(name).")}
        let fd = try ParticleIO.openRelative(root:root,path:tensor.file);defer{close(fd)}
        var data = Data(count:Int(tensor.bytes))
        try data.withUnsafeMutableBytes {try ParticleFileRange(descriptor:fd,offset:Int(tensor.offset),count:Int(tensor.bytes)).read(into:$0.baseAddress!)}
        return (tensor,data)
    }
    public func verifyResidentFiles() throws {
        let expertFiles = Set(manifest.expertHashes.keys)
        for (name,file) in manifest.files where !expertFiles.contains(name) {
            let fd = try ParticleIO.openRelative(root:root,path:name);defer{close(fd)}
            guard try ParticleIO.hashFile(descriptor:fd,size:Int(file.bytes)) == file.sha256 else {throw ModelLibraryError.checksum(name)}
        }
    }
    public func expert(layer: Int, expert: Int) throws -> Data {
        guard (0..<manifest.model.experts).contains(expert),let tensor = manifest.tensors.first(where: {$0.layer == layer}),
              let stride = tensor.expertStride,stride <= 32*1024*1024,
              let expected = manifest.expertHashes[tensor.file]?[expert] else {throw ModelLibraryError.invalid("Invalid expert selection.")}
        let fd = try ParticleIO.openRelative(root:root,path:tensor.file);defer{close(fd)}
        var data = Data(count:Int(stride))
        try data.withUnsafeMutableBytes {try ParticleFileRange(descriptor:fd,offset:expert*Int(stride),count:Int(stride)).read(into:$0.baseAddress!)}
        guard ParticleIO.hash(data) == expected else {throw ModelLibraryError.checksum("\(tensor.file), expert \(expert)")}
        return data
    }
}
