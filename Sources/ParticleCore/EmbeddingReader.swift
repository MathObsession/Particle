import Foundation
@preconcurrency import Metal
import ParticleGGUF

/// Sarvam's output head is untied. Only one packed embedding row is needed
/// per input token, so the full vocabulary matrix need not be GPU-mapped.
struct EmbeddingReader {
    let file: any ParticleModelSource
    let tensor: GGUFTensor
    var rowBytes: Int { tensor.rowBytes }
    init(file: any ParticleModelSource) throws {
        let tensor = try file.tensor("token_embd.weight")
        guard tensor.shape.count == 2 else { throw ParticleError.invalid("Invalid embedding matrix") }
        self.file = file; self.tensor = tensor
    }
    func load(token: Int, into buffer: MTLBuffer) throws -> GPUWeight {
        guard (0..<tensor.shape[1]).contains(token), buffer.length >= rowBytes else {
            throw ParticleError.invalid("Embedding row or buffer is out of range")
        }
        try file.readTensor(tensor,into:buffer.contents(),count:rowBytes,relativeOffset:token*rowBytes)
        return GPUWeight(buffer:buffer,offset:0,columns:tensor.shape[0],rows:1,type:tensor.type)
    }
}
