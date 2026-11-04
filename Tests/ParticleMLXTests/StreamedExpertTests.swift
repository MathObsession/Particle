import Foundation
import Testing
import MLX
import MLXNN
import Metal
import CryptoKit
import ParticleModelLibrary
import ParticleFormat
@testable import ParticleMLXRuntime
private final class MLXTestBundleAnchor: NSObject {}

@Suite("Synthetic MLX streaming kernels",.serialized) @MainActor struct StreamedExpertTests {
    @Test(.enabled(if:MTLCreateSystemDefaultDevice() != nil,"Requires a Metal device; no trained model weights.")) func threeBitExpertMatchesDequantizedReferenceAndSurvivesEviction() throws {
        GPU.metallib = try MLXRuntimeResources.metallib(anchors:[Bundle(for:MLXTestBundleAnchor.self).bundleURL])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("particle-mlx-tests-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root.appendingPathComponent("experts"),withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:root.appendingPathComponent("tokenizer"),withIntermediateDirectories:true)
        func hash(_ data:Data)->String{SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
        var blob = Data(),tensors = [ParticleV2Tensor](),dense = [String:MLXArray]()
        for (index,projection) in ["gate_proj","up_proj","down_proj"].enumerated() {
            let original = MLXArray((0..<4096).map{Float(($0*(index+3)+7)%257-128)/512}).reshaped(64,64)
            let quant = quantized(original,groupSize:64,bits:3),bias = try #require(quant.biases)
            dense[projection] = dequantized(quant.wq,scales:quant.scales,biases:bias,groupSize:64,bits:3)
            for (suffix,array,dtype) in [("weight",quant.wq,"U32"),("scales",quant.scales,"F32"),("biases",bias,"F32")] {
                let data = array.asData(access:.copy).data
                tensors.append(.init(name:"language_model.model.layers.0.mlp.switch_mlp.\(projection).\(suffix)",dtype:dtype,
                    shape:[256]+array.shape,file:"experts/layer_000.bin",offset:Int64(blob.count),bytes:Int64(data.count),layer:0,expertStride:16384))
                blob.append(data)
            }
        }
        blob.append(Data(count:16384-blob.count))
        let payload = root.appendingPathComponent("experts/layer_000.bin")
        FileManager.default.createFile(atPath:payload.path,contents:nil)
        let output = try FileHandle(forWritingTo:payload)
        try output.truncate(atOffset:UInt64(16384*256))
        for id in [3,4] {try output.seek(toOffset:UInt64(id*16384));try output.write(contentsOf:blob)}
        try output.close()
        var files = [String:ParticleV2File]()
        for name in ["common.bin","config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/chat_template.jinja"] {
            try Data().write(to:root.appendingPathComponent(name));files[name] = .init(bytes:0,sha256:hash(Data()))
        }
        // Reader checks each selected expert independently; unselected sparse bytes are never read.
        files["experts/layer_000.bin"] = .init(bytes:16384*256,sha256:String(repeating:"0",count:64))
        let model = ModelDescriptor(id:"synthetic",name:"Synthetic",architecture:.qwen35MoE,artifact:.mlx,provenance:.init(),
            quantization:.init(scheme:"MLX affine",bits:3,groupSize:64),layers:1,experts:256,activeExperts:8,
            contextLimit:128,sourceBytes:0,kvBytesPerToken:8,supportsTools:false)
        var hashes = Array(repeating:hash(Data(count:16384)),count:256);hashes[3] = hash(blob);hashes[4] = hash(blob)
        let manifest = ParticleV2Manifest(model:model,files:files,tensors:tensors,
            sourceFiles:["synthetic":.init(bytes:0,sha256:hash(Data()))],expertHashes:["experts/layer_000.bin":hashes])
        try manifest.validate()
        let encoder = JSONEncoder(),data = try encoder.encode(manifest)
        try data.write(to:root.appendingPathComponent("manifest.json"))
        try encoder.encode(ParticleInstallReceipt(manifestSHA256:hash(data))).write(to:root.appendingPathComponent("verified-install.json"))
        let reader = try ParticleV2Reader(path:root),store = ExpertStore(reader:reader,slots:1)
        let x = MLXArray((0..<192).map{Float($0%37-18)/16}).reshaped(3,64)
        let g = matmul(x,try #require(dense["gate_proj"]).T),u = matmul(x,try #require(dense["up_proj"]).T)
        let expected = matmul(silu(g)*u,try #require(dense["down_proj"]).T)
        let first = try store.get(layer:0,expert:3)(x);eval(first)
        let difference = abs(first-expected).max().item(Float.self)
        #expect(difference < 0.0001)
        let weights = try store.get(layer:0,expert:3)
        let indices = MLXArray([UInt32(0),0,0]).reshaped(3,1)
        for dtype in [DType.float32,DType.bfloat16] {
            let input = x.asType(dtype)
            let upstream = try QuantizedExpert.upstream(input,experts:[weights],indices:indices)
            let gathered = QuantizedExpert.gathered(input,experts:[weights],indices:indices)
            eval(upstream,gathered)
            #expect(abs(upstream-gathered).max().item(Float.self) == 0)
        }
        let bank=(0..<8).map {expert -> QuantizedExpert in
            func projection(_ value:QuantizedExpert.Projection)->QuantizedExpert.Projection {
                let scale=Float(expert+1)/8
                return .init(weight:value.weight,scales:(value.scales*scale).asType(.bfloat16),biases:(value.biases*scale).asType(.bfloat16))
            }
            return .init(gate:projection(weights.gate),up:projection(weights.up),down:projection(weights.down))
        }
        for rows in [1,16,64,128] {
            let input=MLXArray((0..<(rows*64)).map{Float($0%37-18)/16}).reshaped(rows,64).asType(.bfloat16)
            let routes=MLXArray((0..<(rows*8)).map{UInt32(($0*3+1)%8)}).reshaped(rows,8)
            let upstream=try QuantizedExpert.upstream(input,experts:bank,indices:routes)
            let gathered=QuantizedExpert.gathered(input,experts:bank,indices:routes)
            eval(upstream,gathered)
            #expect(abs(upstream-gathered).max().item(Float.self) == 0)
            let positions=(0..<8).map{$0*17+5}
            let fullBank=try QuantizedExpert.upstream(input,experts:bank,indices:routes*17+5,bankIndices:positions,totalExperts:256)
            let compact=QuantizedExpert.gathered(input,experts:bank,indices:routes,totalExperts:256)
            eval(fullBank,compact)
            #expect(abs(fullBank-compact).max().item(Float.self) == 0)
        }
        _ = try store.get(layer:0,expert:3);#expect(store.hits == 2);#expect(store.readBytes == 16384)
        let second = try store.get(layer:0,expert:4)(x);eval(second)
        _ = try store.get(layer:0,expert:3);#expect(store.readBytes == 3*16384)
        #expect(abs(first-second).max().item(Float.self) == 0)
    }
}
