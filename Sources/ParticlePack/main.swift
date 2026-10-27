import Foundation
import Darwin
import ParticleGGUF
import ParticlePackCore

private func log(_ text: String) { FileHandle.standardError.write(Data((text+"\n").utf8)) }
@main struct ParticlePackCLI {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.isEmpty || arguments.contains("--help") {
            print("""
            ParticlePack — bounded, lossless Sarvam GGUF to .particle conversion
            --source <file.gguf> --output <folder.particle> [--tokenizer <folder>]
            --profile native  Preserve the existing quantized bytes (only supported profile)
            --plan       Report sizes without creating files
            --resume     Verify and resume a partial conversion
            --verify <folder.particle>  Verify every payload and expert checksum
            The source GGUF is never modified. Data scratch is at most 8 MiB.
            """); return
        }
        do {
            var source: String?, output: String?, tokenizer: String?, verification: String?
            var resume = false, planOnly = false, i = 0
            func value(_ flag: String) throws -> String {
                i += 1; guard i < arguments.count else { throw ParticleError.invalid("Missing value for \(flag)") }; return arguments[i]
            }
            while i < arguments.count {
                let flag = arguments[i]
                switch flag {
                case "--source": source = try value(flag)
                case "--output": output = try value(flag)
                case "--tokenizer": tokenizer = try value(flag)
                case "--profile": guard try value(flag) == "native" else { throw ParticleError.invalid("Only the lossless native pack profile is implemented") }
                case "--resume": resume = true
                case "--plan": planOnly = true
                case "--verify": verification = try value(flag)
                default: throw ParticleError.invalid("Unknown pack option: \(flag)")
                }; i += 1
            }
            if let verification {
                guard source == nil, output == nil, tokenizer == nil, !resume, !planOnly else { throw ParticleError.invalid("--verify cannot be combined with conversion options") }
                let pack = try ParticlePackFile(path:verification)
                _ = try SarvamConfiguration(file:pack)
                try SarvamConfiguration.validateTensorGraph(file:pack)
                try pack.verifyAll(onFile: { log("Verified \($0)") })
                log("Verified all \(pack.manifest.files.count) files and \(pack.layout.layers.count*128) expert blobs"); return
            }
            guard let source else { throw ParticleError.invalid("Specify --source") }
            let file = try GGUFFile(path:source)
            if planOnly {
                let result = try ParticlePacking.plan(file:file)
                let info: [String:Any] = ["source_bytes":result.sourceBytes,"packed_payload_bytes":result.payloadBytes,
                    "common_bytes":result.fileSizes["model_weights.bin"] ?? 0,
                    "embedding_bytes":result.fileSizes["embeddings.bin"] ?? 0,
                    "expert_bytes":result.layout.layers.reduce(0) { $0+$1.stride*128 },
                    "tensors":result.tensors.count,"expert_layers":result.layout.layers.count,
                    "copy_ranges":result.ranges.count,"payload_scratch_bytes":8*1024*1024,"profile":"native","approximate":false]
                let data = try JSONSerialization.data(withJSONObject:info,options:[.prettyPrinted,.sortedKeys])
                FileHandle.standardOutput.write(data); print(); return
            }
            guard let output else { throw ParticleError.invalid("Specify --output ending in .particle") }
            let folder = tokenizer.map { URL(fileURLWithPath:$0) } ?? URL(fileURLWithPath:file.tokenizerDirectory)
            try ParticlePacking.pack(file:file,tokenizer:folder,output:URL(fileURLWithPath:output),resume:resume,
                progress: { done,total in log(String(format:"Copy %.1f%% (%lld / %lld bytes)",Double(done)*100/Double(total),Int64(done),Int64(total))) },status:log)
        } catch { log("ParticlePack: \(error)"); exit(1) }
    }
}
