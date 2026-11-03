import Foundation
import Testing
@testable import ParticleModelLibrary

/// Metadata captured from the pinned source's real headers; no weights.
@Suite("Pinned Chronos metadata compatibility") struct ChronosCompatibilityTests {
    struct Fixture: Decodable {
        struct Tensor: Decodable {let name:String;let dtype:String;let shape:[Int];let bytes:Int64}
        let tensors:[Tensor]
        let template:String
    }
    private func inspect(change: (inout [String:Any])->Void = {_ in}) throws -> CompatibilityReport {
        let url = try #require(Bundle.module.url(forResource:"chronos-headers",withExtension:"json",subdirectory:"Fixtures"))
        let data = try Data(contentsOf:url),fixture = try JSONDecoder().decode(Fixture.self,from:data)
        let root = try #require(JSONSerialization.jsonObject(with:data) as? [String:Any])
        var configuration = try #require(root["configuration"] as? [String:Any]);change(&configuration)
        let records = Dictionary(uniqueKeysWithValues:fixture.tensors.map{($0.name,SafetensorRecord(name:$0.name,dtype:$0.dtype,shape:$0.shape,offset:0,bytes:$0.bytes))})
        return try ArchitectureRegistry.inspectMLX(configuration:JSONSerialization.data(withJSONObject:configuration),tensors:records,template:fixture.template,
            repository:"Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit",revision:"a982bb2ed776bf0db64375de480a1d0a00475a99",sourceBytes:15_179_224_487,license:"mit")
    }
    @Test func preservesMixedQuantizationWithoutQualifyingRuntime() throws {
        let report = try inspect(),model = try #require(report.descriptor)
        #expect(report.conversionSupported)
        #expect(!report.installable)
        #expect(model.quantization.bits == 3)
        #expect(model.quantization.overrides.count == 80)
        #expect(model.quantization.overrides.values.allSatisfy{$0 == 8})
        #expect(model.activeExperts == 8)
        #expect(!model.supportsTools)
    }
    @Test func rejectsGraphVariantsBeforeDownload() throws {
        for change in ["dense","vision","remote-code","different-routing","different-recurrence"] {
            let report = try inspect {root in
                switch change {
                case "dense":root["model_type"] = "qwen3_5"
                case "vision":root["vision_config"] = ["hidden_size":1024]
                case "remote-code":root["auto_map"] = ["AutoModel":"untrusted.py"]
                default:
                    var text = root["text_config"] as! [String:Any]
                    if change == "different-routing" {text["norm_topk_prob"] = false}
                    else {text["mamba_ssm_dtype"] = "float16"}
                    root["text_config"] = text
                }
            }
            #expect(!report.conversionSupported)
        }
    }
}
