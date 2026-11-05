import Foundation
import MLXLLM
import MLXLMCommon
import ParticleModelLibrary

@MainActor enum ParticleGraphFactory {
    static func contains(_ type:String) async -> Bool {
        if ["qwen2_moe","deepseek","minicpm_moe"].contains(type) {return true}
        return await LLMTypeRegistry.shared.contains(type)
    }
    static func create(_ data:Data,type:String) async throws -> any LanguageModel {
        if ["qwen2_moe","deepseek","minicpm_moe"].contains(type) || (type == "minicpm" && (try? JSONSerialization.jsonObject(with:data) as? [String:Any])?["num_experts"] as? Int ?? 0 > 1) {
            return ClassicMoEModel(try ClassicMoEConfiguration(data))
        }
        return try await LLMTypeRegistry.shared.createModel(configuration:data,modelType:type)
    }
}
