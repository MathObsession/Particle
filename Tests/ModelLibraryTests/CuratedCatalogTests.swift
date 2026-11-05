import Foundation
import Testing
@testable import ParticleModelLibrary

@Suite("Curated model boundary") struct CuratedCatalogTests {
    @Test func collectionHasSarvamDefaultTenMainAndThreeFast() {
        #expect(ModelCatalog.entries.count == 14)
        #expect(ModelCatalog.entries.filter{!$0.fast}.count == 11)
        #expect(ModelCatalog.entries.filter(\.fast).map(\.id) == ["qwen15","deepseek-moe","minicpm"])
        #expect(Set(ModelCatalog.entries.map(\.id)).count == 14)
        #expect(ModelCatalog.allows(.sarvam))
        #expect(ModelCatalog.entries.first?.isDefault == true)
    }
    @Test func checkpointVariantsCannotSmuggleUnrelatedModels() {
        #expect(ModelCatalog.entry(repository:"mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit")?.id == "qwen3-thinking")
        #expect(ModelCatalog.entry(repository:"mlx-community/Qwen3-Coder-30B-A3B-Instruct-3bit")?.id == "qwen3-coder")
        for repo in ["mlx-community/Qwen3-30B-A3B-4bit-evil","publisher/Qwen3-30B-A3B-4bit","mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",ModelRecommendations.chronosRepository,"mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit"] {
            #expect(ModelCatalog.entry(repository:repo) == nil)
        }
    }
    @Test func configurationsMustMatchCatalogShape() throws {
        var object:[String:Any]=["model_type":"qwen3_moe","hidden_size":2048,"num_hidden_layers":48,"num_experts":128]
        #expect(ModelCatalog.entry(configuration:try JSONSerialization.data(withJSONObject:object),repository:"mlx-community/Qwen3-30B-A3B-4bit")?.id == "qwen3")
        object["num_experts"]=256
        #expect(ModelCatalog.entry(configuration:try JSONSerialization.data(withJSONObject:object),repository:"mlx-community/Qwen3-30B-A3B-4bit") == nil)
    }
}
