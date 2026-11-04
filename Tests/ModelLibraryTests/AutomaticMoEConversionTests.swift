import Foundation
import Testing
@testable import ParticleModelLibrary
import ParticleOneCore

@Suite("Configuration-driven MoE conversion") struct AutomaticMoEConversionTests {
    @Test func qwen3RecipeIsDerivedFromConfigurationAndKeepsNativeRouting() throws {
        let object:[String:Any] = ["model_type":"qwen3_moe","hidden_size":1024,"num_hidden_layers":24,"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":512,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":64,"vocab_size":151936,"rms_norm_eps":1e-6,"max_position_embeddings":32768,"quantization":["bits":4,"group_size":32]]
        let schema=try MoEModelSchema(configuration:JSONSerialization.data(withJSONObject:object))
        #expect(schema.architecture == .qwen3MoE)
        #expect(schema.layers == 24 && schema.experts == 32 && schema.active == 4)
        #expect(schema.bits == 4 && schema.groupSize == 32)
        #expect(schema.fullAttentionLayers.count == 24)
        #expect(MoEModelSchema.routedLayer("model.layers.23.mlp.switch_mlp.down_proj.scales") == 23)
        #expect(MoEModelSchema.routedLayer("language_model.model.layers.5.mlp.switch_mlp.up_proj.weight") == 5)
        #expect(MoEModelSchema.routedLayer("model.layers.2.self_attn.q_proj.weight") == nil)
    }
    @Test func expertPreferencesAreBoundToModelNotGlobalEightExpertLimit() throws {
        let descriptor=ModelDescriptor(id:"fixture.generic",name:"Fixture MoE",architecture:.qwen3MoE,artifact:.mlx,provenance:.init(),quantization:.init(scheme:"MLX affine",bits:4,groupSize:32),layers:24,experts:64,activeExperts:16,contextLimit:32768,sourceBytes:0,kvBytesPerToken:4096,supportsTools:false)
        #expect(descriptor.expertRange == 1...16)
        var settings=OneSettings();settings.modelID=descriptor.id;settings.activeExperts=16
        #expect(settings.isValid(for:descriptor))
        #expect(settings.validForPersistence)
        settings.activeExperts=1
        #expect(settings.isValid(for:descriptor))
        settings.activeExperts=17
        #expect(!settings.isValid(for:descriptor))
        #expect(ModelDescriptor.sarvam.expertRange == 2...6)
    }
    @Test func discoveryRecognizesOtherMoEFamiliesWithoutClaimingRuntimeSupport() throws {
        let card=HubModelCard(repository:"publisher/Mixtral-8x7B",revision:String(repeating:"a",count:40),tags:["mixtral"])
        #expect(card.isMoECandidate)
        let unsupported=try ArchitectureRegistry.configurationIncompatibility(JSONSerialization.data(withJSONObject:["model_type":"mixtral"]))
        #expect(unsupported?.contains("graph adapter") == true)
    }
}
