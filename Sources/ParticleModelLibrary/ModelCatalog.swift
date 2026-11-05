import Foundation

/// Deliberately curated product scope. Graph availability never adds a model
/// to this catalog, and a matching name alone never qualifies its runtime.
public enum ModelCatalog {
    public struct Entry:Identifiable,Sendable,Equatable {
        public let id:String,name:String,repository:String,official:String,type:String
        public let hidden:Int,layers:Int,experts:Int
        public let fast:Bool
        public var isDefault:Bool {id == "sarvam"}
        public var matchesSearch:String {(name+" "+repository+" "+official).lowercased()}
        public func matches(_ query:String)->Bool {
            let terms=query.lowercased().split(whereSeparator:{$0.isWhitespace})
            return terms.allSatisfy{matchesSearch.contains($0)}
        }
    }
    public static let entries:[Entry] = [
        .init(id:"sarvam",name:"Sarvam 30B",repository:"MathObsession/sarvam-30b-particle",official:"sarvamai/sarvam-30b",type:"sarvam_moe",hidden:4096,layers:19,experts:128,fast:false),
        .init(id:"qwen36",name:"Qwen3.6-35B-A3B",repository:"mlx-community/Qwen3.6-35B-A3B-4bit",official:"Qwen/Qwen3.6-35B-A3B",type:"qwen3_5_moe",hidden:2048,layers:40,experts:256,fast:false),
        .init(id:"qwen3-coder",name:"Qwen3-Coder-30B-A3B",repository:"mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",official:"Qwen/Qwen3-Coder-30B-A3B-Instruct",type:"qwen3_moe",hidden:2048,layers:48,experts:128,fast:false),
        .init(id:"glm47",name:"GLM-4.7-Flash",repository:"mlx-community/GLM-4.7-Flash-4bit",official:"zai-org/GLM-4.7-Flash",type:"glm4_moe_lite",hidden:2048,layers:47,experts:64,fast:false),
        .init(id:"nemotron35",name:"NVIDIA Nemotron 3.5 Lightning 30B-A3B",repository:"mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",official:"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",type:"nemotron_h",hidden:2688,layers:52,experts:128,fast:false),
        .init(id:"qwen3-thinking",name:"Qwen3-30B-A3B-Thinking-2507",repository:"mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit",official:"Qwen/Qwen3-30B-A3B-Thinking-2507",type:"qwen3_moe",hidden:2048,layers:48,experts:128,fast:false),
        .init(id:"qwen3-instruct",name:"Qwen3-30B-A3B-Instruct-2507",repository:"mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",official:"Qwen/Qwen3-30B-A3B-Instruct-2507",type:"qwen3_moe",hidden:2048,layers:48,experts:128,fast:false),
        .init(id:"gemma4",name:"Gemma 4 26B-A4B",repository:"mlx-community/gemma-4-26b-a4b-it-4bit",official:"google/gemma-4-26B-A4B-it",type:"gemma4",hidden:2816,layers:30,experts:128,fast:false),
        .init(id:"qwen3",name:"Qwen3-30B-A3B",repository:"mlx-community/Qwen3-30B-A3B-4bit",official:"Qwen/Qwen3-30B-A3B",type:"qwen3_moe",hidden:2048,layers:48,experts:128,fast:false),
        .init(id:"gptoss",name:"GPT-OSS-20B",repository:"mlx-community/gpt-oss-20b-MXFP4-Q8",official:"openai/gpt-oss-20b",type:"gpt_oss",hidden:2880,layers:24,experts:32,fast:false),
        .init(id:"deepseek-coder",name:"DeepSeek-Coder-V2-Lite-Instruct",repository:"mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",official:"deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",type:"deepseek_v2",hidden:2048,layers:27,experts:64,fast:false),
        .init(id:"qwen15",name:"Qwen1.5-MoE-A2.7B-Chat",repository:"mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",official:"Qwen/Qwen1.5-MoE-A2.7B-Chat",type:"qwen2_moe",hidden:2048,layers:24,experts:60,fast:true),
        .init(id:"deepseek-moe",name:"DeepSeekMoE-16B-Chat",repository:"BlueMoonlight/deepseek-moe-16b-chat-mlx-4Bit",official:"deepseek-ai/deepseek-moe-16b-chat",type:"deepseek",hidden:2048,layers:28,experts:64,fast:true),
        .init(id:"minicpm",name:"MiniCPM-MoE-8x2B",repository:"openbmb/MiniCPM-MoE-8x2B",official:"openbmb/MiniCPM-MoE-8x2B",type:"minicpm",hidden:2304,layers:40,experts:8,fast:true)
    ]
    public static func fallbackTemplate(configuration:Data,repository:String?)->String? {
        guard entry(configuration:configuration,repository:repository)?.id == "deepseek-moe" else{return nil}
        return "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token + '\\n' }}{% else %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
    }
    public static let outsideCatalog="This model is outside Particle’s curated library. Choose the default Sarvam 30B model, ten other main models, or three Fast models. Existing files and conversations are preserved."
    public static func entry(repository:String)->Entry? {
        let repo=repository.lowercased()
        return entries.first { entry in
            if [entry.repository,entry.official].contains(where:{$0.lowercased() == repo}) {return true}
            let base=entry.repository.split(separator:"/",maxSplits:1).last!.lowercased()
            guard repo.hasPrefix("mlx-community/") else {return false}
            let stem=base.replacingOccurrences(of:"-(?:[234568]bit(?:-mlx)?|mxfp4-q[48])$",with:"",options:.regularExpression)
            return ["","-2bit","-3bit","-4bit","-5bit","-6bit","-8bit","-4bit-mlx","-mxfp4-q4","-mxfp4-q8"].contains{repo == "mlx-community/"+stem+$0}
        }
    }
    public static func entry(configuration:Data,repository:String?)->Entry? {
        guard let c=try? JSONSerialization.jsonObject(with:configuration) as? [String:Any] else{return nil}
        let text=c["text_config"] as? [String:Any] ?? c
        let candidates=entries.filter{ e in
            let type=c["model_type"] as? String
            return (type == e.type || (e.id == "minicpm" && type == nil)) && text["hidden_size"] as? Int == e.hidden && text["num_hidden_layers"] as? Int == e.layers && (text["num_experts"] ?? text["n_routed_experts"] ?? text["num_local_experts"]) as? Int == e.experts
        }
        if let repository {guard let entry=entry(repository:repository) else{return nil};return candidates.first{$0.id == entry.id}}
        if let path=c["_name_or_path"] as? String,let named=entry(repository:path),candidates.contains(named) {return named}
        // Local weights cannot prove their training identity. This only selects
        // the matching supported structural family and retains local provenance.
        return candidates.first(where:{$0.id == "qwen3"}) ?? candidates.first
    }
    public static func entry(for descriptor:ModelDescriptor)->Entry? {
        if descriptor.architecture == .sarvam,descriptor.layers == 19,descriptor.experts == 128 {return entries.first}
        let entry=descriptor.provenance.repository.flatMap{self.entry(repository:$0)} ?? (descriptor.provenance.repository == nil ? entries.first{$0.id == descriptor.catalogID} : nil)
        guard let entry,descriptor.layers == entry.layers,descriptor.experts == entry.experts else{return nil}
        return entry
    }
    public static func allows(_ descriptor:ModelDescriptor)->Bool {entry(for:descriptor) != nil}
}
