import Foundation
import Testing
import ParticleCore
import ParticleOneCore
import ParticleDesktopProtocol

@Suite(.serialized) struct AgentPromptRenderingTests {
    @Test func compactPromptKeepsOfficialToolPrefixStable()async throws {
        let folder=URL(fileURLWithPath:ProcessInfo.processInfo.environment["PARTICLE_TOKENIZER_DIR"] ?? "/Volumes/Particle/Models/sarvam-30b/tokenizer")
        guard FileManager.default.fileExists(atPath:folder.appendingPathComponent("tokenizer.json").path) else{return}
        let tokenizer=try await SarvamTokenizer.load(folder:folder)
        let root="/private/tmp/particle-prompt-project",task="Read main.txt and report the current mode. Use read_file, then give a short answer."
        let context="Root entries: AGENTS.md, main.txt\nRoot AGENTS.md (project instructions):\nOnly modify main.txt. Verify any edit with a command."
        let system=AgentTools.prompt(project:root,planOnly:false)+"\nUser preferences: "+OneSettings().system
        let user=task+"\n\n<project_context>\n"+context+"\n</project_context>"
        let history=[SarvamMessage(role:.system,content:system),.init(role:.user,content:user)]
        let target=try tokenizer.agentChat(messages:history,tools:AgentTools.catalog)
        let call="<tool_call>read_file<arg_key>path</arg_key><arg_value>main.txt</arg_value></tool_call>"
        let committed=target+tokenizer.encode("<think></think>\n"+call)
        let continued=history+[.init(role:.assistant,content:call),.init(role:.tool,content:"{\"success\":true,\"output\":\"mode=red\"}")]
        let next=try tokenizer.agentChat(messages:continued,tools:AgentTools.catalog)
        #expect(ParticlePromptReuse.commonPrefix(committed,next)==committed.count)
        for _ in 0..<10{#expect(try tokenizer.agentChat(messages:continued,tools:AgentTools.catalog)==next)}
        var report:[String:Any]=["candidate_tokens":target.count,"cached_tokens_next_turn":committed.count,"new_tokens_next_turn":next.count-committed.count,"official_template_unchanged":true]
        if let path=ProcessInfo.processInfo.environment["PARTICLE_AGENT_BASELINE"] {
            let baseline=try JSONDecoder().decode(DesktopCommand.self,from:Data(contentsOf:URL(fileURLWithPath:path)))
            let oldSystem=(baseline.system ?? "").replacingOccurrences(of:try #require(baseline.system?.components(separatedBy:"You work in this user-selected project: ").last?.components(separatedBy:". BUILD MODE:").first),with:root)
            let old=try tokenizer.agentChat(messages:[.init(role:.system,content:oldSystem),.init(role:.user,content:task)],tools:baseline.tools ?? [])
            report["reference_tokens"]=old.count;report["token_reduction_fraction"]=1-Double(target.count)/Double(old.count)
            report["reference_command"]=DesktopCommand("agent",messages:[.init(role:"user",content:task)],system:oldSystem,tools:baseline.tools)
            // JSONSerialization cannot encode Codable directly.
            report.removeValue(forKey:"reference_command")
            if let output=ProcessInfo.processInfo.environment["PARTICLE_AGENT_PROMPT_OUTPUT"] {
                var object=report
                object["reference"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(DesktopCommand("agent",messages:[.init(role:"user",content:task)],system:oldSystem,tools:baseline.tools)))
                object["candidate"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode(DesktopCommand("agent",messages:[.init(role:"user",content:user)],system:system,tools:AgentTools.catalog)))
                try JSONSerialization.data(withJSONObject:object,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:output))
            }
            #expect(target.count<old.count)
        }
        print("Agent prompt validation: \(report)")
    }
}
