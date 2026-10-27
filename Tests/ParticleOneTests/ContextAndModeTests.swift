import Foundation
import Testing
import ParticleDesktopProtocol
import ParticleCore
@testable import ParticleOneCore

@Suite struct ContextAndModeTests {
    @Test func memoryTiersAndKVSizeAreBounded() {
        for (gib,expected) in [(4,8192),(8,32768),(12,65536),(16,65536),(24,131072),(64,131072)] {
            let memory=UInt64(gib)*1_073_741_824
            #expect(ParticleContextPolicy.maximum(physicalMemory:memory)==expected)
            #expect(ParticleContextPolicy.available(physicalMemory:memory).last==expected)
        }
        #expect(ParticleContextPolicy.kvBytes(context:32768)==1_275_068_416)
        #expect(ParticleContextPolicy.kvBytes(context:131072)==5_100_273_664)
        #expect(ParticleContextPolicy.available().allSatisfy{$0<=131072})
    }
    @Test func longOutputReachesWorkerWithoutHalfContextOrCodeCap() {
        var value=OneSettings();value.context=32768;value.maxNew=16286
        #expect(value.valid)
        let args=value.arguments()
        #expect(args[args.firstIndex(of:"--max-new")!+1]=="16286")
        value.context=65536;value.maxNew=32768;#expect(value.valid)
        value.context=131072;value.maxNew=65536;#expect(value.valid)
        value.context=262144;#expect(!value.valid)
        #expect(OneSettings().maxNew>=16286)
        #expect(ParticleContextPolicy.outputAllowance(requested:16384,context:16384)==16352)
    }
    @Test func migrationIncreasesCapacityOnceAndPreservesPreferences()throws {
        let legacy=Data(#"{"context":8192,"maxNew":768,"activeExperts":2,"modelPath":"/my/model.particle","system":"Private preference","responseStyle":"thorough"}"#.utf8)
        var value=try JSONDecoder().decode(OneSettings.self,from:legacy)
        #expect(value.context==max(8192,ParticleContextPolicy.recommended()));#expect(value.maxNew==16384)
        #expect(value.activeExperts==2);#expect(value.modelPath=="/my/model.particle")
        #expect(value.system=="Private preference");#expect(value.responseStyle == .thorough)
        #expect(value.codeResponseStyle == .direct)
        value.context=4096;value.maxNew=512;value.codeResponseStyle = .brief
        let again=try JSONDecoder().decode(OneSettings.self,from:JSONEncoder().encode(value))
        #expect(again==value);#expect(again.context==4096);#expect(again.maxNew==512)
    }
    @Test func nativeCodeModesValidateAndLeaveToolPermissionsUnchanged()throws {
        #expect(ResponseStyle.direct.title=="Instant");#expect(ResponseStyle.brief.title=="Thinking")
        let history=[DesktopMessage(role:"user",content:"Read README.md")]
        for budget in [0,64,-1] {
            let command=DesktopCommand("agent",messages:history,reasoningBudget:budget,tools:AgentTools.catalog)
            let restored=try JSONDecoder().decode(DesktopCommand.self,from:command.line())
            #expect(restored.reasoningBudget==budget);#expect(try restored.validatedAgentHistory()==history)
        }
        for invalid in [-2,4097] {
            let command=DesktopCommand("agent",messages:history,reasoningBudget:invalid,tools:AgentTools.catalog)
            #expect(throws:(any Error).self){try command.validatedAgentHistory()}
        }
        let instant=AgentTools.prompt(project:"/project",planOnly:false,style:.direct)
        let thinking=AgentTools.prompt(project:"/project",planOnly:false,style:.brief)
        #expect(instant.contains("do not write thinking"));#expect(!thinking.contains("do not write thinking"))
        #expect(thinking.contains("require host approval"))
        #expect(try AgentCalls.parse("<think><tool_call>write_file</tool_call></think>Done").isEmpty)
    }
    @Test func exactLongContextAdmissionIncludesOutputAndInstantPrefix()throws {
        let target=Array(repeating:42,count:16381)
        let plan=try ParticlePromptReuse(committed:[],target:target,context:32768,outputAllowance:16384+3)
        #expect(plan.suffix.count==16381)
        #expect(throws:(any Error).self){try ParticlePromptReuse(committed:[],target:target+[42],context:32768,outputAllowance:16384+3)}
        let thinking=try ParticlePromptReuse(committed:target,target:target+[1,2,3],context:32768,outputAllowance:16384)
        #expect(thinking.reused==target.count);#expect(thinking.suffix==[1,2,3])
    }
}
