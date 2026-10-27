import Foundation
import Testing
import ParticleDesktopProtocol
@testable import ParticleOneCore

@Suite struct ParticleOneTests {
    @Test func transportPreservesUnicodeAndMultilineAndLiteralCommands() throws {
        for text in ["Hi!","hello\nworld","नमस्ते 🌿","/quit","/new","```swift\nprint(1)\n```",String(repeating:"é",count:3000)] {
            let wire=try DesktopCommand("message",text:text).line();var decoder=DesktopLineDecoder();var lines=[Data]()
            for byte in wire {lines+=try decoder.append(Data([byte]))}
            #expect(lines.count==1)
            #expect(try JSONDecoder().decode(DesktopCommand.self,from:lines[0]).text==text)
        }
    }
    @Test func restoredHistoryCannotInjectSystemOrMalformedRoles() throws {
        let valid=DesktopCommand("restore",messages:[.init(role:"user",content:"hi"),.init(role:"assistant",content:"hello")])
        #expect(try valid.validatedHistory().count==2)
        for roles in [["system","assistant"],["assistant","user"],["user"],["user","user"]] {
            let command=DesktopCommand("restore",messages:roles.map{.init(role:$0,content:"x")})
            #expect(throws:(any Error).self){try command.validatedHistory()}
        }
    }
    @Test func interruptedRepliesArePreservedButNotReplayed() {
        var chat=Conversation();chat.messages=[.init(role:"user",content:"first"),.init(role:"assistant",content:"unfinished",complete:false),.init(role:"user",content:"second"),.init(role:"assistant",content:"finished")]
        #expect(chat.messages.count==4);#expect(chat.history.map(\.content)==["second","finished"])
    }
    @Test func transcriptSeparatesThinkingAndInertCodeDuringStreaming() {
        let pieces=TranscriptParser.parse("<think>reasoning</think>Here:\n```python\nprint('hi')\n```\nDone.")
        #expect(pieces.map(\.kind)==[.thinking,.text,.code,.text]);#expect(pieces[2].language=="python")
        #expect(TranscriptParser.parse("<think>unfinished").first?.kind == .thinking)
        #expect(TranscriptParser.parse("```swift\nprint(1)").first?.kind == .code)
        #expect(TranscriptParser.parse("plain").first?.text=="plain")
    }
    @Test func releasePathsCannotEscapeOrCreateReceipts() {
        for path in ["../x","/tmp/x","a/../../x","a//b","a\\b","https:x","%2e%2e/x",".DS_Store","verified-install.json","install-progress.json","pack.lock"] {#expect(!ModelRelease.safePath(path))}
        for path in ["manifest.json","packed_experts/layer_01.bin","tokenizer/tokenizer.json"] {#expect(ModelRelease.safePath(path))}
    }
    @Test func workspaceRoundTripKeepsTranscriptAndProfile() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:folder)}
        var state=SavedWorkspace();state.settings.activeExperts=3;var chat=Conversation();chat.messages=[.init(role:"user",content:"hi"),.init(role:"assistant",content:"hello")];state.conversations=[chat];state.selected=chat.id
        try WorkspacePersistence.save(state,to:folder);let loaded=try WorkspacePersistence.load(from:folder)
        #expect(loaded.settings.activeExperts==3);#expect(loaded.conversations==state.conversations);#expect(loaded.selected==chat.id)
    }
    @Test func defaultProfileAndBoundedOutputAreHonest() {
        var settings=OneSettings();#expect(settings.activeExperts==6);settings.context=512;settings.maxNew=4096
        let arguments=settings.arguments();let index=arguments.firstIndex(of:"--max-new")!;#expect(arguments[index+1]=="256")
        #expect(arguments.contains("--desktop-events"));#expect(!arguments.contains("--thinking"))
    }
}

@Suite struct ModelReleaseTests {
    private func objects() -> ([String:Any],[String:Any]) {
        let sha=String(repeating:"a",count:64)
        let catalog:[String:Any] = ["repo":"MathObsession/sarvam-30b-particle","revision":String(repeating:"b",count:40),"distributionSHA256":sha,"modelManifestSHA256":sha,"folder":"sarvam-30b-q4_k_m.particle","totalBytes":3]
        let release:[String:Any] = ["version":1,"format":"particle","folder":"sarvam-30b-q4_k_m.particle","architecture":"sarvam_moe","profile":"native","approximate":false,"manifestSHA256":sha,"totalBytes":3,"files":["manifest.json","weights.json","tokenizer/tokenizer.json"].map{["path":$0,"bytes":1,"sha256":sha] as [String:Any]}]
        return (catalog,release)
    }
    @Test func pinnedCatalogRejectsUnexpectedRepositoryAndUnpinnedRevision() throws {
        var (catalog,_)=objects()
        let good=try JSONDecoder().decode(ModelCatalog.self,from:JSONSerialization.data(withJSONObject:catalog));try good.validate()
        for change in [["repo":"other/model"],["revision":"main"],["folder":"../model"]] {
            var copy=catalog;for (k,v) in change{copy[k]=v}
            let bad=try JSONDecoder().decode(ModelCatalog.self,from:JSONSerialization.data(withJSONObject:copy))
            #expect(throws:(any Error).self){try bad.validate()}
        }
        catalog["revision"]="not-a-commit"
    }
    @Test func distributionRejectsDuplicatesWrongTotalsAndApproximateWeights() throws {
        let (catalogObject,releaseObject)=objects()
        let catalog=try JSONDecoder().decode(ModelCatalog.self,from:JSONSerialization.data(withJSONObject:catalogObject))
        let good=try JSONDecoder().decode(ModelRelease.self,from:JSONSerialization.data(withJSONObject:releaseObject));try good.validate(catalog:catalog)
        for field in ["totalBytes","approximate","files"] {
            var copy=releaseObject
            if field=="totalBytes"{copy[field]=4}
            if field=="approximate"{copy[field]=true}
            if field=="files"{var files=copy[field] as! [[String:Any]];files[1]=files[0];copy[field]=files}
            let bad=try JSONDecoder().decode(ModelRelease.self,from:JSONSerialization.data(withJSONObject:copy))
            #expect(throws:(any Error).self){try bad.validate(catalog:catalog)}
        }
    }
    @Test func streamingFileHashAndCancellation() throws {
        let data=Data(repeating:42,count:4*1024*1024+19),url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:url)};try data.write(to:url)
        #expect(try ModelRelease.digestFile(url)==ModelRelease.digest(data))
        #expect(throws:CancellationError.self){try ModelRelease.digestFile(url,cancelled:{true})}
    }
}

@Suite struct InputIdentityTests {
    @Test func greetingBytesSurviveCommandEncodingExactly() throws {
        let greeting="Hi!"
        #expect(Array(greeting.utf8)==[0x48,0x69,0x21])
        let command=try JSONDecoder().decode(DesktopCommand.self,from:DesktopCommand("message",text:greeting).line())
        #expect(command.text==greeting)
        #expect(DesktopTextIdentity.fingerprint(command.text!)==DesktopTextIdentity.fingerprint(greeting))
        #expect(DesktopTextIdentity.fingerprint("Hi! h") != DesktopTextIdentity.fingerprint(greeting))
    }
    @Test func everySupportedExpertCountPersistsAndReachesTheWorker() throws {
        for count in 2...6 {
            var settings=OneSettings();settings.activeExperts=count
            #expect(settings.valid)
            let roundTrip=try JSONDecoder().decode(OneSettings.self,from:JSONEncoder().encode(settings))
            let args=roundTrip.arguments();let flag=args.firstIndex(of:"--active-experts")!
            #expect(args[flag+1]==String(count))
        }
        for count in [1,7] {var settings=OneSettings();settings.activeExperts=count;#expect(!settings.valid)}
    }
}

@Suite struct WorkspacePolishTests {
    @Test func legacyJSONMigratesWithoutChangingOrDiscardingIt() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:folder)}
        let legacy=Data(#"{"version":1,"settings":{"modelPath":"/test.gguf","slots":12,"context":2048,"maxNew":768,"activeExperts":3,"useMirror":false,"system":"My private prompt"},"conversations":[]}"#.utf8)
        try legacy.write(to:folder.appendingPathComponent("workspace.json"))
        let migrated=try WorkspacePersistence.load(from:folder)
        #expect(migrated.settings.activeExperts==3);#expect(migrated.settings.responseStyle == .direct);#expect(migrated.settings.system=="My private prompt")
        #expect(migrated.username.split(separator:" ").count==2)
        #expect(try Data(contentsOf:folder.appendingPathComponent("workspace.json"))==legacy)
        #expect(FileManager.default.fileExists(atPath:folder.appendingPathComponent("particleone.sqlite").path))
        #expect(try WorkspacePersistence.load(from:folder).id==migrated.id)
        let permissions=try FileManager.default.attributesOfItem(atPath:folder.appendingPathComponent("particleone.sqlite").path)[.posixPermissions] as? Int
        #expect(permissions==0o600)
    }
    @Test func workspacesKeepIndependentChatsAndMemories() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:folder)}
        var personal=SavedWorkspace();personal.name="Personal";personal.memories=[.init(text:"My name is Mira")];var chat=Conversation();chat.messages=[.init(role:"user",content:"hello"),.init(role:"assistant",content:"hi")];personal.conversations=[chat];personal.selected=chat.id
        try WorkspacePersistence.save(personal,to:folder)
        var writing=SavedWorkspace();writing.name="Writing";writing.settings.system="Write succinctly";try WorkspacePersistence.save(writing,to:folder)
        #expect(try WorkspacePersistence.load(from:folder).id==writing.id)
        let restored=try WorkspacePersistence.open(personal.id,in:folder)
        #expect(restored.memories==personal.memories);#expect(restored.conversations==personal.conversations)
        #expect(try WorkspacePersistence.open(writing.id,in:folder).memories.isEmpty)
        #expect(try WorkspacePersistence.list(in:folder).count==2)
    }
    @Test func deletingMessagesRemovesTheirContextWithoutShiftingPairs() {
        var state=SavedWorkspace();var chat=Conversation();chat.messages=[.init(role:"user",content:"a"),.init(role:"assistant",content:"b"),.init(role:"user",content:"c"),.init(role:"assistant",content:"d"),.init(role:"user",content:"e"),.init(role:"assistant",content:"f")];state.selected=chat.id;state.conversations=[chat]
        state.deleteMessage(chat.messages[1].id)
        #expect(state.conversations[0].history.map(\.content)==["c","d","e","f"])
        state.deleteMessage(chat.messages[2].id)
        #expect(state.conversations[0].history.map(\.content)==["e","f"])
    }
    @Test func memoryOnlyExtractsExplicitPersonalStatements() {
        #expect(PersonalMemory.statements(in:"My name is Mira. I prefer concise replies. What is two plus two?")==["My name is Mira","I prefer concise replies"])
        #expect(PersonalMemory.statements(in:"The assistant says my name is Mira. My password is secret. I like my bank account.").isEmpty)
        #expect(PersonalMemory.statements(in:"hi").isEmpty)
        var state=SavedWorkspace();state.memories=[.init(text:"I prefer concise replies")]
        #expect(state.systemPrompt.contains("<saved_personal_notes>"));#expect(state.conversations.isEmpty)
        state.memories=[];#expect(!state.systemPrompt.contains("<saved_personal_notes>"))
    }
    @Test func appearanceProfileAndReasoningSettingsSurviveReload() throws {
        var state=SavedWorkspace();state.settings.appearance = .dark;state.settings.responseStyle = .brief;state.username="Curious Maker";state.sidebarHidden=true;state.settings.rememberPersonalDetails=true
        let decoded=try JSONDecoder().decode(SavedWorkspace.self,from:JSONEncoder().encode(state))
        #expect(decoded.settings==state.settings);#expect(decoded.sidebarHidden);#expect(decoded.username=="Curious Maker")
        let args=decoded.settings.arguments();#expect(args[args.firstIndex(of:"--reasoning-budget")!+1]=="64")
        #expect(OneSettings().responseStyle == .direct)
    }
    @Test func corruptedWorkspaceDoesNotSilentlyBecomeEmpty() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:folder)}
        let bad=Data("invalid sqlite file".utf8);try bad.write(to:folder.appendingPathComponent("particleone.sqlite"))
        #expect(throws:(any Error).self){try WorkspacePersistence.load(from:folder)}
        #expect(try Data(contentsOf:folder.appendingPathComponent("particleone.sqlite"))==bad)
    }
}
@Suite struct NoThinkingTests {
    @Test func zeroBudgetClosesBeforeAnyReasoningTokens() {
        var limiter=DesktopReasoningLimiter(budget:0);#expect(!limiter.mustClose)
        limiter.accept(23);#expect(limiter.mustClose);#expect(limiter.used==0)
        limiter.accept(24);#expect(!limiter.thinking);#expect(!limiter.mustClose)
        limiter.accept(10979);#expect(limiter.used==0)
    }
    @Test func boundedThinkingCannotRestartItsBudget() {
        var limiter=DesktopReasoningLimiter(budget:2);limiter.accept(23);limiter.accept(100);#expect(!limiter.mustClose);limiter.accept(101);#expect(limiter.mustClose)
        limiter.accept(24);limiter.accept(23);#expect(limiter.mustClose)
        limiter.accept(24);limiter.accept(23);limiter.accept(24);limiter.accept(23);#expect(limiter.exhausted)
    }
    @Test func nativeModeNeverForcesAnEnd() {
        var limiter=DesktopReasoningLimiter(budget:-1);limiter.accept(23);for _ in 0..<5000{limiter.accept(100)};#expect(!limiter.mustClose)
    }
    @Test func hiddenSystemRoundTripsSeparatelyFromChat() throws {
        let command=DesktopCommand("restore",messages:[.init(role:"user",content:"Hi"),.init(role:"assistant",content:"Hello")],system:"A private instruction")
        let value=try JSONDecoder().decode(DesktopCommand.self,from:command.line());#expect(value.system=="A private instruction");#expect(try value.validatedHistory().allSatisfy{$0.role != "system"})
    }
}

@Suite struct ThinkingControlTests {
    @Test func restoreCarriesValidatedModeWithoutChangingHistory() throws {
        let history=[DesktopMessage(role:"user",content:"Hi!"),DesktopMessage(role:"assistant",content:"Hello!")]
        for budget in [-1,0,64,4096] {
            let command=DesktopCommand("restore",messages:history,system:"Private guidance",reasoningBudget:budget)
            let decoded=try JSONDecoder().decode(DesktopCommand.self,from:command.line())
            #expect(decoded.reasoningBudget==budget)
            #expect(try decoded.validatedHistory()==history)
        }
        for budget in [-2,4097] {
            #expect(throws:(any Error).self){try DesktopCommand("restore",messages:history,reasoningBudget:budget).validatedHistory()}
        }
        #expect(try DesktopCommand("restore",messages:history).validatedHistory()==history)
    }
}

@Suite struct MessageEditingTests {
    @Test func editedContinuationPreservesOriginalAndExcludesFutureTurns() throws {
        var source=Conversation();source.title="Original conversation"
        source.messages=[.init(role:"user",content:"first"),.init(role:"assistant",content:"first answer"),.init(role:"user",content:"old question"),.init(role:"assistant",content:"stale answer"),.init(role:"user",content:"later question"),.init(role:"assistant",content:"later answer")]
        let original=source
        let branch=try #require(source.continuation(before:source.messages[2].id))
        #expect(branch.id != source.id);#expect(branch.messages==Array(source.messages.prefix(2)))
        #expect(branch.history.map(\.content)==["first","first answer"])
        #expect(source==original);#expect(branch.title.hasSuffix(" · edited"))
        #expect(source.continuation(before:source.messages[0].id)?.messages.isEmpty==true)
        #expect(source.continuation(before:source.messages[1].id)==nil)
        #expect(source.continuation(before:UUID())==nil)
    }
    @Test func continuationKeepsCancelledTranscriptButDoesNotReplayIt() throws {
        var source=Conversation();source.messages=[.init(role:"user",content:"cancelled"),.init(role:"assistant",content:"partial",complete:false),.init(role:"user",content:"edit this"),.init(role:"assistant",content:"old answer")]
        let branch=try #require(source.continuation(before:source.messages[2].id))
        #expect(branch.messages.count==2);#expect(branch.history.isEmpty)
    }
    @Test func responseMetricsRoundTripAndUseActualForwardSteps() throws {
        var message=ChatMessage(role:"assistant",content:"Hello")
        message.metrics=try #require(ResponseMetrics(outputTokens:10,decodeSteps:9,decodeSeconds:3,contextTokens:79,contextLimit:2048))
        let decoded=try JSONDecoder().decode(ChatMessage.self,from:JSONEncoder().encode(message))
        #expect(decoded==message);#expect(decoded.metrics?.tokensPerSecond==3)
        #expect(ResponseMetrics(outputTokens:1,decodeSteps:0,decodeSeconds:0)?.tokensPerSecond==nil)
        #expect(ResponseMetrics(outputTokens:-1,decodeSteps:0,decodeSeconds:1)==nil)
        #expect(ResponseMetrics(outputTokens:1,decodeSteps:2,decodeSeconds:1)==nil)
        #expect(ResponseMetrics(outputTokens:1,decodeSteps:1,decodeSeconds:.infinity)==nil)
    }
    @Test func legacyMessagesHaveUnknownCountsRatherThanInventedNumbers() throws {
        let data=Data(("{\"id\":\""+UUID().uuidString+"\",\"role\":\"assistant\",\"content\":\"Hello\",\"complete\":true}").utf8)
        let message=try JSONDecoder().decode(ChatMessage.self,from:data)
        #expect(message.metrics==nil)
    }
}
