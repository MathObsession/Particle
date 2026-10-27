import Foundation
import AppKit
import ParticleOneCore
import ParticleDesktopProtocol

@MainActor enum SmokeCheck {
    static func nativeAgent(_ model:WorkspaceModel,output:String)async{
        guard model.directory.path.hasPrefix("/private/tmp/particle-agent-native-"),ProcessInfo.processInfo.environment["PARTICLEONE_ENGINE_PATH"]==nil else{NSApp.terminate(nil);return}
        var checks=[String:Bool]()
        func wait(_ seconds:Double,_ condition:@MainActor ()->Bool)async->Bool{let end=ProcessInfo.processInfo.systemUptime+seconds;while ProcessInfo.processInfo.systemUptime<end{if condition(){return true};try? await Task.sleep(for:.milliseconds(150))};return false}
        model.state.settings.modelPath=WorkspacePersistence.defaultDirectory.appendingPathComponent("Models/sarvam-30b-q4_k_m.particle").path
        if FileManager.default.fileExists(atPath:WorkspacePersistence.defaultDirectory.appendingPathComponent("Models/churo-chat.particle").path){model.state.settings.modelPath=WorkspacePersistence.defaultDirectory.appendingPathComponent("Models/churo-chat.particle").path}
        model.state.settings.useMirror=false;model.state.settings.context=8192;model.state.settings.maxNew=256;model.state.settings.activeExperts=6;model.state.settings.responseStyle = .direct;model.loadModel()
        checks["model_ready"]=await wait(90){model.connected || model.notice != nil} && model.connected
        if model.connected{
            model.draft="What is your name and who created this assistant? Answer in one short sentence.";model.submit()
            checks["chat_identity_reply"]=await wait(120){!model.busy}
            let identity=model.selected?.messages.last?.content ?? ""
            checks["chat_brand_identity"]=identity.localizedCaseInsensitiveContains("Churo Chat") && identity.contains("Lakshya") && !identity.contains("Sarvam")
            if CommandLine.arguments.contains("--identity-only"){
                model.unload();_=await wait(20){!model.connected && !model.loading}
                let report:[String:Any]=["passed":checks.values.allSatisfy{$0},"checks":checks,"reply":identity,"mode":"Real Churo Chat identity; isolated data; one model"]
                if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))};NSApp.terminate(nil);return
            }
            let project=model.directory.appendingPathComponent("fixture")
            try? FileManager.default.createDirectory(at:project,withIntermediateDirectories:true)
            try? "mode=red\n".write(to:project.appendingPathComponent("main.txt"),atomically:true,encoding:.utf8)
            try? "Only modify main.txt in this temporary fixture. Verify with the exact command requested by the user.\n".write(to:project.appendingPathComponent("AGENTS.md"),atomically:true,encoding:.utf8)
            let pid=model.modelProcessID
            model.setCodeMode(true);model.code.openProject(project)
            model.code.draft="Read main.txt. Use edit_file to replace mode=red with mode=green. Then use run_command with exactly this command: /usr/bin/grep -q 'mode=green' main.txt && /bin/echo verified\nThen finish with a short summary. Do not modify other files."
            model.code.start()
            let end=ProcessInfo.processInfo.systemUptime+420
            var approvedEdits=0,approvedCommands=0,handled=Set<UUID>()
            while model.code.active && ProcessInfo.processInfo.systemUptime<end{
                if let pending=model.code.pending,handled.insert(pending.id).inserted{
                    if ["write_file","edit_file"].contains(pending.call.name),let change=model.code.proposal,change.path=="main.txt",change.after=="mode=green\n"{approvedEdits+=1;model.code.approve(pending.id)}
                    else if pending.call.name=="run_command",let command=pending.call.arguments["command"],["/usr/bin/grep -q 'mode=green' main.txt && /bin/echo verified","/usr/bin/grep -q \"mode=green\" main.txt && /bin/echo verified","grep -q 'mode=green' main.txt && echo verified","cat main.txt","/bin/cat main.txt"].contains(command){approvedCommands+=1;model.code.approve(pending.id)}
                    else if ["write_file","edit_file","run_command","web_search","fetch_url"].contains(pending.call.name){model.code.deny(pending.id)}
                }
                try? await Task.sleep(for:.milliseconds(150))
            }
            if model.code.active{model.code.stop();_=await wait(20){!model.code.active}}
            checks["same_warm_worker"]=pid==model.modelProcessID
            checks["real_edit_approved"]=approvedEdits==1
            checks["real_command_approved"]=approvedCommands>0
            checks["file_changed_correctly"]=(try? String(contentsOf:project.appendingPathComponent("main.txt"),encoding:.utf8))=="mode=green\n"
            checks["native_read_tool"]=model.code.session?.events.contains{$0.call.name=="read_file" && $0.status=="done"}==true
            checks["native_edit_tool"]=model.code.session?.events.contains{$0.call.name=="edit_file" && $0.status=="done"}==true
            checks["native_command_tool"]=model.code.session?.events.contains{$0.call.name=="run_command" && $0.status=="done" && ($0.output.contains("verified") || $0.output.contains("mode=green"))}==true
            checks["agent_finished"] = !model.code.active && model.code.error==nil
            if let change=model.code.session?.changes.last{model.code.undo(change);checks["native_undo"]=(try? String(contentsOf:project.appendingPathComponent("main.txt"),encoding:.utf8))=="mode=red\n"}
        }
        model.unload();_=await wait(20){!model.connected && !model.loading}
        let report:[String:Any]=["passed":!checks.isEmpty && checks.values.allSatisfy{$0},"checks":checks,"mode":"One real local model; isolated temporary project and data; native six-expert profile; no speed qualification","error":model.code.error ?? model.notice ?? "","diagnostics":model.diagnostics]
        if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))};NSApp.terminate(nil)
    }
    static func web(output:String)async{
        var result:[String:Any]=["passed":false,"mode":"Public documentation web tools only; no model"]
        do{let web=AgentWeb();let search=try await web.run(AgentCall(name:"web_search",arguments:["query":"Apple SwiftUI official documentation"]));result["search"]=search;let page=try await web.run(AgentCall(name:"fetch_url",arguments:["url":"https://www.swift.org/documentation/"]));result=["passed":search.contains("https://") && !page.isEmpty,"search":search,"page":page,"mode":"Actual public Exa search and page fetch"]}catch{result["error"]=error.localizedDescription}
        if let data=try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))};NSApp.terminate(nil)
    }
    static func installer(output: String) async {
        var result:[String:Any]=["passed":false,"mode":"pinned public release and small-file resume check; no full model copy"]
        do {
            guard let catalogURL=Bundle.module.url(forResource:"model-catalog",withExtension:"json",subdirectory:"Resources") else {throw ReleaseError.notPublished}
            let catalog=try JSONDecoder().decode(ModelCatalog.self,from:Data(contentsOf:catalogURL));try catalog.validate()
            let base=URL(string:"https://huggingface.co/\(catalog.repo)/resolve/\(catalog.revision)/")!
            let (data,response)=try await URLSession.shared.data(from:base.appendingPathComponent("particle-distribution.json"))
            guard (response as? HTTPURLResponse)?.statusCode==200,ModelRelease.digest(data)==catalog.distributionSHA256 else {throw ReleaseError.invalidManifest}
            let release=try JSONDecoder().decode(ModelRelease.self,from:data);try release.validate(catalog:catalog)
            guard let file=release.files.first(where:{$0.path=="tokenizer/chat_template.jinja"}) else {throw ReleaseError.invalidManifest}
            let url=base.appendingPathComponent(catalog.folder).appendingPathComponent(file.path)
            let (full,_)=try await URLSession.shared.data(from:url)
            guard ModelRelease.digest(full)==file.sha256 else {throw ReleaseError.checksum(file.path)}
            let folder=FileManager.default.temporaryDirectory.appendingPathComponent("particleone-download-test-"+UUID().uuidString)
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:folder)}
            let part=folder.appendingPathComponent("template.part");try full.prefix(100).write(to:part)
            let transfer=ReleaseTransfer(destination:part,expected:file.bytes,progress:{_ in})
            let fetched=try await transfer.start(url)
            let verified=try ModelRelease.digestFile(fetched)==file.sha256
            result=["passed":verified,"revision":catalog.revision,"file":file.path,"bytes":file.bytes,"resumed_prefix_bytes":100,"public_without_token":true,"mode":"small-file resume; no full model copy"]
        } catch {result["error"]=error.localizedDescription}
        if let data=try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))}
        NSApp.terminate(nil)
    }
    static func run(_ model: WorkspaceModel,output: String) async {
        var checks=[String:Bool]()
        func wait(_ condition: @MainActor ()->Bool) async -> Bool {
            for _ in 0..<160 {if condition(){return true};try? await Task.sleep(for:.milliseconds(50))};return false
        }
        guard ProcessInfo.processInfo.environment["PARTICLEONE_ENGINE_PATH"] != nil,model.directory.path.hasPrefix("/private/tmp/particleone-smoke-") else {NSApp.terminate(nil);return}
        model.state.settings.modelPath=model.directory.appendingPathComponent("fixture.gguf").path;model.state.settings.useMirror=false;model.persist()
        let markdown=ParticleMarkdown.render("# A heading\n\n**Bold** and *italic* with `inline code`.\n\n1. First item\n2. Second item\n\n> A quote\n\n| Feature | State |\n| --- | --- |\n| Markdown | Enabled |\n\n[Website](https://example.com)")
        checks["markdown_content"]=markdown.string.contains("A heading") && markdown.string.contains("Markdown") && !markdown.string.contains("**Bold**") && !markdown.string.contains("# A heading")
        var serif=false,mono=false,bold=false,table=false,safeLink=false
        markdown.enumerateAttributes(in:NSRange(location:0,length:markdown.length)) { attributes,_,_ in
            if let font=attributes[.font] as? NSFont {
                serif = serif || font.fontName.hasPrefix("PTSerif")
                mono = mono || font.fontName.hasPrefix("SpaceMono")
                bold = bold || font.fontName=="Poppins-SemiBold"
            }
            if let paragraph=attributes[.paragraphStyle] as? NSParagraphStyle {table = table || !paragraph.textBlocks.isEmpty}
            if let url=attributes[.link] as? URL {safeLink = safeLink || url.scheme=="https"}
        }
        checks["markdown_serif_headings"]=serif;checks["markdown_mono_inline_code"]=mono;checks["markdown_bold"]=bold;checks["markdown_table"]=table;checks["markdown_link"]=safeLink
        let thinking=ParticleMarkdown.render("1. **Analyze the input**\n\n- The greeting is **Hi!**.",compact:true)
        checks["thinking_markdown"]=thinking.string.contains("Analyze the input") && !thinking.string.contains("**")
        model.loadModel();checks["worker_ready"]=await wait{model.connected}
        model.draft="Hi!";model.submit();checks["greeting_complete"]=await wait{model.selected?.messages.last?.complete==true && !model.busy}
        checks["greeting_input_verified"]=model.inputVerified==true
        checks["greeting_visible_exact"]=model.selected?.messages.first?.content=="Hi!"
        model.newConversation()
        model.draft="Hello\nनमस्ते 🌿";model.submit();checks["stream_complete"]=await wait{model.selected?.messages.last?.complete==true && !model.busy}
        checks["unicode_multiline"]=model.selected?.messages.first?.content=="Hello\nनमस्ते 🌿"
        checks["code_received"]=model.selected?.messages.last?.content.contains("```python")==true
        let warmPID=model.modelProcessID
        model.setThinkingEnabled(true)
        checks["composer_thinking_on"]=model.thinkingEnabled && model.settings.responseStyle == .brief
        checks["thinking_keeps_worker"]=model.connected && !model.loading && model.modelProcessID==warmPID
        checks["thinking_preserves_history"]=model.selected?.history.count==2
        checks["thinking_command_on"]=await wait{
            guard let text=try? String(contentsOf:model.directory.appendingPathComponent("commands.jsonl"),encoding:.utf8) else{return false}
            return text.split(separator:"\n").contains{line in (try? JSONDecoder().decode(DesktopCommand.self,from:Data(line.utf8)))?.reasoningBudget==64}
        }
        model.setThinkingEnabled(false)
        checks["composer_thinking_off"] = !model.thinkingEnabled && model.settings.responseStyle == .direct && model.modelProcessID==warmPID
        model.draft="reasoning-limit";model.submit();checks["thinking_restart_stopped"]=await wait{model.selected?.messages.last?.finish=="reasoning_limit" && !model.busy}
        checks["thinking_failure_not_replayed"]=model.selected?.history.count==2
        let first=model.state.selected;model.newConversation();if let first{model.select(first)}
        checks["history_retained"]=model.selected?.history.count==2
        model.draft="slow";model.submit();checks["slow_started"]=await wait{model.selected?.messages.last?.content.contains("working")==true}
        model.setThinkingEnabled(true);checks["thinking_change_blocked_mid_reply"] = !model.thinkingEnabled
        if let chat=model.selected,let question=chat.messages.first {
            let count=model.state.conversations.count
            checks["edit_blocked_mid_reply"] = !model.editAndRegenerate(conversationID:chat.id,messageID:question.id,text:"edited while busy") && model.state.conversations.count==count
        }
        model.stop();checks["cancelled"]=await wait{model.selected?.messages.last?.finish=="cancelled" && !model.busy}
        checks["incomplete_not_replayed"]=model.selected?.history.count==2
        model.setExpertCount(5);checks["expert_five_reload"]=await wait{model.connected && !model.loading && model.settings.activeExperts==5}
        checks["expert_reload_preserves_history"]=model.selected?.history.count==2
        model.setExpertCount(6);checks["expert_six_reload"]=await wait{model.connected && !model.loading && model.settings.activeExperts==6}
        let previousWorkspace=model.state.id
        let previousChat=model.state.selected
        model.saveMemory("My name is Mira")
        checks["memory_saved"]=model.state.memories.count==1 && model.state.systemPrompt.contains("My name is Mira")
        model.saveMemory("My name is Mira");checks["memory_deduplicated"]=model.state.memories.count==1
        model.editMemory(model.state.memories[0].id,text:"My name is Mina");checks["memory_edited"]=model.state.systemPrompt.contains("My name is Mina")
        model.createWorkspace("Writing")
        checks["new_workspace_empty"]=model.state.id != previousWorkspace && model.state.memories.isEmpty && model.selected?.messages.isEmpty==true
        model.openWorkspace(previousWorkspace)
        checks["workspace_restore"]=model.state.id==previousWorkspace && model.state.selected==previousChat && model.state.memories.count==1
        model.removeMemory(model.state.memories[0].id);checks["memory_forgotten"] = !model.state.systemPrompt.contains("My name is Mina")
        model.toggleSidebar();checks["sidebar_collapsed"]=model.state.sidebarHidden;model.toggleSidebar()
        model.setUsername("Curious Maker");checks["username_saved"]=model.state.username=="Curious Maker"
        if let question=model.selected?.messages.first {model.removeMessage(question.id)}
        checks["message_delete_rebuilds_history"]=model.selected?.history.isEmpty==true
        let editor=SendTextView(frame:NSRect(x:0,y:0,width:300,height:100));editor.isRichText=false;editor.string="Hello";editor.setSelectedRange(NSRange(location:5,length:0))
        var sends=0;editor.onSend={sends+=1}
        if let enter=NSEvent.keyEvent(with:.keyDown,location:.zero,modifierFlags:[],timestamp:0,windowNumber:0,context:nil,characters:"\r",charactersIgnoringModifiers:"\r",isARepeat:false,keyCode:36){editor.keyDown(with:enter)}
        checks["enter_sends"]=sends==1 && editor.string=="Hello"
        if let shift=NSEvent.keyEvent(with:.keyDown,location:.zero,modifierFlags:.shift,timestamp:0,windowNumber:0,context:nil,characters:"\r",charactersIgnoringModifiers:"\r",isARepeat:false,keyCode:36){editor.keyDown(with:shift)}
        checks["shift_enter_newline"]=sends==1 && editor.string.contains("\n")
        let syntax=SyntaxHighlighter.render("def is_even(n):\n    return n % 2 == 0 # even",language:"python")
        var syntaxColors=Set<String>();syntax.enumerateAttribute(.foregroundColor,in:NSRange(location:0,length:syntax.length)){value,_,_ in if let color=value as? NSColor{syntaxColors.insert(color.description)}}
        checks["syntax_colors"]=syntaxColors.count>=4
        checks["syntax_preserves_code"]=syntax.string=="def is_even(n):\n    return n % 2 == 0 # even"
        checks["sqlite_saved"]=FileManager.default.fileExists(atPath:model.directory.appendingPathComponent("particleone.sqlite").path)
        model.newConversation();model.draft="First prompt";model.submit();checks["edit_setup_first"]=await wait{!model.busy && model.selected?.messages.last?.complete==true}
        model.draft="Old second question";model.submit();checks["edit_setup_second"]=await wait{!model.busy && model.selected?.messages.last?.complete==true}
        if let source=model.selected,source.messages.count==4 {
            let original=source;let pid=model.modelProcessID;model.draft="Unsent follow-up"
            checks["response_metrics_recorded"]=source.messages[1].metrics?.outputTokens==10 && source.messages[1].metrics?.tokensPerSecond==1
            checks["edit_queued"]=model.editAndRegenerate(conversationID:source.id,messageID:source.messages[2].id,text:"Edited question\nनमस्ते 🌿")
            checks["edit_regenerated"]=await wait{!model.busy && model.selected?.messages.last?.complete==true}
            checks["edit_original_preserved"]=model.state.conversations.first(where:{$0.id==source.id})==original
            checks["edit_new_context"]=model.selected?.history.map(\.content)==["First prompt",source.messages[1].content,"Edited question\nनमस्ते 🌿",source.messages[1].content]
            checks["edit_draft_preserved"]=model.draft=="Unsent follow-up"
            checks["edit_worker_retained"]=model.modelProcessID==pid
            if let text=try? String(contentsOf:model.directory.appendingPathComponent("commands.jsonl"),encoding:.utf8){
                let commands=text.split(separator:"\n").compactMap{try? JSONDecoder().decode(DesktopCommand.self,from:Data($0.utf8))}
                checks["edit_restores_prefix_only"]=commands.last(where:{$0.type=="restore"})?.messages?.map(\.content)==["First prompt",source.messages[1].content]
                checks["edit_text_verified"]=model.inputVerified==true
            }
            let branch=model.state.selected
            model.requestConversationDeletion(source.id);checks["delete_confirmation_target"]=model.pendingConversationDeletion==source.id;model.pendingConversationDeletion=nil
            model.delete(source.id);checks["delete_other_chat"] = !model.state.conversations.contains(where:{$0.id==source.id}) && model.state.selected==branch && model.selected?.history.count==4
            if let branch {model.delete(branch);checks["delete_active_chat"]=model.state.selected != branch && model.selected?.messages.isEmpty==true && model.contextUsed==0}
            checks["delete_persisted"]=(try? WorkspacePersistence.load(from:model.directory).conversations.contains(where:{$0.id==source.id}))==false
        }
        model.unload();checks["unloaded"]=await wait{!model.loading && !model.connected}
        model.loadModel();checks["reloaded"]=await wait{model.connected}
        let fixture=model.directory.appendingPathComponent("agent-project")
        try? FileManager.default.createDirectory(at:fixture,withIntermediateDirectories:true)
        try? "color=red\n".write(to:fixture.appendingPathComponent("main.txt"),atomically:true,encoding:.utf8)
        var codeSettings=model.settings;codeSettings.context=8192;model.updateSettings(codeSettings);_=await wait{model.connected && !model.loading}
        let chatHistory=model.selected?.history
        let agentPID=model.modelProcessID
        model.setCodeMode(true)
        model.setThinkingEnabled(true)
        checks["code_thinking_independent"]=model.thinkingEnabled && model.settings.codeResponseStyle == .brief && model.settings.responseStyle == .direct
        model.code.openProject(fixture);model.code.draft="agent-fixture";model.code.start()
        checks["code_thinking_mid_run_blocked"]={model.setThinkingEnabled(false);return model.settings.codeResponseStyle == .brief}()
        checks["code_thinking_budget_sent"]=await wait{
            guard let text=try? String(contentsOf:model.directory.appendingPathComponent("commands.jsonl"),encoding:.utf8) else{return false}
            return text.split(separator:"\n").contains{line in guard let command=try? JSONDecoder().decode(DesktopCommand.self,from:Data(line.utf8)) else{return false};return command.type=="agent" && command.reasoningBudget==64}
        }
        checks["agent_edit_waits_approval"]=await wait{model.code.proposal != nil}
        checks["agent_file_unchanged_before_approval"]=(try? String(contentsOf:fixture.appendingPathComponent("main.txt"),encoding:.utf8))=="color=red\n"
        let editApproval=model.code.pending?.id
        model.code.approve(editApproval)
        checks["agent_command_waits_approval"]=await wait{model.code.pending?.call.name=="run_command"}
        let commandApproval=model.code.pending?.id
        model.code.approve(editApproval)
        checks["agent_stale_approval_ignored"]=model.code.pending?.id==commandApproval && !FileManager.default.fileExists(atPath:fixture.appendingPathComponent("command-ran.txt").path)
        model.code.approve(commandApproval)
        checks["agent_completed"]=await wait{!model.code.active}
        checks["agent_actual_file_edit"]=(try? String(contentsOf:fixture.appendingPathComponent("main.txt"),encoding:.utf8))=="color=green\n"
        checks["agent_actual_command"]=FileManager.default.fileExists(atPath:fixture.appendingPathComponent("command-ran.txt").path)
        checks["agent_same_worker"]=model.modelProcessID==agentPID
        checks["agent_chat_preserved"]=model.selected?.history==chatHistory
        checks["agent_native_tool_history"]=model.code.session?.history.filter{$0.role=="tool"}.count==3
        if let change=model.code.session?.changes.last{model.code.undo(change);checks["agent_undo"]=(try? String(contentsOf:fixture.appendingPathComponent("main.txt"),encoding:.utf8))=="color=red\n"}
        model.setThinkingEnabled(false)
        checks["code_instant_keeps_worker"]=model.settings.codeResponseStyle == .direct && model.modelProcessID==agentPID
        model.code.newSession();model.code.draft="agent-needs-verification";model.code.start()
        checks["code_instant_budget_sent"]=await wait{
            guard let text=try? String(contentsOf:model.directory.appendingPathComponent("commands.jsonl"),encoding:.utf8) else{return false}
            return text.split(separator:"\n").contains{line in guard let command=try? JSONDecoder().decode(DesktopCommand.self,from:Data(line.utf8)) else{return false};return command.type=="agent" && command.reasoningBudget==0}
        }
        checks["verification_edit_approval"]=await wait{model.code.proposal?.path=="verify.txt"}
        model.code.approve(model.code.pending?.id)
        checks["verification_prompts_for_run"]=await wait{model.code.pending?.call.name=="run_command"}
        checks["verification_premature_final_hidden"]=model.code.session?.entries.contains{$0.displayHidden==true}==true
        model.code.approve(model.code.pending?.id)
        checks["verification_completed"]=await wait{!model.code.active} && model.code.verification == .passed
        let previousSession=model.code.session?.id
        model.code.draft="Unsent project draft"
        let other=fixture.appendingPathComponent("Second project")
        try? FileManager.default.createDirectory(at:other,withIntermediateDirectories:true)
        model.code.switchProject(other.path)
        checks["project_switch_creates_session"]=model.code.project?.root.path==other.standardizedFileURL.resolvingSymlinksInPath().path && model.code.session?.id != previousSession
        checks["project_switch_preserves_old_session"]=model.code.sessions.contains{$0.id==previousSession}
        checks["project_switch_preserves_composer"]=model.code.draft=="Unsent project draft"
        if let selected=model.code.session?.id{model.code.deleteSession(selected)}
        checks["session_delete_restores_project"]=model.code.project?.root.path==fixture.standardizedFileURL.resolvingSymlinksInPath().path && model.code.session?.id==previousSession
        checks["session_delete_restores_draft"]=model.code.draft=="Unsent project draft"
        model.code.openFile("main.txt");model.code.editor+="# local edit\n";model.code.saveEditor()
        checks["editor_save_invalidates_old_check"]=model.code.verification == .needed
        if let change=model.code.session?.changes.last{model.code.undo(change)}
        model.code.switchProject(fixture.path)
        model.code.newSession();model.code.draft="agent-deny";model.code.start()
        checks["agent_new_file_approval"]=await wait{model.code.proposal != nil}
        model.code.deny(model.code.pending?.id);checks["agent_denied_completes"]=await wait{!model.code.active}
        checks["agent_denied_no_write"] = !FileManager.default.fileExists(atPath:fixture.appendingPathComponent("denied.txt").path)
        model.code.newSession();model.code.planOnly=true;model.code.draft="agent-plan";model.code.start()
        checks["agent_plan_finishes"]=await wait{!model.code.active}
        checks["agent_plan_blocks_write"] = !FileManager.default.fileExists(atPath:fixture.appendingPathComponent("denied.txt").path) && model.code.session?.events.last?.status=="failed"
        model.code.planOnly=false;model.code.newSession();model.code.draft="agent-cancel";model.code.start()
        checks["agent_cancel_command_approval"]=await wait{model.code.pending?.call.name=="run_command"}
        model.code.approve(model.code.pending?.id);try? await Task.sleep(for:.milliseconds(100));model.code.stop()
        checks["agent_cancelled"]=await wait{!model.code.active && !model.busy}
        checks["agent_sessions_persisted"]=(try? WorkspacePersistence.load(from:model.directory).agentSessions.count)==model.state.agentSessions.count
        model.setCodeMode(false)
        checks["chat_identity_hidden"]=model.state.systemPrompt.contains("Churo Chat") && model.state.systemPrompt.contains("Lakshya Prajapati") && !(model.selected?.messages.contains{$0.content.contains("Lakshya Prajapati")} ?? true)
        for font in ["PTSerif-Regular","Poppins-Regular","Poppins-Medium","SpaceMono-Regular"]{checks["font_"+font]=NSFont(name:font,size:12) != nil}
        let result:[String:Any] = ["passed":checks.values.allSatisfy{$0},"checks":checks,"mode":"mock worker only; no model or performance claim"]
        if let data=try? JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))}
        NSApp.terminate(nil)
    }
}
