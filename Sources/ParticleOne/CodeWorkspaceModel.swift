import AppKit
import ParticleOneCore
import ParticleDesktopProtocol

@MainActor final class CodeWorkspaceModel:ObservableObject {
    weak var host:WorkspaceModel?
    @Published var pendingSessionDeletion:UUID?
    @Published var draft=""
    @Published var stream=""
    @Published var active=false
    @Published var waitingForModel=false
    @Published var planOnly=false
    @Published var allowProjectEdits=false
    @Published var pending:AgentEvent?
    @Published var proposal:AgentChange?
    @Published var status="Choose a project to begin."
    @Published var error:String?
    @Published var files=[String]()
    @Published var selectedFile:String?
    @Published var editor=""
    @Published var editorSaved=""
    @Published var editorConflict=false
    private var draftSaveTask:Task<Void,Never>?
    @Published var terminalOutput=""
    @Published var panel="Files"
    @Published var inspectingChange:AgentChange?
    @Published var contextUsed=0
    @Published var inspectorVisible=false
    @Published var maximumSteps=48
    @Published var newContextTokens=0
    @Published var reusedContextTokens=0
    @Published var contextProgress=0.0
    @Published var startedAt:Date?
    @Published var phase=""
    private var verificationRepairs=0,runStartedEventCount=0
    var verification:AgentVerification{AgentVerification.state(events:session?.events ?? [])}
    var recentProjects:[String]{Array(Set(sessions.map(\.project))).sorted()}
    func showPanel(_ name:String){panel=name;inspectorVisible=true}
    func switchProject(_ path:String){openProject(URL(fileURLWithPath:path))}
    func runChecks(){
        guard !active,project != nil else{return}
        planOnly=false;draft="Run the relevant tests or a runnable example for the current project changes. Inspect the project to choose the right command. Use run_command, check its actual exit code and output, and report the result briefly. Do not repeat source files."
        start()
    }
    func openInBrowser(){guard !active,let project,let selectedFile,["html","htm"].contains(URL(fileURLWithPath:selectedFile).pathExtension.lowercased()),(try? project.readText(selectedFile,remember:false)) != nil else{return};NSWorkspace.shared.open(project.root.appendingPathComponent(selectedFile))}

    private(set) var project:AgentProject?
    private var queue=[AgentCall](),rounds=0,repairs=0,repeats=[String:Int]()
    private var task:Task<Void,Never>?,cancellation=AgentCancellation(),stopping=false,toolRunning=false
    private let web=AgentWeb()
    var session:AgentSession? {guard let host else{return nil};return host.state.agentSessions.first{$0.id==host.state.selectedAgentSession}}
    var sessions:[AgentSession]{host?.state.agentSessions ?? []}
    func mutate(_ change:(inout AgentSession)->Void){guard let host,let index=host.state.agentSessions.firstIndex(where:{$0.id==host.state.selectedAgentSession}) else{return};change(&host.state.agentSessions[index]);host.state.agentSessions[index].updated=Date();host.persist();objectWillChange.send()}
    private func canLeaveEditor()->Bool{if editor != editorSaved{error="Save or discard the open file before changing context.";return false};return true}
    func restoreSelection(){guard project==nil,let value=session else{return};select(value.id)}
    func workspaceChanged(){guard !active else{return};runAfterLoad=nil;project=nil;selectedFile=nil;editor="";editorSaved="";editorConflict=false;draft="";files=[];allowProjectEdits=false;error=nil;stream="";status="Choose a project to begin."}
    func chooseProject(){guard canLeaveEditor(),!active else{return};let picker=NSOpenPanel();picker.canChooseDirectories=true;picker.canChooseFiles=false;picker.message="Choose the project Particle Code may inspect and edit. Commands and web access ask separately.";if picker.runModal() == .OK,let url=picker.url{openProject(url)}}
    func openProject(_ url:URL){
        guard canLeaveEditor(),!active else{return}
        do{let value=try AgentProject(root:url);let unsent=draft;project=value;allowProjectEdits=false;selectedFile=nil;editor="";editorSaved="";files=[];error=nil;newSession();draft=unsent;refreshFiles();status="Ready in \(url.lastPathComponent)."}catch{self.error=error.localizedDescription}
    }
    func newSession(){guard canLeaveEditor(),!active,let host else{return};guard let project else{chooseProject();return};self.project=try? AgentProject(root:project.root);selectedFile=nil;editor="";editorSaved="";flushDraft();draft="";let value=AgentSession(project:project.root.path);host.state.agentSessions.insert(value,at:0);host.state.selectedAgentSession=value.id;stream="";error=nil;contextUsed=0;newContextTokens=0;reusedContextTokens=0;terminalOutput="";inspectorVisible=false;host.persist();objectWillChange.send()}
    func select(_ id:UUID){guard canLeaveEditor(),!active,let host,let value=host.state.agentSessions.first(where:{$0.id==id}) else{return};do{let selectedProject=try AgentProject(root:URL(fileURLWithPath:value.project));if project != nil{flushDraft()};project=selectedProject;host.state.selectedAgentSession=id;mutate{session in
            for index in session.events.indices where ["pending","running","approval"].contains(session.events[index].status){session.events[index].status="interrupted"}
            if session.entries.last?.role=="assistant",session.entries.last?.text.trimmingCharacters(in:.whitespacesAndNewlines).hasPrefix("<tool_call>")==true{session.entries.append(AgentEntry("tool","{\"success\":false,\"output\":\"Session interrupted. Pending tools were not resumed. An interrupted operation may have changed files; inspect before continuing.\"}"))}
        };selectedFile=nil;editor="";editorSaved="";stream="";allowProjectEdits=false;draft=value.composerDraft ?? "";if let file=value.editorDraftFile,let text=value.editorDraftText,let project {let current=try project.readText(file);selectedFile=file;editorSaved=current;editor=text;editorConflict=value.editorBaseHash != AgentProject.hash(Data(current.utf8));if editorConflict{self.error="The file changed while Particle was closed. Your draft is retained; copy it if needed, then discard and reload before saving."};showPanel("Editor")};host.persist();refreshFiles();objectWillChange.send()}catch{self.error=error.localizedDescription}}
    func deleteSession(_ id:UUID){
        guard !active,canLeaveEditor(),let host else{return}
        let deletingSelected=host.state.selectedAgentSession==id
        host.state.agentSessions.removeAll{$0.id==id}
        if deletingSelected{
            host.state.selectedAgentSession=nil;project=nil;selectedFile=nil;editor="";editorSaved="";draft=""
            if let next=host.state.agentSessions.first?.id{select(next)}else{workspaceChanged()}
        }
        host.persist();objectWillChange.send()
    }

    func refreshFiles(){guard let project else{return};Task{do{let list=try await Task.detached{try project.files(recursive:true)}.value;if self.project?.root==project.root{files=list}}catch{self.error=error.localizedDescription}}}
    func openFile(_ path:String){guard canLeaveEditor(),!active,let project else{return};do{let text=try project.readText(path);selectedFile=path;editor=text;editorSaved=text;editorConflict=false;showPanel("Editor")}catch{self.error=error.localizedDescription}}
    func saveEditor(){guard !editorConflict,!active,let project,let path=selectedFile else{return};do{let change=try project.propose(AgentCall(name:"write_file",arguments:["path":path,"content":editor]));try project.apply(change);mutate{$0.changes.append(change);$0.events.append(AgentEvent(.init(name:"write_file",arguments:["path":path]),status:"done",output:"Saved in the editor; run checks to verify this change."))};editorSaved=editor;flushDraft();refreshFiles()}catch{self.error=error.localizedDescription}}
    func scheduleDraftSave(){draftSaveTask?.cancel();draftSaveTask=Task{try? await Task.sleep(for:.milliseconds(350));guard !Task.isCancelled else{return};flushDraft()}}
    func flushDraft(){mutate{session in session.composerDraft=draft;session.editorDraftFile=editor != editorSaved ? selectedFile : nil;session.editorDraftText=editor != editorSaved ? editor : nil;session.editorBaseHash=editor != editorSaved ? AgentProject.hash(Data(editorSaved.utf8)) : nil}}
    func discardEditor(){editor=editorSaved;editorConflict=false;flushDraft()}
    func undo(_ change:AgentChange){guard !active,let project else{return};do{try project.apply(change,undo:true);mutate{if let index=$0.changes.firstIndex(where:{$0.id==change.id}){$0.changes[index].undone=true};$0.events.append(AgentEvent(.init(name:"edit_file",arguments:["path":change.path]),status:"done",output:"Undid the reviewed change; previous checks describe the earlier file state."))};if selectedFile==change.path{editor=change.before ?? "";editorSaved=editor};refreshFiles()}catch{self.error=error.localizedDescription}}
    func enableCodingContext(){guard let host,!active,!host.busy,!host.loading else{return};var value=host.settings;value.context=max(8192,ParticleContextPolicy.recommended());host.updateSettings(value);error=nil}
    private var runAfterLoad:UUID?
    func loadAndRun(){guard let host,!host.loading else{return};runAfterLoad = draft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty ? nil : session?.id;host.loadModel()}
    func modelReady(){guard let expected=runAfterLoad else{return};runAfterLoad=nil;guard expected==session?.id else{return};Task{await Task.yield();start()}}
    func start(){
        guard canLeaveEditor(),let host,host.connected,!host.busy,!host.loading,!active,let project else{return}
        guard host.settings.context>=4096 else{error="Coding tools need a larger context. Enable coding context below, or choose 4096 or more tokens in Settings.";return}
        let text=draft.trimmingCharacters(in:.whitespacesAndNewlines);guard !text.isEmpty else{return}
        guard text.utf8.count<=800_000 else{error="This task is too large. Try a smaller excerpt.";return}
        if session==nil{newSession()}
        draft="";active=true;host.busy=true;stopping=false;error=nil;rounds=0;repairs=0;verificationRepairs=0;runStartedEventCount=session?.events.count ?? 0;repeats=[:];cancellation=AgentCancellation();startedAt=Date();phase="Preparing project";status="Preparing \(project.root.lastPathComponent)…"
        task=Task{[weak self] in
            let context=try? await Task.detached{try project.startupContext()}.value
            guard let self else{return}
            guard !stopping,!Task.isCancelled else{finish();return}
            var entry=AgentEntry("user",text);entry.projectContext=context
            mutate{if $0.entries.isEmpty{$0.title=String(text.prefix(48))};$0.entries.append(entry);$0.composerDraft=nil}
            nextModelTurn()
        }
    }

    private func nextModelTurn(){
        guard !stopping,let host,let session,let project else{finish();return}
        guard rounds<maximumSteps else{finish("Reached this run’s \(maximumSteps)-step limit. Review the activity, then send another instruction to continue.");return}
        rounds+=1;waitingForModel=true;stream="";phase="Preparing context";status="Preparing Churo Code…";host.busy=true
        let tools=planOnly ? AgentTools.catalog.filter{!["write_file","edit_file","run_command"].contains($0.name)} : AgentTools.catalog
        host.sendCommand(DesktopCommand("agent",messages:session.history,system:AgentTools.prompt(project:project.root.path,planOnly:planOnly,style:host.settings.codeResponseStyle)+"\nUser preferences: "+host.settings.system,reasoningBudget:host.settings.codeResponseStyle.budget,tools:tools))
    }
    func handle(_ event:[String:Any])->Bool{
        guard active,let type=event["event"] as? String else{return false}
        switch type{
        case "text":if waitingForModel{stream+=event["text"] as? String ?? "";phase="Working";status="Churo Code is working…"};return true
        case "agent_context":
            newContextTokens=event["new_tokens"] as? Int ?? 0;reusedContextTokens=event["reused"] as? Int ?? 0;contextProgress=0;phase="Reading context";status="Reading \(newContextTokens) new tokens · \(reusedContextTokens) cached";return true
        case "prefill":let done=event["completed"] as? Int ?? 0,total=max(1,event["total"] as? Int ?? 1);contextProgress=min(1,Double(done)/Double(total));phase="Reading context";status="Reading context · \(done)/\(total) · \(event["reused"] as? Int ?? 0) cached";return true
        case "input":return true
        case "turn":
            guard waitingForModel else{return true};waitingForModel=false
            if stopping{finish();return true}
            let text=event["text"] as? String ?? stream;contextUsed=event["context_used"] as? Int ?? 0
            var entry=AgentEntry("assistant",text)
            if let tokens=event["tokens"] as? Int,let steps=event["decode_steps"] as? Int,let seconds=event["decode_seconds"] as? Double{entry.metrics=ResponseMetrics(outputTokens:tokens,decodeSteps:steps,decodeSeconds:seconds,contextTokens:contextUsed,contextLimit:host?.settings.context)}
            mutate{$0.entries.append(entry)};stream=""
            guard event["finish"] as? String=="stop" else{finish("The reply reached its output or reasoning limit. Incomplete tool calls were not executed. Send a narrower instruction to continue.");return true}
            do{queue=try AgentCalls.parse(text);if queue.isEmpty{
                if !planOnly,[AgentVerification.needed,.failed].contains(verification),verificationRepairs<2,(session?.events.dropFirst(runStartedEventCount).contains{["write_file","edit_file"].contains($0.call.name) && $0.status=="done"} ?? false) {
                    verificationRepairs+=1
                    mutate{if !$0.entries.isEmpty{$0.entries[$0.entries.count-1].displayHidden=true}}
                    observation("The current changes still need a passing check. Use run_command for relevant tests or a runnable example; inspect and fix failures. Do not repeat the code. If no executable check exists, explain the exact blocker instead of claiming it ran.",success:false)
                    nextModelTurn()
                }else{finish()}
            }else{mutate{if !$0.entries.isEmpty{$0.entries[$0.entries.count-1].displayHidden=true}};nextTool()}}
            catch{repairs+=1;if repairs<=2{observation("Invalid tool format: \(error.localizedDescription) Output only a complete native tool call, or give a final answer.",success:false);nextModelTurn()}else{finish("Churo Code could not form a valid tool request. Nothing from that reply was executed. Six experts is recommended for coding-tool reliability.")}}
            return true
        case "error":waitingForModel=false;finish(event["message"] as? String ?? "The model stopped.");return true
        case "cancelled":waitingForModel=false;finish();return true
        case "ready":return true
        default:return false
        }
    }
    private func nextTool(){
        guard !stopping,let call=queue.first else{if stopping{finish()}else{nextModelTurn()};return}
        queue.removeFirst();let signature=call.name+call.arguments.sorted(by:{$0.key<$1.key}).map{$0.key+"="+$0.value}.joined()
        repeats[signature,default:0]+=1
        guard repeats[signature,default:0]<=2 else{finish("Stopped a repeated tool loop. Review the tool results and refine the request.");return}
        let event=AgentEvent(call);mutate{$0.events.append(event)};pending=event;proposal=nil
        if planOnly && ["write_file","edit_file","run_command"].contains(call.name){completeTool("Plan mode blocks edits and terminal commands.",success:false);return}
        if ["write_file","edit_file"].contains(call.name){
            do{guard let project else{throw AgentFailure.message("Project unavailable.")};let change=try project.propose(call)
                guard (session?.changes.reduce(0,{$0+($1.before?.utf8.count ?? 0)+$1.after.utf8.count}) ?? 0)+change.after.utf8.count+(change.before?.utf8.count ?? 0)<=32*1024*1024 else{throw AgentFailure.message("Change history reached 32 MiB. Start a fresh coding session.")}
                proposal=change
                if allowProjectEdits{approve()}else{status="Review file change";updateEvent("approval",output:AgentDiff.text(before:change.before,after:change.after))}
            }catch{completeTool(error.localizedDescription,success:false)}
        }else if ["run_command","web_search","fetch_url"].contains(call.name){status=call.name=="run_command" ? "Approve terminal command" : "Approve public web request";updateEvent("approval")}
        else{approve()}
    }
    func approve(_ expected:UUID?=nil){
        guard active,!stopping,!toolRunning,let event=pending,event.status != "running",expected==nil || expected==event.id,let project else{return}
        pending?.status="running"
        let call=event.call;updateEvent("running");phase="Using tools";status="Running \(call.name.replacingOccurrences(of:"_",with:" "))…";if call.name=="run_command"{showPanel("Terminal")}
        if let change=proposal {
            do{try project.apply(change);mutate{$0.changes.append(change)};completeTool("Applied \(change.path). SHA256: \(change.afterHash)",success:true);refreshFiles()}catch{completeTool(error.localizedDescription,success:false)}
            return
        }
        toolRunning=true;terminalOutput="";let cancellation=self.cancellation
        let onOutput:@Sendable(String)->Void={ [weak self] chunk in
            Task{@MainActor [weak self] in
                guard let self,toolRunning,pending?.id==event.id else{return}
                terminalOutput+=chunk
                if terminalOutput.count>65_536{terminalOutput=String(terminalOutput.suffix(65_536))}
            }
        }
        task=Task{[weak self] in
            guard let self else{return}
            do{
                let output:String,success:Bool
                if ["web_search","fetch_url"].contains(call.name){output=try await web.run(call);success=true}
                else{
                    let result=try await Task.detached {()->(String,Bool) in
                        switch call.name{
                        case "list_files":return (try project.files(path:call.arguments["path"] ?? ".",recursive:call.arguments["recursive"]=="true").prefix(300).joined(separator:"\n"),true)
                        case "read_file":return (try project.read(call.arguments["path"] ?? "",start:Int(call.arguments["start_line"] ?? "1") ?? 1,end:Int(call.arguments["end_line"] ?? "120") ?? 120),true)
                        case "search_files":return (try project.search(call.arguments["query"] ?? "",path:call.arguments["path"] ?? "."),true)
                        case "run_command":
                            let result=try AgentCommand.run(call.arguments["command"] ?? "",root:project.root,timeout:Double(call.arguments["timeout_seconds"] ?? "60") ?? 60,cancellation:cancellation,onOutput:onOutput)
                            return ("Exit code: \(result.exitCode)\n"+result.output,result.exitCode==0 && !result.timedOut && !result.cancelled)
                        default:throw AgentFailure.message("Unknown tool.")
                        }
                    }.value
                    output=result.0;success=result.1
                }
                toolRunning=false;if call.name=="run_command"{terminalOutput=output};if stopping{updateEvent("cancelled",output:output);finish()}else{completeTool(output,success:success)}
            }catch{toolRunning=false;if stopping{finish()}else{completeTool(error.localizedDescription,success:false)}}
        }
    }
    func deny(_ expected:UUID?=nil){guard let pending,!toolRunning,expected==nil || expected==pending.id else{return};completeTool("User denied this operation. Do not retry it without a new instruction.",success:false)}
    private func updateEvent(_ state:String,output:String="") {guard let id=pending?.id else{return};mutate{if let index=$0.events.firstIndex(where:{$0.id==id}){$0.events[index].status=state;if !output.isEmpty{$0.events[index].output=String(output.prefix(12_000))}}}}
    private func completeTool(_ output:String,success:Bool){updateEvent(success ? "done" : "failed",output:output);observation(output,success:success);pending=nil;proposal=nil;if stopping{finish()}else{nextTool()}}
    private func observation(_ output:String,success:Bool){
        let object:[String:Any]=["success":success,"tool":pending?.call.name ?? "protocol","output":String(output.prefix(6000)),"truncated":output.count>6000]
        let data=(try? JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])) ?? Data()
        let text=String(decoding:data,as:UTF8.self).replacingOccurrences(of:"<",with:"\\u003c").replacingOccurrences(of:"[@BOS@]",with:"[BOS text]")
        mutate{$0.entries.append(AgentEntry("tool",text))}
    }
    func stop(){guard active else{return};stopping=true;status="Stopping safely…";cancellation.cancel();task?.cancel();if waitingForModel{host?.interruptModel()}else if !toolRunning{finish()}}
    func cancelForShutdown(){runAfterLoad=nil;draftSaveTask?.cancel();flushDraft();cancellation.cancel();task?.cancel()}
    private func finish(_ failure:String?=nil){
        if stopping,session?.entries.last?.role=="assistant",(session?.entries.last?.text.hasPrefix("<tool_call>") ?? false){observation("Cancelled by the user.",success:false)}
        phase="";active=false;waitingForModel=false;pending=nil;proposal=nil;queue=[];stream="";host?.busy=false;error=failure;status=failure==nil ? (stopping ? "Stopped." : "Ready for the next instruction.") : "Needs attention";host?.persist();objectWillChange.send()
    }
}
