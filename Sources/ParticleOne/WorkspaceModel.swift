import AppKit
import SwiftUI
import ParticleOneCore
import ParticleDesktopProtocol

private final class DesktopPipeReader: @unchecked Sendable {
    private let queue=DispatchQueue(label:"app.particleone.stream")
    private var decoder=DesktopLineDecoder()
    private let receive: @Sendable (Data)->Void
    init(receive: @escaping @Sendable (Data)->Void) { self.receive=receive }
    func accept(_ data: Data) { queue.async { [self] in do { for line in try decoder.append(data) { receive(line) } } catch { receive(Data("{\"event\":\"error\",\"message\":\"The model stream was interrupted. Reload the model.\"}".utf8)) } } }
}

@MainActor final class WorkspaceModel: ObservableObject {
    @Published var state=SavedWorkspace()
    @Published var workspaces=[WorkspaceSummary]()
    @Published var showNewWorkspace=false
    @Published var showMemory=false
    @Published var draft=""
    @Published var search=""
    @Published var busy=false
    @Published var loading=false
    @Published var connected=false
    @Published var status="Model offline"
    @Published var notice: String?
    @Published var pendingConversationDeletion:UUID?
    @Published var updateRestartPending=false
    @Published var showSettings=false
    @Published var showSetup=false
    @Published var rate: Double?
    @Published var contextUsed=0
    @Published var inputVerified: Bool?
    private var expectedInputFingerprint: String?
    @Published var prefillProgress=0.0
    @Published var diagnostics=[String]()
    @Published var codeMode=false
    let code=CodeWorkspaceModel()
    let updates=AppUpdater()
    let installer=ModelInstaller()
    let preview: Bool
    private(set) var directory: URL
    private var saveBlocked=false
    private var unloading=false
    private var expertReloadTask: Task<Void,Never>?
    private var worker: Process?
    private var input: FileHandle?
    private var outputReader: DesktopPipeReader?
    private var errorReader: DesktopPipeReader?
    private var currentReply: UUID?
    private var workerSession=UUID()
    var settings: OneSettings { state.settings }
    var selected: Conversation? { state.conversations.first{$0.id==state.selected} }
    var filtered: [Conversation] { state.conversations.filter { search.isEmpty || $0.title.localizedCaseInsensitiveContains(search) || $0.messages.contains{$0.content.localizedCaseInsensitiveContains(search)} }.sorted{$0.updatedAt>$1.updatedAt} }
    init(preview: Bool = false,chatPreview: Bool = false) {
        self.preview=preview
        let override=ProcessInfo.processInfo.environment["PARTICLEONE_DATA_PATH"]
        let last=UserDefaults.standard.string(forKey:"ParticleOne.lastWorkspaceDirectory").map{URL(fileURLWithPath:$0,isDirectory:true)}
        directory=override.map{URL(fileURLWithPath:$0,isDirectory:true)} ?? last.flatMap{FileManager.default.fileExists(atPath:$0.appendingPathComponent("particleone.sqlite").path) ? $0 : nil} ?? WorkspacePersistence.defaultDirectory
        code.host=self
        if !preview {
            do {state=try WorkspacePersistence.load(from:directory);workspaces=try WorkspacePersistence.list(in:directory)}
            catch {saveBlocked=true;notice="Your saved workspace could not be opened. Saving is paused to preserve the original database and transcripts. Open another workspace folder to continue."}
        }
        if state.conversations.isEmpty { newConversation() }
        else if !state.conversations.contains(where:{$0.id==state.selected}) {state.selected=state.conversations.first?.id}
        if preview && chatPreview {
            var chat=Conversation();chat.title="A small idea, made useful"
            chat.messages=[ChatMessage(role:"user",content:"Write a small Python function that checks if a number is even."),ChatMessage(role:"assistant",content:"Here’s a simple way to do it. The modulo operator gives the remainder after division, so even numbers return zero.\n\n```python\ndef is_even(n):\n    return n % 2 == 0\n\nprint(is_even(12))  # True\n```\n\n## Why this works\n\n- **Readable:** one small function.\n- **Reusable:** works with any integer.\n\n> Even numbers leave no remainder.\n\n[Read about Python](https://docs.python.org/3/)")]
            if CommandLine.arguments.contains("--preview-compact-chat") {chat.messages[1].content="Here’s a simple implementation.\n\n```python\ndef is_even(n):\n    return n % 2 == 0\n```"}
            if CommandLine.arguments.contains("--preview-thinking") {
                chat.messages[1].content="<think>1. **Analyze the input**\n\n- The greeting is **Hi!**.\n- Preserve the user’s exact text.\n\n2. **Choose a response**\n\nKeep the answer brief and friendly.</think>"+chat.messages[1].content
            }
            state.conversations.insert(chat,at:0);state.selected=chat.id
        }
        if preview {if CommandLine.arguments.contains("--preview-thinking-enabled"){state.settings.responseStyle = .brief};status="Design preview";state.username="Curious Maker";state.settings.appearance = CommandLine.arguments.contains("--preview-dark") ? .dark : .light;if CommandLine.arguments.contains("--preview-collapsed"){state.sidebarHidden=true}}
        if preview && CommandLine.arguments.contains("--preview-code"){
            let root=FileManager.default.temporaryDirectory.appendingPathComponent("particle-code-preview-"+UUID().uuidString).appendingPathComponent("Demo Workspace")
            try? FileManager.default.createDirectory(at:root.appendingPathComponent("Sources"),withIntermediateDirectories:true)
            let source="func greeting(name: String) -> String {\n    return \"Hello, \\(name)!\"\n}\n"
            try? source.write(to:root.appendingPathComponent("Sources/Greeting.swift"),atomically:true,encoding:.utf8)
            codeMode=true;code.openProject(root);if !CommandLine.arguments.contains("--preview-code-empty"){code.openFile("Sources/Greeting.swift")}
            if !CommandLine.arguments.contains("--preview-code-empty"){
            code.mutate{session in session.title="A greeting, made personal";session.entries=[AgentEntry("user","Add a greeting function that accepts a name."),AgentEntry("assistant","Added a greeting function with a clear, typed interface.\n\n```swift\nfunc greeting(name: String) -> String {\n    return \"Hello, \\(name)!\"\n}\n```")];session.events=[AgentEvent(AgentCall(name:"read_file",arguments:["path":"Sources/Greeting.swift"]),status:"done",output:source),AgentEvent(AgentCall(name:"write_file",arguments:["path":"Sources/Greeting.swift"]),status:"done",output:"Illustrative design fixture")];session.changes=[AgentChange(path:"Sources/Greeting.swift",before:nil,after:source,beforeHash:nil,afterHash:AgentProject.hash(Data(source.utf8)))]}
            }
        }

    }
    func persist() {
        guard !preview,!saveBlocked else{return}
        do { try WorkspacePersistence.save(state,to:directory);workspaces=try WorkspacePersistence.list(in:directory) } catch { notice="This conversation could not be saved. Check the storage available on your Mac." }
    }
    func inspectModel() {
        guard !preview else { return }
        let candidates=[settings.modelPath,WorkspacePersistence.defaultDirectory.appendingPathComponent("Models/churo-chat.particle").path,directory.appendingPathComponent("Models/sarvam-30b-q4_k_m.particle").path,"/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle"]
        if let path=candidates.first(where:Self.modelExists) { state.settings.modelPath=path;persist() }
        else { showSetup=true;beginInstall() }
    }
    static func modelExists(_ path: String) -> Bool {
        let root=URL(fileURLWithPath:path)
        if path.hasSuffix(".particle") { return FileManager.default.fileExists(atPath:root.appendingPathComponent("verified-install.json").path) && FileManager.default.fileExists(atPath:root.appendingPathComponent("manifest.json").path) }
        return path.lowercased().hasSuffix(".gguf") && FileManager.default.fileExists(atPath:path)
    }
    func beginInstall(root: URL? = nil) {
        guard !preview else { return }
        installer.start(root:root ?? directory.appendingPathComponent("Models",isDirectory:true)) { [weak self] model in
            guard let self else{return};self.state.settings.modelPath=model.path;self.showSetup=false;self.persist();self.loadModel()
        }
    }
    func chooseModel() {
        let panel=NSOpenPanel();panel.canChooseDirectories=true;panel.canChooseFiles=true;panel.allowsMultipleSelection=false
        panel.message="Choose a verified .particle folder or Churo-compatible Q4_K_M GGUF."
        if panel.runModal() == .OK,let url=panel.url {
            guard Self.modelExists(url.path) else { notice="Choose a verified .particle model folder or a GGUF file.";return }
            state.settings.modelPath=url.path;persist();showSetup=false
        }
    }
    func newConversation() {
        guard !busy,!loading else{return}
        var chat=Conversation()
        if let empty=state.conversations.first(where:{$0.messages.isEmpty}) { chat=empty }
        else { state.conversations.insert(chat,at:0) }
        state.selected=chat.id;draft="";rate=nil;contextUsed=0
        if connected { sendCommand(DesktopCommand("reset",system:state.systemPrompt,reasoningBudget:settings.responseStyle.budget)) };persist()
    }
    func select(_ id: UUID) {
        guard !busy,!loading,id != state.selected else{return};state.selected=id;draft="";rate=nil;contextUsed=0
        if connected { sendCommand(DesktopCommand("restore",messages:selected?.history ?? [],system:state.systemPrompt,reasoningBudget:settings.responseStyle.budget)) };persist()
    }
    func requestConversationDeletion(_ id:UUID?=nil){guard !busy,!loading else{return};pendingConversationDeletion=id ?? state.selected}
    func editAndRegenerate(conversationID:UUID,messageID:UUID,text:String)->Bool {
        let replacement=text.trimmingCharacters(in:.whitespacesAndNewlines)
        guard connected,!busy,!loading,!replacement.isEmpty,replacement.utf8.count<900_000 else{return false}
        guard let original=state.conversations.first(where:{$0.id==conversationID}),var branch=original.continuation(before:messageID) else{notice="That message is no longer available in this workspace.";return false}
        if original.messages.first?.id==messageID {branch.title=String(replacement.prefix(36)).replacingOccurrences(of:"\n",with:" ")+" · edited"}
        let savedDraft=draft
        state.conversations.insert(branch,at:0);state.selected=branch.id;search="";restoreSelected()
        draft=replacement;submit();draft=savedDraft
        return true
    }
    func delete(_ id: UUID) {
        guard !busy,!loading else{return};state.conversations.removeAll{$0.id==id}
        if state.selected==id { state.selected=nil;newConversation() };persist()
    }
    func updateSettings(_ value: OneSettings) {
        guard value.valid else{return}
        if busy || loading {state.settings.appearance=value.appearance;persist();return}
        let old=state.settings
        state.settings=value;persist()
        // Appearance and memory preferences never require reading model weights.
        var engineOld=old;engineOld.appearance=value.appearance;engineOld.rememberPersonalDetails=value.rememberPersonalDetails;engineOld.system=value.system;engineOld.responseStyle=value.responseStyle;engineOld.codeResponseStyle=value.codeResponseStyle
        if engineOld != value && connected {reloadWorker()}
        else if (old.system != value.system || old.responseStyle != value.responseStyle) && connected {restoreSelected()}
    }
    func renameWorkspace(_ name:String){let clean=name.trimmingCharacters(in:.whitespacesAndNewlines);guard !clean.isEmpty,clean.count<=80 else{return};state.name=clean;persist()}
    func toggleSidebar(){setSidebarExpanded(state.sidebarHidden)}
    func setSidebarExpanded(_ expanded:Bool){guard state.sidebarHidden==expanded else{return};state.sidebarHidden = !expanded;persist()}
    var modelProcessID:Int32? {worker?.isRunning==true ? worker?.processIdentifier : nil}
    var thinkingEnabled:Bool {(codeMode ? settings.codeResponseStyle : settings.responseStyle) != .direct}
    func setThinkingEnabled(_ enabled:Bool){
        guard !busy,!loading,enabled != thinkingEnabled else{return}
        var value=settings;if codeMode{value.codeResponseStyle=enabled ? .brief : .direct}else{value.responseStyle=enabled ? .brief : .direct};updateSettings(value)
    }
    func setUsername(_ name:String){let clean=name.trimmingCharacters(in:.whitespacesAndNewlines);guard !clean.isEmpty,clean.count<=80 else{return};state.username=clean;persist()}
    func setCodeMode(_ enabled:Bool){guard !busy,!loading else{return};codeMode=enabled;if enabled{code.restoreSelection()}else{restoreSelected()}}
    func restoreSelected(){contextUsed=0;rate=nil;if connected{sendCommand(DesktopCommand("restore",messages:selected?.history ?? [],system:state.systemPrompt,reasoningBudget:settings.responseStyle.budget))}}
    func removeMessage(_ id: UUID){guard !busy,!loading else{return};state.deleteMessage(id);restoreSelected();persist()}
    func saveMemory(_ text:String,source:UUID?=nil){
        let clean=text.trimmingCharacters(in:.whitespacesAndNewlines)
        guard !clean.isEmpty,clean.count<=2000,!busy,!loading else{return}
        guard !state.memories.contains(where:{$0.text.caseInsensitiveCompare(clean) == .orderedSame}) else{return}
        guard state.memories.count<100 else{notice="Memory is full. Remove an older note in Settings before saving another.";return}
        state.memories.append(SavedMemory(text:clean,sourceMessage:source));persist();restoreSelected()
    }
    func removeMemory(_ id:UUID){guard !busy,!loading else{return};state.memories.removeAll{$0.id==id};persist();restoreSelected()}
    func editMemory(_ id:UUID,text:String){guard !busy,!loading,let i=state.memories.firstIndex(where:{$0.id==id}),!text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,text.count<=2000 else{return};state.memories[i].text=text.trimmingCharacters(in:.whitespacesAndNewlines);persist();restoreSelected()}
    func createWorkspace(_ name:String){
        guard !busy,!loading else{return};let clean=name.trimmingCharacters(in:.whitespacesAndNewlines);guard !clean.isEmpty,clean.count<=80 else{return}
        code.flushDraft();code.workspaceChanged();persist();var fresh=SavedWorkspace();fresh.name=clean;fresh.username=state.username;fresh.settings=state.settings;state=fresh;draft="";search="";newConversation();restoreSelected();persist()
    }
    func openWorkspace(_ id:UUID){
        guard !busy,!loading,id != state.id else{return};persist()
        do {let next=try WorkspacePersistence.open(id,in:directory);activate(next)} catch{notice="This workspace could not be opened. Its saved data has been preserved."}
    }
    private func rememberWorkspaceFolder(){if !preview,ProcessInfo.processInfo.environment["PARTICLEONE_DATA_PATH"]==nil {UserDefaults.standard.set(directory.path,forKey:"ParticleOne.lastWorkspaceDirectory")}}
    private func activate(_ next:SavedWorkspace){
        code.flushDraft();code.workspaceChanged()
        var previous=state.settings;previous.responseStyle=next.settings.responseStyle;state=next;draft="";search="";inputVerified=nil
        if state.conversations.isEmpty{newConversation()}
        if previous.arguments(systemOverride:"") != next.settings.arguments(systemOverride:"") && connected {reloadWorker()} else{restoreSelected()}
        persist()
    }
    func openWorkspaceFolder(){
        guard !busy,!loading else{return}
        let panel=NSOpenPanel();panel.canChooseDirectories=true;panel.canChooseFiles=true;panel.message="Open a folder containing a Particle database, or select particleone.sqlite."
        if panel.runModal() == .OK,let url=panel.url {
            let folder=url.pathExtension=="sqlite" ? url.deletingLastPathComponent() : url
            guard FileManager.default.fileExists(atPath:folder.appendingPathComponent("particleone.sqlite").path) || FileManager.default.fileExists(atPath:folder.appendingPathComponent("workspace.json").path) else{notice="That folder has no Particle workspace. Use New workspace to make one.";return}
            do {let next=try WorkspacePersistence.load(from:folder);persist();directory=folder;rememberWorkspaceFolder();saveBlocked=false;activate(next)}catch{notice="That workspace could not be opened. Its saved data has been preserved."}
        }
    }
    func createWorkspaceFolder(){
        guard !busy,!loading else{return};let panel=NSSavePanel();panel.title="Create workspace folder";panel.nameFieldStringValue="Particle Workspace";panel.canCreateDirectories=true
        if panel.runModal() == .OK,let folder=panel.url {
            guard !FileManager.default.fileExists(atPath:folder.path) else{notice="Choose a new folder name to preserve existing files.";return}
            do {var next=SavedWorkspace();next.name=String(folder.lastPathComponent.prefix(80));next.username=state.username;next.settings=state.settings;try WorkspacePersistence.save(next,to:folder);persist();directory=folder;rememberWorkspaceFolder();saveBlocked=false;activate(next)}catch{notice="The workspace folder could not be created."}
        }
    }
    private func reloadWorker(){
        guard worker != nil else{return};unload();expertReloadTask?.cancel()
        expertReloadTask=Task {
            for _ in 0..<150 {guard !Task.isCancelled else{return};if worker==nil{notice=nil;loadModel();return};try? await Task.sleep(for:.milliseconds(100))}
            notice="The model is still stopping. Load it again when it has finished."
        }
    }
    func setExpertCount(_ count: Int) {
        guard !busy,!loading,(2...6).contains(count),count != settings.activeExperts else{return}
        var updated=settings;updated.activeExperts=count;updateSettings(updated)
    }
    func loadModel() {
        guard !loading,!connected,worker==nil,!preview else{return}
        guard settings.context<=ParticleContextPolicy.maximum() else{notice="This Mac supports up to \(ParticleContextPolicy.maximum()) context tokens. Choose a smaller conversation capacity in Settings; your saved messages are retained.";return}
        guard Self.modelExists(settings.modelPath) else { showSetup=true;return }
        let override=ProcessInfo.processInfo.environment["PARTICLEONE_ENGINE_PATH"]
        let bundled=Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/ParticleCLI")
        let sibling=Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("ParticleCLI")
        let engine=override.map{URL(fileURLWithPath:$0)} ?? (FileManager.default.isExecutableFile(atPath:bundled.path) ? bundled : sibling ?? bundled)
        guard FileManager.default.isExecutableFile(atPath:engine.path) else { notice="The model engine is missing. Rebuild or reinstall Particle.";return }
        unloading=false;rate=nil;contextUsed=0
        let process=Process(),stdin=Pipe(),stdout=Pipe(),stderr=Pipe();let session=UUID();workerSession=session
        process.executableURL=engine;process.arguments=settings.arguments(systemOverride:state.systemPrompt);process.standardInput=stdin;process.standardOutput=stdout;process.standardError=stderr
        outputReader=DesktopPipeReader { [weak self] data in Task { @MainActor in guard self?.workerSession==session else{return};self?.receive(data) } }
        errorReader=DesktopPipeReader { [weak self] data in Task { @MainActor in guard let self,self.workerSession==session else{return};self.diagnostics.append(String(decoding:data,as:UTF8.self));self.diagnostics=Array(self.diagnostics.suffix(150)) } }
        let reader=outputReader!,logger=errorReader!
        stdout.fileHandleForReading.readabilityHandler={handle in let data=handle.availableData;if data.isEmpty{handle.readabilityHandler=nil}else{reader.accept(data)}}
        stderr.fileHandleForReading.readabilityHandler={handle in let data=handle.availableData;if data.isEmpty{handle.readabilityHandler=nil}else{logger.accept(data)}}
        process.terminationHandler={ [weak self] process in let code=process.terminationStatus;Task { @MainActor in
            guard let self,self.workerSession==session else{return};self.workerSession=UUID();self.unloading=false;self.worker=nil;self.input=nil;self.connected=false;self.loading=false;self.busy=false;self.status="Model offline"
            if code != 0 { self.notice="The model stopped. Another Particle session may already be open, or the model disk may be unavailable. Details are in Settings." }
            self.persist()
        } }
        do { loading=true;status="Waking up Churo…";notice=nil;try process.run();worker=process;input=stdin.fileHandleForWriting }
        catch { loading=false;status="Model offline";notice="The model engine could not start: \(error.localizedDescription)" }
    }
    func sendCommand(_ command: DesktopCommand) {
        do { guard let input else {return};try input.write(contentsOf:command.line()) }
        catch { notice="The model connection closed. Reload it to continue.";busy=false }
    }
    func submit() {
        let text=draft.trimmingCharacters(in:.whitespacesAndNewlines)
        guard connected,!busy,!loading,!updateRestartPending,!text.isEmpty,let index=state.conversations.firstIndex(where:{$0.id==state.selected}) else{return}
        guard text.utf8.count<900_000 else {notice="This message is too large. Try a smaller excerpt.";return}
        if settings.rememberPersonalDetails {
            for fact in PersonalMemory.statements(in:text) {saveMemory(fact)}
        }
        inputVerified=nil;expectedInputFingerprint=DesktopTextIdentity.fingerprint(text)
        state.conversations[index].messages.append(ChatMessage(role:"user",content:text))
        let reply=ChatMessage(role:"assistant",content:"",complete:false);currentReply=reply.id
        state.conversations[index].messages.append(reply)
        if state.conversations[index].title=="New conversation" {state.conversations[index].title=String(text.prefix(46)).replacingOccurrences(of:"\n",with:" ")}
        state.conversations[index].updatedAt=Date();draft="";busy=true;status="Reading your message…";prefillProgress=0;notice=nil;persist();sendCommand(DesktopCommand("message",text:text))
    }
    func stop() { if code.active{code.stop();return};interruptModel() }
    func interruptModel() { guard let worker,worker.isRunning else{return};status="Stopping safely…";worker.interrupt() }
    func unload() {
        guard let process=worker else{return};unloading=true;sendCommand(DesktopCommand("quit"));if busy || loading {process.interrupt()};connected=false;loading=true;status="Unloading model…"
    }
    func shutdown() { code.cancelForShutdown();expertReloadTask?.cancel();installer.pause();connected=false;loading=true;status="Closing safely…";if busy && !code.active {editReply{$0.finish="cancelled"}};persist();if let process=worker {sendCommand(DesktopCommand("quit"));process.interrupt()} }
    func shutdownWorker() -> Process? {let process=worker;shutdown();return process}
    func finishShutdown() async {
        let process=worker
        for _ in 0..<80 {if process?.isRunning != true {return};try? await Task.sleep(for:.milliseconds(100))}
        if process?.isRunning==true {process?.terminate()}
    }
    private func editReply(_ change: (inout ChatMessage)->Void) {
        guard let conversation=state.conversations.firstIndex(where:{$0.id==state.selected}),let message=state.conversations[conversation].messages.firstIndex(where:{$0.id==currentReply}) else{return}
        change(&state.conversations[conversation].messages[message])
    }
    private func receive(_ data: Data) {
        guard let event=try? JSONSerialization.jsonObject(with:data) as? [String:Any],let type=event["event"] as? String else{return}
        if code.handle(event){return}
        switch type {
        case "ready":
            guard !unloading else{return}
            let first = !connected;connected=true;loading=false;if !busy {status="Ready when you are"}
            if first {sendCommand(DesktopCommand("restore",messages:selected?.history ?? [],system:state.systemPrompt,reasoningBudget:settings.responseStyle.budget));code.modelReady()}
        case "input":
            inputVerified=(event["utf8_sha256"] as? String)==expectedInputFingerprint
            if inputVerified==false {notice="The submitted text did not match the worker acknowledgment. Reload the model before continuing."}
        case "prefill":
            let done=event["completed"] as? Double ?? 0,total=event["total"] as? Double ?? 1
            prefillProgress=done/max(total,1);status="Reading your message · \(Int(done))/\(Int(total))"
        case "text":status="Writing a response…";if let text=event["text"] as? String {editReply{$0.content+=text}}
        case "turn":
            busy=false;status="Ready when you are"
            let steps=event["decode_steps"] as? Double ?? 0,seconds=event["decode_seconds"] as? Double ?? 0
            rate=steps>0 && seconds>0 ? steps/seconds : nil;contextUsed=event["context_used"] as? Int ?? 0
            let metrics:ResponseMetrics?
            if let tokens=event["tokens"] as? Int,let forwards=event["decode_steps"] as? Int {
                metrics=ResponseMetrics(outputTokens:tokens,decodeSteps:forwards,decodeSeconds:seconds,contextTokens:event["context_used"] as? Int,contextLimit:settings.context)
            }else{metrics=nil}
            editReply {if let text=event["text"] as? String{$0.content=text};$0.complete=(event["finish"] as? String) != "reasoning_limit";$0.finish=event["finish"] as? String;$0.metrics=metrics};persist()
        case "cancelled":contextUsed=0;editReply{$0.finish="cancelled"};busy=false;status="Stopped";persist()
        case "error":editReply{$0.finish="error"};notice=event["message"] as? String ?? "The model needs attention.";busy=false;persist()
        default:break
        }
    }
    func exportConversation() {
        guard let chat=selected,!chat.messages.isEmpty else{return};let panel=NSSavePanel();panel.nameFieldStringValue="\(chat.title).md"
        if panel.runModal() == .OK,let url=panel.url {
            let text="# \(chat.title)\n\n"+chat.messages.filter{$0.finish != "deleted"}.map{"## \($0.role == "user" ? "You" : "Churo Chat")\n\n\($0.content)"}.joined(separator:"\n\n")
            do {try text.write(to:url,atomically:true,encoding:.utf8)} catch {notice="The conversation could not be exported."}
        }
    }
}
