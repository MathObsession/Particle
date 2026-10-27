import SwiftUI
import AppKit
import ParticleOneCore
import TurboFieldfareMacPresentation

struct ParticleOneRootView: View {
    @ObservedObject var model: WorkspaceModel
    @StoredState private var showingExperts=false
    @StoredState private var showingWorkspaces=false
    @StoredState private var editingRequest:MessageEditRequest?
    @StoredState private var deleteMessageID:UUID?
    @StoredState private var memoryDraft=""
    @StoredState private var memorySource:UUID?
    @StoredState private var savingMemory=false
    @StoredState private var sidebarExpansionTask:Task<Void,Never>?
    @FocusState private var searchFocused:Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var isWelcome:Bool {model.selected?.messages.isEmpty != false}
    var body: some View {
        HStack(spacing:0) {
            Group {if model.state.sidebarHidden {accessRail.frame(width:68)}else{sidebar.frame(width:242)}}
            Rectangle().fill(Design.line.opacity(0.65)).frame(width:1)
            VStack(spacing:0) {
                if model.codeMode {CodeWorkspaceView(model:model,code:model.code)}else{
                GeometryReader { geometry in
                    VStack(spacing:0){
                        if isWelcome {Spacer(minLength:0).frame(height:max(16,geometry.size.height*0.54-285));welcome}else{transcript}
                        composer
                        if isWelcome {Spacer(minLength:24)}
                    }.frame(width:geometry.size.width,height:geometry.size.height)
                        .animation(reduceMotion ? nil : .easeInOut(duration:0.32),value:isWelcome)
                }
                }
            }.background{Design.paper.overlay{InterfaceGrain()}}
        }
        .disabled(model.updateRestartPending)
        .overlay(alignment:.topTrailing){UpdateIndicator(updater:model.updates,busy:model.busy || model.loading).padding(.top,12).padding(.trailing,16)}
        .animation(reduceMotion ? nil : .easeInOut(duration:0.22),value:model.state.sidebarHidden)
        .onDisappear{sidebarExpansionTask?.cancel()}
        .font(Design.body()).foregroundStyle(Design.ink)
        .frame(minWidth:model.codeMode ? 1100 : 780,minHeight:650)
        .ignoresSafeArea(.container,edges:.top)
        .preferredColorScheme(model.settings.appearance.colorScheme)
        .sheet(isPresented:$model.showSettings) { ParticleOneSettings(model:model) }
        .sheet(isPresented:$model.showNewWorkspace){NewWorkspaceView(model:model)}
        .sheet(isPresented:$model.showMemory){MemoryManagerView(model:model)}
        .sheet(item:$editingRequest){request in EditMessageView(model:model,request:request)}
        .sheet(isPresented:$savingMemory){MemoryDraftView(text:$memoryDraft){model.saveMemory(memoryDraft,source:memorySource)}}
        .alert("Delete this message?",isPresented:Binding(get:{deleteMessageID != nil},set:{if !$0{deleteMessageID=nil}})) {
            Button("Cancel",role:.cancel){deleteMessageID=nil}
            Button("Delete",role:.destructive){if let id=deleteMessageID{model.removeMessage(id)};deleteMessageID=nil}
        } message:{Text("Deleting a question also removes its answer. Deleted messages are removed from future model context.")}
        .sheet(isPresented:$model.showSetup) { ModelSetupView(model:model,installer:model.installer) }
        .alert("Delete this conversation?",isPresented:Binding(get:{model.pendingConversationDeletion != nil},set:{if !$0{model.pendingConversationDeletion=nil}})) {
            Button("Cancel",role:.cancel){model.pendingConversationDeletion=nil}
            Button("Delete",role:.destructive){if let id=model.pendingConversationDeletion{model.delete(id)};model.pendingConversationDeletion=nil}
        } message: {Text("This removes its saved transcript from this Mac.")}
        .onAppear{if !CommandLine.arguments.contains("--smoke-test") && !CommandLine.arguments.contains("--agent-native-smoke"){model.inspectModel()}}
    }
    private var sidebar: some View {
        VStack(alignment:.leading,spacing:0) {
            HStack(spacing:10) {
                ParticleMark(size:26)
                Text("Particle").font(Design.heading(20)).tracking(-0.5).lineLimit(1)
                Spacer(minLength:0)
                Button{model.setSidebarExpanded(false)}label:{Image(systemName:"sidebar.left").font(.system(size:12))}
                    .buttonStyle(QuietButtonStyle()).help("Collapse sidebar").accessibilityLabel("Collapse sidebar")
            }.padding(.top,45).padding(.bottom,22)
            HStack(spacing:4){
                Button{model.setCodeMode(false)}label:{Label("Chat",systemImage:"bubble.left").frame(maxWidth:.infinity).padding(.vertical,8).contentShape(Rectangle())}.buttonStyle(.plain).background(!model.codeMode ? Design.surface : .clear,in:RoundedRectangle(cornerRadius:8))
                Button{model.setCodeMode(true)}label:{Label("Code",systemImage:"chevron.left.forwardslash.chevron.right").frame(maxWidth:.infinity).padding(.vertical,8).contentShape(Rectangle())}.buttonStyle(.plain).background(model.codeMode ? Design.surface : .clear,in:RoundedRectangle(cornerRadius:8))
            }.font(Design.medium(11)).disabled(model.busy || model.loading).padding(.bottom,14)
            Button{if model.codeMode{model.code.newSession()}else{model.newConversation()}} label:{HStack{Image(systemName:"plus").font(.system(size:13));Text(model.codeMode ? "New session" : "New conversation");Spacer();Text("⌘ N").font(Design.body(10)).opacity(0.6)}}
                .buttonStyle(WashedButtonStyle()).disabled(model.busy || model.loading).keyboardShortcut("n",modifiers:.command)
            HStack(spacing:9){Image(systemName:"magnifyingglass").font(.system(size:12));TextField(model.codeMode ? "Find a session" : "Find a conversation",text:$model.search).focused($searchFocused).textFieldStyle(.plain).font(Design.body(11))}
                .foregroundStyle(Design.muted).padding(.horizontal,8).padding(.top,25).padding(.bottom,24)
            Text(model.codeMode ? "SESSIONS" : "YOUR CONVERSATIONS").font(Design.medium(9)).tracking(1.65).foregroundStyle(Design.muted).padding(.leading,8).padding(.bottom,11)
            ScrollView {
                VStack(spacing:5) {
                    if model.codeMode {
                        ForEach(model.code.sessions.filter{model.search.isEmpty || $0.title.localizedCaseInsensitiveContains(model.search)}){session in HStack{Button{model.code.select(session.id)}label:{Text(session.title).font(Design.body(11)).lineLimit(2).frame(maxWidth:.infinity,alignment:.leading).padding(10).contentShape(Rectangle())}.buttonStyle(.plain);Menu{Button("Delete session…",role:.destructive){model.code.pendingSessionDeletion=session.id}}label:{Image(systemName:"ellipsis").frame(width:28,height:32)}.menuStyle(.borderlessButton).menuIndicator(.hidden).frame(width:28)}.disabled(model.busy || model.loading)}
                        if model.code.sessions.isEmpty{Text("Open a project to begin.").font(Design.body(11)).foregroundStyle(Design.muted).padding(10)}
                    }else{
                    ForEach(model.filtered.filter{!$0.messages.isEmpty}) { chat in
                        HStack(spacing:0){
                            Button{model.select(chat.id)}label:{HStack(alignment:.top,spacing:9){Image(systemName:"bubble.left").font(.system(size:12)).padding(.top,2);Text(chat.title).font(Design.body(11)).lineLimit(2).multilineTextAlignment(.leading);Spacer(minLength:0)}.padding(.leading,11).padding(.trailing,3).padding(.vertical,11).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())}.buttonStyle(.plain)
                            Menu{conversationActions(chat)}label:{Image(systemName:"ellipsis").font(.system(size:13)).frame(width:28,height:34).contentShape(Rectangle())}.menuStyle(.borderlessButton).menuIndicator(.hidden).frame(width:28).help("Conversation actions").accessibilityLabel("Actions for \(chat.title)")
                        }.padding(.trailing,4).background(chat.id==model.state.selected ? Design.surface.opacity(0.72) : .clear,in:RoundedRectangle(cornerRadius:10))
                            .disabled(model.busy || model.loading).contextMenu{conversationActions(chat)}
                    }
                    if model.filtered.allSatisfy({$0.messages.isEmpty}) {Text(model.search.isEmpty ? "A fresh page, whenever\nyou need one." : "No conversations found.").font(Design.body(11)).lineSpacing(5).foregroundStyle(Design.muted).frame(maxWidth:.infinity,alignment:.leading).padding(10)}
                }
                    }
            }.scrollIndicators(.hidden)
            Spacer(minLength:15)
            VStack(spacing:7){
                Button{showingExperts.toggle()}label:{sidebarSelector(icon:"cpu",title:model.codeMode ? "Churo Code" : "Churo Chat",detail:"\(model.settings.activeExperts) experts · 4-bit")}.buttonStyle(.plain).popover(isPresented:$showingExperts,arrowEdge:.leading){expertPicker}.accessibilityLabel("Model selector")
                Button{showingWorkspaces.toggle()}label:{sidebarSelector(icon:"square.grid.2x2",title:model.state.name,detail:"Workspace")}.buttonStyle(.plain).disabled(model.busy || model.loading).popover(isPresented:$showingWorkspaces,arrowEdge:.leading){workspacePicker}.accessibilityLabel("Workspace selector")
            }.padding(.bottom,12)
            Button {model.showMemory=true} label:{HStack(spacing:9){Image(systemName:"brain").foregroundStyle(Design.green);Text("Saved memory");Spacer();Text("\(model.state.memories.count)").foregroundStyle(Design.muted)}.font(Design.body(11)).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())}.buttonStyle(HitButtonStyle())
            HStack(spacing:8){
                Image(systemName:"person.crop.circle").font(.system(size:24)).foregroundStyle(Design.green)
                VStack(alignment:.leading,spacing:3){Text(model.state.username).font(Design.medium(11)).lineLimit(1);Text("Private, on this Mac").font(Design.body(9)).foregroundStyle(Design.muted)}
                Spacer(minLength:0)
                Button{model.showSettings=true}label:{Image(systemName:"slider.horizontal.3")}.buttonStyle(QuietButtonStyle()).help("Settings").accessibilityLabel("Settings")
            }.padding(.top,18).padding(.bottom,16)
        }.padding(.horizontal,18).background{Design.sidebar.overlay{InterfaceGrain()}}
    }
    @ViewBuilder private func conversationActions(_ chat:Conversation)->some View {
        Button("Export conversation…"){model.select(chat.id);model.exportConversation()}
        Button("Delete conversation…",role:.destructive){model.requestConversationDeletion(chat.id)}
    }
    private func sidebarSelector(icon:String,title:String,detail:String)->some View {
        HStack(spacing:10){Image(systemName:icon).font(.system(size:15)).foregroundStyle(Design.green).frame(width:20);VStack(alignment:.leading,spacing:3){Text(title).font(Design.medium(11)).lineLimit(1);Text(detail).font(Design.body(9)).foregroundStyle(Design.muted)};Spacer(minLength:4);Image(systemName:"chevron.up.chevron.down").font(.system(size:8)).foregroundStyle(Design.muted)}
            .padding(10).frame(maxWidth:.infinity,alignment:.leading).background(Design.surface.opacity(0.5),in:RoundedRectangle(cornerRadius:11)).overlay(RoundedRectangle(cornerRadius:11).stroke(Design.line.opacity(0.7),lineWidth:1)).contentShape(Rectangle())
    }
    private var workspacePicker:some View {
        VStack(alignment:.leading,spacing:10){
            Text("Workspaces").font(Design.heading(21))
            ForEach(model.workspaces){space in
                Button{showingWorkspaces=false;model.openWorkspace(space.id)}label:{HStack{Text(space.name).font(Design.body(12));Spacer();if space.id==model.state.id{Image(systemName:"checkmark")}}.padding(9).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())}.buttonStyle(.plain)
            }
            Divider()
            Button("New workspace…"){showingWorkspaces=false;model.showNewWorkspace=true}.buttonStyle(HitButtonStyle())
            Button("Open workspace folder…"){showingWorkspaces=false;model.openWorkspaceFolder()}.buttonStyle(HitButtonStyle())
            Button("Create workspace folder…"){showingWorkspaces=false;model.createWorkspaceFolder()}.buttonStyle(HitButtonStyle())
        }.font(Design.body(11)).padding(18).frame(width:270).foregroundStyle(Design.ink).background(Design.paper)
    }
    private var expertPicker: some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Active experts").font(Design.heading(20))
            Text("Fewer experts trade answer quality for speed.").font(Design.body(10)).foregroundStyle(Design.muted)
            ForEach(2...6,id:\.self) { count in
                Button {model.setExpertCount(count);showingExperts=false} label:{
                    HStack {
                        Text("\(count) of 6").font(Design.medium(12))
                        Spacer()
                        Text(count==6 ? "Original" : "Experimental").font(Design.body(10)).foregroundStyle(Design.muted)
                        if count==model.settings.activeExperts {Image(systemName:"checkmark").font(.system(size:11))}
                    }.padding(10).contentShape(Rectangle()).background(count==model.settings.activeExperts ? Design.sidebar : .clear,in:RoundedRectangle(cornerRadius:9))
                }.buttonStyle(.plain).disabled(model.busy || model.loading)
            }
            Divider()
            Button("Model settings…"){showingExperts=false;model.showSettings=true}.buttonStyle(.plain).font(Design.body(11))
        }.padding(18).frame(width:270).foregroundStyle(Design.ink).background(Design.paper)
    }
    private var welcome: some View {
        VStack(spacing:8){
            Text("HELLO, \(model.state.username.uppercased())").font(Design.medium(9)).tracking(2).foregroundStyle(Design.muted).lineLimit(1)
            FadingHeroBanner(height:200)
        }.frame(maxWidth:780).padding(.horizontal,36).padding(.bottom,20).frame(maxWidth:.infinity)
    }
    private func hoverAppMark(_ entered:Bool){
        sidebarExpansionTask?.cancel()
        guard entered,model.state.sidebarHidden else{return}
        sidebarExpansionTask=Task {
            try? await Task.sleep(for:.milliseconds(160))
            guard !Task.isCancelled,model.state.sidebarHidden else{return}
            model.setSidebarExpanded(true)
        }
    }
    private var accessRail:some View {
        VStack(spacing:12){
            Button{model.setSidebarExpanded(true)}label:{ParticleMark(size:30).frame(width:36,height:36).contentShape(Rectangle())}
                .buttonStyle(.plain).onHover(perform:hoverAppMark).help("Expand sidebar").accessibilityLabel("Expand sidebar")
                .padding(.top,45).padding(.bottom,14)
            Button{if model.codeMode{model.code.newSession()}else{model.newConversation()}}label:{Image(systemName:"plus").font(.system(size:17))}.buttonStyle(QuietButtonStyle()).help("New conversation").accessibilityLabel("New conversation").disabled(model.busy || model.loading)
            Button{model.setSidebarExpanded(true);Task{try? await Task.sleep(for:.milliseconds(250));searchFocused=true}}label:{Image(systemName:"magnifyingglass").font(.system(size:15))}.buttonStyle(QuietButtonStyle()).help("Search conversations").accessibilityLabel("Search conversations")
            Menu {
                ForEach(model.filtered.filter{!$0.messages.isEmpty}.prefix(12)){chat in Menu(chat.title){Button("Open conversation"){model.select(chat.id)};conversationActions(chat)}}
                if model.filtered.allSatisfy({$0.messages.isEmpty}){Text("No conversations yet")}
            } label:{Image(systemName:"bubble.left.and.bubble.right").font(.system(size:15)).frame(width:36,height:36).contentShape(Rectangle())}
                .menuStyle(.borderlessButton).menuIndicator(.hidden).frame(width:40).help("Conversations").accessibilityLabel("Conversations").disabled(model.busy || model.loading)
            Button{model.setCodeMode(!model.codeMode)}label:{Image(systemName:model.codeMode ? "bubble.left" : "chevron.left.forwardslash.chevron.right").font(.system(size:16))}.buttonStyle(QuietButtonStyle()).help(model.codeMode ? "Churo Chat" : "Particle Code").disabled(model.busy || model.loading)
            Spacer(minLength:20)
            Button{showingExperts.toggle()}label:{Image(systemName:"cpu").font(.system(size:15))}.buttonStyle(QuietButtonStyle()).help("Model selector").accessibilityLabel("Model selector").popover(isPresented:$showingExperts,arrowEdge:.leading){expertPicker}
            Button{showingWorkspaces.toggle()}label:{Image(systemName:"square.grid.2x2").font(.system(size:15))}.buttonStyle(QuietButtonStyle()).help("Workspace selector").accessibilityLabel("Workspace selector").disabled(model.busy || model.loading).popover(isPresented:$showingWorkspaces,arrowEdge:.leading){workspacePicker}
            Button{model.showMemory=true}label:{Image(systemName:"brain").font(.system(size:16))}.buttonStyle(QuietButtonStyle()).help("Saved memory").accessibilityLabel("Saved memory").disabled(model.busy || model.loading)
            Button{model.showSettings=true}label:{Image(systemName:"slider.horizontal.3").font(.system(size:15))}.buttonStyle(QuietButtonStyle()).help("Settings").accessibilityLabel("Settings")
            Button{model.showSettings=true}label:{Image(systemName:"person.crop.circle").font(.system(size:24)).foregroundStyle(Design.green)}.buttonStyle(QuietButtonStyle()).help(model.state.username).accessibilityLabel("Profile settings")
        }.padding(.horizontal,10).padding(.bottom,16).foregroundStyle(Design.muted).frame(maxWidth:.infinity,maxHeight:.infinity).background{Design.sidebar.overlay{InterfaceGrain()}}
    }
    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment:.leading,spacing:29) {
                    ForEach((model.selected?.messages ?? []).filter{$0.finish != "deleted"}) { message in
                        VStack(alignment:.leading,spacing:8){
                            if message.role=="user" {
                                HStack{Spacer(minLength:70);Text(message.content).font(Design.body(13)).lineSpacing(5).textSelection(.enabled).padding(.horizontal,20).padding(.vertical,15).background(Design.line.opacity(0.52),in:RoundedRectangle(cornerRadius:18))}
                            } else {AssistantMessageView(message:message,busy:model.busy)}
                            HStack(spacing:7){
                                if message.role=="user"{Spacer()}
                                Button{NSPasteboard.general.clearContents();NSPasteboard.general.setString(message.content,forType:.string)}label:{Image(systemName:"doc.on.doc")}.help("Copy message").accessibilityLabel("Copy message")
                                if message.role=="user" {Button{beginEditing(message)}label:{Image(systemName:"pencil")}.help("Edit and regenerate").accessibilityLabel("Edit and regenerate").disabled(model.busy || model.loading)}
                                if message.role=="user" {Button{memoryDraft=message.content;memorySource=message.id;savingMemory=true}label:{Image(systemName:"brain")}.help("Save to memory").accessibilityLabel("Save to memory").disabled(model.busy || model.loading)}
                                Button{deleteMessageID=message.id}label:{Image(systemName:"trash")}.help("Delete message").accessibilityLabel("Delete message").disabled(model.busy || model.loading)
                            }.font(.system(size:11)).foregroundStyle(Design.muted).buttonStyle(QuietButtonStyle())
                        }.contextMenu{
                            Button("Copy"){NSPasteboard.general.clearContents();NSPasteboard.general.setString(message.content,forType:.string)}
                            if message.role=="user" {Button("Edit and regenerate…"){beginEditing(message)}.disabled(model.busy || model.loading)}
                            if message.role=="user" {Button("Save to memory…"){memoryDraft=message.content;memorySource=message.id;savingMemory=true}.disabled(model.busy || model.loading)}
                            Button("Delete message…",role:.destructive){deleteMessageID=message.id}.disabled(model.busy || model.loading)
                        }
                    }
                    Color.clear.frame(height:1).id("bottom")
                }.frame(maxWidth:820).padding(.horizontal,40).padding(.vertical,32).frame(maxWidth:.infinity)
            }.scrollIndicators(.hidden)
                .onChange(of:model.selected?.messages.last?.content) {_,_ in proxy.scrollTo("bottom",anchor:.bottom)}
                .onChange(of:model.selected?.messages.last?.metrics) {_,_ in proxy.scrollTo("bottom",anchor:.bottom)}
                .onChange(of:model.selected?.messages.last?.finish) {_,_ in proxy.scrollTo("bottom",anchor:.bottom)}
                .onChange(of:model.state.selected){_,_ in proxy.scrollTo("bottom",anchor:.bottom)}
        }
    }
    private func beginEditing(_ message:ChatMessage){guard let id=model.state.selected else{return};editingRequest=MessageEditRequest(conversationID:id,message:message)}
    private var thinkingToggle:some View {
        Button{model.setThinkingEnabled(!model.thinkingEnabled)}label:{
            HStack(spacing:6){Image(systemName:model.thinkingEnabled ? "sparkle" : "bolt");Text(model.thinkingEnabled ? "Thinking" : "Instant")}
                .font(Design.medium(11)).foregroundStyle(model.thinkingEnabled ? Design.green : Design.muted)
                .padding(.horizontal,11).frame(height:34).background(model.thinkingEnabled ? Design.green.opacity(0.12) : Design.line.opacity(0.24),in:Capsule()).contentShape(Capsule())
        }.buttonStyle(.plain).disabled(model.busy || model.loading)
            .accessibilityLabel("Response mode").accessibilityValue(model.thinkingEnabled ? "Thinking" : "Instant")
            .help(model.busy || model.loading ? "Wait for this response or stop it before changing thinking." : "Switch between Instant answers and Thinking for the next reply.")
    }
    private var sendControl:some View {
        Group {
            if model.busy {Button{model.stop()}label:{Label("Stop",systemImage:"stop.fill")}.buttonStyle(WashedButtonStyle(compact:true))}
            else if !model.connected {Button{model.loadModel()}label:{HStack(spacing:7){if model.loading{ProgressView().controlSize(.mini)};Text(model.loading ? "Loading…" : "Load model");Image(systemName:"arrow.up.right")}}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.loading)}
            else {Button{model.submit()}label:{Image(systemName:"arrow.up").font(.system(size:14,weight:.medium)).frame(width:18,height:16)}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.draft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty).keyboardShortcut(.return,modifiers:.command).accessibilityLabel("Send message")}
        }
    }
    private var composer:some View {
        VStack(spacing:10){
            if let notice=model.notice {HStack(alignment:.top){Image(systemName:"info.circle");Text(notice).textSelection(.enabled);Spacer();Button{model.notice=nil}label:{Image(systemName:"xmark")}.buttonStyle(QuietButtonStyle())}.font(Design.body(11)).padding(12).background(Design.sidebar,in:RoundedRectangle(cornerRadius:12))}
            VStack(spacing:7){
                HStack(spacing:12){
                    ZStack(alignment:.topLeading){
                        if model.draft.isEmpty {Text("What’s on your mind?").font(Design.body(14)).foregroundStyle(Design.muted).padding(.top,8).padding(.leading,5).allowsHitTesting(false)}
                        ComposerInput(text:$model.draft){model.submit()}.frame(height:isWelcome ? (model.draft.contains("\n") ? 72 : 36) : 66)
                    }.frame(maxWidth:.infinity)
                    if isWelcome {thinkingToggle;sendControl}
                }
                if !isWelcome {
                    HStack(spacing:8){
                        Image(systemName:"circle.dotted").font(.system(size:12)).foregroundStyle(Design.green)
                        Text(model.status).font(Design.body(10)).foregroundStyle(Design.muted).lineLimit(1)
                        if model.settings.activeExperts != 6 {Text("FAST · EXPERIMENTAL").font(Design.medium(8)).foregroundStyle(Design.green).padding(.horizontal,7).padding(.vertical,4).background(Design.line.opacity(0.5),in:Capsule())}
                        Spacer(minLength:4);thinkingToggle;sendControl
                    }
                }
            }.padding(.horizontal,isWelcome ? 18 : 15).padding(.vertical,isWelcome ? 12 : 15)
                .background(Design.surface.opacity(0.88),in:RoundedRectangle(cornerRadius:isWelcome ? 34 : 18))
                .overlay(RoundedRectangle(cornerRadius:isWelcome ? 34 : 18).stroke(Design.line,lineWidth:1))
                .shadow(color:Design.ink.opacity(0.025),radius:12,y:4)
            HStack(spacing:5){
                if isWelcome {Text(model.status);Text("·")}
                Text("Made in India")
            }.font(Design.body(9)).foregroundStyle(Design.muted)
        }.frame(maxWidth:isWelcome ? 760 : 860).padding(.horizontal,isWelcome ? 32 : 40).padding(.top,10).padding(.bottom,isWelcome ? 10 : 18).frame(maxWidth:.infinity)
    }

}

struct AssistantMessageView: View {
    let message: ChatMessage
    var name:String=ParticleIdentity.chat
    let busy: Bool
    @StoredState private var thinkingExpanded=CommandLine.arguments.contains("--preview-thinking")
    var body: some View {
        VStack(alignment:.leading,spacing:16){
            HStack(spacing:9){ParticleMark(size:24);Text(name).font(Design.heading(18))}
            if message.content.isEmpty {HStack(spacing:8){ProgressView().controlSize(.small);Text(busy ? "Preparing your answer…" : "No response was completed.").font(Design.body(12)).foregroundStyle(Design.muted)}}
            ForEach(TranscriptParser.parse(message.content)){block in
                switch block.kind {
                case .code:CodeCard(code:block.text,language:block.language)
                case .thinking:DisclosureGroup(isExpanded:$thinkingExpanded){MarkdownContent(source:block.text,compact:true).padding(.top,10)}label:{Label("Thinking",systemImage:"sparkle").font(Design.medium(11)).foregroundStyle(Design.muted)}.padding(13).background(Design.sidebar.opacity(0.7),in:RoundedRectangle(cornerRadius:12))
                case .text:MarkdownContent(source:block.text)
                }
            }
            if message.finish=="length" {Text("The response reached its output limit. You can continue the conversation or adjust the limit in Settings.").font(Design.body(10)).foregroundStyle(Design.muted)}
            if message.finish=="cancelled" {Text("Stopped · this unfinished reply is not added to model history.").font(Design.body(10)).foregroundStyle(Design.muted)}
            if message.finish=="reasoning_limit" {Text("The model tried to restart thinking repeatedly, so this reply was stopped. Try rephrasing or select Thinking in Settings.").font(Design.body(10)).foregroundStyle(Design.muted)}
            if message.complete || message.finish != nil {ResponseFooter(metrics:message.metrics)}
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
}
struct CodeCard: View {
    let code: String
    var language=""
    @StoredState private var copied=false
    var body: some View {
        VStack(alignment:.leading,spacing:0){
            HStack(spacing:8){Image(systemName:"chevron.left.forwardslash.chevron.right").foregroundStyle(Color(hex:0xB9CFAE));Text(language.isEmpty ? "code" : language).font(Design.mono(10));Spacer();Button{NSPasteboard.general.clearContents();NSPasteboard.general.setString(code,forType:.string);copied=true;DispatchQueue.main.asyncAfter(deadline:.now()+2){copied=false}}label:{Label(copied ? "Copied" : "Copy",systemImage:copied ? "checkmark" : "doc.on.doc").font(Design.body(10))}.buttonStyle(HitButtonStyle())}.foregroundStyle(Color(hex:0xB9C8B9)).padding(.horizontal,18).padding(.vertical,12)
            Rectangle().fill(.white.opacity(0.08)).frame(height:1)
            ScrollView(.horizontal){Text(AttributedString(SyntaxHighlighter.render(code.trimmingCharacters(in:.newlines),language:language))).lineSpacing(5).textSelection(.enabled).fixedSize(horizontal:true,vertical:false).padding(20)}
        }.background(Design.code,in:RoundedRectangle(cornerRadius:14)).overlay(RoundedRectangle(cornerRadius:14).stroke(.white.opacity(0.08),lineWidth:1))
    }
}

struct ResponseFooter:View {
    let metrics:ResponseMetrics?
    var body:some View {
        HStack(spacing:7){
            if let metrics,metrics.isValid {
                Text("\(metrics.outputTokens) tokens")
                if let speed=metrics.tokensPerSecond {Text("·");Text(String(format:"%.2f tok/s",speed))}
                if let context=metrics.contextTokens,let limit=metrics.contextLimit {Text("·");Text("\(context) / \(limit) context")}
            }else{Text("Token count unavailable")}
        }.font(Design.mono(9)).foregroundStyle(Design.muted).textSelection(.enabled).padding(.top,2)
    }
}
