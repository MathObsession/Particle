import SwiftUI
import AppKit
import ParticleOneCore
import TurboFieldfareMacPresentation

struct CodeWorkspaceView:View {
    @ObservedObject var model:WorkspaceModel
    @ObservedObject var code:CodeWorkspaceModel
    @StoredState private var filter=""
    @StoredState private var permissions=false
    private var emptySession:Bool{code.session?.entries.isEmpty != false && !code.active}
    var body:some View {
        VStack(spacing:0){
            toolbar
            HSplitView {
                VStack(spacing:0){
                    if emptySession {
                        Spacer(minLength:35)
                        VStack(spacing:14){ParticleMark(size:40);Text("Particle Code").font(Design.heading(36));Text("Churo Code · local coding agent").font(Design.body(12)).foregroundStyle(Design.muted)}.padding(.bottom,22)
                        composer.frame(maxWidth:790)
                        Spacer(minLength:70)
                    }else{conversation;approval;composer.frame(maxWidth:900)}
                }.frame(minWidth:380,maxWidth:.infinity,maxHeight:.infinity)
                if code.inspectorVisible{inspector.frame(minWidth:310,idealWidth:440,maxWidth:.infinity)}
            }
            if let error=code.error{HStack(alignment:.top,spacing:10){Image(systemName:"exclamationmark.circle").foregroundStyle(Design.green);Text(error).textSelection(.enabled);Spacer();Button{code.error=nil}label:{Image(systemName:"xmark")}.buttonStyle(QuietButtonStyle()).accessibilityLabel("Dismiss message")}.font(Design.body(11)).padding(12).background(Design.surface)}
        }.background(CodeMaterial()).foregroundStyle(Design.ink)
        .sheet(item:$code.inspectingChange){change in CodeChangeReview(change:change)}
        .onChange(of:code.editor){_,_ in code.scheduleDraftSave()}
        .onChange(of:code.draft){_,_ in code.scheduleDraftSave()}
        .alert("Delete session?",isPresented:Binding(get:{code.pendingSessionDeletion != nil},set:{if !$0{code.pendingSessionDeletion=nil}})){Button("Cancel",role:.cancel){code.pendingSessionDeletion=nil};Button("Delete",role:.destructive){if let id=code.pendingSessionDeletion{code.deleteSession(id)};code.pendingSessionDeletion=nil}}message:{Text("This removes its transcript and undo history, without reverting any project files.")}
    }
    private var toolbar:some View {
        HStack(spacing:8){
            Text(emptySession ? "New session" : code.session?.title ?? "Particle Code").font(Design.medium(11)).lineLimit(1).foregroundStyle(Design.muted)
            Spacer()
            if code.active{TimelineView(.periodic(from:.now,by:1)){context in Text(elapsed(context.date)).font(Design.mono(10)).foregroundStyle(Design.muted)}}
            Button{code.inspectorVisible.toggle()}label:{Label(code.inspectorVisible ? "Hide files" : "Files",systemImage:"sidebar.right").font(Design.body(11))}.buttonStyle(HitButtonStyle()).help("Show or hide project tools")
            Button{permissions.toggle()}label:{Image(systemName:"slider.horizontal.3")}.buttonStyle(QuietButtonStyle()).help("Agent controls")
                .popover(isPresented:$permissions){controls}
        }.padding(.horizontal,25).padding(.top,42).padding(.bottom,12)
    }
    private func elapsed(_ now:Date)->String{let seconds=max(0,Int(now.timeIntervalSince(code.startedAt ?? now)));return "\(seconds/60):\(String(format:"%02d",seconds%60))"}
    private var controls:some View {
        VStack(alignment:.leading,spacing:16){
            Text("Agent controls").font(Design.heading(24))
            Toggle("Allow project edits this session",isOn:$code.allowProjectEdits).font(Design.body(11))
            Picker("Steps per run",selection:$code.maximumSteps){Text("24").tag(24);Text("48").tag(48);Text("96").tag(96)}.font(Design.body(11)).disabled(code.active)
            Text("Reads stay inside this project. Terminal commands and public web requests ask first. Commands run with your Mac account’s permissions.").font(Design.body(11)).foregroundStyle(Design.muted).fixedSize(horizontal:false,vertical:true)
            Text("Six experts is recommended for coding tools. Your current profile uses \(model.settings.activeExperts).").font(Design.body(10)).foregroundStyle(Design.muted).fixedSize(horizontal:false,vertical:true)
        }.padding(22).frame(width:340).background(CodeMaterial(panel:true)).foregroundStyle(Design.ink)
    }
    private var conversation:some View {
        ScrollViewReader{proxy in
            ScrollView{LazyVStack(alignment:.leading,spacing:25){
                ForEach(code.session?.entries ?? []){entry in
                    if entry.role=="user"{Text(entry.text).font(Design.body(13)).textSelection(.enabled).padding(16).frame(maxWidth:.infinity,alignment:.leading).background{CodeMaterial(panel:true).clipShape(RoundedRectangle(cornerRadius:18))}.overlay(RoundedRectangle(cornerRadius:18).stroke(Design.line.opacity(0.6),lineWidth:1))}
                    else if entry.role=="assistant",entry.displayHidden != true,!entry.text.trimmingCharacters(in:.whitespacesAndNewlines).hasPrefix("<tool_call>"){
                        CodeAssistantReply(entry:entry,collapseCode:code.session?.changes.isEmpty == false)
                    }
                }
                if code.active && !code.waitingForModel && code.pending==nil{HStack(spacing:9){ProgressView().controlSize(.small);Text(code.status).font(Design.body(11)).foregroundStyle(Design.muted)}}
                if code.waitingForModel {
                    VStack(alignment:.leading,spacing:10){
                        HStack(spacing:10){ProgressView().controlSize(.small);Text(code.status).font(Design.body(11)).foregroundStyle(Design.muted)}
                        if code.phase=="Reading context"{ProgressView(value:code.contextProgress).tint(Design.green).frame(maxWidth:300)}
                        if code.stream.contains("<tool_call>"){Label("Preparing a tool action",systemImage:"wrench.and.screwdriver").font(Design.body(11)).foregroundStyle(Design.muted)}
                        else if !code.stream.isEmpty{CodeStreamPreview(text:code.stream)}
                    }.padding(.vertical,8)
                }
                if !(code.session?.events.isEmpty ?? true){activitySummary}
                if !code.active,code.verification != .none{verificationSummary}
                Color.clear.frame(height:1).id("code-end")
            }.padding(.horizontal,30).padding(.vertical,24).frame(maxWidth:880).frame(maxWidth:.infinity)}
            .onChange(of:code.status){_,_ in if code.active{proxy.scrollTo("code-end",anchor:.bottom)}}
            .onChange(of:code.session?.entries.count){_,_ in proxy.scrollTo("code-end",anchor:.bottom)}
        }
    }
    private var activitySummary:some View {
        VStack(alignment:.leading,spacing:10){
            HStack{Label("Activity",systemImage:"bolt.horizontal.circle").font(Design.medium(11));Spacer();Button("View all"){code.showPanel("Activity")}.buttonStyle(HitButtonStyle()).font(Design.body(10))}
            ForEach(Array((code.session?.events ?? []).suffix(4))){event in
                HStack(spacing:9){Image(systemName:event.status=="done" ? "checkmark" : event.status=="failed" ? "exclamationmark" : "circle.dotted").foregroundStyle(Design.green).frame(width:15);Text(event.call.name.replacingOccurrences(of:"_",with:" ")).font(Design.body(10));Text(event.call.arguments["path"] ?? event.call.arguments["command"] ?? event.call.arguments["query"] ?? "").font(Design.mono(9)).foregroundStyle(Design.muted).lineLimit(1);Spacer();Text(event.status).font(Design.body(9)).foregroundStyle(Design.muted)}
            }
        }.padding(15).background{CodeMaterial(panel:true).clipShape(RoundedRectangle(cornerRadius:16))}.overlay(RoundedRectangle(cornerRadius:16).stroke(Design.line.opacity(0.6),lineWidth:1))
    }
    private var verificationSummary:some View {
        HStack(spacing:10){
            Image(systemName:code.verification == .passed ? "checkmark.seal" : "play.circle").foregroundStyle(Design.green)
            Text(code.verification == .passed ? "Check command completed" : code.verification == .declined ? "Execution was declined" : code.verification == .failed ? "Check command failed" : "Changes saved · execution not verified").font(Design.body(10)).foregroundStyle(Design.muted)
            Spacer();Button("Run checks"){code.runChecks()}.font(Design.body(10)).buttonStyle(HitButtonStyle()).disabled(!model.connected || model.loading)
        }.padding(.horizontal,4)
    }
    @ViewBuilder private var approval:some View {
        if let event=code.pending,event.status != "done",["Review file change","Approve terminal command","Approve public web request"].contains(code.status){
            VStack(alignment:.leading,spacing:10){
                HStack{Image(systemName:"hand.raised").foregroundStyle(Design.green);Text(code.status).font(Design.medium(12));Spacer()}
                if let change=code.proposal{Text(change.path).font(Design.mono(10));Button("Review complete file…"){code.inspectingChange=change}.buttonStyle(HitButtonStyle()).font(Design.body(10));ScrollView([.horizontal,.vertical]){Text(AgentDiff.text(before:change.before,after:change.after)).font(Design.mono(10)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)}.frame(maxHeight:160).padding(10).background(Design.code,in:RoundedRectangle(cornerRadius:9)).foregroundStyle(.white.opacity(0.85))}
                else{ScrollView{Text(event.call.arguments["command"] ?? event.call.arguments["query"] ?? event.call.arguments["url"] ?? "").font(Design.mono(10)).textSelection(.enabled).fixedSize(horizontal:false,vertical:true).frame(maxWidth:.infinity,alignment:.leading)}.frame(maxHeight:150)
                    Text(event.call.name=="run_command" ? "This command runs on your Mac with your account’s permissions." : "Only this query or URL is sent to Exa (or Jina for page-reading fallback). Your model keeps running on this Mac.").font(Design.body(10)).foregroundStyle(Design.muted)}
                HStack{Button("Deny"){code.deny(event.id)}.buttonStyle(HitButtonStyle());Spacer();Button("Allow once"){code.approve(event.id)}.buttonStyle(WashedButtonStyle(compact:true))}
            }.padding(16).background(Design.surface,in:RoundedRectangle(cornerRadius:14)).overlay(RoundedRectangle(cornerRadius:14).stroke(Design.line,lineWidth:1)).padding(.horizontal,18).padding(.bottom,10)
        }
    }
    private var composer:some View {
        VStack(spacing:0){
            VStack(alignment:.leading,spacing:12){
                ZStack(alignment:.topLeading){
                    if code.draft.isEmpty{Text("Describe what you want to build…").font(Design.body(14)).foregroundStyle(Design.muted).padding(.top,8).padding(.leading,4).allowsHitTesting(false)}
                    ComposerInput(text:$code.draft){run()}.frame(height:emptySession ? 66 : 58).disabled(code.active)
                }
                HStack(spacing:10){
                    Menu{Button("Build — implement and verify"){code.planOnly=false};Button("Plan — explore without changes"){code.planOnly=true}}label:{Label(code.planOnly ? "Plan" : "Build",systemImage:code.planOnly ? "map" : "hammer").font(Design.body(11))}.menuStyle(.borderlessButton).fixedSize().disabled(code.active)
                    Button{permissions.toggle()}label:{Label(code.allowProjectEdits ? "Edits allowed" : "Review edits",systemImage:"checkmark.shield").font(Design.body(10))}.buttonStyle(HitButtonStyle()).foregroundStyle(Design.muted)
                    Spacer()
                    Text("Churo Code").font(Design.medium(11)).foregroundStyle(Design.muted)
                    if code.active{Button{code.stop()}label:{Image(systemName:"stop.fill").frame(width:24,height:24)}.buttonStyle(WashedButtonStyle(compact:true)).accessibilityLabel("Stop agent")}
                    else{Button{run()}label:{Image(systemName:model.connected ? "arrow.up" : "power").font(.system(size:15,weight:.medium)).frame(width:24,height:24)}.buttonStyle(WashedButtonStyle(compact:true)).help(model.connected ? "Run task" : "Load Churo Code").accessibilityLabel(model.connected ? "Run task" : "Load Churo Code").keyboardShortcut(.return,modifiers:.command).disabled(model.loading || (model.connected && code.draft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty))}
                }
            }.padding(18).background{CodeMaterial(panel:true).clipShape(RoundedRectangle(cornerRadius:25))}.overlay(RoundedRectangle(cornerRadius:25).stroke(Design.line,lineWidth:1)).shadow(color:Design.ink.opacity(0.035),radius:16,y:6)
            HStack(spacing:12){
                Menu{
                    Button("Change project folder…"){code.chooseProject()}
                    if !code.recentProjects.isEmpty{Divider();ForEach(code.recentProjects,id:\.self){path in Button(path){code.switchProject(path)}}}
                    if let project=code.project{Divider();Button("Show in Finder"){NSWorkspace.shared.open(project.root)}}
                }label:{HStack(spacing:7){Image(systemName:"folder");Text(code.project?.root.lastPathComponent ?? "Choose project folder").lineLimit(1);Image(systemName:"chevron.down").font(.system(size:8))}.font(Design.body(11)).padding(.vertical,9).padding(.horizontal,6).contentShape(Rectangle())}.menuStyle(.borderlessButton).frame(maxWidth:250,alignment:.leading).disabled(code.active)
                Button{code.showPanel("Files")}label:{Label("Files",systemImage:"doc.on.doc").font(Design.body(10))}.buttonStyle(HitButtonStyle()).disabled(code.project==nil)
                if code.session?.changes.isEmpty == false{Button{code.showPanel("Changes")}label:{Label("Changes",systemImage:"square.stack.3d.up").font(Design.body(10))}.buttonStyle(HitButtonStyle())}
                Spacer(minLength:0)
                Image(systemName:"desktopcomputer").foregroundStyle(Design.muted).help("Model runs on this Mac")
            }.foregroundStyle(Design.muted).padding(.horizontal,14).padding(.top,2).padding(.bottom,5).background(Design.sidebar.opacity(0.6),in:UnevenRoundedRectangle(bottomLeadingRadius:18,bottomTrailingRadius:18)).padding(.horizontal,12)
            if model.settings.context<4096{Button("Enable coding context"){code.enableCodingContext()}.font(Design.body(10)).buttonStyle(HitButtonStyle()).disabled(model.loading || code.active)}
            if model.loading{HStack(spacing:7){ProgressView().controlSize(.mini);Text(model.status).font(Design.body(10)).foregroundStyle(Design.muted)}.padding(.top,10)}
        }.padding(.horizontal,26).padding(.top,12).padding(.bottom,20)
    }
    private func run(){
        if code.project==nil{let text=code.draft;code.chooseProject();code.draft=text;guard code.project != nil else{return}}
        if model.connected{code.start()}else{code.loadAndRun()}
    }
    private var inspector:some View {
        VStack(spacing:0){
            Picker("Workspace panel",selection:$code.panel){ForEach(["Files","Editor","Changes","Activity","Terminal"],id:\.self){Text($0)}}.labelsHidden().pickerStyle(.segmented).padding(12)
            Divider().overlay(Design.line)
            switch code.panel {
            case "Files":
                HStack{Image(systemName:"magnifyingglass");TextField("Find a file",text:$filter).textFieldStyle(.plain);Button{code.refreshFiles()}label:{Image(systemName:"arrow.clockwise")}.buttonStyle(QuietButtonStyle())}.font(Design.body(11)).foregroundStyle(Design.muted).padding(13)
                ScrollView{LazyVStack(alignment:.leading,spacing:1){ForEach(code.files.filter{filter.isEmpty || $0.localizedCaseInsensitiveContains(filter)},id:\.self){path in Button{code.openFile(path)}label:{HStack(spacing:8){Image(systemName:"doc.text").foregroundStyle(Design.green);Text(path).font(Design.mono(10)).lineLimit(2);Spacer()}.padding(.vertical,8).padding(.horizontal,12).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle())}.buttonStyle(.plain).disabled(code.active)}}}
            case "Editor":
                if let file=code.selectedFile{HStack{Text(file).font(Design.mono(10)).lineLimit(1);Spacer();if ["html","htm"].contains(URL(fileURLWithPath:file).pathExtension.lowercased()){Button("Open in browser"){code.openInBrowser()}.buttonStyle(HitButtonStyle()).disabled(code.active)};if code.editor != code.editorSaved{Button("Discard"){code.discardEditor()}.buttonStyle(HitButtonStyle());Button("Save"){code.saveEditor()}.buttonStyle(WashedButtonStyle(compact:true)).disabled(code.editorConflict)}}.padding(12);NativeCodeEditor(text:$code.editor,language:URL(fileURLWithPath:file).pathExtension,editable:!code.active).frame(maxHeight:.infinity)}else{empty("Choose a file from the project browser.",icon:"doc.text")}
            case "Changes":
                if code.session?.changes.isEmpty != false{empty("Applied changes appear here, with review and undo.",icon:"square.stack.3d.up")}
                ScrollView{LazyVStack(alignment:.leading,spacing:12){ForEach((code.session?.changes ?? []).reversed()){change in VStack(alignment:.leading,spacing:10){HStack{Image(systemName:change.undone ? "arrow.uturn.backward" : "doc.badge.gearshape").foregroundStyle(Design.green);Text(change.path).font(Design.mono(10));Spacer();if !change.undone{Button("Undo"){code.undo(change)}.buttonStyle(HitButtonStyle()).disabled(code.active)}};DisclosureGroup("Review diff"){ScrollView(.horizontal){Text(AgentDiff.text(before:change.before,after:change.after)).font(Design.mono(10)).textSelection(.enabled).padding(.top,8)}}.font(Design.body(10))}.padding(13).background(Design.surface,in:RoundedRectangle(cornerRadius:10))}}.padding(12)}
            case "Activity":
                ScrollView{LazyVStack(alignment:.leading,spacing:12){ForEach(code.session?.events ?? []){event in DisclosureGroup{VStack(alignment:.leading,spacing:8){ForEach(event.call.arguments.sorted(by:{$0.key<$1.key}),id:\.key){key,value in Text("\(key): \(String(value.prefix(1200)))").font(Design.mono(9)).textSelection(.enabled)};Divider();Text(event.output.isEmpty ? "Waiting…" : event.output).font(Design.mono(10)).textSelection(.enabled)}.frame(maxWidth:.infinity,alignment:.leading).padding(.top,8)}label:{HStack{Image(systemName:event.status=="done" ? "checkmark.circle" : event.status=="failed" ? "exclamationmark.circle" : "circle.dotted").foregroundStyle(Design.green);Text(event.call.name.replacingOccurrences(of:"_",with:" ")).font(Design.medium(11));Spacer();Text(event.status).font(Design.body(9)).foregroundStyle(Design.muted)}}.padding(12).background(Design.surface,in:RoundedRectangle(cornerRadius:10))}}.padding(12)}
            default:
                ScrollView([.vertical,.horizontal]){Text(code.terminalOutput.isEmpty ? "Approved terminal output appears here. Commands have time and output limits; Stop terminates their process group." : code.terminalOutput).font(Design.mono(11)).textSelection(.enabled).foregroundStyle(.white.opacity(0.85)).frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading).padding(15)}.frame(maxWidth:.infinity,maxHeight:.infinity).background(Design.code)
            }
        }.background(CodeMaterial(panel:true))
    }
    private func empty(_ text:String,icon:String)->some View{VStack(spacing:12){Image(systemName:icon).font(.system(size:26)).foregroundStyle(Design.green);Text(text).font(Design.body(11)).foregroundStyle(Design.muted).multilineTextAlignment(.center)}.padding(25).frame(maxWidth:.infinity,maxHeight:.infinity)}
}
struct NativeCodeEditor:NSViewRepresentable {
    @Binding var text:String
    let language:String
    let editable:Bool
    func makeCoordinator()->Coordinator{Coordinator(self)}
    func makeNSView(context:Context)->NSScrollView{
        let scroll=NSScrollView();scroll.hasVerticalScroller=true;scroll.hasHorizontalScroller=true
        let view=NSTextView();view.isRichText=false;view.usesAdaptiveColorMappingForDarkAppearance=false;view.insertionPointColor = .white;view.allowsUndo=true;view.isAutomaticQuoteSubstitutionEnabled=false;view.isAutomaticDashSubstitutionEnabled=false;view.isAutomaticTextReplacementEnabled=false;view.isAutomaticSpellingCorrectionEnabled=false;view.isContinuousSpellCheckingEnabled=false;view.usesFindBar=true;view.isVerticallyResizable=true;view.isHorizontallyResizable=false;view.autoresizingMask=[.width];view.minSize=NSSize(width:0,height:200);view.maxSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude);view.textContainer?.widthTracksTextView=true;view.setFrameSize(NSSize(width:400,height:500));view.textContainerInset=NSSize(width:14,height:14);view.font=NSFont(name:"SpaceMono-Regular",size:12);view.textColor = .white;view.delegate=context.coordinator;scroll.documentView=view;return scroll
    }
    func updateNSView(_ scroll:NSScrollView,context:Context){guard let view=scroll.documentView as? NSTextView else{return};context.coordinator.parent=self;view.isEditable=editable;view.backgroundColor=NSColor(Design.code);if view.string != text{
            view.string=text;context.coordinator.highlight(view)
            // SwiftUI can first size the scroll view after assigning its text.
            // Lay out glyphs at that final width and start new files at the top.
            DispatchQueue.main.async{[weak view] in
                guard let view else{return}
                if let container=view.textContainer{view.layoutManager?.ensureLayout(for:container)}
                view.scrollRangeToVisible(NSRange(location:0,length:0));view.needsDisplay=true
            }
        }}
    @MainActor final class Coordinator:NSObject,NSTextViewDelegate{
        var parent:NativeCodeEditor
        var highlighting=false
        init(_ parent:NativeCodeEditor){self.parent=parent}
        func textDidChange(_ notification:Notification){guard !highlighting,let view=notification.object as? NSTextView else{return};parent.text=view.string;highlight(view)}
        func highlight(_ view:NSTextView){guard view.string.utf8.count<120_000 else{return};highlighting=true;let selection=view.selectedRanges;let highlighted=SyntaxHighlighter.render(view.string,language:parent.language);view.textStorage?.beginEditing();highlighted.enumerateAttributes(in:NSRange(location:0,length:highlighted.length)){attributes,range,_ in view.textStorage?.setAttributes(attributes,range:range)};view.textStorage?.endEditing();view.selectedRanges=selection;highlighting=false}
    }
}

struct CodeChangeReview:View {
    let change:AgentChange
    @StoredState private var selection="Diff"
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(alignment:.leading,spacing:14){
            HStack{VStack(alignment:.leading,spacing:4){Text("Review file change").font(Design.heading(25));Text(change.path).font(Design.mono(11))};Spacer();Button("Close review"){dismiss()}.buttonStyle(WashedButtonStyle(compact:true))}
            Text("\(change.before?.utf8.count ?? 0) bytes before · \(change.after.utf8.count) bytes after").font(Design.body(10)).foregroundStyle(Design.muted)
            Picker("Review",selection:$selection){ForEach(["Diff","Before","After"],id:\.self){Text($0)}}.pickerStyle(.segmented).labelsHidden()
            NativeCodeEditor(text:.constant(selection=="Before" ? change.before ?? "" : selection=="After" ? change.after : AgentDiff.text(before:change.before,after:change.after)),language:selection=="Diff" ? "" : URL(fileURLWithPath:change.path).pathExtension,editable:false)
        }.padding(24).frame(width:820,height:650).background(Design.paper).foregroundStyle(Design.ink)
    }
}

/// Raw native history remains intact; saved code is reviewed in the editor/diff.
struct CodeAssistantReply:View {
    let entry:AgentEntry
    let collapseCode:Bool
    var body:some View {
        VStack(alignment:.leading,spacing:12){
            HStack(spacing:8){ParticleMark(size:21);Text("Churo Code").font(Design.heading(18))}
            ForEach(TranscriptParser.parse(entry.text)){block in
                switch block.kind{
                case .thinking:EmptyView()
                case .text:MarkdownContent(source:block.text)
                case .code:
                    if collapseCode{DisclosureGroup{CodeCard(code:block.text,language:block.language).padding(.top,10)}label:{Label("Code · view if needed",systemImage:"chevron.left.forwardslash.chevron.right").font(Design.body(11)).foregroundStyle(Design.muted)}}
                    else{CodeCard(code:block.text,language:block.language)}
                }
            }
            ResponseFooter(metrics:entry.metrics)
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
}

struct CodeStreamPreview:View {
    let text:String
    private var visible:String{TranscriptParser.parse(text).compactMap{block -> String? in if case .text=block.kind{return block.text};return nil}.joined(separator:"\n")}
    var body:some View {
        VStack(alignment:.leading,spacing:8){
            if !visible.isEmpty{MarkdownContent(source:String(visible.suffix(2200)))}
            if text.contains("```"){Label("Preparing code · changes appear in the editor",systemImage:"chevron.left.forwardslash.chevron.right").font(Design.body(10)).foregroundStyle(Design.muted)}
        }
    }
}
