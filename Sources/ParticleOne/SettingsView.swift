import SwiftUI
import AppKit
import ParticleOneCore
import TurboFieldfareMacPresentation

struct ParticleOneSettings: View {
    @ObservedObject var model: WorkspaceModel
    @StoredState private var value: OneSettings
    @StoredState private var username:String
    @StoredState private var workspaceName:String
    @Environment(\.dismiss) private var dismiss
    init(model: WorkspaceModel) {self.model=model;_value=StoredState(initialValue:model.settings);_username=StoredState(initialValue:model.state.username);_workspaceName=StoredState(initialValue:model.state.name)}
    var body: some View {
        VStack(alignment:.leading,spacing:0){
            HStack{VStack(alignment:.leading,spacing:5){Text("A space that suits you.").font(Design.heading(28));Text("The model, the memory, the little details.").font(Design.body(11)).foregroundStyle(Design.muted)};Spacer();Button{model.setUsername(username);model.renameWorkspace(workspaceName);model.updateSettings(value);dismiss()}label:{Text("Done")}.buttonStyle(WashedButtonStyle(compact:true))}.padding(28)
            Divider().overlay(Design.line)
            ScrollView {
                VStack(alignment:.leading,spacing:25){
                    section("Your space"){
                        HStack{TextField("Username",text:$username).textFieldStyle(.roundedBorder);Button{username=LocalUsername.generate()}label:{Image(systemName:"arrow.triangle.2.circlepath")}.buttonStyle(QuietButtonStyle()).help("Generate a new name")}
                        Text("A name just for this Mac. No account needed.").font(Design.body(10)).foregroundStyle(Design.muted)
                        TextField("Workspace name",text:$workspaceName).textFieldStyle(.roundedBorder)
                        settingRow("Appearance"){Picker("Appearance",selection:$value.appearance){ForEach(AppAppearance.allCases,id:\.self){Text($0.title).tag($0)}}.labelsHidden().pickerStyle(.segmented).frame(width:260)}
                    }
                    section("Responses"){
                        Picker("Thinking",selection:$value.responseStyle){ForEach(ResponseStyle.allCases,id:\.self){Text($0.title).tag($0)}}.pickerStyle(.segmented).labelsHidden()
                        Text("Direct answers starts after an empty thinking block and enforces a zero-token thinking budget. Brief thinking allows up to 64 reasoning tokens. Full thinking uses the model’s original behavior. Hard questions may benefit from thinking.").font(Design.body(10)).lineSpacing(4).foregroundStyle(Design.muted)
                    }.disabled(model.busy || model.loading)
                    section("Your model"){
                        HStack{ParticleMark(size:42);VStack(alignment:.leading,spacing:4){Text(model.codeMode ? "Churo Code" : "Churo Chat").font(Design.medium(14));Text("4-bit · runs on your Mac").font(Design.body(10)).foregroundStyle(Design.muted)};Spacer();if model.connected{Button("Unload"){model.unload()}.disabled(model.busy)}}
                        Text("Stored on: "+URL(fileURLWithPath:value.modelPath).deletingLastPathComponent().lastPathComponent).font(Design.mono(9)).foregroundStyle(Design.muted).lineLimit(2).textSelection(.enabled)
                        HStack{Button("Choose existing model…"){model.chooseModel();value.modelPath=model.settings.modelPath}.disabled(model.connected || model.busy || model.loading);Button("Download model…"){model.showSettings=false;DispatchQueue.main.asyncAfter(deadline:.now()+0.3){model.showSetup=true}}.disabled(model.connected || model.busy || model.loading)}.font(Design.body(11))
                    }
                    section("Make it yours"){
                        settingRow("Active routed experts") {Stepper("\(value.activeExperts) of 6",value:$value.activeExperts,in:2...6).font(Design.mono(12)).frame(width:130)}
                        Text(value.activeExperts==6 ? "The original model’s expert selection." : "Experimental: uses fewer experts for speed. Answers can change; measured quality loss varies by task.").font(Design.body(10)).foregroundStyle(Design.muted)
                        settingRow("RAM expert slots") {Stepper("\(value.slots)",value:$value.slots,in:6...16).font(Design.mono(12)).frame(width:105)}
                        Text(String(format:"Expert buffers: about %.2f GiB. More slots may reduce disk reads; leave room for macOS.",Double(value.slots)*0.1267)).font(Design.body(10)).foregroundStyle(Design.muted)
                        settingRow("Conversation capacity") {Picker("Conversation capacity",selection:$value.context){ForEach([512,1024,2048,4096,8192],id:\.self){Text("\($0) tokens").tag($0)}}.labelsHidden().frame(width:160)}
                        settingRow("Response allowance") {Picker("Response allowance",selection:$value.maxNew){ForEach([128,256,512,768,1024,2048,4096],id:\.self){Text("\($0) tokens").tag($0)}}.labelsHidden().frame(width:160)}
                        Text("The response allowance is capped at half the chosen capacity. Earlier turns are retained; you’ll be told when the conversation is full.").font(Design.body(10)).foregroundStyle(Design.muted)
                        Toggle("Use the prepared SSD mirror when available",isOn:$value.useMirror).font(Design.body(11))
                    }.disabled(model.busy || model.loading)
                    section("Custom system prompt"){
                        Text("Private instructions for every conversation in this workspace. They are sent as the system prompt and never shown as a chat message.").font(Design.body(10)).foregroundStyle(Design.muted)
                        TextEditor(text:$value.system).font(Design.body(11)).scrollContentBackground(.hidden).frame(height:88).padding(9).background(Design.surface,in:RoundedRectangle(cornerRadius:10)).overlay(RoundedRectangle(cornerRadius:10).stroke(Design.line,lineWidth:1)).disabled(model.busy || model.loading)
                        Text("Changing this prompt resets the model’s conversation context; your saved messages are retained.").font(Design.body(10)).foregroundStyle(Design.muted)
                    }
                    section("Private memory"){
                        Toggle("Remember personal details from my messages",isOn:$value.rememberPersonalDetails).font(Design.body(11)).disabled(model.busy || model.loading)
                        Text("Off by default. When enabled, explicit details such as your name, interests, and preferences are saved in this workspace only. You can also save a message manually. Review, edit, or forget any note.").font(Design.body(10)).foregroundStyle(Design.muted)
                        Button("Manage saved memory (\(model.state.memories.count))…"){model.updateSettings(value);model.showSettings=false;DispatchQueue.main.asyncAfter(deadline:.now()+0.25){model.showMemory=true}}.buttonStyle(HitButtonStyle()).font(Design.body(11))
                    }
                    UpdateSettings(updater:model.updates,busy:model.busy || model.loading)
                    DisclosureGroup("Model details"){
                        if let verified=model.inputVerified {Text(verified ? "Last submitted text: verified unchanged by the model worker." : "Last submitted text did not match the worker acknowledgment.").font(Design.body(10)).foregroundStyle(Design.muted)}
                        HStack{Text("\(model.contextUsed) / \(model.settings.context) context tokens");Spacer();if let rate=model.rate{Text(String(format:"%.2f tok/s",rate))}}.font(Design.mono(10)).padding(.top,12)
                        Text(model.diagnostics.suffix(20).joined(separator:"\n")).font(Design.mono(9)).foregroundStyle(Design.muted).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(.top,8)
                        Button("Copy diagnostic log"){NSPasteboard.general.clearContents();NSPasteboard.general.setString(model.diagnostics.joined(separator:"\n"),forType:.string)}.font(Design.body(11))
                    }.font(Design.medium(12))
                    Button("Open source notices…"){if let url=Bundle.module.url(forResource:"Notices",withExtension:"txt",subdirectory:"Resources"){NSWorkspace.shared.open(url)}}.buttonStyle(HitButtonStyle()).font(Design.body(10))
                    Text("Particle \(model.updates.currentVersion) · Made in India.\nPT Serif, Poppins & Space Mono are bundled under the SIL Open Font License. Model distribution: Apache 2.0.").font(Design.body(9)).lineSpacing(5).foregroundStyle(Design.muted)
                }.padding(28)
            }
        }.frame(width:700,height:760).background(Design.paper).foregroundStyle(Design.ink).preferredColorScheme(value.appearance.colorScheme)
    }
    private func section<Content: View>(_ title: String,@ViewBuilder content: ()->Content)->some View {VStack(alignment:.leading,spacing:12){Text(title).font(Design.heading(20));content()}}
    private func settingRow<Content: View>(_ title: String,@ViewBuilder content: ()->Content)->some View {HStack{Text(title).font(Design.body(12));Spacer();content()}}
}

struct ModelSetupView: View {
    @ObservedObject var model: WorkspaceModel
    @ObservedObject var installer: ModelInstaller
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(alignment:.leading,spacing:20){
            FadingHeroBanner(height:168)
            Text("Make yourself at home.").font(Design.heading(31))
            Text("Particle is getting Churo Chat ready for your Mac. Once installed, your conversations work locally.").font(Design.body(12)).lineSpacing(5).foregroundStyle(Design.muted)
            VStack(alignment:.leading,spacing:10){HStack{Text("Churo Chat · native .particle").font(Design.medium(12));Spacer();Text(String(format:"%.0f%%",installer.progress*100)).font(Design.mono(11))};ProgressView(value:installer.progress).tint(Design.green);Text(installer.status).font(Design.body(10)).foregroundStyle(Design.muted).lineLimit(2)}.padding(17).background(Design.sidebar,in:RoundedRectangle(cornerRadius:14))
            if let error=installer.error {Text(error).font(Design.body(11)).foregroundStyle(Design.ink).textSelection(.enabled)}
            Text("About 19.6 GB · a verified public model download. Downloads resume after a pause and every file is verified. No Hugging Face token is needed.").font(Design.body(10)).lineSpacing(4).foregroundStyle(Design.muted)
            HStack {
                Button("Use an existing model…"){installer.pause();model.chooseModel()}.disabled(installer.running)
                Spacer()
                if installer.running {Button("Pause download"){installer.pause()}.buttonStyle(WashedButtonStyle(compact:true))}
                else {Button("Choose disk…"){let panel=NSOpenPanel();panel.canChooseDirectories=true;panel.canChooseFiles=false;panel.message="Choose a model folder on a disk with at least 22 GB free.";if panel.runModal() == .OK,let url=panel.url{model.beginInstall(root:url)}}.font(Design.body(11));Button("Download model"){model.beginInstall()}.buttonStyle(WashedButtonStyle(compact:true))}
            }.font(Design.body(11))
            if !installer.running {Button("Later"){dismiss()}.font(Design.body(10)).foregroundStyle(Design.muted).buttonStyle(.plain)}
        }.padding(28).frame(width:590).background(Design.paper).foregroundStyle(Design.ink).preferredColorScheme(model.settings.appearance.colorScheme).interactiveDismissDisabled(installer.running)
    }
}
