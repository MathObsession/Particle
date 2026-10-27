import SwiftUI
import ParticleOneCore
import TurboFieldfareMacPresentation

struct NewWorkspaceView:View {
    @ObservedObject var model:WorkspaceModel
    @StoredState private var name=""
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(alignment:.leading,spacing:18){
            Text("A fresh workspace.").font(Design.heading(28))
            Text("Keep a project’s conversations, instructions, and memories together. Each workspace has its own context.").font(Design.body(12)).foregroundStyle(Design.muted)
            TextField("For example, Writing or Personal",text:$name).textFieldStyle(.roundedBorder).font(Design.body(13)).onSubmit{create()}
            HStack{Button("Cancel"){dismiss()}.buttonStyle(HitButtonStyle());Spacer();Button("Create workspace"){create()}.buttonStyle(WashedButtonStyle(compact:true)).disabled(name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || name.count>80)}
        }.padding(30).frame(width:450).background(Design.paper).foregroundStyle(Design.ink).preferredColorScheme(model.settings.appearance.colorScheme)
    }
    private func create(){guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=80 else{return};model.createWorkspace(name);dismiss()}
}
struct MemoryDraftView:View {
    @Binding var text:String
    var save:()->Void
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(alignment:.leading,spacing:16){
            Text("Remember this.").font(Design.heading(27))
            Text("Save only the detail you want Particle to remember. It stays in this workspace on this Mac.").font(Design.body(11)).foregroundStyle(Design.muted)
            TextEditor(text:$text).font(Design.body(12)).scrollContentBackground(.hidden).padding(10).frame(height:130).background(Design.surface,in:RoundedRectangle(cornerRadius:12))
            HStack{Text("\(text.count) / 2000").font(Design.body(9)).foregroundStyle(Design.muted);Spacer();Button("Cancel"){dismiss()}.buttonStyle(HitButtonStyle());Button("Save memory"){save();dismiss()}.buttonStyle(WashedButtonStyle(compact:true)).disabled(text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || text.count>2000)}
        }.padding(28).frame(width:500).background(Design.paper).foregroundStyle(Design.ink)
    }
}
struct MemoryManagerView:View {
    @ObservedObject var model:WorkspaceModel
    @StoredState private var editing=false
    @StoredState private var editingID:UUID?
    @StoredState private var text=""
    @Environment(\.dismiss) private var dismiss
    var body:some View {
        VStack(alignment:.leading,spacing:18){
            HStack{VStack(alignment:.leading,spacing:5){Text("A little memory.").font(Design.heading(28));Text("\(model.state.name) · stored privately on this Mac").font(Design.body(11)).foregroundStyle(Design.muted)};Spacer();Button("Done"){dismiss()}.buttonStyle(WashedButtonStyle(compact:true))}
            if model.state.memories.isEmpty {VStack(spacing:15){Image(systemName:"brain").font(.system(size:30)).foregroundStyle(Design.green);Text("Only what you choose to remember.").font(Design.heading(21));Text("Save a message with the memory button, add a note here, or enable remembering personal details in Settings.").font(Design.body(12)).multilineTextAlignment(.center).foregroundStyle(Design.muted)}.padding(.vertical,45).frame(maxWidth:.infinity)}
            else {ScrollView{VStack(spacing:12){ForEach(model.state.memories){memory in
                HStack(alignment:.top){Text(memory.text).font(Design.body(12)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                    Button{editingID=memory.id;text=memory.text;editing=true}label:{Image(systemName:"pencil")}.buttonStyle(QuietButtonStyle()).help("Edit memory")
                    Button{model.removeMemory(memory.id)}label:{Image(systemName:"trash")}.buttonStyle(QuietButtonStyle()).help("Forget memory")
                }.padding(14).background(Design.sidebar,in:RoundedRectangle(cornerRadius:12))
            }}}}
            HStack{Text("\(model.state.memories.count) / 100 saved notes").font(Design.body(10)).foregroundStyle(Design.muted);Spacer();Button("Add memory"){editingID=nil;text="";editing=true}.buttonStyle(WashedButtonStyle(compact:true))}
        }.padding(28).frame(width:620,height:530).background(Design.paper).foregroundStyle(Design.ink).preferredColorScheme(model.settings.appearance.colorScheme)
            .sheet(isPresented:$editing){MemoryDraftView(text:$text){if let id=editingID{model.editMemory(id,text:text)}else{model.saveMemory(text)}}}
            .disabled(model.busy || model.loading)
    }
}

struct MessageEditRequest:Identifiable {
    let conversationID:UUID
    let message:ChatMessage
    var id:UUID {message.id}
}
struct EditMessageView:View {
    @ObservedObject var model:WorkspaceModel
    let request:MessageEditRequest
    @StoredState private var text:String
    @Environment(\.dismiss) private var dismiss
    init(model:WorkspaceModel,request:MessageEditRequest){self.model=model;self.request=request;_text=StoredState(initialValue:request.message.content)}
    private var valid:Bool {!text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && text.utf8.count<900_000}
    var body:some View {
        VStack(alignment:.leading,spacing:18){
            Text("Edit your message").font(Design.heading(27))
            Text("Churo Chat will answer again from this point in an edited continuation. Your original conversation is kept.").font(Design.body(11)).foregroundStyle(Design.muted)
            TextEditor(text:$text).font(Design.body(13)).scrollContentBackground(.hidden).padding(12).frame(height:190).background(Design.surface,in:RoundedRectangle(cornerRadius:12)).overlay(RoundedRectangle(cornerRadius:12).stroke(Design.line,lineWidth:1)).accessibilityLabel("Edited message")
            if !model.connected {Text(model.loading ? "Loading the model…" : "Load the model to generate the edited answer.").font(Design.body(11)).foregroundStyle(Design.muted)}
            HStack{Button("Cancel"){dismiss()}.buttonStyle(HitButtonStyle());Spacer();if !model.connected{Button("Load model"){model.loadModel()}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.loading)};Button("Save & regenerate"){if model.editAndRegenerate(conversationID:request.conversationID,messageID:request.id,text:text){dismiss()}}.buttonStyle(WashedButtonStyle(compact:true)).keyboardShortcut(.return,modifiers:.command).disabled(!valid || !model.connected || model.busy || model.loading)}
        }.padding(28).frame(width:590).background(Design.paper).foregroundStyle(Design.ink).preferredColorScheme(model.settings.appearance.colorScheme)
    }
}
