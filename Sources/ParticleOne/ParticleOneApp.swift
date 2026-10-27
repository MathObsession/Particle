import SwiftUI
import AppKit
import TurboFieldfareMacPresentation

@MainActor final class ParticleOneDelegate: NSObject, NSApplicationDelegate {
    static var model: WorkspaceModel?
    private var previewWindow:NSWindow?
    private var quitting=false
    private var cleanupFinished=false
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        if let model=Self.model {
            model.updates.mayInstall = { [weak model] in model.map { !$0.busy && !$0.loading } ?? false }
            model.updates.prepareForRelaunch = { [weak model] in
                guard let model else { return }
                while model.busy || model.loading {try? await Task.sleep(for:.milliseconds(250))}
                model.updateRestartPending=true;model.persist();_ = model.shutdownWorker();await model.finishShutdown()
            }
            model.updates.start(disabled:model.preview || CommandLine.arguments.contains("--smoke-test"))
        }
        DispatchQueue.main.asyncAfter(deadline:.now()+1) {
            for window in NSApp.windows {window.isMovableByWindowBackground=true;window.titlebarAppearsTransparent=true;window.backgroundColor=NSColor(Design.paper)}
        }
        if let index=CommandLine.arguments.firstIndex(of:"--update-probe"),CommandLine.arguments.indices.contains(index+1),let model=Self.model {
            let output=CommandLine.arguments[index+1]
            Task {
                for _ in 0..<100 {
                    try? await Task.sleep(for:.milliseconds(200))
                    if !model.updates.checking {break}
                }
                let report:[String:Any] = ["configured":model.updates.configured,"checking":model.updates.checking,"available":model.updates.availableVersion ?? "","message":model.updates.message,"current":model.updates.currentVersion]
                if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:output))}
                NSApp.terminate(nil)
            }
        }
        if let index=CommandLine.arguments.firstIndex(of:"--agent-native-smoke"),CommandLine.arguments.indices.contains(index+1),let model=Self.model{let output=CommandLine.arguments[index+1];Task{await SmokeCheck.nativeAgent(model,output:output)}}
        if let index=CommandLine.arguments.firstIndex(of:"--web-smoke"),CommandLine.arguments.indices.contains(index+1){let output=CommandLine.arguments[index+1];Task{await SmokeCheck.web(output:output)}}
        if let index=CommandLine.arguments.firstIndex(of:"--installer-smoke"),CommandLine.arguments.indices.contains(index+1) {
            let output=CommandLine.arguments[index+1];Task{await SmokeCheck.installer(output:output)}
        }
        if let index=CommandLine.arguments.firstIndex(of:"--smoke-test"),CommandLine.arguments.indices.contains(index+1),let model=Self.model {
            let output=CommandLine.arguments[index+1];Task{await SmokeCheck.run(model,output:output)}
        }
        if let position=CommandLine.arguments.firstIndex(of:"--render-preview"),CommandLine.arguments.indices.contains(position+1) {
            let path=CommandLine.arguments[position+1]
            DispatchQueue.main.asyncAfter(deadline:.now()+1) {
                guard let model=Self.model else{return}
                // A deterministic, process-owned preview avoids restored-window
                // state from another running copy of the app. It uses test data.
                for window in NSApp.windows{window.orderOut(nil)}
                let window=NSWindow(contentRect:NSRect(x:0,y:0,width:1280,height:860),styleMask:[.titled,.closable,.resizable,.fullSizeContentView],backing:.buffered,defer:false)
                window.isReleasedWhenClosed=false;window.titlebarAppearsTransparent=true;window.titleVisibility = .hidden
                window.appearance=NSAppearance(named:model.settings.appearance == .dark ? .darkAqua : .aqua)
                window.contentView=NSHostingView(rootView:ParticleOneRootView(model:model))
                window.center();window.makeKeyAndOrderFront(nil);self.previewWindow=window
            }
            DispatchQueue.main.asyncAfter(deadline:.now()+2.5) {self.capturePreview(path,attempt:0)}
        }
    }
    private func capturePreview(_ path: String,attempt: Int) {
        guard let view=(previewWindow ?? NSApp.windows.first(where:{$0.isVisible}))?.contentView,view.bounds.width>0,view.bounds.height>0,
              let bitmap=view.bitmapImageRepForCachingDisplay(in:view.bounds) else {
            if attempt<20 {DispatchQueue.main.asyncAfter(deadline:.now()+0.25){self.capturePreview(path,attempt:attempt+1)}}
            else {NSApp.terminate(nil)}
            return
        }
        view.layoutSubtreeIfNeeded()
        func prepareNativeText(_ parent:NSView){
            if let text=parent as? NSTextView,let container=text.textContainer{text.layoutManager?.ensureLayout(for:container);text.displayIfNeeded()}
            for child in parent.subviews{prepareNativeText(child)}
        }
        prepareNativeText(view);view.cacheDisplay(in:view.bounds,to:bitmap)
        if let data=bitmap.representation(using:.png,properties:[:]) {try? data.write(to:URL(fileURLWithPath:path))}
        func editor(in parent:NSView)->SendTextView? {if let text=parent as? SendTextView{return text};for child in parent.subviews{if let found=editor(in:child){return found}};return nil}
        if let input=editor(in:view),let model=Self.model {
            let frame=input.convert(input.bounds,to:view)
            let layout:[String:Any]=["width":view.bounds.width,"height":view.bounds.height,"flipped":view.isFlipped,"editor_x":frame.minX,"editor_y":frame.minY,"editor_width":frame.width,"editor_height":frame.height,"sidebar_collapsed":model.state.sidebarHidden,"thinking_enabled":model.thinkingEnabled,"welcome":model.selected?.messages.isEmpty != false]
            if let data=try? JSONSerialization.data(withJSONObject:layout,options:[.prettyPrinted,.sortedKeys]){try? data.write(to:URL(fileURLWithPath:path+".layout.json"))}
        }
        NSApp.terminate(nil)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication)->Bool {true}
    func applicationShouldTerminate(_ sender: NSApplication)->NSApplication.TerminateReply {
        if cleanupFinished {return .terminateNow}
        if quitting {return .terminateCancel}
        guard let model=Self.model,let process=model.shutdownWorker(),process.isRunning else {return .terminateNow}
        quitting=true
        // Keep the normal run loop alive while the owned worker stops. A nested
        // AppKit termination loop can otherwise strand a MainActor cleanup task.
        Task {await model.finishShutdown();cleanupFinished=true;NSApp.terminate(nil)}
        return .terminateCancel
    }
}
@main struct ParticleOneApp: App {
    @NSApplicationDelegateAdaptor private var delegate: ParticleOneDelegate
    @StoredState private var model: WorkspaceModel
    init() {
        Design.registerFonts()
        let preview=CommandLine.arguments.contains("--render-preview") || CommandLine.arguments.contains("--preview") || CommandLine.arguments.contains("--installer-smoke") || CommandLine.arguments.contains("--web-smoke") || CommandLine.arguments.contains("--update-probe")
        let value=WorkspaceModel(preview:preview,chatPreview:CommandLine.arguments.contains("--preview-chat"))
        _model=StoredState(initialValue:value);ParticleOneDelegate.model=value
    }
    var body: some Scene {
        Window("Particle",id:"main") {ParticleOneRootView(model:model)}
            .windowStyle(.hiddenTitleBar).defaultSize(width:1280,height:860)
            .commands {
                CommandGroup(replacing:.newItem){Button(model.codeMode ? "New session" : "New conversation"){if model.codeMode{model.code.newSession()}else{model.newConversation()}}.keyboardShortcut("n",modifiers:.command).disabled(model.busy || model.loading)}
                CommandGroup(after:.appInfo){UpdateMenuCommand(updater:model.updates,busy:model.busy || model.loading)}
                CommandGroup(replacing:.appSettings){Button("Settings…"){model.showSettings=true}.keyboardShortcut(",",modifiers:.command)}
                CommandMenu("Workspace") {Button("New workspace…"){model.showNewWorkspace=true}.disabled(model.busy || model.loading);Button("Open workspace folder…"){model.openWorkspaceFolder()}.disabled(model.busy || model.loading);Button("Create workspace folder…"){model.createWorkspaceFolder()}.disabled(model.busy || model.loading);Divider();Button("Saved memory…"){model.showMemory=true};Button("Toggle sidebar"){model.toggleSidebar()}.keyboardShortcut("s",modifiers:[.command,.control])}
                CommandMenu("Conversation") {Button("Delete conversation…",role:.destructive){model.requestConversationDeletion()}.disabled(model.busy || model.loading || model.selected?.messages.isEmpty != false);Button("Export conversation…"){model.exportConversation()}.disabled(model.selected?.messages.isEmpty != false);Button("Load model"){model.loadModel()}.disabled(model.connected || model.loading);Button("Stop response"){model.stop()}.keyboardShortcut(".",modifiers:.command).disabled(!model.busy)}
            }
    }
}
