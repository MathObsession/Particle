import Foundation
import AppKit
import ParticleOneCore
import ParticleModelLibrary
import ParticleDesktopProtocol

extension WorkspaceModel {
    var modelInstallationActive: Bool {installer.running || library.converting || nativeLibraryProcess != nil || nativeLibraryTask != nil}
    var activeConversationModelID:String {codeMode ? (code.session?.modelID ?? settings.modelID) : (selected?.modelID ?? ModelDescriptor.sarvam.id)}
    var conversationUsesCurrentModel:Bool {
        guard activeConversationModelID == settings.modelID else{return false}
        if settings.modelID == ModelDescriptor.sarvam.id {return true}
        return library.installed.contains{$0.id == settings.modelID && $0.path == settings.modelPath && Self.modelExists($0.path)}
    }
    var selectedModelDescriptor: ModelDescriptor {library.installed.first(where: {$0.path == settings.modelPath})?.descriptor ?? .sarvam}
    var selectedModelName: String {selectedModelDescriptor.id == ModelDescriptor.sarvam.id ? (codeMode ? "Churo Code" : "Churo Chat") : selectedModelDescriptor.name}
    var currentSystemPrompt: String {state.systemPrompt(model:selectedModelDescriptor)}
    func prepareConversationModel(_ id: String?, requireTools: Bool = false) -> Bool {
        let id = id ?? ModelDescriptor.sarvam.id
        if requireTools {
            let descriptor = library.installed.first(where:{$0.id == id})?.descriptor ?? .sarvam
            guard descriptor.supportsTools else {notice = "This session’s model has not qualified for coding tools.";return false}
        }
        if id == settings.modelID {
            let installed = library.installed.first(where:{$0.id == id})
            if installed?.path == settings.modelPath, Self.modelExists(settings.modelPath) {return true}
            // Legacy default installations can still use the setup flow without
            // a library record. Reimported packs otherwise restore their new path.
            if id == ModelDescriptor.sarvam.id, installed == nil {return true}
        }
        guard !busy,!loading,let installed = library.installed.first(where: {$0.id == id}),Self.modelExists(installed.path) else {
            notice = "This conversation’s model is not installed. Its transcript is preserved in the sidebar.";return false
        }
        var next = settings;next.modelID = installed.id;next.modelPath = installed.path
        next.slots = installed.profile.slots;next.context = installed.profile.context;next.activeExperts = installed.profile.activeExperts;next.useMirror = false
        next.readWorkers = installed.profile.readWorkers;next.prefillChunk = installed.profile.prefillChunk
        next.context=min(next.context,installed.descriptor.architecture == .sarvam ? ParticleContextPolicy.maximum() : ModelRuntimeProfile(model:installed.descriptor).context)
        guard next.isValid(for:installed.descriptor) else {notice="This model’s saved profile is unavailable on this Mac. Re-import it or choose supported model settings; your transcript is preserved.";return false}
        updateSettings(next);return true
    }
    func useInstalledModel(_ installed: InstalledModel) {
        guard !busy,!loading,!modelInstallationActive,installed.path != settings.modelPath else {return}
        guard ArchitectureRegistry.report(installed.descriptor).installable,Self.modelExists(installed.path) else {notice = "This model is unavailable or its runtime has not qualified yet.";return}
        var next = settings;next.modelPath = installed.path;next.modelID = installed.id;next.useMirror = false
        next.slots = installed.profile.slots;next.context = installed.profile.context;next.activeExperts = installed.profile.activeExperts
        next.readWorkers = installed.profile.readWorkers;next.prefillChunk = installed.profile.prefillChunk
        next.context=min(next.context,installed.descriptor.architecture == .sarvam ? ParticleContextPolicy.maximum() : ModelRuntimeProfile(model:installed.descriptor).context)
        guard next.isValid(for:installed.descriptor) else {notice="This model’s saved profile is unavailable on this Mac. Re-import it or choose supported settings.";return}
        // Model selection opens a new chat; returning to a saved code session
        // restores that session's original model binding.
        code.flushDraft();codeMode=false
        var chat = Conversation();chat.modelID = installed.id
        state.conversations.insert(chat,at:0);state.selected = chat.id;draft = "";rate = nil;contextUsed = 0
        updateSettings(next)
    }
    func installLibraryModel(_ listing: LibraryListing) {
        guard !modelInstallationActive,listing.report.installable else {return}
        // The default uses the existing independently pinned distribution receipt.
        guard listing.isDefault else {beginNativeLibraryInstall(listing);return}
        libraryInstallJob = library.beginDefaultJob(destination:library.managedRoot)
        guard libraryInstallJob != nil else {return}
        beginInstall(root:library.managedRoot,selectWhenComplete:false)
    }
    func resumeLibraryInstall(_ job: ModelInstallJob) {
        guard !installer.running,nativeLibraryProcess == nil,nativeLibraryTask == nil else {return}
        if job.localSource == nil,job.repository == ModelDescriptor.sarvam.provenance.repository,job.revision == ModelDescriptor.sarvam.provenance.revision {
            libraryInstallJob = job.id;beginInstall(root:URL(fileURLWithPath:job.destination,isDirectory:true),selectWhenComplete:false)
        } else {resumeNativeLibraryInstall(job)}
    }
    func cancelLibraryInstall(_ job: ModelInstallJob) {
        if nativeLibraryProcess != nil || nativeLibraryTask != nil {pauseNativeLibraryInstall(cancel:true)}else{installer.pause();libraryInstallJob = nil}
        library.updateJob(job.id,stage:.cancelled,progress:job.totalBytes > 0 ? Double(job.completedBytes)/Double(job.totalBytes) : 0,detail:"Cancelled. Verified partial files are retained for a future installation.")
    }
    func pauseLibraryInstall(_ job: ModelInstallJob) {
        if nativeLibraryProcess != nil || nativeLibraryTask != nil {pauseNativeLibraryInstall()}else{installer.pause()}
    }
    func importLibraryModel() {
        guard !busy,!loading,!code.active,!modelInstallationActive else {return}
        let picker = NSOpenPanel();picker.canChooseFiles = true;picker.canChooseDirectories = true;picker.allowsMultipleSelection = false
        picker.message = "Choose a .particle pack, Sarvam GGUF, or indexed MLX model folder. Files stay on this Mac."
        guard picker.runModal() == .OK,let source = picker.url else {return}
        inspectLocalLibraryModel(source)
    }
}
