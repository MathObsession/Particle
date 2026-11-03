import Foundation
import AppKit
import ParticleOneCore
import ParticleModelLibrary

@MainActor enum LibrarySmokeCheck {
    static func run(_ model:WorkspaceModel,output:String) {
        guard model.preview,model.directory.path.hasPrefix("/private/tmp/particleone-smoke-"),
              ProcessInfo.processInfo.environment["PARTICLEONE_ENGINE_PATH"] != nil else {NSApp.terminate(nil);return}
        var checks=[String:Bool]()
        do {
            let root=model.library.managedRoot
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            func fixture(_ name:String) throws -> URL {
                let path=root.appendingPathComponent(name+".particle")
                try FileManager.default.createDirectory(at:path,withIntermediateDirectories:false)
                // Explicit metadata-only fixtures; preview mode forbids running a worker.
                for name in ["manifest.json","verified-install.json"] {try Data("{}".utf8).write(to:path.appendingPathComponent(name))}
                return path
            }
            let first=try fixture("default"),second=try fixture("other")
            guard model.library.register(.sarvam,path:first.path,managed:true) else {throw ModelLibraryError.invalid("Default fixture registration failed.")}
            var settings=model.settings;settings.modelID=ModelDescriptor.sarvam.id;settings.modelPath=first.path;settings.slots=8;settings.activeExperts=6;settings.useMirror=false
            model.updateSettings(settings)
            let original=model.state.selected
            let other=ModelDescriptor(id:"fixture.second",name:"Fixture Second",architecture:.sarvam,artifact:.particle,provenance:.init(),
                quantization:.init(scheme:"Fixture"),layers:19,experts:128,activeExperts:6,contextLimit:131072,sourceBytes:16,kvBytesPerToken:38912,supportsTools:true)
            guard model.library.register(other,path:second.path,managed:true),let installed=model.library.installed.first(where:{$0.id == other.id}) else {throw ModelLibraryError.invalid("Second fixture registration failed.")}
            model.useInstalledModel(installed)
            checks["switch_opens_bound_conversation"]=model.selected?.modelID == other.id && model.state.selected != original && model.settings.modelID == other.id
            checks["other_identity_keeps_accurate_name"]=model.currentSystemPrompt.contains("Fixture Second") && !model.currentSystemPrompt.contains("Your underlying base model is Sarvam")
            let secondConversation=model.state.selected
            if let original {model.select(original)}
            checks["return_restores_original_model_and_profile"]=model.settings.modelID == ModelDescriptor.sarvam.id && model.settings.modelPath == first.path && model.settings.slots == 8
            checks["conversations_preserved"]=model.state.conversations.contains(where:{$0.id == secondConversation && $0.modelID == other.id})
            let outside=model.directory.appendingPathComponent("user-original.particle")
            try FileManager.default.createDirectory(at:outside,withIntermediateDirectories:false)
            let owned=InstalledModel(descriptor:other,path:outside.path,managed:false)
            model.library.remove(owned)
            checks["user_owned_original_never_removed"]=FileManager.default.fileExists(atPath:outside.path)
            model.library.message=nil
            model.library.remove(installed)
            checks["managed_removal_keeps_conversation"] = !FileManager.default.fileExists(atPath:second.path) && model.state.conversations.contains(where:{$0.id == secondConversation})
            if let secondConversation {model.select(secondConversation)}
            checks["missing_model_transcript_readable_without_fallback"]=model.state.selected == secondConversation && !model.conversationUsesCurrentModel && model.settings.modelID == ModelDescriptor.sarvam.id
            let id=model.library.beginJob(repository:"fixture/model",revision:String(repeating:"a",count:40),destination:root.appendingPathComponent("download.particle"),total:1024)
            let reopened=ModelLibraryStore(root:model.directory)
            checks["restart_pauses_pending_install"]=id != nil && reopened.jobs.first(where:{$0.id == id})?.stage == .paused
            checks["registry_and_profile_survive_restart"]=reopened.installed.count == 1 && reopened.installed.first?.profile.slots == 8
            if let id {model.library.updateJob(id,stage:.cancelled,progress:0,detail:"Fixture stopped")}
            let local=model.library.beginLocalJob(source:outside,identity:"fixture-local",destination:outside,total:1024)
            let localRestart=ModelLibraryStore(root:model.directory)
            checks["local_import_uses_persistent_queue"]=local != nil && localRestart.jobs.first(where:{$0.id == local})?.localSource == outside.path && localRestart.jobs.first(where:{$0.id == local})?.stage == .paused
            checks["recommended_default_available_without_network"]=localRestart.recommended.first?.isDefault == true
            checks["no_worker_started"]=model.modelProcessID == nil && !model.connected
        } catch {checks["fixture_completed"]=false;model.library.message=error.localizedDescription}
        let report:[String:Any]=["passed":checks.values.allSatisfy{$0},"checks":checks,"detail":model.library.message ?? "","mode":"Metadata-only model-library fixtures; isolated app data; no inference"]
        if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]) {try? data.write(to:URL(fileURLWithPath:output))}
        NSApp.terminate(nil)
    }
}
