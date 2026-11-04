import Foundation
import AppKit
import ParticleOneCore
import ParticleModelLibrary

@MainActor enum LibrarySmokeCheck {
    static func runNative(_ model:WorkspaceModel,source:URL,output:String) async {
        guard !model.preview,model.directory.path.hasPrefix("/private/tmp/particle-agent-native-"),
              ProcessInfo.processInfo.environment["PARTICLEONE_ENGINE_PATH"] == nil else {NSApp.terminate(nil);return}
        var checks=[String:Bool](),detail=""
        func wait(_ seconds:Double,_ condition:@MainActor ()->Bool) async -> Bool {
            let deadline=ProcessInfo.processInfo.systemUptime+seconds
            while ProcessInfo.processInfo.systemUptime < deadline {if condition(){return true};try? await Task.sleep(for:.milliseconds(150))}
            return false
        }
        do {
            let before=try ArchitectureRegistry.boundedData(source.appendingPathComponent("manifest.json"),limit:16*1024*1024)
            model.inspectLocalLibraryModel(source)
            checks["import_completed"]=await wait(240){!model.modelInstallationActive} && model.library.jobs.last?.stage == .ready
            guard let installed=model.library.installed.first(where:{$0.path == source.path}) else {throw ModelLibraryError.invalid(model.library.message ?? "The native import did not register its verified pack.")}
            checks["existing_pack_verified_in_place"]=installed.path == source.path && !installed.managed
            checks["import_has_persistent_ready_job"]=model.library.jobs.last?.localSource == source.path && model.library.jobs.last?.stage == .ready
            model.useInstalledModel(installed)
            checks["conversation_bound_to_imported_model"]=model.selected?.modelID == installed.id && model.settings.modelID == installed.id
            var settings=model.settings;settings.context=4096;settings.maxNew=32;settings.temperature=0;settings.responseStyle = .direct
            model.updateSettings(settings);model.loadModel()
            checks["native_worker_handshake"]=await wait(120){model.connected || model.notice != nil} && model.connected
            guard model.connected else {throw ModelLibraryError.invalid(model.notice ?? "The native model did not connect.")}
            model.draft="Reply with only the word ready.";model.submit()
            checks["native_response_completed"]=await wait(120){!model.busy} && model.selected?.messages.last?.complete == true
            checks["input_bytes_acknowledged"]=model.inputVerified == true
            checks["native_reply_received"]=model.selected?.messages.last?.content.lowercased().contains("ready") == true
            checks["source_manifest_unchanged"]=(try ArchitectureRegistry.boundedData(source.appendingPathComponent("manifest.json"),limit:16*1024*1024)) == before
            checks["no_model_payload_copy"]=(try FileManager.default.contentsOfDirectory(at:model.library.managedRoot,includingPropertiesForKeys:nil)).allSatisfy{$0.pathExtension != "particle"}
        } catch {detail=error.localizedDescription}
        model.unload();checks["owned_worker_stopped"]=await wait(30){model.modelProcessID == nil && !model.loading}
        let report:[String:Any]=["passed":detail.isEmpty && checks.values.allSatisfy{$0},"checks":checks,"detail":detail,"mode":"Native app import, full pack verification and chat; isolated data; one packaged worker; no model copy"]
        if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]) {try? data.write(to:URL(fileURLWithPath:output))}
        NSApp.terminate(nil)
    }
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
            guard model.library.register(other,path:outside.path,managed:false) else {throw ModelLibraryError.invalid("Imported fixture registration failed.")}
            for name in ["manifest.json","verified-install.json"] {try Data("{}".utf8).write(to:outside.appendingPathComponent(name))}
            if let imported=model.library.installed.first(where:{$0.id == other.id}) {model.useInstalledModel(imported)}
            model.library.remove(owned)
            checks["import_removal_forgets_entry_and_preserves_original"]=FileManager.default.fileExists(atPath:outside.path) && !model.library.installed.contains(where:{$0.id == other.id})
            checks["removing_current_import_keeps_readonly_transcript"]=model.selected?.modelID == other.id && !model.conversationUsesCurrentModel
            if let original {model.select(original)}
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
