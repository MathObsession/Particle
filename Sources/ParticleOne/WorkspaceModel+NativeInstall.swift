import Foundation
import ParticleModelLibrary

extension WorkspaceModel {
    func beginNativeLibraryInstall(_ listing: LibraryListing) {
        guard listing.report.installable,let card = listing.card,nativeLibraryProcess == nil,nativeLibraryTask == nil,!installer.running else {return}
        let destination = library.managedRoot.appendingPathComponent(listing.id+".particle")
        guard let id = library.beginJob(repository:card.repository,revision:card.revision,destination:destination,total:listing.descriptor.sourceBytes) else {return}
        launchNativeInstall(descriptor:listing.descriptor,jobID:id,destination:destination)
    }
    func resumeNativeLibraryInstall(_ job: ModelInstallJob) {
        guard nativeLibraryProcess == nil,nativeLibraryTask == nil else {return}
        if let path=job.localSource {inspectLocalLibraryModel(URL(fileURLWithPath:path),resuming:job);return}
        nativeLibraryTask = Task { [self] in
            do {
                guard HubRepository.validRevision(job.revision) else {throw ModelLibraryError.invalid("The saved installation has no immutable Hub revision.")}
                let client = HuggingFaceClient(),card = try await client.model(job.repository,revision:job.revision)
                let report = try await client.inspect(card)
                guard report.installable,let descriptor = report.descriptor else {throw ModelLibraryError.incompatible(report.reason)}
                let destination = library.managedRoot.appendingPathComponent(descriptor.id+".particle")
                guard destination.path == job.destination else {throw ModelLibraryError.invalid("The saved installation destination does not match this model.")}
                nativeLibraryTask = nil;launchNativeInstall(descriptor:descriptor,jobID:job.id,destination:destination)
            } catch {
                nativeLibraryTask = nil;library.updateJob(job.id,stage:error is CancellationError ? .paused : .failed,progress:0,detail:error.localizedDescription)
            }
        }
    }
    private func localDestination(_ source:URL,descriptor:ModelDescriptor)->URL {
        if FileManager.default.fileExists(atPath:source.appendingPathComponent("manifest.json").path) {return source}
        return library.managedRoot.appendingPathComponent(ModelDescriptor.identity(repository:"local",revision:descriptor.id,artifact:"particle")+".particle")
    }
    func inspectLocalLibraryModel(_ source:URL,resuming job:ModelInstallJob? = nil) {
        guard !modelInstallationActive,!busy,!loading,!code.active else{return}
        library.converting=true;nativeLibraryIntent=nil;libraryInstallJob=job?.id
        library.message="Inspecting the local configuration, tensor headers and tokenizer…"
        nativeLibraryTask=Task { [self] in
            var handedOff=false
            defer {if !handedOff {nativeLibraryTask=nil;library.converting=false;libraryInstallJob=nil}}
            do {
                let report=try await Task.detached{try ArchitectureRegistry.inspectLocal(source)}.value
                try Task.checkCancellation()
                guard report.installable,let descriptor=report.descriptor else {throw ModelLibraryError.incompatible(report.reason)}
                if job == nil,let existing=library.installed.first(where:{$0.id == descriptor.id}),Self.modelExists(existing.path) {
                    library.message="This model is already in Installed. Select Use model to begin.";return
                }
                let destination=localDestination(source,descriptor:descriptor)
                if let job {guard job.revision == descriptor.id,job.destination == destination.path else {throw ModelLibraryError.invalid("The local source or destination changed. Re-import the current source; existing files were preserved.")}}
                guard let id=job?.id ?? library.beginLocalJob(source:source,identity:descriptor.id,destination:destination,total:descriptor.sourceBytes) else {throw ModelLibraryError.invalid("The import could not be added to the installation queue.")}
                nativeLibraryTask=nil;library.converting=false;handedOff=true
                launchNativeInstall(descriptor:descriptor,jobID:id,destination:destination,localSource:source)
            } catch {
                library.message=error.localizedDescription
                if let job {library.updateJob(job.id,stage:nativeLibraryIntent ?? (error is CancellationError ? .paused : .failed),progress:0,detail:error.localizedDescription)}
            }
        }
    }
    private func launchNativeInstall(descriptor: ModelDescriptor,jobID: UUID,destination: URL,localSource:URL? = nil) {
        var arguments=[String]()
        let verifyingOriginal=localSource.map{FileManager.default.fileExists(atPath:$0.appendingPathComponent("manifest.json").path)} ?? false
        if let localSource {
            arguments=[verifyingOriginal ? "verify" : "convert","--source",localSource.path,"--expected-source-id",descriptor.id]
            if !verifyingOriginal {arguments += ["--output",destination.path,"--resume"]}
        } else {
            guard let repository=descriptor.provenance.repository,let revision=descriptor.provenance.revision else{return}
            arguments=["convert","--repo",repository,"--revision",revision,"--output",destination.path,"--resume"]
        }
        libraryInstallJob = jobID;nativeLibraryIntent = nil
        library.updateJob(jobID,stage:.queued,progress:library.jobs.first(where:{$0.id == jobID}).map{Double($0.completedBytes)/Double(max(1,$0.totalBytes))} ?? 0,detail:"Waiting for the current model to become idle")
        let processArguments=arguments
        let session = UUID();nativeLibrarySession = session
        nativeLibraryTask = Task { [self] in
            do {
                // Reading and conversion share one owner with inference. Wait for a foreground turn.
                while busy || loading || code.active {try await Task.sleep(for:.milliseconds(200));try Task.checkCancellation()}
                library.converting = true
                if connected {
                    unload()
                    for _ in 0..<150 {if !loading {break};try await Task.sleep(for:.milliseconds(100))}
                    guard !loading else {throw ModelLibraryError.invalid("The previous model is still stopping. Resume this installation after it closes.")}
                }
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at:library.managedRoot,withIntermediateDirectories:true)
                let engine = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/ParticleLibraryCLI")
                guard FileManager.default.isExecutableFile(atPath:engine.path) else {throw ModelLibraryError.invalid("The bundled native converter is missing. Reinstall Particle.")}
                let process = Process(),stdout = Pipe(),stderr = Pipe()
                process.executableURL = engine
                process.arguments = processArguments
                process.standardOutput = stdout;process.standardError = stderr;process.standardInput = FileHandle.nullDevice
                let output = DesktopPipeReader{[weak self] data in Task{@MainActor in
                    guard let self,self.nativeLibrarySession == session,
                          let object = try? JSONSerialization.jsonObject(with:data) as? [String:Any],object["event"] as? String == "conversion",
                          let completed = object["completed"] as? Int64,let total = object["total"] as? Int64,let detail = object["detail"] as? String else {return}
                    self.library.updateJobBytes(jobID,stage:detail.hasPrefix("Verifying") ? .verifying : .converting,completed:completed,total:total,detail:detail)
                }}
                let errors = DesktopPipeReader{[weak self] data in Task{@MainActor in
                    guard let self,self.nativeLibrarySession == session else{return}
                    self.library.message = String(String(decoding:data,as:UTF8.self).prefix(2000))
                }}
                stdout.fileHandleForReading.readabilityHandler={handle in let data = handle.availableData;if data.isEmpty{handle.readabilityHandler = nil}else{output.accept(data)}}
                stderr.fileHandleForReading.readabilityHandler={handle in let data = handle.availableData;if data.isEmpty{handle.readabilityHandler = nil}else{errors.accept(data)}}
                process.terminationHandler={ [weak self] process in
                    let status = process.terminationStatus
                    Task { @MainActor in
                        guard let self,self.nativeLibrarySession == session else{return}
                        self.nativeLibrarySession = UUID();self.nativeLibraryProcess = nil;self.nativeLibraryTask = nil;self.library.converting = false
                        defer{self.libraryInstallJob = nil}
                        if status == 0 {
                            do {
                                let report = try ArchitectureRegistry.inspectLocal(destination)
                                let convertedGGUF=localSource?.pathExtension.lowercased() == "gguf"
                                guard report.installable,let actual=report.descriptor,actual.architecture == descriptor.architecture,convertedGGUF || actual.id == descriptor.id else {throw ModelLibraryError.invalid("The converted model does not match this installation.")}
                                guard self.library.register(actual,path:destination.path,managed:!verifyingOriginal) else {throw ModelLibraryError.invalid("The model was verified, but its library record could not be saved. The model files were preserved.")}
                                self.library.updateJob(jobID,stage:.ready,progress:1,detail:"Verified and ready. Select Use model to begin.")
                            } catch {self.library.updateJob(jobID,stage:.failed,progress:0,detail:error.localizedDescription)}
                        } else {
                            let pressurePaused = self.library.message?.contains("Conversion paused because the Mac needs memory") == true
                            self.library.updateJob(jobID,stage:self.nativeLibraryIntent ?? (pressurePaused ? .paused : .failed),progress:self.library.jobs.first(where:{$0.id == jobID}).map{Double($0.completedBytes)/Double(max(1,$0.totalBytes))} ?? 0,
                                detail:self.nativeLibraryIntent == nil ? (self.library.message ?? "Conversion stopped. Verified progress is retained.") : "Paused. Verified progress is retained.")
                        }
                    }
                }
                library.updateJob(jobID,stage:.inspecting,progress:0,detail:"Inspecting the pinned source before conversion")
                try process.run();nativeLibraryProcess = process;nativeLibraryTask = nil
            } catch {
                nativeLibraryTask = nil;library.converting = false
                library.updateJob(jobID,stage:nativeLibraryIntent ?? (error is CancellationError ? .paused : .failed),progress:0,detail:error.localizedDescription)
                libraryInstallJob = nil
            }
        }
    }
    func pauseNativeLibraryInstall(cancel: Bool = false) {
        guard nativeLibraryProcess != nil || nativeLibraryTask != nil else {return}
        nativeLibraryIntent = cancel ? .cancelled : .paused
        nativeLibraryTask?.cancel();nativeLibraryProcess?.interrupt()
        if let id = libraryInstallJob {library.updateJob(id,stage:cancel ? .cancelled : .paused,progress:library.jobs.first(where:{$0.id == id}).map{Double($0.completedBytes)/Double(max(1,$0.totalBytes))} ?? 0,detail:"Stopped safely. Verified progress is retained.")}
    }
}
