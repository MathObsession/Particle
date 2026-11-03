import Foundation
import AppKit
import ParticleModelLibrary
import ParticleOneCore

struct LibraryListing: Codable, Identifiable, Sendable {
    var id: String { descriptor.id }
    let descriptor: ModelDescriptor
    let card: HubModelCard?
    let report: CompatibilityReport
    let readme: String?
    init(descriptor: ModelDescriptor, card: HubModelCard?, report: CompatibilityReport, readme: String? = nil) {
        self.descriptor = descriptor;self.card = card;self.report = report;self.readme = readme
    }
    var isDefault: Bool { descriptor.id == ModelDescriptor.sarvam.id }
}

@MainActor final class ModelLibraryStore: ObservableObject {
    @Published var registry = InstalledModelRegistry()
    @Published var listings = [LibraryListing]()
    @Published var jobs = [ModelInstallJob]()
    @Published var searching = false
    @Published var converting = false
    @Published var optimizing = false
    @Published var message: String?
    @Published var search = ""
    @Published var repository = ""
    @Published var next: URL?
    private var work: Task<Void,Never>?
    private let client = HuggingFaceClient()
    private let root: URL
    private var saveBlocked = false
    private var retryAfter: Date?
    private var searchGeneration = UUID()
    var managedRoot: URL { root.appendingPathComponent("Models",isDirectory:true) }
    var installed: [InstalledModel] { registry.models.sorted { $0.descriptor.name.localizedCaseInsensitiveCompare($1.descriptor.name) == .orderedAscending } }
    var recommended: [LibraryListing] {
        let eligible=ModelRecommendations.eligible(listings.map(\.descriptor))
        return eligible.compactMap { descriptor in listings.first { $0.id == descriptor.id && $0.report.installable } }
    }
    init(root: URL) {
        self.root = root
        let standard = LibraryListing(descriptor:.sarvam,card:nil,report:ArchitectureRegistry.report(.sarvam))
        listings = [standard]
        do {
            registry = try InstalledModelRegistry.load(from:root.appendingPathComponent("installed-models.json"))
        } catch {saveBlocked = true;message = "The installed-model registry could not be opened. Existing records and weights have been preserved. \(error.localizedDescription)"}
        do {
            let cache = root.appendingPathComponent("model-library-cache.json")
            if FileManager.default.fileExists(atPath:cache.path) {
                let decoded = try JSONDecoder().decode([LibraryListing].self,from:ArchitectureRegistry.boundedData(cache,limit:4*1024*1024))
                if let cached = decoded.first(where:{$0.isDefault}) {listings = [.init(descriptor:.sarvam,card:cached.card,report:standard.report,readme:cached.readme)]}
                listings += decoded.filter { $0.id != standard.id && (try? $0.descriptor.validate()) != nil && ArchitectureRegistry.report($0.descriptor).installable }
            }
        } catch {message = "Cached browsing is unavailable. Refresh Discover to fetch model information again."}
        do {
            let queue = root.appendingPathComponent("model-install-queue.json")
            if FileManager.default.fileExists(atPath:queue.path) {
                jobs = try JSONDecoder().decode([ModelInstallJob].self,from:ArchitectureRegistry.boundedData(queue,limit:1_048_576))
                for index in jobs.indices where ![.ready,.cancelled,.failed].contains(jobs[index].stage) {jobs[index].stage = .paused;jobs[index].detail = "Paused after app restart"}
            }
        } catch {message = "The saved download queue could not be read. Installed models and existing partial files have been preserved."}
    }
    @discardableResult func register(_ descriptor: ModelDescriptor, path: String, managed: Bool) -> Bool {
        guard !saveBlocked else {return false}
        do {
            let previous=registry.models.first(where:{$0.id == descriptor.id})
            var value=previous ?? InstalledModel(descriptor:descriptor,path:path,managed:managed)
            if value.path != path {value=InstalledModel(descriptor:descriptor,path:path,managed:managed);if let previous{value.profile=previous.profile}}
            value.profile.expertBytesPerSlot = Self.bytesPerSlot(path)
            var next = registry;try next.register(value);try next.save(to:root.appendingPathComponent("installed-models.json"));registry = next;return true
        }
        catch {message = error.localizedDescription;return false}
    }
    func rememberExistingSarvam(_ path: String, settings: OneSettings) {
        guard !saveBlocked,!registry.models.contains(where: {$0.path == path}), WorkspaceModel.modelExists(path) else {return}
        if path.hasSuffix(".particle") {
            guard let data = try? ArchitectureRegistry.boundedData(URL(fileURLWithPath:path).appendingPathComponent("manifest.json"),limit:16*1024*1024),
                  let manifest = try? JSONSerialization.jsonObject(with:data) as? [String:Any],manifest["architecture"] as? String == "sarvam_moe" else {return}
        }
        do {
            var model = InstalledModel(descriptor:.sarvam,path:path,managed:URL(fileURLWithPath:path).deletingLastPathComponent().standardizedFileURL == managedRoot.standardizedFileURL)
            model.profile.slots = settings.slots;model.profile.context = settings.context;model.profile.activeExperts = settings.activeExperts
            model.profile.expertBytesPerSlot = Self.bytesPerSlot(path)
            try registry.register(model);try registry.save(to:root.appendingPathComponent("installed-models.json"))
        } catch {message = error.localizedDescription}
    }
    func updateProfile(path: String, settings: OneSettings) {
        guard var value = registry.models.first(where: {$0.path == path}), !saveBlocked else {return}
        value.profile.slots = settings.slots;value.profile.context = settings.context;value.profile.activeExperts = settings.activeExperts
        value.profile.readWorkers = settings.readWorkers;value.profile.prefillChunk = settings.prefillChunk
        do {try registry.register(value);try registry.save(to:root.appendingPathComponent("installed-models.json"))}catch{message = error.localizedDescription}
    }
    func saveCalibration(_ record:ModelCalibrationRecord,model:InstalledModel) throws {
        guard !saveBlocked,let index=registry.models.firstIndex(where:{$0.id == model.id}),registry.models[index].profile == record.baseline else {throw ModelLibraryError.invalid("The profile changed during optimization. Current settings were retained.")}
        var value=registry.models[index];value.calibration=record;value.profile=record.selected
        var next=registry;try next.register(value);try next.save(to:root.appendingPathComponent("installed-models.json"));registry=next
    }
    func discover(loadMore: Bool = false) {
        work?.cancel()
        let generation = UUID();searchGeneration = generation
        if let retryAfter, retryAfter > Date() {message = ModelLibraryError.rateLimited(retryAfter).localizedDescription;return}
        let query = search,page = loadMore ? next : nil
        work = Task {
            do {
                try await Task.sleep(for:.milliseconds(400));try Task.checkCancellation();searching = true;message = nil
                if !loadMore,let card = try? await client.model(ModelDescriptor.sarvam.provenance.repository!) {
                    let pinned = HubModelCard(repository:card.repository,revision:ModelDescriptor.sarvam.provenance.revision!,downloads:card.downloads,likes:card.likes,license:card.license,files:card.files)
                    let readme = try? await client.metadata(pinned,path:"README.md",limit:524_288)
                    try Task.checkCancellation();guard searchGeneration == generation else {return}
                    if let index = listings.firstIndex(where:{$0.isDefault}) {listings[index] = .init(descriptor:.sarvam,card:pinned,report:ArchitectureRegistry.report(.sarvam),readme:readme.map{Self.cardText($0)})}
                }
                let results = try await client.search(query,next:page)
                guard searchGeneration == generation else {return}
                // Retain cached cards until a replacement page succeeds, so
                // an offline refresh never erases the browseable library.
                if !loadMore {listings = listings.filter(\.isDefault)}
                var summaries=results.models
                let featured="Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit"
                if !loadMore,(query.isEmpty || featured.localizedCaseInsensitiveContains(query)),!summaries.contains(where:{$0.repository == featured}),
                   let card=try? await client.model(featured,revision:"a982bb2ed776bf0db64375de480a1d0a00475a99") {summaries.insert(card,at:0)}
                for summary in summaries {
                    try Task.checkCancellation()
                    guard summary.repository != ModelDescriptor.sarvam.provenance.repository else {continue}
                    // Tags only reduce requests; each surviving candidate still undergoes tensor inspection.
                    guard summary.tags.contains("qwen3_5_moe") || summary.repository.localizedCaseInsensitiveContains("Chronos") else {continue}
                    do {
                        let card = try await client.model(summary.repository,revision:summary.repository == featured ? "a982bb2ed776bf0db64375de480a1d0a00475a99" : nil)
                        let report = try await client.inspect(card)
                        try Task.checkCancellation();guard searchGeneration == generation else {return}
                        if report.installable,let descriptor = report.descriptor,!listings.contains(where: {$0.id == descriptor.id}) {
                            let readme = try? await client.metadata(card,path:"README.md",limit:524_288)
                            try Task.checkCancellation()
                            listings.append(.init(descriptor:descriptor,card:card,report:report,readme:readme.map{Self.cardText($0)}))
                        }
                    } catch is CancellationError {throw CancellationError()}
                    catch let error as ModelLibraryError {if case .rateLimited = error {throw error}}
                    catch {continue}
                }
                next = results.next;searching = false
                if listings.allSatisfy(\.isDefault),!query.isEmpty {message = "No additional qualified MoE models matched. You can inspect a repository directly for a compatibility explanation."}
                if !saveBlocked {try JSONEncoder().encode(listings).write(to:root.appendingPathComponent("model-library-cache.json"),options:.atomic)}
            } catch is CancellationError {if searchGeneration == generation {searching = false}}
            catch {
                guard searchGeneration == generation else {return}
                if let error = error as? ModelLibraryError,case .rateLimited(let date) = error {retryAfter = date}
                message = error.localizedDescription;searching = false
            }
        }
    }
    func inspectRepository() {
        work?.cancel()
        let generation = UUID();searchGeneration = generation
        work = Task {
            do {
                let repo = try HubRepository.parse(repository);searching = true;message = nil
                if repo == ModelDescriptor.sarvam.provenance.repository {message = "The default Sarvam release is available in Discover.";searching = false;return}
                let card = try await client.model(repo),report = try await client.inspect(card)
                guard searchGeneration == generation else {return}
                if report.installable,let descriptor = report.descriptor {
                    listings.removeAll {$0.id == descriptor.id};listings.append(.init(descriptor:descriptor,card:card,report:report));message = "Compatible model added to Discover."
                } else {message = report.reason}
                searching = false
            } catch {guard searchGeneration == generation else{return};message = error.localizedDescription;searching = false}
        }
    }
    func beginDefaultJob(destination: URL) -> UUID? {
        guard !saveBlocked,!jobs.contains(where: {[.queued,.inspecting,.downloading,.converting,.verifying].contains($0.stage)}) else {return nil}
        do {
            if let index=jobs.firstIndex(where:{$0.repository == ModelDescriptor.sarvam.provenance.repository && $0.revision == ModelDescriptor.sarvam.provenance.revision && $0.destination == destination.path && $0.stage != .ready}) {
                jobs[index].stage = .downloading;jobs[index].detail="Resuming the verified default release";try saveQueue();return jobs[index].id
            }
            var job = try ModelInstallJob(repository:ModelDescriptor.sarvam.provenance.repository!,revision:ModelDescriptor.sarvam.provenance.revision!,destination:destination.path)
            job.stage = .downloading;job.totalBytes = ModelDescriptor.sarvam.sourceBytes;job.detail = "Downloading the verified default release"
            jobs.append(job);try saveQueue();return job.id
        } catch {message = error.localizedDescription;return nil}
    }
    func updateJob(_ id: UUID, stage: ModelInstallStage, progress: Double, detail: String) {
        guard let index = jobs.firstIndex(where: {$0.id == id}) else {return}
        jobs[index].stage = stage;jobs[index].completedBytes = Int64(Double(jobs[index].totalBytes)*min(1,max(0,progress)));jobs[index].detail = detail
        do {try saveQueue()}catch{message = error.localizedDescription}
    }
    func beginJob(repository: String, revision: String, destination: URL, total: Int64) -> UUID? {
        guard !saveBlocked,!jobs.contains(where:{[.queued,.inspecting,.downloading,.converting,.verifying].contains($0.stage)}) else {return nil}
        do {
            var job = try ModelInstallJob(repository:repository,revision:revision,destination:destination.path)
            job.totalBytes = total;jobs.append(job);try saveQueue();return job.id
        } catch {message = error.localizedDescription;return nil}
    }
    func beginLocalJob(source:URL,identity:String,destination:URL,total:Int64)->UUID? {
        guard !saveBlocked,!jobs.contains(where:{[.queued,.inspecting,.downloading,.converting,.verifying].contains($0.stage)}) else{return nil}
        do {
            if let existing=jobs.first(where:{$0.localSource == source.path && $0.revision == identity && $0.destination == destination.path && $0.stage != .ready}) {return existing.id}
            var job=try ModelInstallJob(localSource:source,identity:identity,destination:destination);job.totalBytes=total;jobs.append(job);try saveQueue();return job.id
        } catch {message=error.localizedDescription;return nil}
    }
    func updateJobBytes(_ id: UUID, stage: ModelInstallStage, completed: Int64, total: Int64, detail: String) {
        guard let index = jobs.firstIndex(where:{$0.id == id}),completed >= 0,total > 0,completed <= total else {return}
        jobs[index].completedBytes = completed;jobs[index].totalBytes = total;jobs[index].stage = stage;jobs[index].detail = String(detail.prefix(1024))
        do {try saveQueue()}catch{message = error.localizedDescription}
    }
    func remove(_ model: InstalledModel) {
        guard !saveBlocked else {return}
        do {
            let url = try InstalledModelRegistry.removableURL(model,managedRoot:managedRoot)
            // Rename on the same volume first; a failed registry commit can
            // restore the model without copying its payload.
            let tombstone = managedRoot.appendingPathComponent(".removing-"+UUID().uuidString+".particle")
            try FileManager.default.moveItem(at:url,to:tombstone)
            var next = registry;next.forget(model.id)
            do {try next.save(to:root.appendingPathComponent("installed-models.json"))}
            catch {try? FileManager.default.moveItem(at:tombstone,to:url);throw error}
            registry = next
            try FileManager.default.removeItem(at:tombstone)
        } catch {message = error.localizedDescription}
    }
    private func saveQueue() throws {guard !saveBlocked else{return};try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);try JSONEncoder().encode(jobs).write(to:root.appendingPathComponent("model-install-queue.json"),options:.atomic)}
    func cancelDiscovery() {work?.cancel()}
    private static func bytesPerSlot(_ path: String) -> Int64? {
        let root = URL(fileURLWithPath:path)
        if let data = try? ArchitectureRegistry.boundedData(root.appendingPathComponent("packed_experts/layout.json"),limit:16*1024*1024),
           let object = try? JSONSerialization.jsonObject(with:data) as? [String:Any],let layers = object["layers"] as? [[String:Any]] {
            return layers.compactMap{$0["stride"] as? Int64}.reduce(0,+)
        }
        if let data = try? ArchitectureRegistry.boundedData(root.appendingPathComponent("manifest.json"),limit:16*1024*1024),
           let manifest = try? JSONDecoder().decode(ParticleV2Manifest.self,from:data) {
            let layers = Dictionary(grouping:manifest.tensors.filter{$0.layer != nil},by:{$0.layer!})
            return layers.values.compactMap{$0.first?.expertStride}.reduce(0,+)
        }
        return nil
    }
    private static func cardText(_ data: Data) -> String {
        var text = String(decoding:data,as:UTF8.self)
        if text.hasPrefix("---\n"),let boundary = text.dropFirst(4).range(of:"\n---") {text = String(text[boundary.upperBound...]).trimmingCharacters(in:.whitespacesAndNewlines)}
        return String(text.prefix(12_000))
    }
}
