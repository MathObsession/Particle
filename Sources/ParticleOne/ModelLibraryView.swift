import SwiftUI
import AppKit
import ParticleModelLibrary
import TurboFieldfareMacPresentation

struct ModelLibraryView: View {
    @ObservedObject var model: WorkspaceModel
    @ObservedObject var library: ModelLibraryStore
    @Environment(\.dismiss) private var dismiss
    @StoredState private var tab = "Discover"
    @StoredState private var removal: InstalledModel?
    @StoredState private var showRepository = false
    var body: some View {
        VStack(alignment:.leading,spacing:0) {
            HStack(alignment:.top) {
                VStack(alignment:.leading,spacing:7){Text("Your model library").font(Design.heading(29));Text("Discover and install models that run on this Mac.").font(Design.body(12)).foregroundStyle(Design.muted)}
                Spacer()
                Button{model.importLibraryModel()}label:{Label("Import model…",systemImage:"square.and.arrow.down")}.buttonStyle(HitButtonStyle()).disabled(model.busy || model.loading || library.converting || model.installer.running)
                Button{dismiss()}label:{Image(systemName:"xmark")}.buttonStyle(QuietButtonStyle()).accessibilityLabel("Close model library")
            }.padding(26).background {
                WashSurface(radius:0).mask(LinearGradient(stops:[.init(color:.black,location:0),.init(color:.black,location:0.3),.init(color:.black.opacity(0.75),location:0.6),.init(color:.clear,location:1)],startPoint:.top,endPoint:.bottom))
            }
            HStack(spacing:16){
                ForEach(["Discover","Installed","Downloads"],id:\.self){name in
                    Button{tab = name}label:{Text(name).font(Design.medium(12)).padding(.horizontal,14).padding(.vertical,10).frame(minHeight:40).contentShape(Rectangle())}.buttonStyle(.plain).background(tab == name ? Design.surface : .clear,in:Capsule())
                }
                Spacer()
            }.padding(.horizontal,24).padding(.vertical,14)
            if let text = library.message {Text(text).font(Design.body(11)).foregroundStyle(Design.muted).padding(.horizontal,28).padding(.bottom,12).textSelection(.enabled)}
            if library.optimizing {Button("Cancel optimization"){model.cancelModelOptimization()}.buttonStyle(HitButtonStyle()).padding(.horizontal,26).padding(.bottom,12)}
            ScrollView {
                VStack(alignment:.leading,spacing:16) {
                    if tab == "Discover" {discover}
                    else if tab == "Installed" {installed}
                    else {downloads}
                }.padding(.horizontal,26).padding(.bottom,26)
            }
            Text("MoE models only · Public Hugging Face sources · Conversion preserves supplied quantization").font(Design.body(9)).foregroundStyle(Design.muted).padding(20)
        }.frame(width:840,height:690).foregroundStyle(Design.ink).background{Design.paper.overlay{InterfaceGrain()}}
            .preferredColorScheme(model.settings.appearance.colorScheme)
            .alert("Remove this model?",isPresented:Binding(get:{removal != nil},set:{if !$0{removal = nil}})) {
                Button("Cancel",role:.cancel){removal = nil}
                Button("Remove model",role:.destructive){if let value = removal {library.remove(value)};removal = nil}
            } message:{Text(removal?.managed == false ? "Remove this entry from the library. Your original model files, conversations and memories are preserved." : "Only Particle’s managed model files are removed. Saved conversations and memories remain on this Mac.")}
            .onDisappear{library.cancelDiscovery()}
            .onAppear{if !model.preview{library.discover()}}
    }
    private var discover: some View {
        VStack(alignment:.leading,spacing:18) {
            HStack {
                Image(systemName:"magnifyingglass").foregroundStyle(Design.green)
                TextField("Search any MoE name or paste a Hugging Face link",text:$library.search).textFieldStyle(.plain).onChange(of:library.search){_,_ in library.discover()}
                if library.searching {ProgressView().controlSize(.small)}
                Button{showRepository.toggle()}label:{Image(systemName:"link").frame(width:30,height:30)}.buttonStyle(.plain).help("Add Hugging Face repository")
            }.font(Design.body(12)).padding(13).background(Design.surface,in:RoundedRectangle(cornerRadius:13))
            if showRepository {
                HStack{TextField("owner/model or Hugging Face URL",text:$library.repository).textFieldStyle(.roundedBorder);Button("Inspect"){library.inspectRepository()}.buttonStyle(HitButtonStyle()).disabled(library.searching)}.font(Design.body(11))
            }
            if library.search.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!library.recommended.isEmpty {
                VStack(alignment:.leading,spacing:5) {
                    Text("Recommended").font(Design.heading(23))
                    Text("A good place to start. Install once, then use your model on this Mac.").font(Design.body(11)).foregroundStyle(Design.muted)
                }
                ForEach(library.recommended) { listing in modelCard(listing,recommended:true) }
            }
            ForEach(library.listings.filter { listing in
                let searching = !library.search.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty
                return searching ? (!listing.isDefault || listing.descriptor.name.localizedCaseInsensitiveContains(library.search) || (listing.descriptor.provenance.repository?.localizedCaseInsensitiveContains(library.search) ?? false)) : !library.recommended.contains(where: { $0.id == listing.id })
            }) { listing in modelCard(listing,recommended:false) }
            ForEach(library.candidates.filter {candidate in !library.listings.contains(where:{$0.card?.repository == candidate.id})}) {candidate in
                card {
                    VStack(alignment:.leading,spacing:9) {
                        HStack(alignment:.top) {
                            Image(systemName:candidate.inspected ? "shippingbox" : "magnifyingglass").foregroundStyle(Design.green)
                            Text(candidate.card.repository).font(Design.medium(12)).textSelection(.enabled)
                            Spacer()
                            if !candidate.inspected {ProgressView().controlSize(.small)}
                            else {Text("Adapter needed").font(Design.body(10)).foregroundStyle(Design.muted)}
                        }
                        Text(candidate.detail).font(Design.body(11)).foregroundStyle(Design.muted).fixedSize(horizontal:false,vertical:true)
                        Text("\(candidate.card.downloads.formatted()) downloads · \(candidate.card.likes.formatted()) likes · \(candidate.card.license ?? "See source license")").font(Design.body(9)).foregroundStyle(Design.muted)
                        if let url=URL(string:"https://huggingface.co/"+candidate.id) {Link("View source and available variants ↗",destination:url).font(Design.body(10)).tint(Design.green)}
                    }
                }
            }
            Text("Particle automatically inspects the backend graph, identifies expert matrix operations and compiles a streamed .particle execution plan. Models with unsupported graph operations, tensor layouts or tokenizers show a specific explanation.").font(Design.body(11)).foregroundStyle(Design.muted).lineSpacing(4)
            if library.next != nil {Button("Load more"){library.discover(loadMore:true)}.buttonStyle(HitButtonStyle()).disabled(library.searching)}
        }
    }
    private func modelCard(_ listing: LibraryListing,recommended:Bool)->some View {
        card {
                    HStack(alignment:.top,spacing:16) {
                        Image(systemName:"cpu").font(.system(size:25)).foregroundStyle(Design.green).padding(.top,4)
                        VStack(alignment:.leading,spacing:7) {
                            HStack{Text(listing.descriptor.name).font(Design.heading(22));if listing.isDefault {Text("DEFAULT").font(Design.medium(8)).tracking(1).padding(6).background(Design.line,in:Capsule())} else if listing.descriptor.architecture != .sarvam {Text("EARLY ACCESS").font(Design.medium(8)).tracking(0.6).padding(6).background(Design.line,in:Capsule())}}
                            Text(listing.descriptor.provenance.repository ?? "Local model").font(Design.body(10)).foregroundStyle(Design.muted)
                            if recommended,let reason=ModelRecommendations.reason(for:listing.descriptor) {Text(reason).font(Design.body(11)).foregroundStyle(Design.muted).fixedSize(horizontal:false,vertical:true)}
                            Text("\(listing.descriptor.layers) layers · \(listing.descriptor.experts) total experts").font(Design.body(10)).foregroundStyle(Design.muted)
                            if let graph=listing.descriptor.graph {Text("Automatically adapted · \(graph.projections.count) streamed expert projections").font(Design.body(10)).foregroundStyle(Design.green)}
                            Text("\(listing.descriptor.activeExperts) active experts · \(listing.descriptor.quantization.label)").font(Design.body(11))
                            Text(ByteCountFormatter.string(fromByteCount:listing.descriptor.sourceBytes,countStyle:.file)+" download · "+(listing.descriptor.provenance.license ?? "See model license")).font(Design.body(10)).foregroundStyle(Design.muted)
                            Text("Allow "+ByteCountFormatter.string(fromByteCount:listing.descriptor.sourceBytes+2_147_483_648,countStyle:.file)+" free for installation").font(Design.body(10)).foregroundStyle(Design.muted)
                            let profile=ModelRuntimeProfile(model:listing.descriptor,physicalMemory:ProcessInfo.processInfo.physicalMemory)
                            Text((listing.descriptor.supportsTools ? "Chat and Code · " : "Chat only · ")+"\(profile.context.formatted()) context on this Mac · Internal SSD recommended").font(Design.body(10)).foregroundStyle(Design.muted)
                            if let card = listing.card {Text("\(card.downloads.formatted()) downloads · \(card.likes.formatted()) likes on Hugging Face").font(Design.body(9)).foregroundStyle(Design.muted)}
                            if let source = listing.descriptor.provenance.repository,let url = URL(string:"https://huggingface.co/"+source) {Link("Model card ↗",destination:url).font(Design.body(10)).tint(Design.green)}
                        }
                        Spacer(minLength:10)
                        if let installed = library.installed.first(where: {$0.id == listing.id}),WorkspaceModel.modelExists(installed.path) {
                            Button(model.settings.modelPath == installed.path ? "Selected" : "Use model"){model.useInstalledModel(installed)}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.busy || model.loading || model.settings.modelPath == installed.path)
                        } else {
                            Button("Install"){model.installLibraryModel(listing);tab = "Downloads"}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.modelInstallationActive || !listing.report.installable)
                        }
                    }
                    if let readme = listing.readme,!readme.isEmpty {DisclosureGroup("About this model"){MarkdownContent(source:readme,compact:true).padding(.top,10)}.font(Design.body(10)).padding(.top,12)}
                }
    }
    private var installed: some View {
        VStack(alignment:.leading,spacing:14) {
            if library.installed.isEmpty {Text("Install the default model or import an existing model to begin.").font(Design.body(12)).foregroundStyle(Design.muted)}
            ForEach(library.installed){value in
                card {
                    VStack(alignment:.leading,spacing:12) {
                        HStack{Text(value.descriptor.name).font(Design.heading(21));Spacer();Button("Use model"){model.useInstalledModel(value)}.buttonStyle(WashedButtonStyle(compact:true)).disabled(model.busy || model.loading || model.settings.modelPath == value.path || !WorkspaceModel.modelExists(value.path))}
                        if !WorkspaceModel.modelExists(value.path) {Text("Model files are unavailable. Reinstall from Discover or import the source again; conversations are retained.").font(Design.body(10)).foregroundStyle(Design.muted)}
                        Text(value.descriptor.quantization.label+" · "+ByteCountFormatter.string(fromByteCount:value.descriptor.sourceBytes,countStyle:.file)).font(Design.body(11)).foregroundStyle(Design.muted)
                        HStack {
                            Text("Active experts").font(Design.medium(11))
                            Picker("Active experts",selection:Binding(get:{value.profile.activeExperts},set:{count in
                                if model.settings.modelPath == value.path {model.setExpertCount(count)}
                                else {library.setExpertCount(count,for:value)}
                            })) {ForEach(Array(value.descriptor.expertRange),id:\.self){count in Text(count == value.descriptor.activeExperts ? "\(count) · Native" : "\(count)").tag(count)}}
                                .labelsHidden().frame(width:130)
                            Text("of \(value.descriptor.experts) total").font(Design.body(10)).foregroundStyle(Design.muted)
                        }.disabled(model.busy || model.loading)
                        if value.profile.activeExperts != value.descriptor.activeExperts {Text("Reduced routing is experimental: fewer experts can improve speed but can reduce answer quality. A change reloads the model context; your messages are retained.").font(Design.body(10)).foregroundStyle(Design.muted)}
                        HStack {
                            Button("Model settings…"){model.useInstalledModel(value);dismiss();model.showSettings = true}.buttonStyle(HitButtonStyle()).disabled(model.busy || model.loading)
                            Button("Optimize for this Mac"){model.optimizeLibraryModel(value)}.buttonStyle(HitButtonStyle()).disabled(model.busy || model.loading || model.modelInstallationActive || !ArchitectureRegistry.report(value.descriptor).installable)
                            Button("Show in Finder"){NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath:value.path)])}.buttonStyle(HitButtonStyle())
                            Spacer()
                            if model.connected && model.settings.modelPath == value.path {Button("Unload"){model.unload()}.buttonStyle(HitButtonStyle()).disabled(model.busy || model.loading)}
                            Button(value.managed ? "Remove…" : "Remove from library…",role:.destructive){removal = value}.buttonStyle(HitButtonStyle()).disabled(model.busy || model.loading || model.modelInstallationActive || (model.connected && model.settings.modelPath == value.path))
                        }.font(Design.body(10))
                        if let report=value.calibration {Text("Last optimized \(report.createdAt.formatted(date:.abbreviated,time:.omitted)). Forced-continuation tests; results apply to this engine, Mac and model location.").font(Design.body(9)).foregroundStyle(Design.muted)}
                    }
                }
            }
        }
    }
    private var downloads: some View {
        VStack(alignment:.leading,spacing:14) {
            if library.jobs.isEmpty {Text("No downloads yet. Choose a model in Discover.").font(Design.body(12)).foregroundStyle(Design.muted)}
            ForEach(library.jobs){job in
                card {
                    VStack(alignment:.leading,spacing:10) {
                        HStack{Text(job.repository).font(Design.medium(12));Spacer();Text(job.stage.rawValue.capitalized).font(Design.body(10)).foregroundStyle(Design.muted)}
                        ProgressView(value:job.totalBytes > 0 ? Double(job.completedBytes)/Double(job.totalBytes) : 0).tint(Design.green)
                        Text(job.detail).font(Design.body(10)).foregroundStyle(Design.muted)
                        if job.stage != .ready {
                            HStack{
                                if job.id == model.libraryInstallJob && model.modelInstallationActive {Button("Pause"){model.pauseLibraryInstall(job)}.buttonStyle(HitButtonStyle())}
                                else {Button("Resume"){model.resumeLibraryInstall(job)}.buttonStyle(HitButtonStyle()).disabled(model.modelInstallationActive)}
                                if job.stage != .cancelled {Button("Cancel",role:.destructive){model.cancelLibraryInstall(job)}.buttonStyle(HitButtonStyle()).disabled(model.modelInstallationActive && job.id != model.libraryInstallJob)}
                            }.font(Design.body(11))
                        }
                    }
                }
            }
        }
    }
    private func card<Content:View>(@ViewBuilder content:()->Content)->some View {
        VStack(alignment:.leading,spacing:0){content()}.padding(20).frame(maxWidth:.infinity,alignment:.leading).background{Design.surface.opacity(0.7).overlay{InterfaceGrain()}}.clipShape(RoundedRectangle(cornerRadius:18)).overlay(RoundedRectangle(cornerRadius:18).stroke(Design.line.opacity(0.8),lineWidth:1))
    }
}
