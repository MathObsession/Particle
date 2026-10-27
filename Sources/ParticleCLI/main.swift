import Foundation
import Darwin
import ParticleGGUF
import ParticleCore
import ParticleDesktopProtocol

private func log(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
private struct Options {
    var model = ProcessInfo.processInfo.environment["PARTICLE_MODEL_PATH"] ?? "/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf"
    var tokenizer: String?, prompt: String?, system: String?
    var maxNew = 64, context = 2048, slots = 12, topK = 50, prefillChunk = 128
    var temperature: Float = 0, topP: Float = 0.95
    var execution = ParticleExecutionOptions()
    var idlePreparation = true, gpuGreedy = false, idlePrefillChunk = 8
    var timingsPath: String?, sessionBenchmarkPath: String?, comparisonModel: String?, approximationBenchmarkPath: String?
    var seed: UInt64 = 42
    var benchmark = false, benchmarkAttention = false, benchmarkIO = false, benchmarkHybrid = false
    var desktop = false
    var reasoningBudget: Int?
    var interactive = false, ssdCache = true, cacheMiB = 4096
    var storageProfile: ParticleStorageProfile = .legacy
    var cachePolicy: ParticleSSDCachePolicy = .victim
    var mirrorReadHint: ParticleMirrorReadHint = .standard
    var prepareCache = false
    var cacheDirectory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/Particle/Experts").path
    var raw = false, thinking = false, inspect = false, tokenize = false, validate = false, verbose = false, comparePrefill = false
    static let help = """
    ParticleCLI — local Churo model engine for Apple Silicon

    Usage: ParticleCLI --prompt "Explain what an MoE model is." [options]
           ParticleCLI --interactive [options]
           ParticleCLI --inspect
           ParticleCLI --validate-kernels

      --model <path>            Compatible GGUF or verified .particle folder (defaults to PARTICLE_MODEL_PATH)
      --tokenizer <folder>      Official tokenizer sidecars (default: beside model/tokenizer)
      --prompt <text>           User message; use '-' to read UTF-8 from stdin
      --prompt-file <path>      Read a UTF-8 prompt file (up to 1 MiB)
      --desktop-events         Opt-in JSONL desktop transport (requires --interactive)
      --interactive            Keep the model warm; /new resets chat, /quit exits
      --ssd-cache <folder>      Expert cache location (default: ~/Library/Caches/Particle/Experts)
      --ssd-cache-mib <n>       Bounded SSD payload, 1...18432 MiB (default 4096)
      --no-ssd-cache            Disable the internal SSD tier
      --storage-profile <name> legacy/sd-only/sd-hybrid/ssd (default legacy)
      --ssd-cache-policy <name> victim/read-through/mirror (default victim)
      --mirror-read-hint <mode> default/nocache for expert files only (default default)
      --prepare-cache          Prepare verified common/expert SSD mirror, then exit
                               Requires .particle, mirror policy, sufficient explicit budget
      --system <text>           Optional system guidance
      --raw                    Unframed text completion instead of instruction chat
      --thinking               Enable reasoning in the official chat template
      --reasoning-budget <n>   Desktop reasoning: 0 Instant, 64 Thinking, -1 extended
      --max-new <n>            Output token limit (default 64)
      --max-context <n>        Total input + output, 32...131072; memory-tier limited (default 2048)
      --expert-cache-slots <n> RAM expert slots per MoE layer, 6...16 (default 12)
                               Fewer: less RAM; more: may reduce disk reads
                               Expert buffers: 6 ~0.76, 8 ~1.01, 12 ~1.52, 16 ~2.03 GiB
                               Excludes KV, common weights, scratch and system memory
      --prefill-chunk <n>      Reuse prompt experts over 1...256 rows (above 128 experimental) (default 128; 1 is scalar)
      --idle-preparation       Prepare canonical history while idle (default)
      --idle-prefill-chunk <n> Idle checkpoint size, 1...128 (default 8)
      --no-idle-preparation    Do all history preparation after submission
      --attention <mode>      auto/reference/fast/tiled/split/tensor (explicit candidates)
      --projection <mode>     auto/reference/vector/staged (explicit candidates)
      --gpu-greedy            Experimental compact GPU sample at temperature zero
      --hybrid-prefill        Experimental two-worker CPU/GPU expert-row queue
      --source-read-workers <n> 1, 2, or 4 slow-source workers (default 4)
      --ssd-read-workers <n>   1, 2, or 4 SSD workers (default 2)
      --serial-expert-loads   Disable resident-expert/read overlap for comparison
      --deferred-decode       Experimental queued decode; fewer host waits
      --gpu-routing           Experimental compact top-six GPU router
      --grouped-experts       Experimental two-dispatch multi-expert decode
      --ffn-vector            Experimental native-block FFN metadata reuse
      --cooperative-quant     Experimental SIMD-shared native metadata; requires vector/FFN vector
      --pipelined-verification Experimental two-worker mirror checksum pipeline
      --active-experts <n>    Approximate routed experts per layer, 2...6 (default 6)
                               Smaller budgets change quality; shared expert stays enabled
      --coalesce-reads        Experimental demand-only reads, max 10% amplification
      --split-kv-block <n>    256 or 512 positions (default 256)
      --timings-json <path>   Append JSONL turn/idle metrics; profile attention
      --metrics-json <path>   Append metrics without attention profiling overhead
      --compare-prefill        Compare scalar/batched logits, continuation, KV reuse and reads
      --compare-model <path>   Compare full logits/continuation against reference; one runner at a time
      --temperature <value>    0 for greedy (default); positive for sampling
      --top-k <n>              Sampling candidates (default 50)
      --top-p <value>          Sampling nucleus cutoff (default 0.95)
      --seed <n>               Sampling seed (default 42)
      --tokenize               Print prompt token IDs as JSON without loading weights
      --inspect                Validate GGUF metadata and show model information
      --benchmark-hybrid      Compare resident expert CPU/GPU row sharing
      --benchmark-io          Compare 1/2/4 source and SSD read workers
      --benchmark-session <json> Real-history AB/BA benchmark configuration (requires --raw)
      --benchmark-speed-quality <json> Screen 6/4/3/2 expert profiles with one runner (requires --raw)
      --benchmark-attention   Compare reference/tiled/split at 512...8192 positions
      --benchmark-kernels      Time scalar versus batched resident K-quant GPU work
      --validate-kernels       Run independent model-free GPU/CPU numerical checks
      --verbose                Show per-layer progress during slow storage reads
      --help                   Show this help

    Text streams to stdout; progress and measured timings go to stderr.
    Control-C cancels after the current GPU/read work completes.
    """
    init(_ args: [String]) throws {
        var i = 0
        var explicitCache = false, explicitNoCache = false
        func next(_ flag: String) throws -> String {
            i += 1; guard i < args.count else { throw ParticleError.invalid("Missing value for \(flag)") }
            return args[i]
        }
        func integer(_ flag: String) throws -> Int {
            let s = try next(flag); guard let n = Int(s) else { throw ParticleError.invalid("Invalid integer for \(flag)") }; return n
        }
        func number(_ flag: String) throws -> Float {
            let s = try next(flag); guard let n = Float(s), n.isFinite else { throw ParticleError.invalid("Invalid number for \(flag)") }; return n
        }
        while i < args.count {
            let flag = args[i]
            switch flag {
            case "--idle-preparation": idlePreparation = true
            case "--idle-prefill-chunk": idlePrefillChunk = try integer(flag)
            case "--no-idle-preparation": idlePreparation = false
            case "--attention":
                guard let mode = ParticleAttentionMode(rawValue:try next(flag)) else { throw ParticleError.invalid("Invalid attention mode") }
                execution.attention = mode
            case "--projection":
                guard let mode = ParticleProjectionMode(rawValue:try next(flag)) else { throw ParticleError.invalid("Invalid projection mode") }
                execution.projection = mode
            case "--gpu-greedy": gpuGreedy = true
            case "--hybrid-prefill": execution.hybrid = true
            case "--source-read-workers": execution.sourceWorkers = try integer(flag)
            case "--ssd-read-workers": execution.ssdWorkers = try integer(flag)
            case "--serial-expert-loads": execution.overlapExpertReads = false
            case "--deferred-decode": execution.deferredDecode = true
            case "--gpu-routing": execution.gpuRouting = true
            case "--grouped-experts": execution.groupedExperts = true
            case "--ffn-vector": execution.ffnVector = true
            case "--active-experts": execution.activeExperts = try integer(flag)
            case "--pipelined-verification": execution.pipelinedVerification = true
            case "--cooperative-quant": execution.cooperativeQuant = true
            case "--coalesce-reads": execution.coalesceReads = true
            case "--split-kv-block": execution.splitBlock = try integer(flag)
            case "--timings-json": timingsPath = try next(flag); execution.profileAttention = true
            case "--metrics-json": timingsPath = try next(flag)
            case "--benchmark-hybrid": benchmarkHybrid = true
            case "--benchmark-io": benchmarkIO = true
            case "--benchmark-session": sessionBenchmarkPath = try next(flag)
            case "--benchmark-speed-quality": approximationBenchmarkPath = try next(flag)
            case "--benchmark-attention": benchmarkAttention = true
            case "--benchmark-kernels": benchmark = true
            case "--interactive": interactive = true
            case "--desktop-events": desktop = true
            case "--reasoning-budget": reasoningBudget = try integer(flag)
            case "--ssd-cache": cacheDirectory = try next(flag); ssdCache = true; explicitCache = true
            case "--ssd-cache-mib": cacheMiB = try integer(flag)
            case "--no-ssd-cache": ssdCache = false; explicitNoCache = true
            case "--storage-profile":
                guard let value = ParticleStorageProfile(rawValue:try next(flag)) else { throw ParticleError.invalid("Invalid storage profile") }
                storageProfile = value
            case "--ssd-cache-policy":
                guard let value = ParticleSSDCachePolicy(rawValue:try next(flag)) else { throw ParticleError.invalid("Invalid SSD cache policy") }
                cachePolicy = value
            case "--mirror-read-hint":
                guard let hint=ParticleMirrorReadHint(rawValue:try next(flag)) else { throw ParticleError.invalid("Invalid mirror read hint") }
                mirrorReadHint=hint
            case "--prepare-cache": prepareCache = true
            case "--model": model = try next(flag)
            case "--tokenizer": tokenizer = try next(flag)
            case "--prompt": guard prompt == nil else { throw ParticleError.invalid("Specify one prompt") }; prompt = try next(flag)
            case "--prompt-file":
                guard prompt == nil else { throw ParticleError.invalid("Specify one prompt") }
                let path = try next(flag), handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
                defer { try? handle.close() }
                let data = try handle.read(upToCount: 1_048_577) ?? Data()
                guard data.count <= 1_048_576, let text = String(data: data, encoding: .utf8) else { throw ParticleError.invalid("Prompt must be UTF-8 and at most 1 MiB") }
                prompt = text
            case "--system": system = try next(flag)
            case "--max-new": maxNew = try integer(flag)
            case "--max-context": context = try integer(flag)
            case "--expert-cache-slots": slots = try integer(flag)
            case "--prefill-chunk": prefillChunk = try integer(flag)
            case "--top-k": topK = try integer(flag)
            case "--temperature": temperature = try number(flag)
            case "--top-p": topP = try number(flag)
            case "--seed": guard let n = UInt64(try next(flag)) else { throw ParticleError.invalid("Invalid seed") }; seed = n
            case "--raw": raw = true
            case "--thinking": thinking = true
            case "--inspect": inspect = true
            case "--tokenize": tokenize = true
            case "--validate-kernels": validate = true
            case "--verbose": verbose = true
            case "--compare-prefill": comparePrefill = true
            case "--compare-model": comparisonModel = try next(flag)
            default: throw ParticleError.invalid("Unknown option: \(flag)")
            }
            i += 1
        }
        guard (1...ParticleContextPolicy.modelLimit).contains(maxNew), (32...ParticleContextPolicy.modelLimit).contains(context), (6...16).contains(slots), (1...256).contains(prefillChunk),
              (2...6).contains(execution.activeExperts), (1...18432).contains(cacheMiB), (1...128).contains(idlePrefillChunk), temperature >= 0, (1...262144).contains(topK), topP > 0, topP <= 1 else {
            throw ParticleError.invalid("Options out of range; see --help")
        }
        guard [1,2,4].contains(execution.sourceWorkers), [1,2,4].contains(execution.ssdWorkers), [256,512].contains(execution.splitBlock) else {
            throw ParticleError.invalid("Read workers must be 1, 2, or 4; split block must be 256 or 512")
        }
        guard !(explicitCache && explicitNoCache), !([ParticleStorageProfile.sdOnly,.ssd].contains(storageProfile) && explicitCache) else {
            throw ParticleError.invalid("Conflicting SSD cache flags or storage profile")
        }
        if storageProfile == .sdOnly || storageProfile == .ssd { ssdCache = false }
        guard storageProfile != .sdHybrid || ssdCache else { throw ParticleError.invalid("sd-hybrid requires an SSD tier") }
        guard cachePolicy != .mirror || ssdCache else { throw ParticleError.invalid("Mirror policy requires an SSD tier") }
        guard !prepareCache || (ssdCache && cachePolicy == .mirror && !interactive && prompt == nil && !inspect && !tokenize && !comparePrefill && !benchmark && !benchmarkIO && !benchmarkHybrid && !benchmarkAttention && !validate && sessionBenchmarkPath == nil) else {
            throw ParticleError.invalid("--prepare-cache requires mirror policy and cannot run a prompt or inspection")
        }
        guard comparisonModel == nil || (!interactive && !comparePrefill && !prepareCache && !tokenize) else { throw ParticleError.invalid("Model comparison cannot combine with interactive, tokenize, prepare-cache, or compare-prefill") }
        guard !(interactive && (tokenize || comparePrefill || prompt == "-")) else { throw ParticleError.invalid("Interactive mode cannot use stdin prompt, --tokenize, or --compare-prefill") }
        guard sessionBenchmarkPath == nil || (raw && !interactive && !comparePrefill) else {
            throw ParticleError.invalid("Session benchmark needs --raw and cannot use interactive/compare-prefill")
        }
        guard approximationBenchmarkPath == nil || (raw && !interactive && !comparePrefill && sessionBenchmarkPath == nil && comparisonModel == nil && !tokenize && !inspect && !benchmark && !benchmarkIO && !benchmarkHybrid && !benchmarkAttention && !validate && !prepareCache) else {
            throw ParticleError.invalid("Speed/quality screening requires --raw and cannot combine interactive or comparison modes")
        }
        guard !execution.cooperativeQuant || (execution.projection == .vector && execution.ffnVector) else { throw ParticleError.invalid("--cooperative-quant requires --projection vector --ffn-vector") }
        guard !execution.pipelinedVerification || cachePolicy == .mirror else { throw ParticleError.invalid("Pipelined verification requires mirror policy") }
        guard mirrorReadHint == .standard || cachePolicy == .mirror else { throw ParticleError.invalid("--mirror-read-hint nocache requires mirror policy") }
        guard !desktop || (interactive && !raw && prompt == nil && !inspect && !tokenize && !validate && !benchmark && !benchmarkIO && !benchmarkHybrid && !benchmarkAttention && sessionBenchmarkPath == nil && approximationBenchmarkPath == nil && comparisonModel == nil && !comparePrefill && !prepareCache) else { throw ParticleError.invalid("Desktop transport requires interactive chat only") }
        guard reasoningBudget == nil || (desktop && (-1...4096).contains(reasoningBudget!)) else {throw ParticleError.invalid("Reasoning budget requires desktop mode and -1...4096")}
        guard !(raw && (system != nil || thinking)) else { throw ParticleError.invalid("--raw cannot be combined with --system or --thinking") }
    }
}

@main struct ParticleCLI {
    @MainActor static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.isEmpty || args.contains("--help") || args.contains("-h") { print(Options.help); return }
        do { try await run(Options(args)) }
        catch is CancellationError { log("\nParticle: cancelled; no further tokens computed."); if args.contains("--desktop-events") { desktopEvent(["event":"error","message":"Loading was cancelled."]) }; exit(130) }
        catch { log("Particle: \(error)"); if args.contains("--desktop-events") { desktopEvent(["event":"error","message":String(describing:error)]) }; exit(1) }
    }
    @MainActor private static func run(_ o: Options) async throws {
        if o.benchmarkHybrid {
            let encoder=JSONEncoder(); encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(ParticleValidation.benchmarkHybrid())); print(); return
        }
        if o.benchmarkAttention {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(ParticleValidation.benchmarkAttention())); print(); return
        }
        if o.benchmark {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(ParticleValidation.benchmarkKernels(projection:o.execution.projection,cooperative:o.execution.cooperativeQuant))); print(); return
        }
        if o.validate {
            for report in try ParticleValidation.kernels(execution:o.execution) { log("PASS \(report)") }
            if o.execution.cooperativeQuant { for report in try ParticleValidation.cooperativeKernels() { log("PASS \(report)") } }
            return
        }
        let start = ProcessInfo.processInfo.systemUptime
        if o.execution.activeExperts != 6 {
            log("Particle: APPROXIMATE profile: \(o.execution.activeExperts)/6 routed experts; renormalized native routing weights. Quality and 4 tokens/s are unqualified.")
        }
        log("Particle: reading model metadata")
        var file: any ParticleModelSource = try ParticleModels.open(path: o.model)
        _ = try SarvamConfiguration(file: file)
        try SarvamConfiguration.validateTensorGraph(file:file)
        if o.storageProfile == .ssd {
            var modelStat = stat(), homeStat = stat()
            guard stat(o.model,&modelStat) == 0, stat(FileManager.default.homeDirectoryForCurrentUser.path,&homeStat) == 0,
                  modelStat.st_dev == homeStat.st_dev else { throw ParticleError.invalid("ssd profile requires the model on the internal SSD; use sd-hybrid for a mirrored SD pack") }
        }
        if o.prepareCache {
            guard let pack = file as? ParticlePackFile else { throw ParticleError.invalid("Mirror preparation requires a verified .particle pack") }
            let lease = try ParticleModelLease(); defer { withExtendedLifetime(lease) {} }
            let cancellation = ParticleCancellation(), previous = Darwin.signal(SIGINT,SIG_IGN)
            let interrupt = DispatchSource.makeSignalSource(signal:SIGINT,queue:.global())
            interrupt.setEventHandler(handler:cancellation.cancellationHandler()); interrupt.resume()
            defer { interrupt.cancel(); Darwin.signal(SIGINT,previous) }
            log("Particle: preparing SSD mirror within \(o.cacheMiB) MiB; replaces the older expert cache")
            var lastReport = -1
            try ParticleMirror.prepare(pack:pack,directory:URL(fileURLWithPath:o.cacheDirectory),maxBytes:o.cacheMiB*1048576,cancellation:cancellation) { _,done,total in
                let percent = Int(Double(done)*100/Double(max(total,1)))
                if percent != lastReport { lastReport = percent; log("Particle: verified mirror copy \(percent)%") }
            }
            log("Particle: SSD mirror ready; original pack and GGUF retained"); return
        }
        if o.benchmarkIO {
            let lease=try ParticleModelLease(); defer { withExtendedLifetime(lease) {} }
            let rows=try ParticleValidation.benchmarkReads(file:file,ssdDirectory:o.ssdCache ? URL(fileURLWithPath:o.cacheDirectory) : nil)
            let encoder=JSONEncoder(); encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(rows)); print(); return
        }
        if o.inspect {
            var types: [String:Int] = [:]
            for tensor in file.tensors.values { types[String(describing: tensor.type), default: 0] += 1 }
            let info: [String:Any] = ["path":file.path,"bytes":file.size,"architecture":"sarvam_moe","format":file.formatName,"layers":19,
                "experts":128,"experts_per_token":6,"tensors":file.tensors.count,"quantization_types":types,
                "context_limit_v1":ParticleContextPolicy.modelLimit,"model_position_limit":ParticleContextPolicy.modelLimit,
                "system_context_limit":ParticleContextPolicy.maximum(),"kv_bytes_per_token":ParticleContextPolicy.kvBytesPerToken,
                "embedding_access":"bounded packed row reads",
                "mapped_common_bytes":file.tensors.values.filter { !$0.name.contains("_exps.weight") && $0.name != "token_embd.weight" }.reduce(0) { $0+$1.byteCount },
                "tokenizer_directory":o.tokenizer ?? file.tokenizerDirectory,
                "mirror_required_bytes":(file as? ParticlePackFile).map { ParticleMirror.requiredBytes(for:$0) } ?? 0]
            let data = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted,.sortedKeys])
            FileHandle.standardOutput.write(data); print(); return
        }
        guard o.prompt != nil || o.interactive else { throw ParticleError.invalid("Provide --prompt or --interactive") }
        var prompt = o.prompt ?? ""
        if prompt == "-" {
            let data = try FileHandle.standardInput.read(upToCount: 1_048_577) ?? Data()
            guard data.count <= 1_048_576, let text = String(data:data,encoding:.utf8) else { throw ParticleError.invalid("stdin must be UTF-8 and at most 1 MiB") }
            prompt = text
        }
        guard (o.interactive || !prompt.isEmpty), prompt.utf8.count <= 1_048_576 else { throw ParticleError.invalid("Prompt is empty or exceeds 1 MiB") }
        let folder = o.tokenizer.map { URL(fileURLWithPath:$0) } ?? URL(fileURLWithPath:file.tokenizerDirectory)
        log("Particle: validating tokenizer vocabulary")
        let tokenizer = try await SarvamTokenizer.load(folder: folder, vocabulary: file.metadata["tokenizer.ggml.tokens"]?.strings)
        let ids = prompt.isEmpty ? [] : try o.raw ? tokenizer.encode(prompt) : tokenizer.chat(prompt:prompt,system:o.system,thinking:o.thinking)
        guard (o.interactive || !ids.isEmpty), ids.allSatisfy({ (0..<262144).contains($0) }) else { throw ParticleError.invalid("Tokenizer returned invalid prompt IDs") }
        if o.tokenize {
            let data = try JSONSerialization.data(withJSONObject: ["ids":ids,"decoded":try tokenizer.decode(ids)], options: [.prettyPrinted,.sortedKeys])
            FileHandle.standardOutput.write(data); print(); return
        }
        let allowance = o.comparePrefill ? 1 : o.maxNew
        guard ids.count + allowance <= o.context else {
            throw ParticleError.invalid("Prompt (\(ids.count)) plus output allowance (\(allowance)) exceeds context \(o.context)")
        }
        let lease = try ParticleModelLease()
        defer { withExtendedLifetime(lease) {} }
        let cancellation = ParticleCancellation()
        let previous = Darwin.signal(SIGINT, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal:SIGINT,queue:.global())
        interrupt.setEventHandler(handler:cancellation.cancellationHandler()); interrupt.resume()
        defer { interrupt.cancel(); Darwin.signal(SIGINT,previous) }
        if o.cachePolicy == .mirror {
            guard let pack = file as? ParticlePackFile else { throw ParticleError.invalid("Mirror policy requires a .particle pack; prepare it first") }
            file = try ParticleMirror.open(pack:pack,directory:URL(fileURLWithPath:o.cacheDirectory),maxBytes:o.cacheMiB*1048576,expertReadHint:o.mirrorReadHint)
        }
        if let path = o.comparisonModel {
            let reference = try ParticleModels.open(path:path)
            let report = try ParticleModelComparison.compare(reference:reference,candidate:file,tokens:ids,context:o.context,slots:o.slots,execution:o.execution,cancellation:cancellation,progress:log)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(report)); print()
            guard report.passed else { throw ParticleError.invalid("Model format comparison failed bitwise full-logit/continuation parity") }
            return
        }
        log("Particle: storage \(o.storageProfile.rawValue), source \(file.formatName), SSD policy \(o.ssdCache ? o.cachePolicy.rawValue : "disabled")")
        log("Particle: mapping common weights; \(o.slots) bounded expert slots per MoE layer")
        if o.ssdCache { log("Particle: SSD expert cache at \(o.cacheDirectory), limit \(o.cacheMiB) MiB") }
        let runner = try SarvamRunner(file:file,maxContext:o.context,expertCacheSlots:o.slots,
            ssdCacheDirectory:o.ssdCache && o.cachePolicy != .mirror ? URL(fileURLWithPath:o.cacheDirectory) : nil,ssdCacheBytes:o.cacheMiB*1048576,ssdCachePolicy:o.cachePolicy,execution:o.execution)
        defer { runner.flushSSDCache() }
        if let path=o.approximationBenchmarkPath {
            let handle=try FileHandle(forReadingFrom:URL(fileURLWithPath:path)); defer { try? handle.close() }
            let data=try handle.read(upToCount:1_048_577) ?? Data()
            guard data.count<=1_048_576 else { throw ParticleError.invalid("Speed/quality configuration exceeds 1 MiB") }
            let configuration=try JSONDecoder().decode(ParticleApproximationConfiguration.self,from:data)
            let report=try runner.benchmarkApproximation(seed:ids,tokenizer:tokenizer,options:configuration,chunkSize:o.prefillChunk,cancellation:cancellation,progress:{ log("Particle screening: \($0)") })
            let encoder=JSONEncoder(); encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(report)); print(); return
        }
        if let path=o.sessionBenchmarkPath {
            let configuration=try JSONDecoder().decode(ParticleSessionBenchmarkOptions.self,from:Data(contentsOf:URL(fileURLWithPath:path)))
            let passes=try runner.benchmarkSession(seed:ids,options:configuration,candidate:o.execution,chunkSize:o.prefillChunk,cancellation:cancellation,progress:{ log("Particle benchmark: \($0)") })
            let encoder=JSONEncoder(); encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(passes)); print()
            guard passes.allSatisfy(\.parityPassed) else { throw ParticleError.invalid("Session benchmark failed strict scalar parity; candidate remains experimental") }
            return
        }
        if o.comparePrefill {
            let comparison = try runner.comparePrefill(tokens:ids,chunkSize:o.prefillChunk,cancellation:cancellation,
                onProgress:{ mode,start,count,layer in
                    if o.verbose || layer == 19 { log("  \(mode): prompt \(start+1)...\(start+count)/\(ids.count), layer \(layer)/19") }
                })
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(comparison)); print()
            log("Particle: PASS full-logit and continuation parity for all passes, including KV rewind/replay")
            return
        }
        let loaded = ProcessInfo.processInfo.systemUptime
        if o.interactive {
            try interactive(o,tokenizer:tokenizer,runner:runner,cancellation:cancellation,initial:prompt,loadSeconds:loaded-start)
        } else {
            _ = try generate(o,tokenizer:tokenizer,runner:runner,cancellation:cancellation,ids:ids,loadSeconds:loaded-start,reused:0)
        }
    }
    @MainActor private static func desktopEvent(_ payload: [String:Any]) {
        if let data=try? JSONSerialization.data(withJSONObject:payload,options:[.sortedKeys]) {
            FileHandle.standardOutput.write(data+Data([10]))
        }
    }
    @MainActor private static func interactive(_ initialOptions: Options, tokenizer: SarvamTokenizer, runner: SarvamRunner,
                                               cancellation: ParticleCancellation, initial: String, loadSeconds: Double) throws {
        var o=initialOptions
        log("Particle: warm session ready. Type a message; /new clears chat, /quit exits. Raw mode uses independent completions.")
        let input = ParticleInputMailbox(); input.startStandardInput()
        var inputPromptShown = false
        var sessionSystem=o.system
        var cachedHistoricalPrefix: [Int]?
        var pendingCanonical: (prompt: [Int], assistant: String)?
        var messages: [SarvamMessage] = [], committed: [Int] = [], pending: String? = initial.isEmpty ? nil : initial
        if let system = sessionSystem { messages.append(SarvamMessage(role:.system,content:system)) }
        while true {
            let prompt: String, submitted: Double
            if let first = pending { prompt = first; pending = nil; submitted = ProcessInfo.processInfo.systemUptime }
            else {
                if !inputPromptShown { log("You> "); if o.desktop { desktopEvent(["event":"ready"]) } }; inputPromptShown = false
                guard let line = input.next() else { break }; submitted = line.receivedAt
                if o.desktop {
                    do {
                        guard line.text.utf8.count <= 1_048_576 else { throw ParticleError.invalid("Message exceeds 1 MiB") }
                        let command=try JSONDecoder().decode(DesktopCommand.self,from:Data(line.text.utf8))
                        switch command.type {
                        case "quit": break
                        case "reset", "restore":
                            let history=try command.validatedHistory()
                            if let budget=command.reasoningBudget {o.reasoningBudget=budget;o.thinking=budget != 0}
                            if let system=command.system {sessionSystem=system}
                            messages=history.map { SarvamMessage(role:$0.role == "user" ? .user : .assistant,content:$0.content) }
                            if let system=sessionSystem { messages.insert(SarvamMessage(role:.system,content:system),at:0) }
                            committed.removeAll(); cachedHistoricalPrefix=nil; pendingCanonical=nil; runner.reset(); cancellation.reset()
                            desktopEvent(["event":"restored","messages":history.count]); continue
                        case "agent":
                            do {
                                let history=try command.validatedAgentHistory()
                                var agentOptions=o
                                agentOptions.reasoningBudget=command.reasoningBudget ?? 0
                                agentOptions.thinking=agentOptions.reasoningBudget != 0
                                let prefixCount=agentOptions.thinking ? 0 : 3
                                var candidate=try history.map { message -> SarvamMessage in
                                    guard let role=SarvamMessage.Role(rawValue:message.role) else{throw ParticleError.invalid("Invalid agent role")}
                                    return SarvamMessage(role:role,content:message.content)
                                }
                                if let system=command.system {candidate.insert(SarvamMessage(role:.system,content:system),at:0)}
                                cancellation.reset();cachedHistoricalPrefix=nil;pendingCanonical=nil
                                let templateStart=ProcessInfo.processInfo.systemUptime
                                let target=try tokenizer.agentChat(messages:candidate,tools:command.tools ?? [],thinking:agentOptions.thinking)
                                let plan=try ParticlePromptReuse(committed:committed,target:target,context:o.context,outputAllowance:agentOptions.maxNew+prefixCount)
                                let templateSeconds=ProcessInfo.processInfo.systemUptime-templateStart
                                desktopEvent(["event":"agent_context","new_tokens":plan.suffix.count+prefixCount,"reused":plan.reused,"template_seconds":templateSeconds,"target_tokens":target.count])
                                log("Particle agent: reusing \(plan.reused) KV tokens; processing \(plan.suffix.count+prefixCount) new tokens; template \(templateSeconds)s")
                                try runner.rewind(to:plan.reused)
                                let result=try generate(agentOptions,tokenizer:tokenizer,runner:runner,cancellation:cancellation,ids:plan.suffix,loadSeconds:loadSeconds,reused:plan.reused,submitted:submitted,templateSeconds:templateSeconds,replayedHistory:0,historyClassificationKnown:false,cachedTemplateTokens:0)
                                if result.usable {committed=target+result.forwarded}else{runner.reset();committed.removeAll()}
                            } catch is CancellationError {runner.reset();committed.removeAll();cancellation.reset();desktopEvent(["event":"cancelled"])}
                              catch {runner.reset();committed.removeAll();desktopEvent(["event":"error","message":String(describing:error)])}
                            continue
                        case "message":
                            guard let text=command.text, !text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else { throw ParticleError.invalid("Message is empty") }
                        default: throw ParticleError.invalid("Unknown desktop command")
                        }
                        if command.type == "quit" { break }
                        prompt=command.text ?? ""
                    } catch { desktopEvent(["event":"error","message":String(describing:error)]); continue }
                } else { prompt=line.text }
            }
            if !o.desktop && (prompt == "/quit" || prompt == "/exit") { break }
            if !o.desktop && prompt == "/new" {
                messages.removeAll(); if let system = sessionSystem { messages.append(SarvamMessage(role:.system,content:system)) }
                committed.removeAll(); cachedHistoricalPrefix=nil; pendingCanonical=nil; runner.reset(); cancellation.reset()
                log("Particle: new chat; weights and expert caches remain warm."); continue
            }
            if !o.desktop && prompt == "/help" { log("Send a message, /new, or /quit. Control-C during generation cancels that turn and clears KV state."); continue }
            if prompt.isEmpty { continue }
            guard prompt.utf8.count <= 1_048_576 else { log("Particle: message exceeds 1 MiB"); continue }
            if o.desktop {desktopEvent(["event":"input","utf8_sha256":DesktopTextIdentity.fingerprint(prompt),"utf8_bytes":prompt.utf8.count])}
            let candidate = messages+[SarvamMessage(role:.user,content:prompt)]
            var startedModel = false
            cancellation.reset()
            do {
                let templateStart = ProcessInfo.processInfo.systemUptime
                let hasHistory = messages.contains { $0.role == .assistant }
                var historicalPrefix: [Int]?
                if !o.raw && hasHistory && tokenizer.supportsIncrementalChat {
                    if cachedHistoricalPrefix==nil, let pending=pendingCanonical {
                        cachedHistoricalPrefix=try tokenizer.completedChatPrefix(promptTokens:pending.prompt,assistant:pending.assistant,thinking:o.thinking)
                    }
                    historicalPrefix=cachedHistoricalPrefix
                    if historicalPrefix==nil { historicalPrefix=try? tokenizer.historicalPrefix(messages:messages,thinking:o.thinking) }
                    cachedHistoricalPrefix=historicalPrefix
                }
                let historicalCount = historicalPrefix?.count ?? 0
                let classificationKnown = o.raw || !hasHistory || (tokenizer.supportsIncrementalChat && historicalPrefix != nil)
                let target: [Int]
                var incrementalPrompt=false
                if o.raw { target=tokenizer.encode(prompt) }
                else if let prefix=historicalPrefix, let incremental=try tokenizer.continuingChat(historicalPrefix:prefix,prompt:prompt,thinking:o.thinking) { target=incremental; incrementalPrompt=true }
                else { target=try tokenizer.chat(messages:candidate,thinking:o.thinking) }
                let plan = try ParticlePromptReuse(committed:o.raw ? [] : committed,target:target,context:o.context,outputAllowance:o.maxNew+(o.reasoningBudget==0 ? 3 : 0))
                let templateSeconds = ProcessInfo.processInfo.systemUptime-templateStart
                if cancellation.isCancelled { throw CancellationError() }
                startedModel = true
                if o.raw { runner.reset() } else { try runner.rewind(to:plan.reused) }
                log("Particle: reusing \(plan.reused) KV tokens; processing \(plan.suffix.count) new/reformatted tokens")
                let result = try generate(o,tokenizer:tokenizer,runner:runner,cancellation:cancellation,ids:plan.suffix,loadSeconds:loadSeconds,reused:plan.reused,
                    submitted:submitted,templateSeconds:templateSeconds,replayedHistory:max(0,historicalCount-plan.reused),historyClassificationKnown:classificationKnown,cachedTemplateTokens:incrementalPrompt ? historicalCount : 0)
                if !result.usable {runner.reset();committed.removeAll();runner.flushSSDCache();continue}
                committed = target+result.forwarded
                if !o.raw {
                    messages = candidate+[SarvamMessage(role:.assistant,content:result.text)]
                    cachedHistoricalPrefix=nil; pendingCanonical=(target,result.text)
                }
                runner.flushSSDCache()
                if o.idlePreparation && !o.raw && tokenizer.supportsIncrementalChat {
                    log("You> "); if o.desktop { desktopEvent(["event":"ready"]) }; inputPromptShown = true
                    cancellation.reset()
                    if input.beginIdle(cancellation) {
                        defer { input.endIdle() }
                        cachedHistoricalPrefix=prepareIdle(o,tokenizer:tokenizer,runner:runner,cancellation:cancellation,messages:messages,promptTokens:target,committed:&committed)
                        if cachedHistoricalPrefix != nil { pendingCanonical=nil }
                    }
                }
            } catch is CancellationError {
                // A partially evaluated layer is not a valid KV lineage.
                runner.reset(); committed.removeAll(); cancellation.reset()
                runner.flushSSDCache(); log("\nParticle: turn cancelled; next message replays completed history with warm weights.")
                if o.desktop { desktopEvent(["event":"cancelled"]) }
            } catch {
                if startedModel { runner.reset(); committed.removeAll() }
                log("Particle: \(error)")
                if o.desktop { desktopEvent(["event":"error","message":String(describing:error)]) }
            }
        }
    }
    @MainActor private static func prepareIdle(_ o: Options, tokenizer: SarvamTokenizer, runner: SarvamRunner,
        cancellation: ParticleCancellation, messages: [SarvamMessage], promptTokens: [Int], committed: inout [Int]) -> [Int]? {
        var ledger=committed
        defer { committed=ledger }
        let start=ProcessInfo.processInfo.systemUptime, before=runner.metrics
        var replayed=0, reused=ledger.count, began=false, status="complete"
        var canonical: [Int]?, templateSeconds=0.0
        do {
            if cancellation.isCancelled { return nil }
            let incremental=try tokenizer.completedChatPrefix(promptTokens:promptTokens,assistant:messages.last?.content ?? "",thinking:o.thinking)
            let target=try incremental ?? tokenizer.historicalPrefix(messages:messages,thinking:o.thinking)
            canonical=target; templateSeconds=ProcessInfo.processInfo.systemUptime-start
            if target.count>o.context { status="context_limit" }
            else if cancellation.isCancelled { status="interrupted" }
            else {
            reused=ParticlePromptReuse.commonPrefix(ledger,target)
            try runner.rewind(to:reused); ledger=Array(target.prefix(reused))
            let suffix=Array(target.dropFirst(reused))
            if !suffix.isEmpty {
                began=true
                try runner.prefill(tokens:suffix,chunkSize:min(o.prefillChunk,o.idlePrefillChunk),emitLogits:false,cancellation:cancellation,
                    onChunkComplete:{ completed in
                        replayed=completed; ledger=Array(target.prefix(reused+completed))
                    })
            }
            }
        } catch is CancellationError {
            status="interrupted"
            if began { do { try runner.recoverCancelledPrefix() } catch { runner.reset(); ledger.removeAll() } }
        } catch {
            status="failed"
            if began { runner.reset(); ledger.removeAll() }
            log("Particle: idle preparation skipped: \(error)")
        }
        let after=runner.metrics, seconds=ProcessInfo.processInfo.systemUptime-start
        if replayed>0 || status != "complete" { log(String(format:"Particle: idle preparation %@, %d history tokens completed in %.2fs",status,replayed,seconds)) }
        writeTiming(o,["event":"idle","chunk_size":min(o.prefillChunk,o.idlePrefillChunk),"status":status,"seconds":seconds,"reused_kv":reused,"completed_tokens":replayed,
            "source_read_bytes":after.sourceReadBytes-before.sourceReadBytes,"ssd_read_bytes":after.ssdReadBytes-before.ssdReadBytes,
            "attention_gpu_seconds":after.attentionGPUSeconds-before.attentionGPUSeconds,"template_seconds":templateSeconds])
        return canonical
    }
    private static func writeTiming(_ o: Options, _ record: [String:Any]) {
        guard let path=o.timingsPath else { return }
        do {
            var payload=record
            payload["attention_mode"]=o.execution.attention.rawValue; payload["projection_mode"]=o.execution.projection.rawValue
            payload["hybrid"]=o.execution.hybrid
            payload["model_path"]=o.model; payload["storage_profile"]=o.storageProfile.rawValue
            payload["ssd_cache_policy"]=o.ssdCache ? o.cachePolicy.rawValue : "disabled"
            payload["ssd_cache_limit_bytes"]=o.ssdCache ? o.cacheMiB*1048576 : 0
            payload["ram_expert_slots_per_layer"]=o.slots
            payload["deferred_decode"]=o.execution.deferredDecode; payload["gpu_routing"]=o.execution.gpuRouting && o.execution.activeExperts == 6; payload["grouped_experts"]=o.execution.groupedExperts; payload["ffn_vector"]=o.execution.ffnVector
            payload["cooperative_quant"]=o.execution.cooperativeQuant
            payload["pipelined_verification"]=o.execution.pipelinedVerification
            payload["mirror_read_hint"]=o.mirrorReadHint.rawValue
            payload["active_experts"]=o.execution.activeExperts; payload["approximate_execution"]=o.execution.activeExperts != 6
            payload["attention_profiled"]=o.execution.profileAttention
            payload["chunk_size"]=payload["chunk_size"] ?? o.prefillChunk
            payload["source_workers"]=o.execution.sourceWorkers; payload["ssd_workers"]=o.execution.ssdWorkers
            payload["coalesced_reads"]=o.execution.coalesceReads; payload["overlap_expert_reads"]=o.execution.overlapExpertReads
            payload["timestamp"]=ISO8601DateFormatter().string(from:Date())
            let data=try JSONSerialization.data(withJSONObject:payload,options:[.sortedKeys])
            if !FileManager.default.fileExists(atPath:path) { guard FileManager.default.createFile(atPath:path,contents:nil) else { throw ParticleError.invalid("Cannot create timings file") } }
            let handle=try FileHandle(forWritingTo:URL(fileURLWithPath:path)); defer { try? handle.close() }
            try handle.seekToEnd(); try handle.write(contentsOf:data+Data([10]))
        } catch { log("Particle: cannot append timing report: \(error)") }
    }
    private struct TurnResult { let text: String; let forwarded: [Int]; let usable: Bool }
    @MainActor private static func generate(_ o: Options, tokenizer: SarvamTokenizer, runner: SarvamRunner,
                                            cancellation: ParticleCancellation, ids: [Int], loadSeconds: Double, reused: Int,
                                            submitted: Double? = nil, templateSeconds: Double = 0, replayedHistory: Int = 0, historyClassificationKnown: Bool = true, cachedTemplateTokens: Int = 0) throws -> TurnResult {
        let loaded = ProcessInfo.processInfo.systemUptime
        var peak = runner.metrics.footprintBytes
        log(String(format:"Particle: ready in %.2fs; prefilling %d tokens",loadSeconds,ids.count))
        let beforePrefill = runner.metrics
        // Continue the assistant after an already-completed empty reasoning
        // block. These three IDs are evaluated together in prefill, not decoded
        // one at a time, and are retained in the exact KV lineage.
        let assistantPrefix=o.reasoningBudget==0 ? tokenizer.encode("<think></think>\n") : []
        guard o.reasoningBudget != 0 || assistantPrefix == [23,24,107] else{throw ParticleError.invalid("Unsupported no-thinking prefix tokenization")}
        let prefillTokens=ids+assistantPrefix
        try runner.prefill(tokens:prefillTokens,chunkSize:o.prefillChunk,cancellation:cancellation,
            onLayer:o.verbose ? { start,count,layer in log("  prompt \(start+1)...\(start+count)/\(ids.count), layer \(layer)/19") } : nil,
            onChunkComplete:{ completed in
                peak = max(peak,runner.metrics.footprintBytes); log("Prefill \(completed)/\(ids.count)")
                if o.desktop { desktopEvent(["event":"prefill","completed":completed,"total":prefillTokens.count,"reused":reused]) }
            })
        let prefilled = ProcessInfo.processInfo.systemUptime
        let afterPrefill = runner.metrics
        var sampler = ParticleSampler(seed:o.seed), decoder = SarvamTextDecoder(), generated = 0, steps = 0
        var firstToken = 0.0, finish = "length"
        var text = "", forwarded: [Int] = assistantPrefix
        var reasoning=DesktopReasoningLimiter(budget:o.reasoningBudget ?? -1)
        var forcedClosures=0
        while generated < o.maxNew {
            if cancellation.isCancelled { throw CancellationError() }
            if reasoning.exhausted {finish="reasoning_limit";break}
            let forced=reasoning.mustClose
            // Sarvam's verified tokenizer assigns </think> to ID 24. This
            // token is evaluated normally, not spliced into display-only text.
            let id = forced ? 24 : try o.gpuGreedy && o.temperature == 0 ? runner.greedyToken() : sampler.sample(runner.currentLogits(),temperature:o.temperature,topK:o.topK,topP:o.topP)
            if forced {forcedClosures+=1}
            reasoning.accept(id)
            if generated == 0 { firstToken = ProcessInfo.processInfo.systemUptime - (submitted ?? loaded) }
            generated += 1
            if tokenizer.stopIDs.contains(id) { finish = "stop"; break }
            let piece = decoder.append(try tokenizer.token(id)); text += piece
            if o.desktop { desktopEvent(["event":"text","text":piece]) } else { FileHandle.standardOutput.write(Data(piece.utf8)) }
            if generated < o.maxNew {
                try runner.forward(token:id,cancellation:cancellation,onLayer:o.verbose ? { layer in log("  output \(generated), layer \(layer)/19") } : nil)
                forwarded.append(id); steps += 1; peak = max(peak,runner.metrics.footprintBytes)
            }
        }
        let tail = decoder.finish(); text += tail
        if o.desktop { if !tail.isEmpty { desktopEvent(["event":"text","text":tail]) } } else { FileHandle.standardOutput.write(Data((tail+"\n").utf8)) }
        let done = ProcessInfo.processInfo.systemUptime, metrics = runner.metrics
        let seconds = done-prefilled
        if o.desktop { desktopEvent(["event":"turn","text":text,"finish":finish,"tokens":generated,"decode_steps":steps,"decode_seconds":seconds,"prefill_seconds":prefilled-loaded,"reused":reused,"context_used":runner.position,"active_experts":o.execution.activeExperts,"forced_thinking_closures":forcedClosures,"assistant_prefix_tokens":assistantPrefix.count,"context_limit":runner.maxContext,"output_limit":o.maxNew]) }
        log(String(format:"Particle: prompt=%d output=%d finish=%@ load=%.2fs prefill=%.2fs first_token=%.2fs decode=%.2fs",ids.count,generated,finish,loadSeconds,prefilled-loaded,firstToken,seconds))
        log(steps > 0 ? String(format:"Particle: decode_steps=%d decode_rate=%.3f tok/s",steps,Double(steps)/max(seconds,1e-9)) : "Particle: decode rate n/a (no post-prefill forward step)")
        log(String(format:"Particle: expert_reads=%.1f MiB cache_hits=%d cache_misses=%d peak_sampled_footprint=%.1f MiB",Double(metrics.expertReadBytes)/1048576,metrics.cacheHits,metrics.cacheMisses,Double(peak)/1048576))
        log(String(format:"Particle: prefill_chunk=%d prefill_expert_reads=%.1f MiB prefill_read_wait=%.2fs prefill_gpu_wait=%.2fs",o.prefillChunk,
            Double(afterPrefill.expertReadBytes-beforePrefill.expertReadBytes)/1048576,
            afterPrefill.expertReadWaitSeconds-beforePrefill.expertReadWaitSeconds,afterPrefill.gpuWaitSeconds-beforePrefill.gpuWaitSeconds))
        log(String(format:"Particle: decode_expert_reads=%.1f MiB decode_read_wait=%.2fs decode_gpu_wait=%.2fs",
            Double(metrics.expertReadBytes-afterPrefill.expertReadBytes)/1048576,
            metrics.expertReadWaitSeconds-afterPrefill.expertReadWaitSeconds,metrics.gpuWaitSeconds-afterPrefill.gpuWaitSeconds))
        log(String(format:"Particle: reused_kv=%d source_expert_reads=%.1f MiB ssd_reads=%.1f MiB ssd_writes=%.1f MiB ssd_hits=%d ssd_stored=%.1f MiB ssd_corruptions=%d",reused,
            Double(metrics.sourceReadBytes-beforePrefill.sourceReadBytes)/1048576,Double(metrics.ssdReadBytes-beforePrefill.ssdReadBytes)/1048576,
            Double(metrics.ssdWriteBytes-beforePrefill.ssdWriteBytes)/1048576,metrics.ssdHits-beforePrefill.ssdHits,Double(metrics.ssdStoredBytes)/1048576,metrics.ssdCorruptions))
        log(String(format:"Particle: prefill_source_reads=%.1f MiB prefill_ssd_reads=%.1f MiB decode_source_reads=%.1f MiB decode_ssd_reads=%.1f MiB",
            Double(afterPrefill.sourceReadBytes-beforePrefill.sourceReadBytes)/1048576,Double(afterPrefill.ssdReadBytes-beforePrefill.ssdReadBytes)/1048576,
            Double(metrics.sourceReadBytes-afterPrefill.sourceReadBytes)/1048576,Double(metrics.ssdReadBytes-afterPrefill.ssdReadBytes)/1048576))
        if let warning = metrics.ssdWarning { log("Particle: \(warning)") }
        writeTiming(o,["event":"turn","reused_kv":reused,"prefill_tokens":ids.count,"replayed_history_tokens":replayedHistory,
            "new_prompt_tokens":max(0,ids.count-replayedHistory),"history_classification_known":historyClassificationKnown,"cached_template_tokens":cachedTemplateTokens,"template_seconds":templateSeconds,"prefill_seconds":prefilled-loaded,
            "first_token_seconds":firstToken,"decode_seconds":seconds,"decode_steps":steps,"output_tokens":generated,"finish":finish,
            "source_read_bytes":metrics.sourceReadBytes-beforePrefill.sourceReadBytes,"ssd_read_bytes":metrics.ssdReadBytes-beforePrefill.ssdReadBytes,
            "ssd_write_bytes":metrics.ssdWriteBytes-beforePrefill.ssdWriteBytes,"ssd_stored_bytes":metrics.ssdStoredBytes,
            "ssd_corruptions":metrics.ssdCorruptions,"storage_warning":metrics.ssdWarning ?? "",
            "prefill_read_wait_seconds":afterPrefill.expertReadWaitSeconds-beforePrefill.expertReadWaitSeconds,
            "decode_read_wait_seconds":metrics.expertReadWaitSeconds-afterPrefill.expertReadWaitSeconds,
            "prefill_gpu_wait_seconds":afterPrefill.gpuWaitSeconds-beforePrefill.gpuWaitSeconds,
            "decode_gpu_wait_seconds":metrics.gpuWaitSeconds-afterPrefill.gpuWaitSeconds,
            "prefill_source_bytes":afterPrefill.sourceReadBytes-beforePrefill.sourceReadBytes,
            "decode_source_bytes":metrics.sourceReadBytes-afterPrefill.sourceReadBytes,
            "prefill_ssd_read_bytes":afterPrefill.ssdReadBytes-beforePrefill.ssdReadBytes,
            "decode_ssd_read_bytes":metrics.ssdReadBytes-afterPrefill.ssdReadBytes,
            "cache_hits":metrics.cacheHits-beforePrefill.cacheHits,"cache_misses":metrics.cacheMisses-beforePrefill.cacheMisses,
            "ssd_hits":metrics.ssdHits-beforePrefill.ssdHits,
            "prefill_attention_gpu_seconds":afterPrefill.attentionGPUSeconds-beforePrefill.attentionGPUSeconds,
            "decode_attention_gpu_seconds":metrics.attentionGPUSeconds-afterPrefill.attentionGPUSeconds,"peak_sampled_footprint_bytes":peak])
        return TurnResult(text:text,forwarded:forwarded,usable:finish != "reasoning_limit")
    }
}
