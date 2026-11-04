import Foundation
import Darwin
import MLX
import MLXNN
import MLXLMCommon
import Tokenizers
import Hub
import ParticleCore
import ParticleModelLibrary
import ParticleDesktopProtocol

@MainActor public enum ParticleMLXWorker {
    struct Options {
        var model = "",prompt = "",context = 32768,maxNew = 256,slots = 12,experts = 8,chunk = 128,budget = 0,readWorkers = 2
        var temperature: Float = 1.0
        var validateReference = false,gathered = true
        var validationSteps = 64
        var benchmarkSession: String?
        var desktop = false,interactive = false,experimental = false
        init(_ args:[String]) throws {
            var i = 0
            func value() throws -> String {i += 1;guard i < args.count else {throw ModelLibraryError.invalid("Missing option value.")};return args[i]}
            while i < args.count {
                let flag = args[i]
                switch flag {
                case "--model":model = try value()
                case "--prompt":prompt = try value()
                case "--max-context":context = Int(try value()) ?? 0
                case "--max-new":maxNew = Int(try value()) ?? 0
                case "--expert-cache-slots":slots = Int(try value()) ?? 0
                case "--active-experts":experts = Int(try value()) ?? 0
                case "--prefill-chunk":chunk = Int(try value()) ?? 0
                case "--read-workers":readWorkers = Int(try value()) ?? 0
                case "--temperature":temperature = Float(try value()) ?? -1
                case "--validate-reference":validateReference = true
                case "--validation-steps":validationSteps = Int(try value()) ?? 0
                case "--benchmark-session":benchmarkSession = try value()
                case "--expert-execution":let mode = try value();guard ["grouped","gathered"].contains(mode) else {throw ModelLibraryError.invalid("Unknown expert execution mode.")};gathered = mode == "gathered"
                case "--reasoning-budget":budget = Int(try value()) ?? -2
                case "--desktop-events":desktop = true
                case "--interactive":interactive = true
                case "--allow-unqualified-run":experimental = true
                case "--thinking":budget = 64
                default:throw ModelLibraryError.invalid("Unknown MLX worker option: \(flag).")
                };i += 1
            }
            guard !model.isEmpty,(4096...262144).contains(context),(1...16384).contains(maxNew),maxNew < context,
                  (8...32).contains(slots),(1...64).contains(experts),[32,64,128].contains(chunk),(-1...4096).contains(budget),[1,2,4].contains(readWorkers),temperature.isFinite,(0...2).contains(temperature),(1...128).contains(validationSteps) else {throw ModelLibraryError.invalid("Invalid Chronos execution settings.")}
            guard experimental || ArchitectureRegistry.adapter(for:.qwen35MoE).qualification == .qualified else {throw ModelLibraryError.incompatible("Chronos is awaiting runtime qualification. Development runs require --allow-unqualified-run.")}
            guard benchmarkSession == nil || (!interactive && !validateReference && !prompt.isEmpty) else {throw ModelLibraryError.invalid("Calibration requires a prompt and a non-interactive worker.")}
        }
    }
    static func event(_ payload:[String:Any]) {if let data = try? JSONSerialization.data(withJSONObject:payload,options:.sortedKeys){FileHandle.standardOutput.write(data+Data([10]))}}
    static func log(_ text:String) {FileHandle.standardError.write(Data((text+"\n").utf8))}
    static func sample(logits:MLXArray,using sampler:any LogitSampler) -> Int {
        // MLX LM's top-k sampler indexes [batch, vocabulary]. Greedy sampling
        // also accepts a vector, which previously concealed this mismatch.
        sampler.sample(logits:logits.expandedDimensions(axis:0)).item(Int.self)
    }
    public static func main() async {
        if CommandLine.arguments.contains("--help") {
            print("ParticleMLXCLI --model <v2.particle> --prompt <text> [--interactive] [--desktop-events] --allow-unqualified-run\n--max-context 32768 --max-new 256 --active-experts 8 --expert-cache-slots 12 --prefill-chunk 128 --reasoning-budget 0\n--expert-execution gathered (upstream math; default) | grouped (experimental; full-logit parity has failed)\nExperimental Qwen3.5 MoE streaming worker. Native eight-expert routing is the reference; reduced counts change the model.");return
        }
        do {
            var options = try Options(Array(CommandLine.arguments.dropFirst()))
            let lease = try ParticleModelLease();defer{withExtendedLifetime(lease){}}
            GPU.metallib = try MLXRuntimeResources.metallib()
            Memory.cacheLimit = 96*1024*1024
            let cancellation = ParticleCancellation()
            signal(SIGINT,SIG_IGN)
            let interrupt = DispatchSource.makeSignalSource(signal:SIGINT,queue:.global());interrupt.setEventHandler(handler:cancellation.cancellationHandler());interrupt.resume();defer{interrupt.cancel()}
            let model = try ChronosModel(path:URL(fileURLWithPath:options.model),slots:options.slots,activeExperts:options.experts,readWorkers:options.readWorkers,cancellation:cancellation)
            model.setGatheredExecution(options.gathered)
            let folder = URL(fileURLWithPath:options.model).appendingPathComponent("tokenizer")
            let configuration = LanguageModelConfigurationFromHub(modelFolder:folder)
            guard let config = try await configuration.tokenizerConfig else {throw ModelLibraryError.invalid("Missing tokenizer configuration.")}
            let tokenizer = try await AutoTokenizer.from(tokenizerConfig:config,tokenizerData:configuration.tokenizerData)
            let template = try String(contentsOf:folder.appendingPathComponent("chat_template.jinja"),encoding:.utf8)
            guard let opening = tokenizer.convertTokenToId("<think>"),let closing = tokenizer.convertTokenToId("</think>"),
                  let end = tokenizer.convertTokenToId("<|im_end|>"),let endOfText = tokenizer.convertTokenToId("<|endoftext|>") else {throw ModelLibraryError.invalid("Unsupported Chronos tokenizer control tokens.")}
            let stopIDs:Set<Int> = [end,endOfText]
            if let path=options.benchmarkSession {
                let data=try ArchitectureRegistry.boundedData(URL(fileURLWithPath:path),limit:1_048_576)
                let configuration=try JSONDecoder().decode(ParticleSessionBenchmarkOptions.self,from:data)
                guard !configuration.histories.isEmpty,configuration.histories.count <= 4,!configuration.suffixes.isEmpty,configuration.suffixes.count <= 4,
                      configuration.histories.allSatisfy({$0 > 0}),configuration.suffixes.allSatisfy({(1...128).contains($0)}),
                      (64...128).contains(configuration.decodeTokens),configuration.histories.max()!+configuration.suffixes.max()!+configuration.decodeTokens <= options.context else {throw ModelLibraryError.invalid("Invalid calibration workloads or context allowance.")}
                let seed=tokenizer.encode(text:options.prompt,addSpecialTokens:false)
                guard !seed.isEmpty else {throw ModelLibraryError.invalid("Calibration requires input tokens.")}
                struct Workload: Encodable {let history:Int;let suffix:Int;let prefixExtensionSeconds:Double;let report:ChronosParityReport}
                var results=[Workload]()
                var seeded=0
                for history in configuration.histories.sorted() {
                    model.setReference(false)
                    let prefix=(0..<history).map{seed[$0%seed.count]},start=ProcessInfo.processInfo.systemUptime
                    if seeded > 0 {_=try model.rewindForPrompt(Array(prefix.prefix(seeded)))}
                    for offset in stride(from:seeded,to:history,by:options.chunk) { _=try model.evaluate(Array(prefix[offset..<min(history,offset+options.chunk)])) }
                    model.saveCheckpoint();seeded=history
                    let prefixSeconds=ProcessInfo.processInfo.systemUptime-start
                    for suffix in configuration.suffixes {
                        log("Calibration: retained history \(history), new suffix \(suffix)")
                        let ids=(0..<(history+suffix)).map{seed[$0%seed.count]}
                        let report=try ChronosValidation.compare(model:model,ids:ids,chunk:options.chunk,steps:configuration.decodeTokens,retainedHistory:prefix,rounds:configuration.rounds,candidateOnly:configuration.candidateOnly == true)
                        results.append(.init(history:history,suffix:suffix,prefixExtensionSeconds:prefixSeconds,report:report))
                        log("Benchmark result: "+String(decoding:try JSONEncoder().encode(results.last!),as:UTF8.self))
                        log("Checked history \(history), suffix \(suffix): reference compared=\(report.referenceCompared), parity=\(report.parityPassed), maximum logit difference=\(report.maxLogitDifference), continuation difference=\(report.maxContinuationDifference)")
                        guard report.parityPassed || (!report.referenceCompared && report.executionFinite) else {
                            FileHandle.standardOutput.write(try JSONEncoder().encode(results)+Data([10]))
                            throw ModelLibraryError.incompatible("Warm-history reference parity failed. The adapter and current profile were not promoted.")
                        }
                    }
                }
                FileHandle.standardOutput.write(try JSONEncoder().encode(results)+Data([10]))
                guard results.allSatisfy({$0.report.parityPassed || (!$0.report.referenceCompared && $0.report.executionFinite)}) else {throw ModelLibraryError.incompatible("Calibration failed reference parity. The current profile was retained.")};return
            }
            if options.validateReference {
                guard !options.prompt.isEmpty,!options.interactive else {throw ModelLibraryError.invalid("Parity validation requires a prompt and a non-interactive worker.")}
                let ids = try tokenizer.applyChatTemplate(messages:[["role":"user","content":options.prompt]],chatTemplate:.literal(template),addGenerationPrompt:true,truncation:false,maxLength:nil,tools:nil,additionalContext:["enable_thinking":false])
                let report = try ChronosValidation.compare(model:model,ids:ids,chunk:options.chunk,steps:options.validationSteps)
                FileHandle.standardOutput.write(try JSONEncoder().encode(report)+Data([10]))
                guard report.parityPassed else {throw ModelLibraryError.incompatible("Full-logit or continuation parity failed; the adapter remains experimental.")};return
            }
            let mailbox = ParticleInputMailbox()
            if options.interactive {mailbox.startStandardInput()}
            var messages:[Message] = []
            var next = options.prompt.isEmpty ? nil : options.prompt
            if options.desktop {event(["event":"capabilities","protocol_version":2,"architecture":model.reader.manifest.model.architecture.rawValue,"native_experts":model.reader.manifest.model.activeExperts,"tools":false,"reasoning":true,"context_limit":model.reader.manifest.model.contextLimit])}
            while true {
                let text:String
                if let initial = next {text = initial;next = nil}
                else {
                    guard options.interactive else {break}
                    if options.desktop {event(["event":"ready"])}else{log("You> ")}
                    guard let line = mailbox.next() else {break}
                    if options.desktop {
                        let command = try JSONDecoder().decode(DesktopCommand.self,from:Data(line.text.utf8))
                        if command.type == "quit" {break}
                        if ["reset","restore"].contains(command.type) {
                            let history = try command.validatedHistory()
                            messages = history.map{["role":$0.role,"content":$0.content]}
                            if let system = command.system {messages.insert(["role":"system","content":system],at:0)}
                            if let budget = command.reasoningBudget {options.budget = budget}
                            // Restore replaces the transcript, not the model state.
                            // The next rendered prompt must match token IDs before
                            // the recurrent checkpoint can be reused. Explicit New
                            // still clears every state immediately.
                            if command.type == "reset" {try model.reset()};continue
                        }
                        guard command.type == "message",let content = command.text,content.utf8.count <= 900_000 else {
                            event(["event":"error","message":"This experimental worker supports chat only; tool execution is not qualified."]);continue
                        }
                        text = content
                    } else {
                        if line.text == "/quit" {break}
                        if line.text == "/new" {messages = [];try model.reset();continue}
                        text = line.text
                    }
                }
                cancellation.reset()
                let previousMessages = messages
                messages.append(["role":"user","content":text])
                var modelChanged=false
                do {
                    let started = ProcessInfo.processInfo.systemUptime,readsBefore = model.expertReadBytes
                    let ids = try tokenizer.applyChatTemplate(messages:messages,chatTemplate:.literal(template),addGenerationPrompt:true,truncation:false,maxLength:nil,tools:nil,additionalContext:["enable_thinking":options.budget != 0])
                    let historicalInput = try tokenizer.applyChatTemplate(messages:messages,chatTemplate:.literal(template),addGenerationPrompt:false,truncation:false,maxLength:nil,tools:nil,additionalContext:["enable_thinking":options.budget != 0])
                    let stableBoundary = ids.starts(with:historicalInput) ? historicalInput.count : 0
                    guard ids.count+options.maxNew <= options.context else {throw ModelLibraryError.invalid("The prompt and reserved response exceed the configured context. Earlier turns were retained.")}
                    if options.desktop {event(["event":"input","utf8_sha256":DesktopTextIdentity.fingerprint(text)])}
                    modelChanged=true
                    let reused = try model.rewindForPrompt(ids)
                    var logits:MLXArray? = model.currentLogits
                    var start = reused
                    while start < ids.count {
                        if cancellation.isCancelled {throw CancellationError()}
                        var end = min(ids.count,start+options.chunk)
                        if start < stableBoundary {end = min(end,stableBoundary)}
                        logits = try model.evaluate(Array(ids[start..<end]))
                        if end == stableBoundary {model.saveCheckpoint()}
                        if options.desktop {event(["event":"prefill","completed":end,"total":ids.count])}
                        start = end
                    }
                    guard var last = logits else {throw ModelLibraryError.invalid("A generation boundary must contain new prompt tokens.")}
                    if stableBoundary == 0 {model.saveCheckpoint()}
                    let prefilled = ProcessInfo.processInfo.systemUptime
                    let opened=ids.lastIndex(of:opening),closed=ids.lastIndex(of:closing)
                    let thinkingInPrompt=opened.map{index in closed.map{index > $0} ?? true} ?? false
                    let leadingPrefix=thinkingInPrompt ? "<think>" : ""
                    var output = [Int](),emitted = leadingPrefix,steps = 0,finish = "length"
                    var byteDecoder = ChronosByteDecoder()
                    let sampler = GenerateParameters(temperature:options.temperature,topP:0.95,topK:20).sampler()
                    var limiter = DesktopReasoningLimiter(budget:options.budget,openingToken:opening,closingToken:closing)
                    if thinkingInPrompt {limiter.accept(opening)}
                    if !leadingPrefix.isEmpty {
                        if options.desktop {event(["event":"text","text":leadingPrefix,"prefilled":true])}
                        else {FileHandle.standardOutput.write(Data(leadingPrefix.utf8))}
                    }
                    for index in 0..<options.maxNew {
                        if cancellation.isCancelled {throw CancellationError()}
                        let token = autoreleasepool {limiter.mustClose ? closing : sample(logits:last,using:sampler)}
                        if stopIDs.contains(token) {finish = "stop";break}
                        if limiter.exhausted {finish = "reasoning_limit";break}
                        output.append(token);limiter.accept(token)
                        guard let spelling = tokenizer.convertIdToToken(token) else {throw ModelLibraryError.invalid("The tokenizer is missing a generated token.")}
                        if let body = try byteDecoder.append(spelling) {
                            let decoded=leadingPrefix+body
                            guard decoded.utf8.starts(with:emitted.utf8) else {throw ModelLibraryError.invalid("The decoded byte stream changed an emitted prefix.")}
                            let suffix = try ChronosByteDecoder.suffix(decoded,emittedBytes:emitted.utf8.count);emitted = decoded
                            if options.desktop {event(["event":"text","text":suffix])}else{FileHandle.standardOutput.write(Data(suffix.utf8))}
                        }
                        if index+1 < options.maxNew {last = try model.evaluate([token]);steps += 1}
                    }
                    let finished = ProcessInfo.processInfo.systemUptime
                    if String(data:byteDecoder.bytes,encoding:.utf8) == nil {finish = "unicode_boundary";messages = previousMessages;try model.reset()}
                    else if !(leadingPrefix+tokenizer.decode(tokens:output)).utf8.elementsEqual(emitted.utf8) {throw ModelLibraryError.incompatible("The tokenizer decoder does not match the qualified byte-level contract.")}
                    else if finish == "reasoning_limit" {messages = previousMessages;try model.reset()}
                    else {messages.append(["role":"assistant","content":emitted])}
                    if options.desktop {event(["event":"turn","text":emitted,"tokens":output.count,"decode_steps":steps,"decode_seconds":finished-prefilled,
                        "context_used":model.tokens.count,"context_limit":options.context,"finish":finish,"reused":reused,"prefill_seconds":prefilled-started,"source_read_bytes":model.expertReadBytes-readsBefore,"assistant_prefix_tokens":thinkingInPrompt ? 1 : 0,"active_experts":options.experts])}
                    else {print();log(String(format:"Particle: %d decode steps in %.2fs · %.2f tok/s · %d retained tokens",steps,finished-prefilled,Double(steps)/max(0.001,finished-prefilled),reused))}
                } catch {
                    messages = previousMessages;if modelChanged {try model.reset()}
                    if options.desktop {event(["event":error is CancellationError ? "cancelled" : "error","message":error is CancellationError ? "Stopped safely. Incomplete model state was discarded." : error.localizedDescription])}
                    else {log(error.localizedDescription)}
                    if !options.interactive {throw error}
                }
            }
        } catch {log("Particle: "+error.localizedDescription);exit(1)}
    }
}
