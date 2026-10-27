import Foundation
import ParticleGGUF

public struct ParticleSessionBenchmarkOptions: Codable, Sendable {
    public var histories: [Int]
    public var suffixes: [Int]
    public var rounds: Int
    public var decodeTokens: Int
    public init(histories: [Int] = [512,2048,4096,8000], suffixes: [Int] = [1,16,64,128], rounds: Int = 3, decodeTokens: Int = 64) {
        self.histories=histories; self.suffixes=suffixes; self.rounds=rounds; self.decodeTokens=decodeTokens
    }
}
public struct ParticleSessionBenchmarkPass: Codable, Sendable {
    public let mode: String, round: Int, history: Int, suffix: Int, decodeSteps: Int
    public let prefixExtensionSeconds: Double, prefillSeconds: Double, decodeSeconds: Double
    public let sourceBytes: Int, ssdReadBytes: Int, ssdWriteBytes: Int, ssdStoredBytes: Int
    public let footprintBytes: UInt64
    public let prefillReadWaitSeconds: Double, decodeReadWaitSeconds: Double
    public let prefillGPUWaitSeconds: Double, decodeGPUWaitSeconds: Double
    public let decodeExpertReadBytes: Int, decodeSSDReadBytes: Int
    public let maxLogitDifference: Float, maxContinuationDifference: Float
    public let continuationMatches: Bool, parityPassed: Bool
}
extension SarvamRunner {
    /// Extends one real KV lineage to each history length, then rewinds only
    /// its suffix between AB/BA passes. No second model or KV snapshot exists.
    /// The seed is a raw benchmark token stream; stop IDs are deliberately
    /// processed to measure the requested number of decode forward passes.
    public func benchmarkSession(seed: [Int], options: ParticleSessionBenchmarkOptions,
                                 candidate: ParticleExecutionOptions, chunkSize: Int = 128, cancellation: ParticleCancellation? = nil,
                                 progress: ((String)->Void)? = nil) throws -> [ParticleSessionBenchmarkPass] {
        guard !seed.isEmpty, seed.allSatisfy({ (0..<configuration.vocabulary).contains($0) }),
              !options.histories.isEmpty, !options.suffixes.isEmpty, (1...9).contains(options.rounds),
              (1...256).contains(options.decodeTokens), options.histories.allSatisfy({ $0>0 }),
              options.suffixes.allSatisfy({ (1...256).contains($0) }),
              options.histories.max()!+options.suffixes.max()!+options.decodeTokens<=maxContext else {
            throw ParticleError.invalid("Benchmark histories, suffixes and decode allowance exceed the configured context")
        }
        let length=options.histories.max()!+options.suffixes.max()!
        let stream=(0..<length).map { seed[$0%seed.count] }
        var control=candidate
        control.attention = .reference; control.projection = .reference; control.hybrid=false; control.deferredDecode=false; control.gpuRouting=false; control.groupedExperts=false; control.ffnVector=false; control.pipelinedVerification=false; control.cooperativeQuant=false
        control.overlapExpertReads=false; control.coalesceReads=false
        var passes: [ParticleSessionBenchmarkPass]=[], seeded=0
        reset()
        for history in options.histories.sorted() {
            try configureExecution(control); try rewind(to:seeded)
            let prefixStart=ProcessInfo.processInfo.systemUptime
            if history>seeded {
                progress?("Extending real history from \(seeded) to \(history) tokens")
                try prefill(tokens:Array(stream[seeded..<history]),chunkSize:chunkSize,emitLogits:false,cancellation:cancellation)
            }
            seeded=history
            let prefixSeconds=ProcessInfo.processInfo.systemUptime-prefixStart
            for suffix in options.suffixes {
                var baseline: [Float]=[], baselineEnd: [Float]=[], baselineTokens: [Int]=[]
                for round in 0..<options.rounds {
                    for isCandidate in (round%2==0 ? [false,true] : [true,false]) {
                        let mode=isCandidate ? "candidate" : "reference"
                        progress?("\(mode), history \(history), suffix \(suffix), round \(round+1)")
                        try configureExecution(isCandidate ? candidate : control); try rewind(to:history)
                        let before=metrics, start=ProcessInfo.processInfo.systemUptime
                        try prefill(tokens:Array(stream[history..<history+suffix]),chunkSize:chunkSize,cancellation:cancellation)
                        let prefilled=ProcessInfo.processInfo.systemUptime, afterPrefill=metrics, logits=try currentLogits()
                        var sampler=ParticleSampler(seed:42), generated: [Int]=[]
                        for _ in 0..<options.decodeTokens {
                            let token=try sampler.sample(currentLogits(),temperature:0,topK:50,topP:0.95)
                            generated.append(token); try forward(token:token,cancellation:cancellation)
                        }
                        let done=ProcessInfo.processInfo.systemUptime, final=try currentLogits(), after=metrics
                        if baseline.isEmpty { baseline=logits; baselineEnd=final; baselineTokens=generated }
                        let difference=zip(logits,baseline).reduce(Float(0)) { max($0,abs($1.0-$1.1)) }
                        let finalDifference=zip(final,baselineEnd).reduce(Float(0)) { max($0,abs($1.0-$1.1)) }
                        let matching=generated==baselineTokens
                        passes.append(ParticleSessionBenchmarkPass(mode:mode,round:round+1,history:history,suffix:suffix,decodeSteps:generated.count,
                            prefixExtensionSeconds:prefixSeconds,prefillSeconds:prefilled-start,decodeSeconds:done-prefilled,
                            sourceBytes:after.sourceReadBytes-before.sourceReadBytes,ssdReadBytes:after.ssdReadBytes-before.ssdReadBytes,
                            ssdWriteBytes:after.ssdWriteBytes-before.ssdWriteBytes,ssdStoredBytes:after.ssdStoredBytes,footprintBytes:after.footprintBytes,
                            prefillReadWaitSeconds:afterPrefill.expertReadWaitSeconds-before.expertReadWaitSeconds,
                            decodeReadWaitSeconds:after.expertReadWaitSeconds-afterPrefill.expertReadWaitSeconds,
                            prefillGPUWaitSeconds:afterPrefill.gpuWaitSeconds-before.gpuWaitSeconds,
                            decodeGPUWaitSeconds:after.gpuWaitSeconds-afterPrefill.gpuWaitSeconds,
                            decodeExpertReadBytes:after.expertReadBytes-afterPrefill.expertReadBytes,
                            decodeSSDReadBytes:after.ssdReadBytes-afterPrefill.ssdReadBytes,
                            maxLogitDifference:difference,maxContinuationDifference:finalDifference,continuationMatches:matching,
                            parityPassed:matching && difference<=0.0002 && finalDifference<=0.0002))
                    }
                }
            }
        }
        try configureExecution(candidate)
        return passes
    }
}
