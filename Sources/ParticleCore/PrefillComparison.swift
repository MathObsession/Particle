import Foundation
import ParticleGGUF

public struct PrefillPass: Codable, Sendable {
    public let mode: String
    public let promptTokens: Int
    public let chunkSize: Int
    public let reusedTokens: Int
    public let prefillSeconds: Double
    public let expertReadMiB: Double
    public let sourceReadMiB: Double
    public let ssdReadMiB: Double
    public let continuationSourceMiB: Double
    public let continuationSSDMiB: Double
    public let expertReadWaitSeconds: Double
    public let gpuWaitSeconds: Double
    public let continuationSeconds: Double
    public let continuationReadMiB: Double
    public let continuationReadWaitSeconds: Double
    public let continuationGPUWaitSeconds: Double
    public let nextToken: Int
    public let continuationToken: Int
    public let maxLogitDifference: Float
    public let maxContinuationDifference: Float
    public let bitwiseLogitMismatches: Int
    public let footprintMiB: Double
}

public struct PrefillComparison: Codable, Sendable {
    public let passes: [PrefillPass]
    public let cachePolicy: String
    public let orderingNote: String
}

extension SarvamRunner {
    /// A scalar/grouped/scalar sandwich plus a prefix replay keeps one model owner and compares
    /// full logits plus a continuation step. Expert slots are cleared between
    /// passes; filesystem caches are deliberately not flushed.
    public func comparePrefill(tokens: [Int], chunkSize: Int, cancellation: ParticleCancellation? = nil,
                               onProgress: ((String,Int,Int,Int) -> Void)? = nil) throws -> PrefillComparison {
        guard (2...256).contains(chunkSize), tokens.count < maxContext else {
            throw ParticleError.invalid("Comparison needs chunk size 2...128 and room for one continuation token")
        }
        let candidate=executionOptions
        var control=candidate; control.attention = .reference; control.projection = .reference; control.hybrid=false; control.deferredDecode=false; control.gpuRouting=false; control.groupedExperts=false; control.ffnVector=false; control.pipelinedVerification=false; control.cooperativeQuant=false
        defer { _ = try? configureExecution(candidate) }
        var passes: [PrefillPass] = [], reference: [Float] = [], continued: [Float] = []
        var referenceNext = 0, referenceContinuation = 0
        var sampler = ParticleSampler(seed:42)
        for (index,size) in [1,chunkSize,1,chunkSize].enumerated() {
            let mode = ["scalar-first","grouped","scalar-repeat","reused-prefix"][index]
            try configureExecution(size == 1 ? control : candidate)
            let reused = index == 3 ? tokens.count/2 : 0
            if index == 3 { try rewind(to:reused) }
            else { reset(clearExpertCache:true) }
            let before = metrics, start = ProcessInfo.processInfo.systemUptime
            try prefill(tokens:Array(tokens.dropFirst(reused)),chunkSize:size,cancellation:cancellation,
                        onLayer:{ onProgress?(mode,$0+reused,$1,$2) })
            let seconds = ProcessInfo.processInfo.systemUptime-start, after = metrics
            let logits = try currentLogits()
            let next = try sampler.sample(logits,temperature:0,topK:50,topP:0.95)
            let continuationStart = ProcessInfo.processInfo.systemUptime
            try forward(token:next,cancellation:cancellation)
            let continuationSeconds = ProcessInfo.processInfo.systemUptime-continuationStart
            let continuation = try currentLogits(), end = metrics
            let nextAfter = try sampler.sample(continuation,temperature:0,topK:50,topP:0.95)
            if index == 0 {
                reference = logits; continued = continuation
                referenceNext = next; referenceContinuation = nextAfter
            }
            var difference: Float = 0, continuationDifference: Float = 0, mismatches = 0
            for i in logits.indices {
                difference = max(difference,abs(logits[i]-reference[i]))
                continuationDifference = max(continuationDifference,abs(continuation[i]-continued[i]))
                if logits[i].bitPattern != reference[i].bitPattern { mismatches += 1 }
            }
            guard difference <= 0.0002, continuationDifference <= 0.0002,
                  next == referenceNext, nextAfter == referenceContinuation else {
                throw ParticleError.invalid("\(mode) failed prefill/continuation parity: logits delta \(difference), continuation delta \(continuationDifference)")
            }
            passes.append(PrefillPass(mode:mode,promptTokens:tokens.count,chunkSize:size,reusedTokens:reused,prefillSeconds:seconds,
                expertReadMiB:Double(after.expertReadBytes-before.expertReadBytes)/1048576,
                sourceReadMiB:Double(after.sourceReadBytes-before.sourceReadBytes)/1048576,
                ssdReadMiB:Double(after.ssdReadBytes-before.ssdReadBytes)/1048576,
                continuationSourceMiB:Double(end.sourceReadBytes-after.sourceReadBytes)/1048576,
                continuationSSDMiB:Double(end.ssdReadBytes-after.ssdReadBytes)/1048576,
                expertReadWaitSeconds:after.expertReadWaitSeconds-before.expertReadWaitSeconds,
                gpuWaitSeconds:after.gpuWaitSeconds-before.gpuWaitSeconds,continuationSeconds:continuationSeconds,
                continuationReadMiB:Double(end.expertReadBytes-after.expertReadBytes)/1048576,
                continuationReadWaitSeconds:end.expertReadWaitSeconds-after.expertReadWaitSeconds,
                continuationGPUWaitSeconds:end.gpuWaitSeconds-after.gpuWaitSeconds,nextToken:next,continuationToken:nextAfter,
                maxLogitDifference:difference,maxContinuationDifference:continuationDifference,bitwiseLogitMismatches:mismatches,
                footprintMiB:Double(max(after.footprintBytes,end.footprintBytes))/1048576))
        }
        return PrefillComparison(passes:passes,cachePolicy:"RAM expert slots cleared for the first three passes; final pass rewinds existing KV; SSD expert cache retained if enabled",
            orderingNote:"scalar/grouped/scalar/prefix-replay in one process; OS filesystem caches not flushed; enabled SSD cache can warm between passes; short diagnostic, not a throughput benchmark")
    }
}
