import Foundation
import MLX
import ParticleModelLibrary
import CryptoKit

struct ChronosParityReport: Codable {
    struct Pass: Codable {
        let mode: String
        let prefillSeconds: Double
        let decodeSeconds: Double
        let decodeSteps: Int
        let sourceReadBytes: Int64
        let maxLogitDifference: Float
        let maxContinuationDifference: Float
        let continuationMatches: Bool
        let logitSHA256: String
        let continuationSHA256: String
        let tokensSHA256: String
    }
    let test: String
    let forcedContinuation: Bool
    let maxLogitDifference: Float
    let maxContinuationDifference: Float
    let continuationMatches: Bool
    let parityPassed: Bool
    let passes: [Pass]
    let referenceCompared:Bool
    let executionFinite:Bool
}
@MainActor enum ChronosValidation {
    static func compare(model: ChronosModel, ids: [Int], chunk: Int, steps: Int,retainedHistory:[Int]? = nil,rounds:Int=1,candidateOnly:Bool=false) throws -> ChronosParityReport {
        var baseline = [Float](),baselineEnd = [Float](),baselineTokens = [Int](),passes = [ChronosParityReport.Pass]()
        var difference:Float = 0,endDifference:Float = 0,matching = true
        func digest<T>(_ values:[T])->String {values.withUnsafeBytes{SHA256.hash(data:Data($0)).map{String(format:"%02x",$0)}.joined()}}
        guard (1...9).contains(rounds) else {throw ModelLibraryError.invalid("Invalid validation round count.")}
        for reference in (0..<rounds).flatMap({_ in candidateOnly ? [false] : [true,false,false,true]}) {
            let retained:Int
            if let history=retainedHistory {
                guard ids.starts(with:history),model.checkpointPosition == history.count else {throw ModelLibraryError.invalid("Missing stable recurrent prefix for warm-history validation.")}
                retained=try model.rewindForPrompt(ids)
                guard retained >= history.count else {throw ModelLibraryError.invalid("Warm-history restoration unexpectedly discarded its valid prefix.")}
            } else {try model.reset();retained=0}
            model.setReference(reference)
            let reads = model.expertReadBytes,start = ProcessInfo.processInfo.systemUptime
            var logits:MLXArray?=model.currentLogits
            for offset in stride(from:retained,to:ids.count,by:chunk) {logits = try model.evaluate(Array(ids[offset..<min(ids.count,offset+chunk)]))}
            guard var last = logits else {throw ModelLibraryError.invalid("An input is required for parity validation.")}
            let prefilled = ProcessInfo.processInfo.systemUptime,initial = last.asType(.float32).asArray(Float.self)
            var tokens = [Int]()
            for _ in 0..<steps {
                let token = argMax(last).item(Int.self);tokens.append(token);last = try model.evaluate([token])
            }
            let finished = ProcessInfo.processInfo.systemUptime,final = last.asType(.float32).asArray(Float.self)
            guard initial.allSatisfy(\.isFinite),final.allSatisfy(\.isFinite) else {throw ModelLibraryError.invalid("Non-finite validation logits.")}
            if baseline.isEmpty {baseline = initial;baselineEnd = final;baselineTokens = tokens}
            let initialDifference = zip(initial,baseline).map{abs($0-$1)}.max() ?? .infinity
            let finalDifference = zip(final,baselineEnd).map{abs($0-$1)}.max() ?? .infinity
            difference = max(difference,initialDifference)
            endDifference = max(endDifference,finalDifference)
            matching = matching && tokens == baselineTokens
            passes.append(.init(mode:reference ? "upstream-selected-expert-reference" : "streamed-candidate",prefillSeconds:prefilled-start,decodeSeconds:finished-prefilled,decodeSteps:steps,sourceReadBytes:model.expertReadBytes-reads,maxLogitDifference:initialDifference,maxContinuationDifference:finalDifference,continuationMatches:tokens == baselineTokens,logitSHA256:digest(initial),continuationSHA256:digest(final),tokensSHA256:digest(tokens.map(Int32.init))))
        }
        if let history=retainedHistory {_=try model.rewindForPrompt(history)}else{try model.reset()}
        model.setReference(false)
        return .init(test:candidateOnly ? "storage-profile trial, one model process; no upstream comparison" : "full logits and greedy continuation, ABBA, one model process",forcedContinuation:true,
            maxLogitDifference:difference,maxContinuationDifference:endDifference,continuationMatches:!candidateOnly && matching,
            parityPassed:!candidateOnly && difference <= 0.001 && endDifference <= 0.001 && matching,passes:passes,referenceCompared:!candidateOnly,executionFinite:true)
    }
}
