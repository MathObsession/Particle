import Foundation
import ParticleGGUF

public struct ParticleModelComparisonReport: Codable, Sendable {
    public let referenceFormat: String, candidateFormat: String
    public let promptTokens: Int, continuationSteps: Int, comparedLogits: Int
    public let bitwiseMismatches: Int, maximumAbsoluteDifference: Float
    public let continuationTokens: [Int], passed: Bool
}

/// Allocates one runner at a time. The retained reference consists only of
/// a few output vectors and token IDs, never a second model or KV lineage.
public enum ParticleModelComparison {
    public static func compare(reference: any ParticleModelSource, candidate: any ParticleModelSource,
                               tokens: [Int], context: Int, slots: Int, steps: Int = 2,
                               execution: ParticleExecutionOptions = .init(), cancellation: ParticleCancellation? = nil,
                               progress: ((String) -> Void)? = nil) throws -> ParticleModelComparisonReport {
        guard !tokens.isEmpty, (1...8).contains(steps), tokens.count+steps <= context else {
            throw ParticleError.invalid("Model comparison requires a prompt and room for 1...8 continuation steps")
        }
        func run(_ source: any ParticleModelSource, name: String) throws -> ([[Float]],[Int]) {
            try autoreleasepool {
                progress?("\(name): loading \(source.formatName)")
                let runner = try SarvamRunner(file:source,maxContext:context,expertCacheSlots:slots,execution:execution)
                var logits: [[Float]] = [], generated: [Int] = []
                for (index,token) in tokens.enumerated() {
                    try runner.forward(token:token,emitLogits:index == tokens.count-1,cancellation:cancellation)
                }
                for step in 0...steps {
                    let values = try runner.currentLogits()
                    logits.append(values)
                    // Stable tie-breaking matches scalar greedy sampling.
                    let next = values.indices.max { values[$0] == values[$1] ? $0 > $1 : values[$0] < values[$1] }!
                    generated.append(next)
                    progress?("\(name): checked vector \(step+1)/\(steps+1)")
                    if step < steps { try runner.forward(token:next,cancellation:cancellation) }
                }
                return (logits,generated)
            }
        }
        let baseline = try run(reference,name:"reference")
        let other = try run(candidate,name:"candidate")
        var mismatches = 0, maximum: Float = 0, count = 0
        guard baseline.0.count == other.0.count else { throw ParticleError.invalid("Comparison vector count mismatch") }
        for (a,b) in zip(baseline.0,other.0) {
            guard a.count == b.count else { throw ParticleError.invalid("Comparison vocabulary mismatch") }
            for (x,y) in zip(a,b) { count += 1; if x.bitPattern != y.bitPattern { mismatches += 1 }; maximum = max(maximum,abs(x-y)) }
        }
        return ParticleModelComparisonReport(referenceFormat:reference.formatName,candidateFormat:candidate.formatName,
            promptTokens:tokens.count,continuationSteps:steps,comparedLogits:count,bitwiseMismatches:mismatches,
            maximumAbsoluteDifference:maximum,continuationTokens:other.1,passed:mismatches == 0 && baseline.1 == other.1)
    }
}
