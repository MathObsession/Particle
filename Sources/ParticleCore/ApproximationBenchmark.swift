import Foundation
import ParticleGGUF

public struct ParticleQualityQuestion: Codable, Sendable {
    public var id: String, category: String, prompt: String
    public var choices: [String]
    public var correctIndex: Int
    public init(id: String, category: String, prompt: String, choices: [String], correctIndex: Int) {
        self.id=id; self.category=category; self.prompt=prompt; self.choices=choices; self.correctIndex=correctIndex
    }
}
public struct ParticleApproximationConfiguration: Codable, Sendable {
    public var profiles: [Int], decodeTokens: Int, rounds: Int
    public var qualityQuestions: [ParticleQualityQuestion]
    public init(profiles: [Int] = [6,4,3,2], decodeTokens: Int = 64, rounds: Int = 2, qualityQuestions: [ParticleQualityQuestion]) {
        self.profiles=profiles; self.decodeTokens=decodeTokens; self.rounds=rounds; self.qualityQuestions=qualityQuestions
    }
    public func validate() throws {
        guard profiles.contains(6), Set(profiles).count==profiles.count, profiles.allSatisfy({ (2...6).contains($0) }),
              (16...256).contains(decodeTokens), (1...3).contains(rounds), (4...256).contains(qualityQuestions.count),
              Set(qualityQuestions.map(\.id)).count==qualityQuestions.count else { throw ParticleError.invalid("Invalid approximation benchmark profiles, steps, rounds or questions") }
        for q in qualityQuestions {
            guard !q.id.isEmpty, !q.category.isEmpty, !q.prompt.isEmpty, q.prompt.utf8.count<=65536,
                  q.choices.count==4, q.choices.allSatisfy({ !$0.isEmpty && $0.utf8.count<=4096 }), (0..<4).contains(q.correctIndex) else {
                throw ParticleError.invalid("Invalid four-way quality question: \(q.id)")
            }
        }
    }
}
public enum ParticleChoiceScoring {
    /// Stable ties and a four-choice conditional log loss; this is not perplexity
    /// over a corpus and is not a free-form answer-quality measurement.
    public static func score(logits: [Float], correctIndex: Int) throws -> (prediction: Int, negativeLogLikelihood: Double) {
        guard logits.count==4, logits.allSatisfy(\.isFinite), (0..<4).contains(correctIndex) else {
            throw ParticleError.invalid("Invalid finite multiple-choice logits")
        }
        var best=0
        for i in 1..<4 where logits[i]>logits[best] { best=i }
        let maximum=Double(logits[best]), logSum=log(logits.reduce(0.0) { $0+exp(Double($1)-maximum) })+maximum
        return (best,logSum-Double(logits[correctIndex]))
    }
}
public struct ParticleQuestionResult: Codable, Sendable {
    public let id: String, category: String
    public let predictedIndex: Int, correctIndex: Int
    public let negativeLogLikelihood: Double
    public var correct: Bool { predictedIndex==correctIndex }
}
public struct ParticleApproximationPass: Codable, Sendable {
    public let activeExperts: Int, round: Int, decodeSteps: Int
    public let prefillSeconds: Double, decodeSeconds: Double, tokensPerSecond: Double
    public let p50StepSeconds: Double, p95StepSeconds: Double
    public let sourceBytes: Int, ssdReadBytes: Int, ssdWriteBytes: Int
    public let readWaitSeconds: Double, gpuWaitSeconds: Double, footprintBytes: UInt64
    public let generatedText: String
}
public struct ParticleProfileQuality: Codable, Sendable {
    public let activeExperts: Int, accuracy: Double, relativeAccuracy: Double?
    public let medianTokensPerSecond: Double, meanChoiceLogLoss: Double
    public let meetsPilotQualityFloor: Bool, meetsPilotSpeedTarget: Bool
    public let questions: [ParticleQuestionResult]
}
public struct ParticleApproximationReport: Codable, Sendable {
    public let labelTokens: [Int]
    public let passes: [ParticleApproximationPass]
    public let profiles: [ParticleProfileQuality]
    public let qualityMetric: String
    public let qualified: Bool
    public let notes: String
}
extension SarvamRunner {
    /// One live runner. Changing the expert budget requires clearing KV first;
    /// quality questions always get a newly evaluated prefix for their profile.
    public func benchmarkApproximation(seed: [Int], tokenizer: SarvamTokenizer, options: ParticleApproximationConfiguration,
                                       chunkSize: Int = 128, cancellation: ParticleCancellation? = nil,
                                       progress: ((String)->Void)? = nil) throws -> ParticleApproximationReport {
        try options.validate()
        guard !seed.isEmpty, seed.count+options.decodeTokens<=maxContext else { throw ParticleError.invalid("Approximation speed prompt exceeds context") }
        let labels=["A","B","C","D"]
        let prefixed=labels.map { tokenizer.encode(" "+$0) }, plain=labels.map { tokenizer.encode($0) }
        let encodedLabels=prefixed.allSatisfy({ $0.count==1 }) ? prefixed : plain
        guard encodedLabels.allSatisfy({ $0.count==1 }), Set(encodedLabels.flatMap { $0 }).count==4 else {
            throw ParticleError.invalid("Approximation screening requires four distinct single-token answer labels")
        }
        let labelIDs=encodedLabels.map { $0[0] }
        let questionTokens=options.qualityQuestions.map { q in
            tokenizer.encode("Question: "+q.prompt+"\n"+q.choices.enumerated().map { labels[$0.offset]+". "+$0.element }.joined(separator:"\n")+"\nAnswer:")
        }
        guard questionTokens.allSatisfy({ !$0.isEmpty && $0.count<=maxContext }) else { throw ParticleError.invalid("A quality question exceeds context") }
        let original=executionOptions
        defer { reset(); try? configureExecution(original) }
        var passes=[ParticleApproximationPass](), qualities=[Int:[ParticleQuestionResult]]()
        for round in 0..<options.rounds {
            let order=round%2==0 ? options.profiles : Array(options.profiles.reversed())
            for experts in order {
                reset()
                var profile=original; profile.activeExperts=experts
                try configureExecution(profile)
                progress?("speed diagnostic: \(experts) experts, round \(round+1)/\(options.rounds)")
                let start=ProcessInfo.processInfo.systemUptime
                try prefill(tokens:seed,chunkSize:chunkSize,cancellation:cancellation)
                let prefilled=ProcessInfo.processInfo.systemUptime, before=metrics
                var generated=[Int](), sampler=ParticleSampler(seed:42), steps=[Double]()
                for _ in 0..<options.decodeTokens {
                    if cancellation?.isCancelled == true { throw CancellationError() }
                    let began=ProcessInfo.processInfo.systemUptime
                    let id=try sampler.sample(currentLogits(),temperature:0,topK:50,topP:0.95)
                    generated.append(id); try forward(token:id,cancellation:cancellation)
                    steps.append(ProcessInfo.processInfo.systemUptime-began)
                }
                let finished=ProcessInfo.processInfo.systemUptime, after=metrics, sorted=steps.sorted(), seconds=finished-prefilled
                passes.append(ParticleApproximationPass(activeExperts:experts,round:round+1,decodeSteps:generated.count,
                    prefillSeconds:prefilled-start,decodeSeconds:seconds,tokensPerSecond:Double(generated.count)/seconds,
                    p50StepSeconds:sorted[(sorted.count-1)/2],p95StepSeconds:sorted[Int(ceil(Double(sorted.count)*0.95))-1],
                    sourceBytes:after.sourceReadBytes-before.sourceReadBytes,ssdReadBytes:after.ssdReadBytes-before.ssdReadBytes,
                    ssdWriteBytes:after.ssdWriteBytes-before.ssdWriteBytes,readWaitSeconds:after.expertReadWaitSeconds-before.expertReadWaitSeconds,
                    gpuWaitSeconds:after.gpuWaitSeconds-before.gpuWaitSeconds,footprintBytes:after.footprintBytes,
                    generatedText:try tokenizer.decode(generated)))
                if qualities[experts] == nil {
                    var results=[ParticleQuestionResult]()
                    for (index,q) in options.qualityQuestions.enumerated() {
                        if cancellation?.isCancelled == true { throw CancellationError() }
                        progress?("quality screening: \(experts) experts, \(q.id), \(index+1)/\(questionTokens.count)")
                        reset(); try prefill(tokens:questionTokens[index],chunkSize:chunkSize,cancellation:cancellation)
                        let logits=try currentLogits()
                        guard logits.allSatisfy(\.isFinite) else { throw ParticleError.invalid("Non-finite full logits in quality screening") }
                        let scores=labelIDs.map { logits[$0] }
                        let scored=try ParticleChoiceScoring.score(logits:scores,correctIndex:q.correctIndex)
                        results.append(ParticleQuestionResult(id:q.id,category:q.category,predictedIndex:scored.prediction,
                            correctIndex:q.correctIndex,negativeLogLikelihood:scored.negativeLogLikelihood))
                    }
                    qualities[experts]=results
                }
            }
        }
        let native=Double(qualities[6]!.filter(\.correct).count)/Double(options.qualityQuestions.count)
        let profiles=options.profiles.map { experts in
            let results=qualities[experts]!, accuracy=Double(results.filter(\.correct).count)/Double(results.count)
            let relative=native>0 ? accuracy/native : nil
            let rates=passes.filter { $0.activeExperts==experts }.map(\.tokensPerSecond).sorted()
            let middle=rates.count/2, median=rates.count%2==0 ? (rates[middle-1]+rates[middle])/2 : rates[middle]
            return ParticleProfileQuality(activeExperts:experts,accuracy:accuracy,relativeAccuracy:relative,medianTokensPerSecond:median,
                meanChoiceLogLoss:results.reduce(0) { $0+$1.negativeLogLikelihood }/Double(results.count),
                meetsPilotQualityFloor:native>=0.5 && (relative ?? 0)>=0.75,meetsPilotSpeedTarget:median>=4,questions:results)
        }
        return ParticleApproximationReport(labelTokens:labelIDs,passes:passes,profiles:profiles,
            qualityMetric:"Relative accuracy on a fixed raw four-choice screening suite; 75% of six-expert accuracy is the pilot floor, not a model-wide quality claim.",
            qualified:false,
            notes:"Single runner, reversed profile order on alternate rounds; KV reset at every profile/question. RAM/file caches retained. Speed passes force the requested decode steps, ignoring stop IDs, and are diagnostic only. Quality is screened once per profile. Neither target qualification nor a 20–25% general quality-loss claim follows from this pilot. Natural completions, long contexts, task-specific quality and sustained AC-power performance remain required.")
    }
}
