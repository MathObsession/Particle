import Foundation
import Testing
import ParticleGGUF
@testable import ParticleCore

@Suite struct ApproximationTests {
    @Test func nativeRoutingIsUnchangedAndReducedBudgetsKeepNativeRanking() throws {
        let logits=(0..<128).map { sin(Float($0)*0.31)*4 }
        var bias=[Float](repeating:0,count:128); bias[127]=100
        let native=try SarvamRouting.select(logits:logits,bias:bias)
        let explicit=try SarvamRouting.select(logits:logits,bias:bias,activeExperts:6)
        #expect(native.experts==explicit.experts)
        #expect(native.weights.map(\.bitPattern)==explicit.weights.map(\.bitPattern))
        for k in 2...6 {
            let route=try SarvamRouting.select(logits:logits,bias:bias,activeExperts:k)
            #expect(route.experts==Array(native.experts.prefix(k)))
            #expect(route.weights.count==k)
            #expect(abs(route.weights.reduce(0,+)-2.5)<0.000001)
            let scores=route.experts.map { id -> Float in let x=logits[id];return x>=0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x)) }
            let denominator=scores.reduce(Float(1e-20),+)
            #expect(route.weights.map(\.bitPattern)==scores.map { $0/denominator*2.5 }.map(\.bitPattern))
        }
        for k in [0,1,7,128] { #expect(throws:ParticleError.self) { try SarvamRouting.select(logits:logits,bias:bias,activeExperts:k) } }
        let equal=try SarvamRouting.select(logits:[Float](repeating:1000,count:128),bias:[Float](repeating:0,count:128),activeExperts:3)
        #expect(equal.experts==[0,1,2]);let finite=equal.weights.allSatisfy { $0.isFinite };#expect(finite)
    }
    @Test func promptPlannerTracksOnlySelectedRanks() throws {
        for k in 2...6 {
            let routes=(0..<9).map { row in RoutingResult(experts:(0..<k).map { ($0+row)%13 },weights:[Float](repeating:2.5/Float(k),count:k)) }
            let plan=try ExpertBatchPlan(routes:routes,cached:[0,1],capacity:6)
            #expect(plan.batches.allSatisfy { $0.count<=6 })
            #expect(plan.uses.values.reduce(0) { $0+$1.count }==routes.count*k)
            for (expert,uses) in plan.uses { for use in uses { #expect(use.rank<k);#expect(routes[use.row].experts[use.rank]==expert) } }
        }
        #expect(throws:ParticleError.self) { try ExpertBatchPlan(routes:[RoutingResult(experts:[0,1,2],weights:[1,1])],cached:[],capacity:6) }
    }
    @Test func fourChoiceScoringHasStableTiesAndRejectsInvalidScores() throws {
        let uniform=try ParticleChoiceScoring.score(logits:[1,1,1,1],correctIndex:2)
        #expect(uniform.prediction==0);#expect(abs(uniform.negativeLogLikelihood-log(4))<1e-10)
        let high=try ParticleChoiceScoring.score(logits:[-1000,1000,0,1],correctIndex:1)
        #expect(high.prediction==1);#expect(high.negativeLogLikelihood==0)
        #expect(throws:ParticleError.self) { try ParticleChoiceScoring.score(logits:[.nan,0,0,0],correctIndex:0) }
        #expect(throws:ParticleError.self) { try ParticleChoiceScoring.score(logits:[0,0,0,0],correctIndex:4) }
    }
    @Test func screeningConfigurationRequiresNativeBaselineAndUniqueQuestions() throws {
        let questions=(0..<4).map { ParticleQualityQuestion(id:"q\($0)",category:"test",prompt:"Choose one",choices:["one","two","three","four"],correctIndex:0) }
        var configuration=ParticleApproximationConfiguration(qualityQuestions:questions)
        try configuration.validate()
        configuration.profiles=[3,2];#expect(throws:ParticleError.self) { try configuration.validate() }
        configuration.profiles=[6,6];#expect(throws:ParticleError.self) { try configuration.validate() }
        configuration.profiles=[6,3];configuration.qualityQuestions[1].id=configuration.qualityQuestions[0].id
        #expect(throws:ParticleError.self) { try configuration.validate() }
    }
}
