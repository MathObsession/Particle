import Foundation
import Testing
@testable import ParticleCore

/// Exercises the actual Metal address/mask path beyond the previous 8K cap.
/// Sparse V and zero Q/K give an analytic uniform-softmax oracle, including
/// grouped-head mapping, the final position, and a poisoned future KV row.
@Suite(.serialized) struct LongContextAttentionTests {
    @Test func longPositionMaskAndGroupedHeads()throws {
        let gpu=try ParticleGPU(execution:.init())
        for context in [32768,65536,131072] {
            try autoreleasepool {
                let capacity=context+1
                let keys=try gpu.buffer(capacity*256),values=try gpu.buffer(capacity*256)
                memset(keys.contents(),0,keys.length);memset(values.contents(),0,values.length)
                let v=values.contents().assumingMemoryBound(to:Float.self)
                for g in 0..<4 {for d in 0..<64 {
                    v[g*64+d]=Float(g+1)
                    v[(context-1)*256+g*64+d]=16384*Float(g+1)
                    v[context*256+g*64+d]=1e9 // must remain causally invisible
                }}
                let queries=try (0..<2).map{_ in try gpu.buffer(4096)}
                let out=try (0..<2).map{_ in try gpu.buffer(4096)}
                for q in queries{memset(q.contents(),0,q.length)}
                let cb=try gpu.command("Particle long-context mask validation")
                try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:out,base:context-2,capacity:capacity,mode:.reference)
                try gpu.finish(cb)
                for row in 0..<2 {for head in 0..<64 {for d in 0..<64 {
                    let expected=Float(head/16+1)*(row==0 ? 1 : 16385)/Float(context-1+row)
                    let actual=out[row].contents().assumingMemoryBound(to:Float.self)[head*64+d]
                    #expect(actual.isFinite && abs(actual-expected)<1e-4)
                }}}
            }
        }
    }
}
