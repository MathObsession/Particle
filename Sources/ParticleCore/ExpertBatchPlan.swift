import Foundation
import ParticleGGUF

struct ExpertUse {
    let row: Int
    let rank: Int
}

/// Serve each distinct expert once per prompt chunk. Cached experts go first
/// so later misses cannot evict a weight before its scheduled consumers run.
/// Frequently used cold experts go last to leave useful entries in the cache.
struct ExpertBatchPlan {
    let uses: [Int:[ExpertUse]]
    let batches: [[Int]]
    init(routes: [RoutingResult], cached: Set<Int>, capacity: Int) throws {
        guard !routes.isEmpty, routes.count <= 256, (6...16).contains(capacity) else {
            throw ParticleError.invalid("Invalid prompt expert batch")
        }
        var uses: [Int:[ExpertUse]] = [:]
        for (row,route) in routes.enumerated() {
            guard (2...6).contains(route.experts.count), route.weights.count == route.experts.count, Set(route.experts).count == route.experts.count,
                  route.experts.allSatisfy({ (0..<128).contains($0) }), route.weights.allSatisfy(\.isFinite) else {
                throw ParticleError.invalid("Invalid prompt routing")
            }
            for (rank,expert) in route.experts.enumerated() {
                uses[expert,default:[]].append(ExpertUse(row:row,rank:rank))
            }
        }
        let ordered = uses.keys.sorted {
            let a = cached.contains($0), b = cached.contains($1)
            if a != b { return a }
            if uses[$0]!.count != uses[$1]!.count { return uses[$0]!.count < uses[$1]!.count }
            return $0 < $1
        }
        self.uses = uses
        batches = stride(from:0,to:ordered.count,by:capacity).map { Array(ordered[$0..<min($0+capacity,ordered.count)]) }
    }
}
