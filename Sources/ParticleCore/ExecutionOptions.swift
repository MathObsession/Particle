import Foundation

public enum ParticleAttentionMode: String, Sendable, CaseIterable {
    case automatic = "auto", reference, tiled, split, tensor, fast
}
public enum ParticleProjectionMode: String, Sendable { case automatic = "auto", reference, staged, vector }

/// Experimental math and scheduling require explicit selection. Automatic
/// dispatch retains the validated FP32 accumulation path until promotion.
public struct ParticleExecutionOptions: Sendable {
    public var attention: ParticleAttentionMode = .automatic
    public var projection: ParticleProjectionMode = .automatic
    public var hybrid = false
    public var profileAttention = false
    public var splitBlock = 256
    public var sourceWorkers = 4
    public var ssdWorkers = 2
    public var coalesceReads = false
    public var overlapExpertReads = true
    /// Candidate schedule: queue dependent work without redundant CPU waits.
    public var deferredDecode = false
    public var gpuRouting = false
    public var groupedExperts = false
    public var ffnVector = false
    /// Explicit approximation: retain the first k bias-ranked native experts.
    public var activeExperts = 6
    public var pipelinedVerification = false
    public var cooperativeQuant = false
    public init() {}
}
