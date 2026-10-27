import Foundation

/// Shared by the desktop settings and the native runner. Context includes both
/// input and output. The fixed Sarvam graph uses FP32 K/V: 19 layers × 2 ×
/// 4 KV heads × 64 dimensions × 4 bytes per token. No RoPE extrapolation.
public enum ParticleContextPolicy {
    public static let modelLimit = 131_072
    public static let defaultOutput = 16_384
    public static let kvBytesPerToken = 38_912
    public static let capacities = [512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072]
    public static func kvBytes(context: Int) -> UInt64 {
        UInt64(max(0, min(context, modelLimit))) * UInt64(kvBytesPerToken)
    }
    /// Conservative tiers leave room for resident weights, the existing expert
    /// slots, prompt scratch and macOS. They are admission limits, not measured
    /// guarantees of long-context quality or speed under other app workloads.
    public static func maximum(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        let gib = physicalMemory / 1_073_741_824
        if gib >= 24 { return modelLimit }
        if gib >= 12 { return 65_536 }
        if gib >= 8 { return 32_768 }
        return 8192
    }
    public static func available(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> [Int] {
        capacities.filter { $0 <= maximum(physicalMemory: physicalMemory) }
    }
    public static func recommended(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Int {
        maximum(physicalMemory: physicalMemory)
    }
    public static func outputAllowance(requested: Int, context: Int) -> Int {
        // Leave at least 32 positions for a prompt. Admission later checks the
        // exact rendered prompt plus the requested output and Instant prefix.
        min(requested, max(1, context - 32))
    }
}
