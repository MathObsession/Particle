import ParticleGGUF

/// The ledger contains only tokens whose forward passes actually completed.
/// A sampled terminal token or capped final token has no KV row yet.
public struct ParticlePromptReuse: Sendable {
    public let reused: Int
    public let suffix: [Int]
    public static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var common=0
        for (x,y) in zip(a,b) { if x != y { break }; common += 1 }
        return common
    }
    public init(committed: [Int], target: [Int], context: Int, outputAllowance: Int) throws {
        guard !target.isEmpty, outputAllowance > 0, outputAllowance <= context, target.count <= context-outputAllowance else {
            throw ParticleError.invalid("Conversation plus output allowance exceeds context; use /new or increase --max-context")
        }
        let common = Self.commonPrefix(committed,target)
        // Recompute the last row if the whole target already exists, because a
        // rewind intentionally invalidates logits from the previous suffix.
        reused = min(common,target.count-1)
        suffix = Array(target.dropFirst(reused))
    }
}

public struct SarvamMessage: Sendable {
    public enum Role: String, Sendable { case system, user, assistant, tool }
    public let role: Role
    public let content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}
