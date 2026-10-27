import Foundation

public struct ParticleInputLine: Sendable {
    public let text: String
    public let receivedAt: Double
}
/// Input is independent of the single model owner. A submitted line cancels
/// idle work only; it never cancels an active foreground generation.
public final class ParticleInputMailbox: @unchecked Sendable {
    private let condition=NSCondition()
    private var lines: [ParticleInputLine]=[]
    private var ended=false
    private var idle: ParticleCancellation?
    public init() {}
    public func submit(_ text: String) {
        condition.lock(); defer { condition.unlock() }
        while lines.count>=16 && !ended { condition.wait() }
        guard !ended else { return }
        lines.append(ParticleInputLine(text:text,receivedAt:ProcessInfo.processInfo.systemUptime))
        idle?.cancel(); condition.broadcast()
    }
    public func finish() {
        condition.lock(); ended=true; idle?.cancel(); condition.broadcast(); condition.unlock()
    }
    public func next() -> ParticleInputLine? {
        condition.lock(); defer { condition.unlock() }
        while lines.isEmpty && !ended { condition.wait() }
        guard !lines.isEmpty else { return nil }
        let line=lines.removeFirst(); condition.broadcast(); return line
    }
    public func beginIdle(_ cancellation: ParticleCancellation) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard lines.isEmpty && !ended else { return false }
        idle=cancellation; return true
    }
    public func endIdle() { condition.lock(); idle=nil; condition.unlock() }
    public func startStandardInput() {
        Thread.detachNewThread { [self] in
            while let line=readLine() { submit(line) }
            finish()
        }
    }
}
