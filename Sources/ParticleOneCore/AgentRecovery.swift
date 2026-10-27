import Foundation

/// File mutation outcomes come from host events, never the model's prose.
/// A successful retry of the same target resolves its failure; unrelated reads,
/// writes or passing commands cannot erase it. Events before this run may be
/// excluded by the caller so an old failure does not block a new instruction.
public enum AgentMutationRecovery {
    public static func target(_ call:AgentCall,project:AgentProject)->String {
        let supplied=call.arguments["path"] ?? ""
        return (try? project.relativePath(supplied)) ?? supplied
    }
    public static func unresolved(_ events:[AgentEvent],project:AgentProject)->[AgentEvent] {
        var pending=[String:AgentEvent]()
        for event in events where ["write_file","edit_file"].contains(event.call.name) {
            let path=target(event.call,project:project)
            if event.status=="done"{pending.removeValue(forKey:path)}
            else if event.status=="failed"{pending[path]=event}
        }
        return pending.values.sorted{$0.created<$1.created}
    }
    public static func mayRetry(_ event:AgentEvent)->Bool {
        event.retryAllowed ?? (!event.output.contains("User denied") && !event.output.contains("Plan mode blocks"))
    }
    public static func instruction(_ failures:[AgentEvent])->String {
        let details=failures.prefix(4).map{event in
            "\(event.call.name) [\(event.call.arguments["path"] ?? "missing path")]: \(String(event.output.prefix(500)))"
        }.joined(separator:"\n")
        return "Host file-operation failure: these operations did not create or overwrite their files. Do not claim success or repeat the generated contents. Correct the tool arguments and retry within the selected project; read an existing file before editing, and await approval. If blocked, stop and report the actual failure.\n"+details
    }
}
