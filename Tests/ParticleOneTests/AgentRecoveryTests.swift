import Foundation
import Testing
import Darwin
@testable import ParticleOneCore

@Suite(.serialized) struct AgentRecoveryTests {
    private func folder()throws->URL {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("particle-agent-recovery-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        return root.resolvingSymlinksInPath()
    }
    @Test func screenshotAbsoluteWriteListsReadsEditsAndUndoesWithinRoot()throws {
        let root=try folder();defer{try? FileManager.default.removeItem(at:root)}
        let project=try AgentProject(root:root),file=root.appendingPathComponent("showcase.html")
        let html="<!DOCTYPE html>\n<html><body>नमस्ते 🌿</body></html>\n"
        let proposal=try project.propose(.init(name:"write_file",arguments:["path":file.path,"content":html]))
        #expect(proposal.path=="showcase.html")
        #expect(!FileManager.default.fileExists(atPath:file.path)) // still awaits approval
        try project.apply(proposal)
        #expect(try project.files(path:root.path)==["showcase.html"])
        #expect(try project.readText(file.path)==html)
        // The absolute read must authorize the relative edit under one hash key.
        let edit=try project.propose(.init(name:"edit_file",arguments:["path":"./showcase.html","old_string":"नमस्ते","new_string":"Hello"]))
        try project.apply(edit);#expect(try project.readText("showcase.html").contains("Hello"))
        try project.apply(edit,undo:true);#expect(try project.readText(file.path)==html)
        try project.apply(proposal,undo:true);#expect(!FileManager.default.fileExists(atPath:file.path))
    }
    @Test func selectedRootSystemAliasesShareOneReadHashAndMutationTarget()throws {
        let root=try folder();defer{try? FileManager.default.removeItem(at:root)}
        let project=try AgentProject(root:root)
        let canonical=try #require(realpath(root.path,nil));defer{free(canonical)}
        let posixRoot=String(cString:canonical)
        let file=root.appendingPathComponent("alias.txt");try "before".write(to:file,atomically:true,encoding:.utf8)
        #expect(try project.files(path:posixRoot)==["alias.txt"])
        #expect(try project.readText(posixRoot+"/alias.txt")=="before")
        let change=try project.propose(.init(name:"write_file",arguments:["path":file.path,"content":"after"]))
        try project.apply(change);#expect(try project.readText("alias.txt")=="after")
        #expect(throws:(any Error).self){try project.relativePath(posixRoot+"-sibling/file")}
    }
    @Test func normalizationCannotEscapeRootOrBypassCredentialAndSymlinkGuards()throws {
        let root=try folder(),outside=try folder();defer{try? FileManager.default.removeItem(at:root);try? FileManager.default.removeItem(at:outside)}
        let project=try AgentProject(root:root)
        for path in [outside.appendingPathComponent("file").path,root.path+"-sibling/file",root.path+"/../escape",root.path+"/src/../../escape",root.path+"/.env",root.path+"/.git/config","../file","x\\y","bad\0path"] {
            #expect(throws:(any Error).self){try project.propose(.init(name:"write_file",arguments:["path":path,"content":"blocked"]))}
        }
        try "outside".write(to:outside.appendingPathComponent("secret.txt"),atomically:true,encoding:.utf8)
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("link"),withDestinationURL:outside)
        #expect(throws:(any Error).self){try project.readText(root.path+"/link/secret.txt")}
        let proposal=try project.propose(.init(name:"write_file",arguments:["path":root.path+"/link/new/deep.txt","content":"blocked"]))
        #expect(throws:(any Error).self){try project.apply(proposal)}
        #expect(!FileManager.default.fileExists(atPath:outside.path+"/new/deep.txt"))
        #expect(try project.relativePath(root.path+"/src//./file.txt")=="src/file.txt")
        #expect(try project.relativePath(root.path)==".")
    }
    @Test func failedMutationCannotBeResolvedByProseReadsOtherFilesOrChecks()throws {
        let root=try folder();defer{try? FileManager.default.removeItem(at:root)}
        let project=try AgentProject(root:root)
        let failed=AgentEvent(.init(name:"write_file",arguments:["path":root.path+"/showcase.html"]),status:"failed",output:"Disk write failed")
        let read=AgentEvent(.init(name:"read_file",arguments:["path":"showcase.html"]),status:"done")
        let other=AgentEvent(.init(name:"write_file",arguments:["path":"different.html"]),status:"done")
        let check=AgentEvent(.init(name:"run_command",arguments:["command":"true"]),status:"done")
        #expect(AgentMutationRecovery.unresolved([failed,read,other,check],project:project).map(\.id)==[failed.id])
        let repaired=AgentEvent(.init(name:"edit_file",arguments:["path":"./showcase.html"]),status:"done")
        #expect(AgentMutationRecovery.unresolved([failed,read,other,check,repaired],project:project).isEmpty)
        #expect(AgentMutationRecovery.instruction([failed]).contains("did not create or overwrite"))
    }
    @Test func deniedAndPlanBlockedMutationsAreNotAutomaticRetryCandidates()throws {
        let root=try folder();defer{try? FileManager.default.removeItem(at:root)}
        var denied=AgentEvent(.init(name:"write_file",arguments:["path":"a.txt"]),status:"failed",output:"User denied this operation.")
        #expect(!AgentMutationRecovery.mayRetry(denied))
        denied.output="A host decision";denied.retryAllowed=false
        #expect(!AgentMutationRecovery.mayRetry(denied))
        let decoded=try JSONDecoder().decode(AgentEvent.self,from:JSONEncoder().encode(denied))
        #expect(decoded.retryAllowed==false)
        let blocked=AgentEvent(.init(name:"edit_file",arguments:[:]),status:"failed",output:"Plan mode blocks edits and terminal commands.")
        #expect(!AgentMutationRecovery.mayRetry(blocked))
    }
}
