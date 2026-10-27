import Foundation
import CryptoKit
import Darwin

public enum AgentFailure:Error,LocalizedError {
    case message(String)
    public var errorDescription:String?{switch self{case .message(let s):s}}
}
public final class AgentProject:@unchecked Sendable {
    public let root:URL
    private let rootAliases:[String]
    private let lock=NSLock()
    private var readHashes=[String:String]()
    public init(root:URL)throws{
        self.root=root.standardizedFileURL.resolvingSymlinksInPath()
        var directory:ObjCBool=false
        guard FileManager.default.fileExists(atPath:self.root.path,isDirectory:&directory),directory.boolValue,self.root.path != "/",self.root.path != FileManager.default.homeDirectoryForCurrentUser.path else{throw AgentFailure.message("Choose a specific project folder, rather than your whole Mac or home folder.")}
        // Foundation may render /private/tmp as /tmp. Resolve the selected
        // root only, never a supplied child path; children still use O_NOFOLLOW.
        var aliases=[self.root.path,root.standardizedFileURL.path]
        if let resolved=realpath(self.root.path,nil){aliases.append(String(cString:resolved));free(resolved)}
        rootAliases=Array(Set(aliases)).sorted{$0.count>$1.count}
    }
    public static func hash(_ data:Data)->String{SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    public static func privatePath(_ path:String)->Bool{
        path.split(separator:"/").contains{piece in let s=piece.lowercased();return s==".git" || s==".ssh" || s==".aws" || s==".codex" || s==".agents" || s==".npmrc" || s==".netrc" || s.hasPrefix(".env") || s.hasSuffix(".pem") || s.hasSuffix(".key") || s=="credentials" || s.hasPrefix("id_rsa") || s.hasPrefix("id_ed25519")}
    }
    /// Accept the model's absolute project paths without widening access.
    /// Validate raw components before normalization, then continue using the
    /// existing descriptor-relative, O_NOFOLLOW filesystem operations.
    public func relativePath(_ path:String)throws->String {
        guard !path.contains("\0"),!path.contains("\\"),path.utf8.count<=4096,
              !path.split(separator:"/").contains("..") else{
            throw AgentFailure.message("Use a path inside the selected project, without parent traversal or backslashes.")
        }
        let relative:String
        if path.hasPrefix("/") {
            guard let prefix=rootAliases.first(where:{path==$0 || path.hasPrefix($0+"/")}) else{
                throw AgentFailure.message("That absolute path is outside the selected project. Use a project-relative path, such as showcase.html.")
            }
            relative=path==prefix ? "." : String(path.dropFirst(prefix.count+1))
        }else{relative=path}
        let components=relative.split(separator:"/").filter{$0 != "."}
        let normalized=components.joined(separator:"/")
        guard !Self.privatePath(normalized) else{throw AgentFailure.message("That path contains private credentials or protected project data.")}
        return normalized.isEmpty ? "." : normalized
    }
    private func parts(_ path:String)throws->[String]{
        try relativePath(path).split(separator:"/").filter{$0 != "."}.map(String.init)
    }
    private func parent<T>(_ path:String,create:Bool=false,_ action:(Int32,String)throws->T)throws->T{
        let components=try parts(path);guard let leaf=components.last else{throw AgentFailure.message("A file path is required.")}
        var fd=open(root.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd>=0 else{throw AgentFailure.message("Project folder is unavailable.")}
        defer{close(fd)}
        for component in components.dropLast(){
            if create && mkdirat(fd,component,0o755) != 0 && errno != EEXIST{throw AgentFailure.message("Cannot create project folder.")}
            let next=openat(fd,component,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard next>=0 else{throw AgentFailure.message("Project paths must not cross symlinks or missing directories.")}
            close(fd);fd=next
        }
        return try action(fd,leaf)
    }
    private func data(parent:Int32,leaf:String)throws->Data?{
        let fd=openat(parent,leaf,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK)
        if fd<0 && errno==ENOENT{return nil}
        guard fd>=0 else{throw AgentFailure.message("Cannot read this file; symlinks are not allowed.")}
        defer{close(fd)};var st=stat()
        guard fstat(fd,&st)==0,(st.st_mode & S_IFMT)==S_IFREG,st.st_size<=1_048_576 else{throw AgentFailure.message("Only regular UTF-8 files up to 1 MiB are supported. Use a specific small source file.")}
        var bytes=Data();let buffer=UnsafeMutableRawPointer.allocate(byteCount:64*1024,alignment:16);defer{buffer.deallocate()}
        while true {let n=Darwin.read(fd,buffer,64*1024);if n==0{break};if n<0 && errno==EINTR{continue};guard n>0,bytes.count+n<=1_048_576 else{throw AgentFailure.message("File changed or exceeded the size limit.")};bytes.append(buffer.assumingMemoryBound(to:UInt8.self),count:n)}
        return bytes
    }
    public func readText(_ path:String,remember:Bool=true)throws->String{
        let path=try relativePath(path)
        return try parent(path){fd,leaf in
            guard let bytes=try data(parent:fd,leaf:leaf),let text=String(data:bytes,encoding:.utf8),!text.contains("\0") else{throw AgentFailure.message("File is missing or is not UTF-8 text.")}
            if remember{lock.lock();readHashes[path]=Self.hash(bytes);lock.unlock()};return text
        }
    }
    public func read(_ path:String,start:Int=1,end:Int=120)throws->String{
        guard start>0,end>=start,end-start<200 else{throw AgentFailure.message("Read 1–200 lines at a time, starting from line 1.")}
        let text=try readText(path),lines=text.components(separatedBy:"\n")
        let selected=lines.enumerated().filter{$0.offset+1>=start && $0.offset+1<=end}.map{"\($0.offset+1): \($0.element)"}.joined(separator:"\n")
        return "File: \(path)\nSHA256: \(Self.hash(Data(text.utf8)))\nLines: \(lines.count)\n"+String(selected.prefix(6000))+(selected.count>6000 ? "\n[Output truncated; request a narrower line range.]" : "")
    }
    public func files(path:String=".",recursive:Bool=false)throws->[String]{
        let components=try parts(path)
        var fd=open(root.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);guard fd>=0 else{throw AgentFailure.message("Project unavailable.")};defer{close(fd)}
        for component in components{let next=openat(fd,component,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);guard next>=0 else{throw AgentFailure.message("Invalid folder or symlink.")};close(fd);fd=next}
        let excluded:Set<String>=["node_modules",".build",".swiftpm","dist","build",".next",".venv","venv","__pycache__","target","vendor",".DS_Store"]
        var result=[String](),visited=0
        func walk(_ directory:Int32,_ prefix:String,_ depth:Int)throws{
            guard depth<16,visited<10_000,result.count<2000 else{return}
            let duplicate=dup(directory);guard duplicate>=0,let stream=fdopendir(duplicate) else{if duplicate>=0{close(duplicate)};throw AgentFailure.message("Cannot list folder.")}
            defer{closedir(stream)}
            while let entry=readdir(stream){
                if Task.isCancelled{throw CancellationError()}
                let name=withUnsafePointer(to:entry.pointee.d_name){pointer in pointer.withMemoryRebound(to:CChar.self,capacity:MemoryLayout.size(ofValue:entry.pointee.d_name)){String(cString:$0)}}
                if name=="." || name==".."{continue}
                visited+=1;let relative=prefix.isEmpty ? name : prefix+"/"+name
                if Self.privatePath(relative) || excluded.contains(name){continue}
                var info=stat();guard fstatat(directory,name,&info,AT_SYMLINK_NOFOLLOW)==0 else{continue}
                switch info.st_mode & S_IFMT{
                case S_IFDIR:
                    if recursive{let child=openat(directory,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);if child>=0{defer{close(child)};try walk(child,relative,depth+1)}}else{result.append(relative+"/")}
                case S_IFREG:result.append(relative)
                default:break
                }
                if visited>=10_000 || result.count>=2000{break}
            }
        }
        try walk(fd,components.joined(separator:"/"),0)
        return result.sorted()
    }
    /// Small, local startup context avoids separate list/instructions model turns.
    /// Source contents are never scanned wholesale or sent to a web service.
    public func startupContext() throws -> String {
        let names=String(try files().prefix(36).joined(separator:", ").prefix(1100))
        var result="Root entries: "+names
        if let rules=try? readText("AGENTS.md",remember:false){
            result+="\nRoot AGENTS.md (project instructions):\n"+String(rules.prefix(1800))
            if rules.count>1800{result+="\n[Truncated: read AGENTS.md for the remaining instructions before editing.]"}
        }
        return result+"\nInspect applicable nested AGENTS.md before editing within a subfolder."
    }
    public func search(_ query:String,path:String=".")throws->String{
        guard !query.isEmpty,query.count<=512 else{throw AgentFailure.message("Use a search query of 1–512 characters.")}
        var results=[String]()
        for file in try files(path:path,recursive:true).prefix(500){
            if Task.isCancelled{throw CancellationError()}
            guard let text=try? readText(file,remember:false) else{continue}
            for (line,text) in text.components(separatedBy:"\n").enumerated() where text.localizedCaseInsensitiveContains(query){results.append("\(file):\(line+1): \(String(text.prefix(200)))");if results.count>=60{return results.joined(separator:"\n")+"\n[60 matches; narrow your query.]"}}
        }
        return results.isEmpty ? "No matches in the scanned source files." : results.joined(separator:"\n")
    }
    public func propose(_ call:AgentCall)throws->AgentChange{
        guard let suppliedPath=call.arguments["path"] else{throw AgentFailure.message("Missing file path; no file was changed.")}
        let path=try relativePath(suppliedPath)
        if call.name=="write_file",call.arguments["content"]==nil{throw AgentFailure.message("write_file requires content; no file was changed.")}
        return try parentForProposal(path){before in
            if let before {
                lock.lock();let known=readHashes[path];lock.unlock()
                guard known==Self.hash(Data(before.utf8)) else{throw AgentFailure.message("Read this file again before editing; it is new to this session or changed since it was read.")}
            }
            let after:String
            if call.name=="write_file",let content=call.arguments["content"]{after=content}
            else if call.name=="edit_file",let before,let old=call.arguments["old_string"],let new=call.arguments["new_string"],!old.isEmpty,let match=before.range(of:old,options:.literal),before.range(of:old,options:.literal,range:match.upperBound..<before.endIndex)==nil{after=before.replacingCharacters(in:match,with:new)}
            else{throw AgentFailure.message("Edit must match exactly one nonempty occurrence in an existing file.")}
            guard after.utf8.count<=1_048_576,!after.contains("\0"),after != before else{throw AgentFailure.message("No change, non-text content, or edit exceeds 1 MiB.")}
            return AgentChange(path:path,before:before,after:after,beforeHash:before.map{Self.hash(Data($0.utf8))},afterHash:Self.hash(Data(after.utf8)))
        }
    }
    private func parentForProposal<T>(_ path:String,_ action:(String?)throws->T)throws->T{
        _=try parts(path)
        do{return try parent(path){fd,leaf in let bytes=try data(parent:fd,leaf:leaf);if let bytes{guard let text=String(data:bytes,encoding:.utf8),!text.contains("\0") else{throw AgentFailure.message("Only UTF-8 text may be edited.")};return try action(text)};return try action(nil)}}
        catch{ // New nested paths are checked component-by-component at apply time.
            let url=root.appendingPathComponent(path)
            if !FileManager.default.fileExists(atPath:url.deletingLastPathComponent().path){return try action(nil)}
            throw error
        }
    }
    public func apply(_ change:AgentChange,undo:Bool=false)throws{
        let canonicalPath=try relativePath(change.path)
        try parent(change.path,create:!undo){fd,leaf in
            let existing=try data(parent:fd,leaf:leaf),expected=undo ? change.afterHash : change.beforeHash
            guard existing.map(Self.hash)==expected else{throw AgentFailure.message("File changed after review. Nothing was overwritten; read and review it again.")}
            let text=undo ? change.before : change.after
            if text==nil{guard unlinkat(fd,leaf,0)==0 else{throw AgentFailure.message("Could not undo new file.")};return}
            let name=".particle-"+UUID().uuidString
            let output=openat(fd,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600);guard output>=0 else{throw AgentFailure.message("Cannot stage file change.")}
            defer{close(output);unlinkat(fd,name,0)}
            var mode:mode_t=0o644,st=stat();if fstatat(fd,leaf,&st,AT_SYMLINK_NOFOLLOW)==0{mode=st.st_mode & 0o777}
            let bytes=Data(text!.utf8)
            try bytes.withUnsafeBytes { buffer in var offset=0;while offset<bytes.count{let n=Darwin.write(output,buffer.baseAddress!.advanced(by:offset),bytes.count-offset);if n<0 && errno==EINTR{continue};guard n>0 else{throw AgentFailure.message("Write failed.")};offset+=n}}
            guard fchmod(output,mode)==0,fsync(output)==0,try data(parent:fd,leaf:leaf).map(Self.hash)==expected,renameat(fd,name,fd,leaf)==0 else{throw AgentFailure.message("Could not commit change, or file changed during write.")}
            lock.lock();readHashes[canonicalPath]=Self.hash(bytes);lock.unlock()
        }
    }
}
public enum AgentDiff {
    public static func text(before:String?,after:String)->String{
        let a=(before ?? "").components(separatedBy:"\n"),b=after.components(separatedBy:"\n")
        var prefix=0;while prefix<min(a.count,b.count) && a[prefix]==b[prefix]{prefix+=1}
        var suffix=0;while suffix<min(a.count-prefix,b.count-prefix) && a[a.count-1-suffix]==b[b.count-1-suffix]{suffix+=1}
        let context=a[max(0,prefix-3)..<prefix].map{" "+$0}
        let removed=a[prefix..<(a.count-suffix)].map{"-"+$0},added=b[prefix..<(b.count-suffix)].map{"+"+$0}
        let lines=["@@ line \(prefix+1) @@"]+context+removed+added+Array(b[(b.count-suffix)..<min(b.count,b.count-suffix+3)]).map{" "+$0}
        return lines.prefix(600).joined(separator:"\n")+(lines.count>600 ? "\n[Preview limited to 600 lines; review the complete proposed file before approving.]" : "")
    }
}
