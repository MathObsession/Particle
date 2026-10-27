import Foundation
import Darwin

public final class AgentCancellation:@unchecked Sendable {
    private let lock=NSLock();private var stopped=false
    public init(){}
    public func cancel(){lock.lock();stopped=true;lock.unlock()}
    public var cancelled:Bool{lock.lock();defer{lock.unlock()};return stopped}
}
public struct AgentCommandResult:Sendable {public let output:String;public let exitCode:Int32;public let timedOut:Bool;public let cancelled:Bool}
public enum AgentCommand {
    /// Every command has its own process group and a bounded output reader.
    /// This is a permission-gated terminal, not an OS security sandbox.
    public static func run(_ command:String,root:URL,timeout:Double=60,cancellation:AgentCancellation,onOutput:@Sendable(String)->Void={_ in})throws->AgentCommandResult{
        guard !command.isEmpty,command.utf8.count<=16_384,(1...300).contains(timeout),!command.contains("\0") else{throw AgentFailure.message("Invalid command or timeout (1–300 seconds).")}
        let descriptors=UnsafeMutablePointer<Int32>.allocate(capacity:2);defer{descriptors.deallocate()}
        guard pipe(descriptors)==0 else{throw AgentFailure.message("Cannot open terminal output.")}
        let readFD=descriptors[0],writeFD=descriptors[1];defer{close(readFD)}
        var actions:posix_spawn_file_actions_t?,attr:posix_spawnattr_t?
        posix_spawn_file_actions_init(&actions);posix_spawnattr_init(&attr)
        defer{posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attr)}
        posix_spawn_file_actions_addchdir(&actions,root.path)
        posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0)
        posix_spawn_file_actions_adddup2(&actions,writeFD,STDOUT_FILENO);posix_spawn_file_actions_adddup2(&actions,writeFD,STDERR_FILENO)
        posix_spawn_file_actions_addclose(&actions,readFD);posix_spawn_file_actions_addclose(&actions,writeFD)
        posix_spawnattr_setflags(&attr,Int16(POSIX_SPAWN_SETPGROUP));posix_spawnattr_setpgroup(&attr,0)
        let arguments=["/bin/zsh","-f","-c",command].map{item -> UnsafeMutablePointer<CChar>? in item.withCString{strdup($0)}}
        let environment=["HOME=\(FileManager.default.homeDirectoryForCurrentUser.path)","PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin","LANG=en_US.UTF-8","TERM=dumb","TMPDIR=\(NSTemporaryDirectory())"].map{item -> UnsafeMutablePointer<CChar>? in item.withCString{strdup($0)}}
        defer{for p in arguments+environment{free(p)}}
        var argv=arguments+[nil],env=environment+[nil],pid:pid_t=0
        let code=posix_spawn(&pid,"/bin/zsh",&actions,&attr,&argv,&env);close(writeFD)
        guard code==0 else{throw AgentFailure.message("Could not start terminal command (\(code)).")}
        defer{kill(-pid,SIGKILL)}
        _=fcntl(readFD,F_SETFL,O_NONBLOCK)
        var output=Data(),status:Int32=0,exited=false,timedOut=false,cancelled=false,terminated:Double?
        let start=ProcessInfo.processInfo.systemUptime,buffer=UnsafeMutableRawPointer.allocate(byteCount:8192,alignment:16);defer{buffer.deallocate()}
        while true {
            let now=ProcessInfo.processInfo.systemUptime
            if !exited && (cancellation.cancelled || Task.isCancelled || now-start>timeout) {
                cancelled=cancellation.cancelled || Task.isCancelled;timedOut = !cancelled
                if terminated==nil{kill(-pid,SIGTERM);terminated=now}
                else if now-terminated!>0.5{kill(-pid,SIGKILL)}
            }
            var info=pollfd(fd:readFD,events:Int16(POLLIN),revents:0);_ = poll(&info,1,50)
            while true {
                let count=Darwin.read(readFD,buffer,8192)
                if count<=0{break}
                let take=min(count,max(0,65_536-output.count))
                if take>0{let chunk=Data(bytes:buffer,count:take);output.append(chunk);onOutput(String(decoding:chunk,as:UTF8.self))}
            }
            if !exited {let result=waitpid(pid,&status,WNOHANG);if result==pid{exited=true;kill(-pid,SIGTERM)}else if result<0 && errno != EINTR{exited=true}}
            if exited {
                while true {let count=Darwin.read(readFD,buffer,8192);if count<=0{break};let take=min(count,max(0,65_536-output.count));if take>0{let chunk=Data(bytes:buffer,count:take);output.append(chunk);onOutput(String(decoding:chunk,as:UTF8.self))}}
                break
            }
        }
        var text=String(decoding:output,as:UTF8.self)
        if output.count>=65_536{text+="\n[Output truncated to 64 KiB.]"}
        if timedOut{text+="\n[Command timed out; its process group was stopped.]"}
        if cancelled{text+="\n[Command cancelled.]"}
        let exitCode=(status & 0x7f)==0 ? (status >> 8)&0xff : 128+(status & 0x7f)
        return AgentCommandResult(output:text,exitCode:exitCode,timedOut:timedOut,cancelled:cancelled)
    }
}
