import Foundation
import Darwin
import ParticleGGUF

public struct ParticleReadTiming: Codable, Sendable {
    public let storage: String, workers: Int, coalesced: Bool, bytes: Int
    public let medianSeconds: Double, requestedMiBPerSecond: Double
    public let note: String
}
private struct TimedRead: Sendable { let fd: Int32, offset: Int, count: Int }
private final class IOFailures: @unchecked Sendable {
    let lock=NSLock(); private var first: (any Error)?
    func record(_ error: any Error) { lock.lock(); if first==nil { first=error }; lock.unlock() }
    var error: (any Error)? { lock.lock(); defer { lock.unlock() }; return first }
}
extension ParticleValidation {
    /// Bounded demand-shaped read test, with ordinary runtime source-fd policy.
    /// Repeated reads can hit the OS cache: these are not cold media rates.
    public static func benchmarkReads(file: any ParticleModelSource, ssdDirectory: URL?) throws -> [ParticleReadTiming] {
        var source: [TimedRead]=[]
        for name in ["gate","up","down"] {
            let tensor=try file.tensor("blk.1.ffn_\(name)_exps.weight")
            for expert in 0..<6 {
                if name == "gate", let packed = try file.expertRead(layer:1,expert:expert) {
                    source.append(TimedRead(fd:packed.range.descriptor,offset:packed.range.offset,count:packed.range.count))
                } else if file.formatName == "gguf" {
                    let range = try file.location(for:tensor,relativeOffset:expert*tensor.expertBytes)
                    source.append(TimedRead(fd:range.descriptor,offset:range.offset,count:tensor.expertBytes))
                }
            }
        }
        var ssd: [TimedRead]=[], owned: [Int32]=[]
        defer { for fd in owned { close(fd) } }
        if let directory=ssdDirectory?.appendingPathComponent("kquant-v1"),
           let files=try? FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:nil) {
            for url in files.sorted(by:{ $0.path<$1.path }) where url.pathExtension=="expert" {
                let fd=open(url.path,O_RDONLY|O_NOFOLLOW)
                guard fd>=0 else { continue }
                var info=stat()
                guard fstat(fd,&info)==0, (info.st_mode&S_IFMT)==S_IFREG, info.st_size>0, info.st_size<=8*1024*1024 else { close(fd); continue }
                _=fcntl(fd,F_NOCACHE,1); owned.append(fd)
                ssd.append(TimedRead(fd:fd,offset:0,count:Int(info.st_size)))
                if ssd.count==6 { break }
            }
        }
        var rows: [(String,Int,Bool,[TimedRead])]=[]
        for workers in [1,2,4] { rows.append(("source",workers,false,source)); if !ssd.isEmpty { rows.append(("ssd",workers,false,ssd)) } }
        let dummy=UnsafeMutableRawPointer.allocate(byteCount:1,alignment:1); defer { dummy.deallocate() }
        let grouped=ParticleReadPlan.groups(source.map { ParticleReadRange(offset:$0.offset,count:$0.count,destination:dummy) })
            .map { TimedRead(fd:source[0].fd,offset:$0.offset,count:$0.count) }
        rows.append(("source",1,true,grouped))
        var timings=[[Double]](repeating:[],count:rows.count)
        for round in 0..<4 {
            for offset in rows.indices {
                let index=(offset+round)%rows.count, row=rows[index]
                let workers=min(row.1,row.3.count), errors=IOFailures(), start=ProcessInfo.processInfo.systemUptime
                DispatchQueue.concurrentPerform(iterations:workers) { worker in
                    let buffer=UnsafeMutableRawPointer.allocate(byteCount:8*1024*1024,alignment:64); defer { buffer.deallocate() }
                    for index in stride(from:worker,to:row.3.count,by:workers) {
                        let request=row.3[index]; var done=0
                        while done<request.count {
                            let bytes=pread(request.fd,buffer.advanced(by:done),request.count-done,off_t(request.offset+done))
                            if bytes<0 && errno==EINTR { continue }
                            guard bytes>0 else { errors.record(ParticleError.invalid("Benchmark read failed")); break }
                            done += bytes
                        }
                    }
                }
                if let error=errors.error { throw error }
                if round>0 { timings[index].append(ProcessInfo.processInfo.systemUptime-start) }
            }
        }
        return rows.indices.map { index in
            let row=rows[index], seconds=timings[index].sorted()[1], bytes=row.3.reduce(0) { $0+$1.count }
            return ParticleReadTiming(storage:row.0,workers:row.1,coalesced:row.2,bytes:bytes,medianSeconds:seconds,
                requestedMiBPerSecond:Double(bytes)/1048576/seconds,
                note:"Six synthetic demanded experts; one warmup, three rotating timed rounds. OS caches retained. SSD uses runtime F_NOCACHE policy. Excludes checksums, scatter copies, routing and GPU compute; not physical cold-device bandwidth.")
        }
    }
}
