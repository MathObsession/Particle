import Foundation
import ParticleOneCore

/// One bounded, resumable file transfer. URLSession streams directly into a .part
/// file; neither a full expert layer nor the complete model is held in memory.
final class ReleaseTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let destination: URL
    private let expected: Int64
    private let progress: @Sendable (Int64) -> Void
    private var handle: FileHandle?
    private var received: Int64=0
    private var lastProgressTime=0.0
    private var result: CheckedContinuation<URL,Error>?
    private var failure: Error?
    private let lock=NSLock()
    private var task: URLSessionDataTask?
    private var cancelled=false
    private var session: URLSession?
    init(destination: URL,expected: Int64,progress: @escaping @Sendable (Int64)->Void) {
        self.destination=destination;self.expected=expected;self.progress=progress
    }
    func start(_ url: URL) async throws -> URL {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                do {
                    if FileManager.default.fileExists(atPath:destination.path) {
                        guard !(try destination.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink ?? false) else { throw ReleaseError.invalidManifest }
                    } else { guard FileManager.default.createFile(atPath:destination.path,contents:nil) else { throw CocoaError(.fileWriteUnknown) } }
                    handle=try FileHandle(forWritingTo:destination);received=Int64(try handle!.seekToEnd())
                    guard received<=expected else { throw ReleaseError.invalidManifest }
                    if received==expected { try handle?.close();handle=nil;continuation.resume(returning:destination);return }
                    var request=URLRequest(url:url);request.timeoutInterval=60
                    if received>0 { request.setValue("bytes=\(received)-",forHTTPHeaderField:"Range") }
                    let configuration=URLSessionConfiguration.ephemeral;configuration.timeoutIntervalForResource=14_400
                    let queue=OperationQueue();queue.maxConcurrentOperationCount=1
                    session=URLSession(configuration:configuration,delegate:self,delegateQueue:queue)
                    lock.lock();result=continuation;let next=session!.dataTask(with:request);task=next;let stop=cancelled;lock.unlock()
                    next.resume();if stop { next.cancel() }
                } catch { try? handle?.close();handle=nil;continuation.resume(throwing:error) }
            }
        } onCancel: { self.lock.lock();self.cancelled=true;let current=self.task;self.lock.unlock();current?.cancel() }
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive response: URLResponse,completionHandler: @escaping @Sendable (URLSession.ResponseDisposition)->Void) {
        do {
            guard let response=response as? HTTPURLResponse,[200,206].contains(response.statusCode) else { throw ReleaseError.network((response as? HTTPURLResponse)?.statusCode ?? 0) }
            if response.statusCode==206 {
                guard let range=response.value(forHTTPHeaderField:"Content-Range"),range.hasPrefix("bytes \(received)-"),range.hasSuffix("/\(expected)") else { throw ReleaseError.invalidManifest }
            } else if received>0 { try handle?.truncate(atOffset:0);try handle?.seek(toOffset:0);received=0 }
            guard response.expectedContentLength < 0 || response.expectedContentLength <= expected-received else { throw ReleaseError.invalidManifest }
            completionHandler(.allow)
        } catch { failure=error;completionHandler(.cancel) }
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive data: Data) {
        do { guard Int64(data.count)<=expected-received else { throw ReleaseError.invalidManifest };try handle?.write(contentsOf:data);received+=Int64(data.count);let now=ProcessInfo.processInfo.systemUptime;if now-lastProgressTime>=0.1 || received==expected {lastProgressTime=now;progress(received)} }
        catch { failure=error;dataTask.cancel() }
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,didCompleteWithError error: Error?) {
        try? handle?.close();handle=nil
        lock.lock();let continuation=result;result=nil;lock.unlock()
        if let failure=failure ?? error { continuation?.resume(throwing:failure) }
        else if received != expected { continuation?.resume(throwing:ReleaseError.invalidManifest) }
        else { continuation?.resume(returning:destination) }
        session.finishTasksAndInvalidate();self.session=nil
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,newRequest request: URLRequest,completionHandler: @escaping @Sendable (URLRequest?)->Void) {
        guard request.url?.scheme=="https" else { completionHandler(nil);return };var safe=request;safe.setValue(nil,forHTTPHeaderField:"Authorization");completionHandler(safe)
    }
}

@MainActor final class ModelInstaller: ObservableObject {
    @Published var running=false
    @Published var progress=0.0
    @Published var status="Checking your model…"
    @Published var error: String?
    private var work: Task<Void,Never>?
    func pause() { work?.cancel() }
    func start(root: URL,onComplete: @escaping @MainActor (URL)->Void) {
        guard !running else { return };running=true;error=nil
        work=Task { [self] in
            do {
                guard let catalogURL=Bundle.module.url(forResource:"model-catalog",withExtension:"json",subdirectory:"Resources") else { throw ReleaseError.notPublished }
                let catalog=try JSONDecoder().decode(ModelCatalog.self,from:Data(contentsOf:catalogURL));try catalog.validate()
                let base=URL(string:"https://huggingface.co/\(catalog.repo)/resolve/\(catalog.revision)/")!
                status="Reading the verified release…"
                let (data,response)=try await URLSession.shared.data(from:base.appendingPathComponent("particle-distribution.json"))
                guard (response as? HTTPURLResponse)?.statusCode==200,data.count<1_048_576,ModelRelease.digest(data)==catalog.distributionSHA256 else { throw ReleaseError.invalidManifest }
                let release=try JSONDecoder().decode(ModelRelease.self,from:data);try release.validate(catalog:catalog)
                try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
                let partial=root.appendingPathComponent(catalog.folder+".installing",isDirectory:true),final=root.appendingPathComponent(catalog.folder,isDirectory:true)
                guard !FileManager.default.fileExists(atPath:final.path) else { throw CocoaError(.fileWriteFileExists) }
                try FileManager.default.createDirectory(at:partial,withIntermediateDirectories:true)
                guard !(try partial.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink ?? false) else { throw ReleaseError.invalidManifest }
                var existing:Int64=0
                for file in release.files {
                    for suffix in ["",".part"] {
                        let f=partial.appendingPathComponent(file.path+suffix)
                        if let n=(try? FileManager.default.attributesOfItem(atPath:f.path)[.size]) as? NSNumber { existing+=min(n.int64Value,file.bytes) }
                    }
                }
                let free=(try root.resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey])).volumeAvailableCapacityForImportantUsage ?? 0
                guard free>=max(0,release.totalBytes-existing)+1_073_741_824 else { throw ReleaseError.storage }
                var completed:Int64=0
                for (index,file) in release.files.enumerated() {
                    try Task.checkCancellation()
                    let destination=partial.appendingPathComponent(file.path),piece=partial.appendingPathComponent(file.path+".part")
                    var parent=destination.deletingLastPathComponent()
                    while parent.path.hasPrefix(partial.path),parent.path != root.path {
                        if FileManager.default.fileExists(atPath:parent.path), (try parent.resourceValues(forKeys:[.isSymbolicLinkKey])).isSymbolicLink==true { throw ReleaseError.invalidManifest }
                        if parent==partial { break };parent.deleteLastPathComponent()
                    }
                    try FileManager.default.createDirectory(at:destination.deletingLastPathComponent(),withIntermediateDirectories:true)
                    status="File \(index+1) of \(release.files.count) · \(file.path)"
                    if FileManager.default.fileExists(atPath:destination.path) {
                        guard !(try destination.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink ?? false) else { throw ReleaseError.invalidManifest }
                        let matches=try await Task.detached { try ModelRelease.digestFile(destination)==file.sha256 }.value
                        if matches { completed+=file.bytes;progress=Double(completed)/Double(release.totalBytes);continue }
                        try FileManager.default.removeItem(at:destination)
                    }
                    let baseCompleted=completed,total=release.totalBytes
                    let transfer=ReleaseTransfer(destination:piece,expected:file.bytes) { bytes in
                        Task { @MainActor [weak self] in self?.progress=Double(baseCompleted+bytes)/Double(total) }
                    }
                    let fetched=try await transfer.start(base.appendingPathComponent(catalog.folder).appendingPathComponent(file.path))
                    status="Verifying \(file.path)…"
                    let hash=try await Task.detached { try ModelRelease.digestFile(fetched) }.value
                    try Task.checkCancellation()
                    guard hash==file.sha256 else { try FileManager.default.removeItem(at:fetched);throw ReleaseError.checksum(file.path) }
                    try FileManager.default.moveItem(at:fetched,to:destination);completed+=file.bytes
                }
                guard try ModelRelease.digestFile(partial.appendingPathComponent("manifest.json"))==catalog.modelManifestSHA256 else { throw ReleaseError.invalidManifest }
                let receipt=try JSONSerialization.data(withJSONObject:["complete":true,"manifestSHA256":catalog.modelManifestSHA256],options:[.sortedKeys])
                try receipt.write(to:partial.appendingPathComponent("verified-install.json"),options:.atomic)
                try FileManager.default.moveItem(at:partial,to:final);progress=1;status="Your model is ready.";running=false;onComplete(final)
            } catch is CancellationError { status="Download paused. Your progress is saved.";running=false }
            catch { self.error=(error as? URLError)?.code == .cancelled ? nil : error.localizedDescription;status=self.error == nil ? "Download paused. Your progress is saved." : "Download needs attention.";running=false }
        }
    }
}
