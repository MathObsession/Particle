import Foundation

final class PublicHubRequestRouter: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var requests = [Int:BoundedHubRequest]()
    func register(_ request: BoundedHubRequest, task: URLSessionTask) {lock.withLock{requests[task.taskIdentifier] = request}}
    private func lookup(_ task: URLSessionTask) -> BoundedHubRequest? {lock.withLock{requests[task.taskIdentifier]}}
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive response: URLResponse,completionHandler: @escaping @Sendable (URLSession.ResponseDisposition)->Void) {
        guard let request = lookup(dataTask) else {completionHandler(.cancel);return}
        request.urlSession(session,dataTask:dataTask,didReceive:response,completionHandler:completionHandler)
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive data: Data) {lookup(dataTask)?.urlSession(session,dataTask:dataTask,didReceive:data)}
    func urlSession(_ session: URLSession,task: URLSessionTask,didCompleteWithError error: Error?) {
        let request = lock.withLock{requests.removeValue(forKey:task.taskIdentifier)}
        request?.urlSession(session,task:task,didCompleteWithError:error)
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,newRequest request: URLRequest,completionHandler: @escaping @Sendable (URLRequest?)->Void) {
        guard let pending = lookup(task) else {completionHandler(nil);return}
        pending.urlSession(session,task:task,willPerformHTTPRedirection:response,newRequest:request,completionHandler:completionHandler)
    }
}

/// URLSession delivers blocks directly to a bounded accumulator. Iterating bytes
/// individually makes large range-based conversions unnecessarily expensive.
final class BoundedHubRequest: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    static func retryDate(_ value:String?,now:Date=Date())->Date {
        if let value,let delay=Double(value),delay.isFinite,delay >= 0 {return now.addingTimeInterval(max(1,delay))}
        let formatter=DateFormatter();formatter.locale=Locale(identifier:"en_US_POSIX");formatter.timeZone=TimeZone(secondsFromGMT:0);formatter.dateFormat="EEE, dd MMM yyyy HH:mm:ss zzz"
        if let value,let date=formatter.date(from:value) {return max(now.addingTimeInterval(1),date)}
        return now.addingTimeInterval(300)
    }
    let request: URLRequest
    let limit: Int
    let range: Range<Int64>?
    private var data = Data()
    private var response: HTTPURLResponse?
    private var failure: Error?
    private let lock = NSLock()
    private var continuation: CheckedContinuation<(Data,HTTPURLResponse),Error>?
    private var task: URLSessionDataTask?
    private var cancelled = false
    init(request: URLRequest, limit: Int, range: Range<Int64>?) {self.request = request;self.limit = limit;self.range = range}
    func start(session: URLSession, router: PublicHubRequestRouter) async throws -> (Data,HTTPURLResponse) {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { result in
                let next = session.dataTask(with:request)
                let stop = lock.withLock {continuation = result;task = next;return cancelled}
                router.register(self,task:next)
                next.resume();if stop {next.cancel()}
            }
        } onCancel: {
            let current = self.lock.withLock {self.cancelled = true;return self.task};current?.cancel()
        }
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition)->Void) {
        do {
            guard let http = response as? HTTPURLResponse else {throw ModelLibraryError.invalid("Missing Hub response.")}
            if http.statusCode == 429 {
                throw ModelLibraryError.rateLimited(Self.retryDate(http.value(forHTTPHeaderField:"Retry-After")))
            }
            guard http.statusCode == (range == nil ? 200 : 206) else {
                if [401,403].contains(http.statusCode) {throw ModelLibraryError.incompatible("Only public, ungated models are supported. No credentials were sent.")}
                throw ModelLibraryError.network(http.statusCode)
            }
            if let range {
                guard let content = http.value(forHTTPHeaderField:"Content-Range"),content.hasPrefix("bytes \(range.lowerBound)-\(range.upperBound-1)/") else {throw ModelLibraryError.invalid("The Hub did not honor the requested byte range.")}
            }
            guard http.expectedContentLength < 0 || http.expectedContentLength <= limit else {throw ModelLibraryError.invalid("Hub response exceeded its expected size.")}
            self.response = http;completionHandler(.allow)
        } catch {failure = error;completionHandler(.cancel)}
    }
    func urlSession(_ session: URLSession,dataTask: URLSessionDataTask,didReceive piece: Data) {
        guard piece.count <= limit-data.count else {failure = ModelLibraryError.invalid("Hub response exceeded its safety limit.");dataTask.cancel();return}
        data.append(piece)
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,didCompleteWithError error: Error?) {
        let result = lock.withLock {let value = continuation;continuation = nil;return value}
        if let problem = failure ?? error {result?.resume(throwing:problem)}
        else if let response,range.map({$0.count == data.count}) ?? true {result?.resume(returning:(data,response))}
        else {result?.resume(throwing:ModelLibraryError.invalid("The download was truncated."))}
        lock.withLock{self.task = nil}
    }
    func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,completionHandler: @escaping @Sendable (URLRequest?)->Void) {
        guard let url = request.url,PublicHubHTTP.allowed(url) else {completionHandler(nil);return}
        var safe = request;safe.setValue(nil,forHTTPHeaderField:"Authorization");completionHandler(safe)
    }
}
