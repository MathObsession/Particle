import Foundation
import Dispatch

/// Conversion has a small working set, but must yield its disk/memory budget
/// when macOS reports critical pressure. Completed ranges remain resumable.
public final class ModelMemoryPressure: @unchecked Sendable {
    private let lock = NSLock()
    private var critical = false
    private let source: DispatchSourceMemoryPressure
    public init() {
        source = DispatchSource.makeMemoryPressureSource(eventMask:[.normal,.warning,.critical],queue:.global(qos:.utility))
        source.setEventHandler { [weak self] in
            guard let self else{return}
            let events = self.source.data
            self.lock.withLock {self.critical = events.contains(.critical)}
        }
        source.resume()
    }
    deinit {source.cancel()}
    public func check() throws {
        guard !lock.withLock({critical}) else {throw ModelLibraryError.memoryPressure}
    }
}
