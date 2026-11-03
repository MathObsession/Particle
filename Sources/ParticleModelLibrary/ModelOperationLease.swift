import Foundation
import Darwin

/// Shares the existing Particle inference lease without depending on a GPU backend.
/// Metadata inspection takes no lease; conversion must not compete with a model owner.
public final class ModelOperationLease {
    private let descriptor: Int32
    public init() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("particle-model-\(getuid()).lock").path
        let fd = open(path,O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0o600)
        guard fd >= 0 else {throw ModelLibraryError.invalid("Cannot open the Particle model-operation lock.")}
        var st = stat()
        guard fstat(fd,&st) == 0,st.st_mode & S_IFMT == S_IFREG,st.st_uid == getuid(),flock(fd,LOCK_EX|LOCK_NB) == 0 else {
            close(fd);throw ModelLibraryError.invalid("Another Particle model or conversion is active. Finish it before starting this operation.")
        }
        descriptor = fd
    }
    deinit {flock(descriptor,LOCK_UN);close(descriptor)}
}
