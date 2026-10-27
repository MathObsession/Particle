import Foundation
import Darwin
import ParticleGGUF

/// A per-user advisory lease prevents two Particle model owners from filling
/// expert caches simultaneously. Inspection and tokenization do not take it.
public final class ParticleModelLease {
    private let descriptor: Int32
    public init() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("particle-model-\(getuid()).lock").path
        let fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw ParticleError.invalid("Cannot open the Particle inference lock") }
        var st = stat()
        guard fstat(fd,&st) == 0, st.st_mode & S_IFMT == S_IFREG, st.st_uid == getuid(), flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd); throw ParticleError.invalid("Another Particle model process is running, or its lock is unavailable")
        }
        descriptor = fd
    }
    deinit { flock(descriptor,LOCK_UN); close(descriptor) }
}
