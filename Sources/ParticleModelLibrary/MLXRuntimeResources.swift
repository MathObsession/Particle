import Foundation

public enum MLXRuntimeResources {
    /// Resolve shipped resources explicitly. Never compile shaders or search an SDK at runtime.
    public static func metallib(anchors: [URL] = []) throws -> URL {
        let roots = anchors+[Bundle.main.bundleURL]+[Bundle.main.executableURL,Bundle.main.resourceURL].compactMap{$0}+Bundle.allBundles.map(\.bundleURL)
        for root in roots {
            var directory = root
            for _ in 0..<7 {
                for suffix in ["mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                               "Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
                               "mlx-swift_Cmlx.bundle/default.metallib"] {
                    let url = directory.appendingPathComponent(suffix)
                    if FileManager.default.fileExists(atPath:url.path) {return url}
                }
                let parent = directory.deletingLastPathComponent();if parent == directory {break};directory = parent
            }
        }
        throw ModelLibraryError.invalid("The bundled MLX shader library is missing. Reinstall the complete Particle app.")
    }
}
