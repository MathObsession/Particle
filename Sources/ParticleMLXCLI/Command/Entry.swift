import ParticleMLXRuntime

@MainActor @main struct ParticleMLXCLI {
    static func main() async {await ParticleMLXWorker.main()}
}
