# Particle workspace setup

Updated 23 October 2026.

- Main Swift package: `/Users/lakshyaprajapati/Repositories/Particle`.
- Package name: Particle. Sarvam modules: `ParticleGGUF`, `ParticleCore`, and `ParticleCLI`. Upstream Gemma module names remain intact.
- Xcode: `/Volumes/Particle/Xcode.app` (27.1, build 27A9275). Compiler: Swift 6.4.
- Metal Toolchain: 27A266a, installed as an Apple-managed component. The exported installer is retained under `/Volumes/Particle/Caches/Particle/Downloads/`.
- Model: `/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf` (19,582,216,032 bytes). Particle reads it directly; no full copy or conversion is required.
- Official tokenizer sidecars: `/Volumes/Particle/Models/sarvam-30b/tokenizer/`, pinned to Hugging Face revision `071ae95e933605ca1104a6b4524a36a98488efa4` with checksums in `Scripts/fetch_sarvam_tokenizer.py`.
- Build and Derived Data: `/Volumes/Particle/Build/Particle/`.
- Dependency/compiler caches: `/Volumes/Particle/Caches/Particle/`.
- `Scripts/particle` and `Scripts/particle-env.sh` select Xcode locally without changing system-wide `xcode-select`.
- `Particle.code-workspace` configures the editor, terminal, and ParticleCLI build task for the same tools.
- Pinned dependency resolution and release compilation succeeded with the SD-card Xcode.
- The Metal shaders compile and pass independent CPU comparisons on this M2. See the [CLI guide](PARTICLE_CLI.md) for validation commands.
- The root package can be opened with `Scripts/particle xcode`.
- Current CLI defaults: 128-token batched prefill, bounded embedding rows, 12 RAM expert slots, and the approved exclusive SSD expert cache capped at 4096 MiB in `~/Library/Caches/Particle/Experts/`. Use `--no-ssd-cache` for original-volume-only expert reads. Fifteen focused tests pass; see [current performance](PARTICLE_PERFORMANCE.md).

The earlier Xcode launch failure applied to the prior installation. Historical setup notes and the original Derived Data preference backup are retained under `setup/`; they do not describe the active toolchain.

See the [implementation plan](IMPLEMENTATION_PLAN.md) for the Sarvam integration and optimization milestones.
