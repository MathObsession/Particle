# Particle

Particle is a native, local AI workspace for Apple Silicon Macs, created by Lakshya Prajapati. **Churo Chat** handles conversations; **Particle Code**, powered by **Churo Code**, reads and edits projects, runs approved commands, and searches public documentation.

The SwiftUI app uses a green, grain-textured ParticleDesign interface with PT Serif, Poppins, and Space Mono. It stores conversations and memories locally, supports multiple workspaces, and shares one warm inference worker between chat and coding.

- [Desktop setup and usage](docs/particleone/README.md)
- [Coding agent capabilities and permissions](docs/particleone/code/README.md)
- [Building from source](docs/BUILDING_PARTICLE.md)

The underlying engine is a TurboFieldfare fork adapted to the Sarvam 30B base model, with Swift/Metal inference and bounded MoE expert streaming from GGUF or native `.particle` packs. Churo names describe the assistant experience, not newly trained weights. Original licenses and technical attribution are retained. This source repository contains no model weights, credentials, chats, or build archives.

## Build and run

```sh
Scripts/particle doctor
Scripts/particle build -c release --product ParticleCLI
Scripts/particle run --interactive --max-new 64 --max-context 512
```

The model can emit reasoning before its answer; a small output limit may end during reasoning. Generated text goes to stdout. Progress, timing, cache activity, and sampled memory use go to stderr. Add `--verbose` for per-layer progress on slow storage; Control-C cancels at the next layer boundary after current work completes.

```sh
Scripts/particle run --inspect
Scripts/particle run --tokenize --prompt "नमस्ते"
Scripts/particle run --validate-kernels
Scripts/particle test -c release --filter 'ParticleTests|OptimizationTests'
```

See the [CLI guide](docs/PARTICLE_CLI.md) for input files, sampling, tokenizer setup, and limits, and the [implementation plan](docs/IMPLEMENTATION_PLAN.md) for completed modules and remaining optimization work.

The current optimization pass adds actual batched native K-quant GPU work, 128-token prompt chunks, fused gate/up activation, the approved bounded internal SSD expert cache, and warm multi-turn sessions with exact KV-prefix reuse. Use `/new` to clear history while retaining warm weights, and `/quit` to exit. `--no-ssd-cache` keeps expert reads on the original model volume.

The final short eight-token checks measured 0.42 tokens/s with a partially warm cache and 0.72 tokens/s on the warm repeat. New routes and cold prompt processing can still be slow on SD.

See the [current performance record](docs/PARTICLE_PERFORMANCE.md) for tests, measured throughput and limits, and the [complete method audit](docs/PARTICLE_OPTIMIZATION_AUDIT.md) for all 104 upstream experiment entries, including rejected methods and unfinished Sarvam ports. Faster isolated kernels do not establish a model-wide token rate.

## Particle desktop app

The new [Particle desktop guide](docs/particleone/README.md) covers the native SwiftUI chat interface, local conversations, model setup, green ParticleDesign artwork, bundled PT Serif/Poppins/Space Mono fonts and Homebrew cask packaging. Build with `Scripts/build_particleone.sh`. The separate `ParticleOne` target connects to the Sarvam `ParticleCLI` worker; the inherited Gemma app remains separate. Public model-release status is in `docs/particleone/model-release/status.json`.

## Native packs and SSD storage

The [native pack guide](docs/PARTICLE_PACK.md) covers lossless `.particle` conversion, resumable installation, read-through SSD caching, and a complete common/expert SSD mirror within the approved 18 GiB tier. Defaults preserve legacy behavior. The native SD pack is verified and the complete SSD mirror is prepared. A short, alternating reference/candidate benchmark measured **1.92 → 2.61 tokens/s** with matching full logits and continuations; [results and limits](docs/validation/particle-hybrid-decode/20261024T134049Z/README.md). [Validation logs](docs/validation/particle-pack-20261024/README.md) distinguish completed checks from pending sustained workload qualification.

Experimental `--deferred-decode`, `--gpu-routing`, and `--grouped-experts` reduce decode scheduling/dispatch work. They remain opt-in pending whole-model latency gates. Native-six has not qualified the sustained 4 tokens/s internal-SSD goal.

The new [speed/quality experiment guide](docs/PARTICLE_SPEED_QUALITY.md) documents explicit `--active-experts 2...6` routing budgets and a single-runner screening tool. Reduced counts change the model and remain experimental. The final three-expert candidate completed two repeated code responses at **4.20 and 4.30 tokens/s**; an English chat response was slower at **3.71 tokens/s** and hit its reasoning cap. The small quality screen scored 14/20 versus native 15/20. These results do **not** certify 4 tokens/s across workloads or a global 25% quality-loss ceiling. [Complete results and limits](docs/validation/particle-speed-quality-20261025/README.md).

## Choose RAM usage

Set `--expert-cache-slots` to any integer from **6 to 16** when starting Particle. The default remains **12 slots per MoE layer**. More slots may reduce disk reads; the speed benefit depends on the routes used and available RAM.

```sh
# Smaller RAM cache
Scripts/particle run --interactive --expert-cache-slots 6

# Default balance
Scripts/particle run --interactive --expert-cache-slots 12

# Larger RAM cache; compare speed and memory pressure
Scripts/particle run --interactive --expert-cache-slots 16
```

Expert buffers across the 18 MoE layers use approximately **0.76 GiB at 6 slots**, **1.01 GiB at 8**, **1.52 GiB at 12**, or **2.03 GiB at 16**. These are buffer capacities for the installed model, excluding common weights, KV, scratch, file caches and the rest of the system. Choose the setting before launch; changing it requires restarting Particle. Conversation context and the separate SSD payload budget (4096 MiB default, up to 18432 MiB explicitly selected) are independent of this setting.

## Workspace

Main folder: `/Users/lakshyaprajapati/Repositories/Particle`. Open `Package.swift` in Xcode or `Particle.code-workspace` in VS Code. `Sources/`, `Tests/`, `Scripts/`, and `docs/` are at the root.

```text
/Volumes/Particle/
  Xcode.app
  Models/sarvam-30b/sarvam-30b-q4_k_m.gguf
  Models/sarvam-30b/tokenizer/
  Build/Particle/SwiftPM/
  Build/Particle/DerivedData/
  Caches/Particle/
```

`Scripts/particle` selects the SD-card Xcode, build directory, and compiler caches. `source Scripts/particle-env.sh` applies those settings to a shell. The helper limits builds to two jobs. The ignored `.build` link directs ordinary SwiftPM builds to the card. Keep it mounted as `Particle`. The model is used in place; no full copy or conversion is needed. An internal SSD cache of recent RAM expert evictions is enabled by default, capped at 4096 MiB of payload. This increases effective coverage without increasing the RAM slot count. It is separate from the SD compiler caches.

## Current scope

The Sarvam path consists of `ParticleFormat`, `ParticleGGUF`, `ParticlePackCore`, `ParticlePack`, `ParticleCore`, and `ParticleCLI`. It supports the installed 19-layer, 128-expert/top-6 Sarvam artifact, its official tokenizer/chat template, and direct Q4_K/Q5_K/Q6_K operations. Prompt processing groups expert consumers over 128-token chunks by default with batched GPU projections; activations and KV cache use FP32. Embeddings use bounded row reads, and the expert cache defaults to 12 slots per layer. The scalar path remains available for validation.

Target hardware is M2 with 8 GB RAM. The temporary SD card can make cold reads very slow. The 6–8 token/s internal-SSD goal requires measurements and optimization; it is not a current performance claim. Use small contexts on this Mac. The revised 12-slot default was tested with context 64; six slots remain available when memory is tight.

The inherited `TurboFieldfare*` app, server, CLI, and repacker remain Gemma-specific. Use **ParticleCLI** for Sarvam. Preserve the upstream licenses and [third-party notices](THIRD_PARTY_NOTICES.md).

See [workspace setup](docs/WORKSPACE_SETUP.md), [codebase analysis](docs/CODEBASE_ANALYSIS.md), and [upstream documentation](UPSTREAM_README.md). Consult `AGENTS.md` before model-using tests or inference.
