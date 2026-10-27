# Particle workspace

The main Swift package is this directory, `/Users/lakshyaprajapati/Repositories/Particle`. The package display name is Particle. Existing Swift target/module and product identifiers retain the upstream names until the model port; do not mass-rename implementation symbols as workspace housekeeping.

Use `Scripts/particle` for doctor, resolve, build, manifest, test, and xcode commands. The default toolchain is `/Volumes/Particle/Xcode.app/Contents/Developer`. Build outputs and compiler/dependency caches belong on the SD card. The user explicitly approved a separate, bounded internal SSD expert-data cache: `~/Library/Caches/Particle/Experts/`, 4096 MiB of payload by default. The user subsequently approved up to 18432 MiB for the new bounded SSD tier and a verified .particle model conversion. Keep the original GGUF intact. Legacy victim caching remains available; read-through and complete-mirror profiles are explicit choices. A complete mirror must replace the partial cache within one budget, never add a redundant full copy. Use `--no-ssd-cache` for SD-only expert-read controls. `PARTICLE_TOOLCHAIN=clt` explicitly selects installed Command Line Tools as a fallback.

The Sarvam model is `/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf`, exposed as `PARTICLE_MODEL_PATH`. It is a Q4_K_M GGUF containing Q4_K, Q5_K, Q6_K, and FP32 tensors. The legacy TurboFieldfare engine is Gemma-specific. ParticleCLI uses the separate ParticleGGUF/ParticleCore Sarvam path; never pass the Sarvam GGUF to the legacy loader.

Target: M2, 8 GB RAM, CLI-first Sarvam-focused fork. Temporary SD-card testing accepts 0.5–1 token/s; the current required internal-SSD target is at least 4 token/s sustained, subject to actual validation. The SD/hybrid target is at least 0.6 token/s, aiming for 0.9–1+. Keep a single real-model process, bounded memory, and no duplicate full model copies. For authorized Sarvam inference, check macOS/Swift, free disk, `memory_pressure -Q`, the installed GGUF and tokenizer, and that no other real-model process is running. Use the GGUF in place of the inherited Gemma-install precondition for ParticleCLI; the legacy Gemma tests retain their original model requirements. Run one real-model process at a time. ParticleCLI supports `--interactive` to keep weights, experts and verified KV prefixes warm; `/new` clears history without unloading. Sarvam defaults are prefill chunk 128 and 12 RAM expert slots. Current execution controls, full-context policy, experimental paths and benchmark commands are in `docs/PARTICLE_PREFILL_OPTIMIZATION.md`; the 23 October baseline is in `docs/PARTICLE_PERFORMANCE.md`; all 104 upstream experiment entries are mapped in `docs/PARTICLE_OPTIMIZATION_AUDIT.md`. Focused checks use `Scripts/particle test -c release --filter 'ParticleTests|OptimizationTests'`.

On 25 October 2026 the user explicitly accepted approximately 20–25% measured quality loss in exchange for a required 4 tokens/s target. Approximate `--active-experts 2...6` experiments are authorized; six remains the default. Preserve native weights and clearly report approximation, per-profile quality and speed, and KV reset requirements. A small multiple-choice or forced-decode pilot does not qualify the global quality/sustained-speed target. See `docs/PARTICLE_SPEED_QUALITY.md`.

ParticleOne is the separate Sarvam SwiftUI app in `Sources/ParticleOne`, using `ParticleOneCore` and the opt-in JSONL `ParticleDesktopProtocol`/`ParticleCLI --desktop-events`. Build/package with `Scripts/build_particleone.sh`; keep CLI defaults and legacy Gemma targets unchanged. Use the invoked ParticleDesign style, bundled PT Serif/Poppins/Space Mono, and `StoredState`. First-launch installation uses only the pinned public MathObsession/sarvam-30b-particle catalog after publication. The user explicitly approved that public model repository on 25 October 2026. Never bundle publisher tokens, local receipts/progress or logs; public downloads need no authentication. Model upload is independent of model execution; do not start inference during the large upload just to test UI. Model-free UI smoke checks use isolated temporary state and an explicit mock engine. See `docs/particleone/README.md`.

Standalone runtime setup (26 October 2026): the installed app is `~/Applications/ParticleOne.app`; its permanent native model is `~/Library/Application Support/ParticleOne/Models/sarvam-30b-q4_k_m.particle`. Both existing workspaces use that path with the SSD mirror disabled. The former internal mirror payload was promoted into this permanent model without a second payload allocation; `mirror-v1` no longer exists. Do not automatically recreate a redundant full mirror or remove this permanent model as cache housekeeping. The original SD GGUF/.particle pack and development tools remain intact. The installed engine successfully generated a real reply with the SD volume unmounted, no developer paths and its bundled shaders. Development scripts still require the SD toolchain; the installed app does not. See `docs/particleone/standalone/README.md`.

Particle 0.3.0 (26 October 2026): the product is now Particle, installed at `~/Applications/Particle.app`; the permanent model was renamed in place to `~/Library/Application Support/ParticleOne/Models/churo-chat.particle`. Both existing workspaces use it; weights and upstream format identifiers are unchanged. Churo Chat / Churo Code are assistant identities with hidden built-in prompts, not new quantizations. Particle Code is a native project/terminal/web agent using the same single warm worker. Source targets/data directory/update bundle ID retain ParticleOne compatibility names. Use the signed local release pipeline; never start a second user model or replace an active bundle. Direct file tools are scoped/approved; shell execution is explicitly approved and not an OS sandbox. See `docs/particleone/code/README.md` and `docs/particleone/validation-0.3.0/README.md`. Focused tests include AgentHarnessTests; desktop mock fixtures require isolated /private/tmp/particleone-smoke-* state; real harness tests require /private/tmp/particle-agent-native-* and no mock engine. Preserve the user's saved expert preference; six is recommended/validated for tool reliability but no quality/speed promise is made.

The following inherited instructions govern existing Gemma behavior and tests; use them alongside the user's authorized Sarvam work.

---

# TurboFieldfare

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/TurboFieldfareFormat/` owns the Foundation-only `.gturbo` v1 wire
contract. `Sources/TurboFieldfare/` is the runtime; `Sources/TurboFieldfareRepack/`,
`Sources/TurboFieldfareCLI/`, `Sources/TurboFieldfareServer/`, and
`Sources/TurboFieldfareApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

```bash
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo --resume
swift build -c release
.build/release/TurboFieldfareMac
Scripts/build_app.sh
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output scratch/gemma4.gturbo`.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## SwiftUI state and previews

Use `@StoredState` from `TurboFieldfareMacPresentation` for view-owned state.
It aliases Apple's `SwiftUI.State<Value>` property wrapper, avoiding the State
macro plugin missing from some CLT installations. Keep reference construction
explicit; the wrapper evaluates inline initializers eagerly. Use
`PreviewProvider` for debug previews. Do not introduce direct State attributes
or Preview macros. Run `ruby Scripts/check_swiftui_macros.rb --self-test` and
`ruby Scripts/tests/swift_macro_probe_test.rb` when changing these surfaces.

## Test rules

Before a model run, require macOS 26+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process from `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For isolated app checks, set `TURBO_FIELDFARE_APP_DATA_PATH` to a temporary
directory so settings and history stay separate. `TURBO_FIELDFARE_MODEL_PATH`
overrides the saved model location and disables location changes in the UI.
See [Build a Mac app](UPSTREAM_README.md#build-a-mac-app) for bundle instructions.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through the pinned Gemma 4 IT chat format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.2`, Top-K `64`, and Top-P `0.95`.
The app saves conversation history locally and retains one live FP16 KV lineage.
Browsing the sidebar preserves that lineage; continuing another saved chat
replays its recorded tokens and replaces it. See
[Conversation history](UPSTREAM_README.md#conversation-history) for an overview.
The HUD shows generation rate, context use, decode-service memory, and,
on hover, cached-token reuse; Last run also shows time to first token and I/O.
Use **New Chat** to clear the transcript, KV lineage, gauge, and retained
images. Reloading or unloading keeps the transcript but marks it outside the
new model context. Build the app with its sibling
`TurboFieldfareDecodeService`; it never loads a second in-process model. See
[README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).

## Images

Image support is an optional `<name>.vision.gturbo` companion pack that sits
beside the text model. Without it the text runtime behaves exactly as before.

```bash
swift run -c release TurboFieldfareRepack \
  --vision-output scratch/gemma4.vision.gturbo \
  --text-model scratch/gemma4.gturbo
```

The model-free suites cover preprocessing, the tower kernels, the companion
format, the installer transaction, prompt rendering, and the server ingress.
Cases that need a real installed pack skip themselves when
`scratch/gemma4.gturbo` and its companion are missing, so a checkout without a
model still runs green.

Keep the image path fail-closed. If the pack is missing or invalid, say image
support is unavailable. Never accept an image and then answer as though it had
not been sent.
