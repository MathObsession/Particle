# Native .particle packs and SSD storage

Implementation started 24 October 2026. The native converter and loader preserve the installed Sarvam Q4_K_M tensor bytes, including Q4_K, Q5_K, Q6_K and FP32. A `.particle` is a self-contained directory; renaming a GGUF does not create one. Original tokenizer assets are copied unchanged. This implementation does not yet establish the 4 tokens/s SSD target or introduce a lower-bit model derivative.

## Convert the installed model

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/particle build -c release --product ParticlePack
Scripts/particle build -c release --product ParticleCLI
Scripts/particle pack --source /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf --plan
Scripts/particle pack \
  --source /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf \
  --output /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle
```

If interrupted, repeat the conversion command with `--resume`. Source identity and committed destination ranges must still match. The converter refuses to overwrite a completed pack. It uses a source-ordered pass with 8 MiB of payload scratch, durable checkpoints every 64 MiB, contiguous-first preallocation, checksummed indexes and expert blobs, and atomic final publication. Preallocation/checkpoint flushing can stall for minutes on the SD card; provisioning is separate from inference latency. The GGUF remains intact.

The installed model's exact planned payload is 19,566,018,560 bytes: common weights 1,547,288,576, embeddings 603,979,776, and experts 17,414,750,208. Tokenizer/index files are additional. Expert gate/up/down projections share one contiguous aligned blob, allowing one read per RAM miss rather than three scattered reads. The quantization itself is unchanged.

```sh
Scripts/particle verify /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle
Scripts/particle run --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle --inspect
```

## Storage choices

Defaults retain legacy GGUF/victim-cache behavior. Explicit profiles:

| Profile | Model placement | Internal SSD tier |
| --- | --- | --- |
| `legacy` | Supplied model | Existing flags; victim policy by default |
| `sd-only` | Supplied SD model | Disabled |
| `sd-hybrid` | SD pack | Bounded victim, read-through, or prepared mirror |
| `ssd` | Requires model on the Mac's internal SSD | Disabled to avoid a redundant tier |

`--ssd-cache-mib` accepts 1–18432 MiB and defaults to 4096. This is disk payload capacity. RAM remains independently controlled by `--expert-cache-slots 6...16`, default 12 per MoE layer. Twelve native slots use about 1.52 GiB across the MoE layers, in addition to common mappings, KV, scratch and system memory.

`read-through` admits demanded experts and retains valid SSD hits instead of deleting and rewriting them during RAM exchange. It uses probation/protected LRU eviction with up to 80% protected capacity. The legacy exclusive victim policy remains selectable. Bulk I/O and checksum work run outside the metadata lock. Reservations, in-flight pins, and a cache-wide exclusive lease protect the payload budget.

A complete native mirror requires **18,962,038,784 bytes**, approximately 17.66 GiB, fitting the explicitly approved 18 GiB maximum. It contains common weights and all routed experts. Embeddings remain row-readable on SD. Preparation replaces the old expert-cache namespace within the one SSD budget; it does not allocate the mirrored model in RAM.

```sh
Scripts/particle prepare-cache \
  --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle \
  --storage-profile sd-hybrid --ssd-cache-policy mirror --ssd-cache-mib 18432

Scripts/particle run \
  --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle \
  --storage-profile sd-hybrid --ssd-cache-policy mirror --ssd-cache-mib 18432 \
  --interactive --expert-cache-slots 12
```

Mirror preparation verifies completed files on restart, copies incomplete files again with a 4 MiB buffer, checks SHA-256, and publishes a receipt bound to the pack identity. Every SSD expert load is verified. A damaged expert falls back to the original SD pack and reports degraded storage. A damaged common-weight mirror fails at startup with a repair instruction. Re-run preparation after closing the active model to repair the tier. Mirror and victim/read-through pools cannot operate concurrently in the same cache directory.

Complete-mirror performance must be described as SSD-backed hybrid inference. Source-expert counters exclude embedding-row reads and mapped common-weight page faults. A fresh process is not proof of physically cold media.

## Validation and opt-in decode work

```sh
Scripts/particle run \
  --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle \
  --compare-model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf \
  --no-ssd-cache --raw --prompt 'The capital of France is' --max-new 1 --max-context 64
```

This compares every vocabulary logit at the prompt boundary and two continuation steps, plus greedy token IDs. It creates one runner at a time and retains only the small reference vectors. Packing/storage changes require zero bitwise mismatches. It is a correctness diagnostic, not a throughput benchmark.

`--deferred-decode` queues embedding, leading dense work and ready experts without unnecessary CPU waits, combines fetched experts with the ordered residual submission, and carries that submission into the next layer’s required routing wait. All dependent GPU commands remain on one queue; completion checks every tracked command, and failure/cancellation drains outstanding consumers before buffer reuse. The CPU still waits for actual router/read dependencies.

`--gpu-routing` computes deterministic top-six selection on the GPU and transfers six IDs/weights. Selection-only bias and unbiased sigmoid mixture weights remain intact. GPU transcendental arithmetic requires full-model numerical validation; the option remains experimental. `--grouped-experts` evaluates a selected batch of up to six experts with two dispatches and separate 24 KiB total gate scratch, preserving the six rank outputs and their ordered reduction. Gate/up/down native encodings are explicit. All three options default off. Existing `--attention` and `--projection` reference/candidate selectors are retained. `--metrics-json <path>` records storage, cache, reuse and latency metrics without the attention-profiling command boundaries used by `--timings-json`.

Focused checks:

```sh
Scripts/particle test -c release --filter 'ParticleTests|ParticlePackTests|OptimizationTests|AttentionOptimizationTests|ProjectionOptimizationTests'
```

Current evidence and conversion/build logs are recorded under [the pack validation directory](validation/particle-pack-20261024/README.md). Full real-model parity, SSD provisioning and sustained workload qualification are tracked there. Whole-model promotion of grouped experts, verified multi-token generation, calibrated Q3/Q2 derivatives and the remainder of [the performance plan](PARTICLE_FORMAT_AND_SPEED_PLAN.md) are subsequent work, not shipped capabilities of the native pack.

## Compare the next decode candidates

After closing the active chat with `/quit`, run:

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/particle-hybrid-check
```

This uses the existing prepared mirror. It first verifies scalar/candidate full-logit and continuation parity, then alternates reference and vector/deferred/grouped-expert execution over three pairs with 64 decode forwards per pass. It records JSON results and progress under `docs/validation/particle-hybrid-decode/`. The short repeated history is diagnostic; it does not qualify natural-answer speed, long-context performance or a default promotion. The script never stops another model process or rebuilds/replaces the active model.

## Current measured decode option

The native pack and complete SSD mirror are installed on the development M2/8 GB Mac. The [short-history comparison](validation/particle-hybrid-decode/20261024T134049Z/README.md) measured median **1.918 → 2.613 tokens/s**, with all full-logit and continuation checks passing. Candidate passes ranged from 2.60 to 2.64 tokens/s. This is an experimental bundle, not a sustained 4 tokens/s qualification or a default promotion.

To try that bundle in a warm interactive session:

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/particle run \
  --model /Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle \
  --storage-profile sd-hybrid --ssd-cache-policy mirror --ssd-cache-mib 18432 \
  --expert-cache-slots 12 --ssd-read-workers 2 \
  --attention reference --projection vector --deferred-decode --grouped-experts \
  --interactive --max-new 256 --max-context 2048
```

The 16-slot/four-worker follow-up had substantial run-to-run variation and increased system swap; it is not selected as an improvement. Retain 12 slots/two workers for now.

`--ffn-vector` is a separate experimental candidate for grouped expert and shared/dense FFN projections. It reuses each native K-quant block's metadata over eight weights, keeps FP32 lane/reduction order, and falls back for unsupported shapes/types. It does not expand whole weights or change model quantization. Model-free checks cover Q4_K/Q5_K/Q6_K, mixed expert projection types, production dimensions, and partial/256-row prompt batches; its [full-model short-history test](validation/particle-hybrid-decode/20261024-ffn-vector-slots12-workers2/README.md) passed exact logits and continuations at 2.68 tokens/s. Cross-run differences do not isolate a speed gain over the 2.61 bundle, so it remains experimental and is not included in the command above.

The benchmark helper accepts `--slots`, `--ssd-workers`, `--history`, `--suffix`, `--rounds`, `--decode-tokens`, and `--ffn-vector`. Results include per-phase exposed read/GPU waits, expert byte counts, source fingerprints and before/after system swap counters. GPU wait counters include host scheduling and mapped page faults; they are not pure GPU execution times.
