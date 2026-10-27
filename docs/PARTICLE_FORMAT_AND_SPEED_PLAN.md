# Particle model format and performance plan

24 October 2026. Research and implementation specification. Native packing, loading, storage profiles, bounded SSD mirror/read-through caching, and initial opt-in decode candidates are now implemented; see [implementation status](PARTICLE_PACK.md) for validation and outstanding work. This plan is broader than the current implementation. No throughput target is established by this specification.

## 1. Goals and evidence

The primary acceptance target is **at least 4 generated tokens/s on the M2 / 8 GiB Mac with the active model pack on the internal SSD**. The secondary target is **at least 0.6 tokens/s, aiming for 0.9 and then 1+, with the primary pack on SD and a bounded internal SSD tier**. These are engineering acceptance targets, not measured results or guarantees.

Decisions agreed with the user:

- `.particle` is a self-contained model **directory**, following the `.gturbo` architecture.
- Preserve the existing Q4_K_M model as the reference. Also investigate explicit, speed-first approximate profiles, with measured quality losses and no automatic substitution.
- Allow up to **18 GiB / 18,432 MiB of internal SSD tier payload**. Keep the existing default of 4,096 MiB until a larger value is selected. This is disk capacity, not a RAM allocation.
- Keep adjustable RAM expert capacity, official Sarvam conversation formatting, all conversation turns within the configured context, and one model process.
- Do not replace Sarvam with an unrelated model or treat an untrained graph modification as a compatible checkpoint. Training a derivative or a draft head is a separate research result requiring its own weights and validation.

### What we actually know

The installed GGUF is 19,582,216,032 bytes, about 18.24 GiB. It has 19 layers, one leading dense layer, 18 MoE layers, 128 routed experts per MoE layer, top-six routing, and an untied output head. Existing implementation and measurements are recorded in [the current optimization guide](PARTICLE_PREFILL_OPTIMIZATION.md) and [24 October validation](validation/particle-prefill-20261024/README.md).

- Twelve expert slots per MoE layer have approximately 1,557 MiB total capacity. The entire routed pool is about **16.22 GiB**. Common mapped weights excluding embeddings are about **1.44 GiB**; together they are about **17.66 GiB**, before new alignment and metadata overhead. The packer must calculate exact padded sizes before provisioning.
- Measured SD-backed varied conversation: **0.200–0.221 tokens/s**. A repetitive raw completion reached 0.495. Varied decode spent about 84–87% of its wall time in exposed expert-loading waits, including secondary-cache transactions.
- RAM misses requested about **287–395 MiB per generated step** in those longer runs. Internal SSD entry reads measured about **1.6 GiB/s**, excluding checksums, scheduling and GPU work.
- A tiny, completely warm one-step diagnostic reached about 0.158 seconds per step. It demonstrates a favorable execution case, not sustained 6+ tokens/s.
- A new read-only probe read eight distinct 8 MiB GGUF ranges at offsets 256, 2304, 4352, 6400, 8448, 10496, 12544 and 14592 MiB. Elapsed seconds were 0.2593, 0.2525, 0.2519, 0.2504, 0.2526, 0.2600, 0.2506 and 0.2506: **31.56 MiB/s** requested throughput. It used Darwin `F_NOCACHE`, no cache purge and no inference. This is a small storage probe, not a guaranteed physical-cold benchmark or a replacement for runtime measurements.

Four tokens/s permits **250 ms per committed token**. Serial reads of the observed misses alone take approximately 174–239 ms at the measured SSD rate. Adding the tiny resident diagnostic gives roughly 2.5–3 tokens/s in a deliberately simplified, non-overlapping cost model. This is not a prediction or ceiling; it explains why moving the file alone does not establish the new target. We need fewer transferred bytes, less execution overhead, useful overlap, or more committed tokens per verification cycle.

With no expert reuse, six experts across all MoE layers require about 778.5 MiB per token. That is about 24.7 seconds of reads even at the new 31.56 MiB/s probe rate. A directory suffix cannot remove this volume. SD-only, empty-cache performance must be reported even if it misses the target. A prepared 18 GiB SSD tier can potentially cover the complete expert pool and critical common weights: this is explicitly **hybrid SSD-backed inference**, with SD retaining the primary installation.

### Research coverage and conclusions

Screened all **194 issue/PR entries** returned by the upstream API: 86 issues and 108 pull requests, across open and closed states. Reviewed relevant bodies, selected comment threads, local source and the existing 104-experiment audit. Upstream `main` was `2e9ea50e8c35a65d9170e3bff9d1063be8931ec0`. GitHub Discussions are disabled there; issues and PR comment threads supply the discussions. This is a bounded review of relevant evidence, not a claim to exhaust the internet.

| Evidence | Consequence for Particle |
| --- | --- |
| [TurboFieldfare system design](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md), [shared-format issue #83](https://github.com/drumih/turbo-fieldfare/issues/83), [PR #87](https://github.com/drumih/turbo-fieldfare/pull/87) | Use a shared format contract, separated common/expert data, aligned contiguous expert blobs, bounded loading and transactional installation. Keep Sarvam graph and native quantization distinct from Gemma. |
| [Upstream I/O experiments](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md) and [cache experiments](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/03-expert-cache-prediction-and-layout.md) | Establish demand-read controls. Compression, larger caches, speculative I/O and trace-specific layouts can regress. Re-test only against a stated mechanism. |
| [Early-read PR #187](https://github.com/drumih/turbo-fieldfare/pull/187), [fitted predictor PR #188](https://github.com/drumih/turbo-fieldfare/pull/188) | Both were open/unmerged at inspection. Their separate staging buffers and idle-only reads merit an opt-in Sarvam experiment; their other-host results are not M2/SD predictions. |
| [Preallocation PR #152](https://github.com/drumih/turbo-fieldfare/pull/152) | Closed and unmerged at inspection. Test contiguous-first APFS preallocation when creating the new files; do not describe its submitted gains as upstream production behavior. |
| [Expert-count issue #167 and maintainer responses](https://github.com/drumih/turbo-fieldfare/issues/167) | Reducing active experts caused arithmetic/code errors in the reported Gemma test. Any Sarvam expert reduction is an approximate experiment, never a transparent cache optimization. |
| [Apple: LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language), [paper](https://arxiv.org/html/2312.11514v2) | Apply contiguous bundling, reuse and a measured I/O cost model. Its neuron-sparsity results rely on activation sparsity including ReLU; Sarvam's SwiGLU does not inherit those zeros automatically. |
| [Mixtral offloading](https://arxiv.org/abs/2312.17238), [implementation](https://github.com/dvmazur/mixtral-offloading), [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Use expert-granularity caching and trace-based policy studies. Host-RAM-to-GPU transfer results do not establish slow-SD performance. |
| [Flash-MoE](https://github.com/danveloper/flash-moe), [streamlx](https://github.com/srcterm/streamlx), [Slipstream results](https://github.com/Schero94/slipstream/blob/main/BENCHMARKS.md) | Study deferred submissions, grouped expert-major processing, direct cache-buffer consumption and storage placement. Their machines, RAM and models differ. Conflicting overlap/prefetch results require a device-specific benchmark. |
| [Upstream Qwen-port proposal #105](https://github.com/drumih/turbo-fieldfare/pull/105), [SwiftMoE](https://github.com/jpurnell/swiftMoE), [moe-stream](https://github.com/GOBA-AI-Labs/moe-stream), [llama.cpp expert-cache PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) | Useful scheduling, cache and porting comparisons. Do not transfer M3/M4/large-RAM or CUDA performance numbers to this M2. |
| [Fiddler](https://arxiv.org/abs/2402.07033), [SP-MoE](https://arxiv.org/abs/2510.10302), [SpecMoEOff](https://arxiv.org/abs/2508.21706) | CPU/GPU and speculative scheduling depend on bandwidth and workload. Particle's existing hybrid arithmetic was slower; keep it off unless a new complete-latency comparison wins. |
| [Prompt Lookup](https://github.com/apoorvumang/prompt-lookup-decoding), [Draft & Verify](https://arxiv.org/abs/2309.08168), [SpecFit](https://arxiv.org/abs/2610.15417), [Specter v3](https://arxiv.org/abs/2511.14102v3) | Build a small verified-block interface. Optimize accepted tokens per total cycle time, including the union of demanded experts; do not assume longer draft blocks are cheaper. |
| [LayerSkip](https://arxiv.org/abs/2404.16710), [EAGLE-3](https://arxiv.org/abs/2503.01840) | These can need specially trained weights. Do not assume the installed Sarvam has an MTP/early-exit head. A trained drafter is a later, separately measured derivative. |
| [HQQ](https://github.com/dropbox/hqq), [HIGGS](https://arxiv.org/abs/2411.17525), [MC-MoE](https://arxiv.org/abs/2410.06270) | Investigate expert-specific low precision and calibration. CUDA kernels are not M2 kernels, and re-quantizing our existing quantized source adds error. |
| [CORM](https://openreview.net/pdf/9a5ef195830f63bfedef042e385cb6f91b10a95f.pdf), [TD-MoE](https://zeyiwen.github.io/papers/iclr26-td-moe.pdf) | Fine-grained neuron streaming and shared expert factorization are research alternatives if simpler changes fail. They change approximation/layout and need calibration, bounded prototypes and separate quality gates. |
| [TurboQuant-MLX Sarvam report](https://github.com/manjunathshiva/turboquant-mlx/blob/main/README.md#sarvam-moe), [Sarvam byte-token issue #30879](https://github.com/ggml-org/llama.cpp/issues/30879), [V-cache issue #22627](https://github.com/ggml-org/llama.cpp/issues/22627) | Test long-output degeneration, exact official tokenization and KV layouts. These external reports do not establish defects in Particle's current GGUF path. |

## 2. Build a real `.particle` pack

### Version 1 layout

Create `/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle/` containing:

```text
manifest.json
weights.json
model_weights.bin
embeddings.bin
tokenizer/                     # unchanged official sidecars
packed_experts/
  layout.json
  layer_01.bin
  ...
  layer_18.bin
verified-install.json
```

This imitates the upstream storage/runtime architecture, not Gemma's binary compatibility. Use a Foundation-only `ParticleFormat` module shared by the packer, loader, verifier and fixtures. Version 1 uses canonical sorted-key JSON indexes and raw little-endian payload files; it does not pretend to be a `.gturbo` resident binary index.

- Manifest magic `PARTICLE`, version `1.0`, architecture `sarvam_moe`; include the exact graph configuration, immutable tokenizer hashes, tensor/pack hashes, source identity, conversion profile and quantization lineage. Runtime KV/cache settings are not model weights and do not alter pack identity.
- `weights.json` describes each common tensor's logical name, shape, native encoding, row bytes, file, offset and byte length. Keep the untied output head in common weights and embeddings in their own row-readable file.
- Each MoE layer has its own 16 KiB-aligned expert stride, because K-quant types/sizes differ between layers. An expert blob stores gate, up and down in that order with 256-byte subregion alignment and zero padding to the layer stride. Store explicit logical expert IDs, offsets, lengths, encodings, shapes and per-blob SHA-256. The initial physical order is logical-ID order.
- Preserve Q4_K/Q5_K/Q6_K and FP32 bytes exactly in the reference pack. One contiguous `pread` fills a whole expert slot. No whole-weight floating-point expansion.
- Bound each structural JSON index to 16 MiB; keep tokenizer handling under its existing separate rules. Validate integer overflow, shape/byte-count agreement, non-overlap, alignment, required tensors, filenames and supported encodings. Resolve the outer model path once and reject internal symlink/path escapes and incomplete installs.

### Packer and loader

Add `ParticlePack` and helper commands `Scripts/particle pack` and `Scripts/particle verify`. `pack --source <gguf> --output <directory.particle> --tokenizer <folder> --profile native` performs a metadata-only size/space plan before copying. Require destination capacity plus 2 GiB headroom. Never overwrite/delete the GGUF or replace an existing completed pack automatically.

Use a deterministic source-offset-ordered copy plan, at most 8 MiB data scratch, and preallocated destination ranges. Read source bytes sequentially, calculate the source hash in that pass and write the planned destination offsets. Commit resumable checkpoints after at most 64 MiB source progress: flush dirty destinations, then atomically update the checkpoint. After restart, verify committed ranges and rehash the source prefix needed to restore its digest; replay incomplete work. A final bounded destination verification establishes all file/blob hashes before writing the receipt and atomically renaming the `.partial` directory. Stop on source changes, short reads, corruption, removal of the card, or insufficient space; preserve resumable progress.

Contiguous-first `F_PREALLOCATE` falls back to ordinary allocation with a reported status when unsupported. Measure it; allocation success alone is not evidence of a speed win. Conversion and verification can take many minutes on SD and must be reported separately from inference speed.

Introduce a storage-source protocol exposing configuration, logical tensor descriptors, embedding-row access, common mappings and expert-blob reads. Supply GGUF and `.particle` adapters. Keep the existing GGUF runner initializer as a compatibility wrapper. Derive persistent cache identity from immutable pack/tensor content and encoding, not its pathname. Benchmark helpers and cache validation use the same abstraction. Never route Sarvam through the inherited Gemma loader.

The verified `.particle` must run without its source GGUF. Package tokenizer sidecars unchanged and continue validating their token IDs and official template. Merely changing the file extension must fail validation.

## 3. Implement and measure optimizations in this order

### A. Storage modes and the 18 GiB tier

Add `--storage-profile legacy|sd-only|sd-hybrid|ssd`; default `legacy` preserves existing command behavior. `sd-only` disables the secondary tier; `sd-hybrid` uses the configured SSD directory; `ssd` requires the active pack on internal SSD and disables the redundant disk cache. Keep `--no-ssd-cache` compatible and reject contradictory new profile/legacy flags clearly.

Retain `--ssd-cache-mib` default 4096 and raise its selectable maximum to **18432**. Add `--ssd-cache-policy victim|read-through|mirror`: victim remains the legacy default; read-through becomes the new hybrid candidate. A read-through hit stays on SSD rather than being deleted and rewritten as part of every RAM exchange. Use bounded probation/protected LRU segments (20/80% of available expert bytes), checksum every SSD-to-RAM load, and perform metadata reservations/commits outside bulk reads/hashing. Compare it against the exclusive victim cache at equal capacity; duplication can be unfavorable at 4 GiB.

`mirror` requires room for every padded routed blob plus common weights. Add `Scripts/particle prepare-cache --model <pack> --ssd-cache-mib 18432 --ssd-cache-policy mirror`; verify/provision it before inference, with resumability and no concurrent model worker. Keep common weights and the complete routed pool on SSD, embeddings on the SD primary. Once complete, unfamiliar expert routes do not require SD reads. If exact padded sizes exceed the cap, fail with the required byte count rather than silently claiming full coverage. Cache metadata, masks and small indexes are reported separately; all payload, old namespaces and in-flight temporary payload share the one cap.

Mirror mode maps common weights from its verified SSD copy and uses packed expert reads from SSD. It does not allocate the mirror in RAM. Losing the tier falls back to the SD primary with a visible degraded-storage event and invalidates any 0.6/4-token/s claim for that interval. Never combine a full mirror with a second redundant victim pool.

For the SSD-primary target, activate the pack on the Mac's internal SSD, approximately 18.24 GiB for the native pack before exact padding. This primary installation is separate from the optional 18 GiB cache budget; **disable that secondary cache**, so there is no extra 18 GiB copy. The original SD pack can remain a backup. Copy/provision only as an implementation step, with preflight and verification; no model was copied during this planning work.

### B. Direct reads, scheduling and cache policy

- Read packed experts into reusable aligned Metal-visible buffers, preserve reservation/pin/read/complete lifetimes, and use the same bytes in kernels. Keep six rank outputs and the original reduction order for the reference path.
- Retain 12 RAM slots by default and the existing 6–16 manual setting. Sweep 6/8/12/16 at fixed context. Permit 24/32 only as explicit experimental settings after planned common + expert + KV + scratch allocations fit both device working-set and measured system headroom. For compressed packs, calculate actual strides instead of using native-pack memory estimates. Refuse or reduce only the experimental cache allocation under pressure; never silently discard conversation context.
- Record routing traces by `(layer, expert, bytes, tier, latency)`, with calibration and held-out workloads separated. Simulate LFU, LRU, aged LFU, S3-FIFO and offline Belady bounds before implementing a new policy. Keep LFU unless a complete-latency comparison wins. Size-aware policies must optimize saved read time per byte, not just hit count.
- Tune 1/2/4 SD and SSD workers independently. Compare caching/read-ahead hints and physical-offset ordering; merge only already demanded adjacent ranges with at most 10% extra bytes and one 8 MiB staging buffer. No speculative read amplification is hidden from metrics.
- Compare controlled overlap and serial scheduling on this M2: cached experts/shared branch may run while misses load, but unified-memory contention can reverse the benefit. Do not transplant another machine's schedule as a default.
- Experiment with one then two early-read buffers outside the foreground slots, at most 16 MiB total. Start after current demand reads finish, cancel/deprioritize for demand, and adopt a prediction only after the actual router requests that expert. SD speculation starts disabled. Enable only when incremental saved wait exceeds wasted traffic and GPU slowdown. A fitted predictor is trained from Sarvam traces; Gemma predictor weights are incompatible.

### C. Reduce decode execution cost toward the 250 ms budget

Add command-buffer GPU timestamps and CPU encode/wait attribution separately for attention, shared/routed FFN, head, selection, checksum and I/O. Distinguish mapped-weight page faults from GPU execution. Take detailed profiles separately from uninstrumented throughput runs.

The current runner waits after embedding, attention/router, ready experts, fetched experts and the ordered residual. First remove host waits that are not true routing/readiness dependencies: serialize GPU dependencies on the command queue, defer the routed tail and carry it into the next layer's attention submission. Keep GPU/read errors propagated, input buffers owned until completion and cancellation recovery correct. At the CPU routing boundary, an ordered command-buffer wait makes the needed router result visible; no CPU spin polling.

Then implement independently selectable candidates:

1. GPU top-six sigmoid selection with selection-only bias, deterministic ties and unchanged normalized routing weights scaled by 2.5. Return six IDs/weights rather than copying two 128-value arrays and sorting on CPU.
2. Persistent multi-expert gate/up and down kernels, bounded per-rank output scratch, and fused ordered reduction/residual. Reduce the current per-expert dispatch count without changing logical experts. Avoid argument-buffer rings until ownership and a whole-model benefit are proven.
3. Native wide-load K-quant decode kernels, block metadata reuse and projection-specific launch shapes; isolate attention, common/dense/shared matrices, routed experts and the 262,144-row output head. Test aligned and live-offset bindings. No whole-weight dequantized copies.
4. Optional lossless interleaving of native gate/up blocks, with an explicit format encoding and reversible byte validation; keep it only if both SD/SSD loading and the matching Metal kernel improve complete latency. Keep native layout readable as the fallback.
5. Tiled output-head projection with GPU argmax/top-k reduction. Greedy mode never transfers all logits to CPU; sampled mode preserves its distribution and has separate timing/validation.
6. Finish promotion tests for the existing grouped/split/tiled attention and shape-selected staged prefill kernels. Maintain full attention and official GQA mapping. FP32 KV remains the reference; FP16 KV is separately selectable and evaluated for memory and quality, not assumed to solve expert I/O.

The first internal latency allocation to investigate is approximately 100 ms compute/attention/head, 110 ms exposed I/O and 40 ms host/synchronization. This is a profiling budget summing to 250 ms, not a claim that the allocations are already achievable.

### D. Verified multi-token generation and prefill

Add a verifier that computes logits/greedy IDs for every candidate position, rather than only the last prefill row. Verify blocks of 2/4/8 tokens with the existing expert-major causal execution; share every demanded expert across its matching rows. Return only committed tokens. Commit KV only through the first rejection, mask/overwrite speculative suffix rows, and handle output caps, EOS, cancellation, Unicode streaming and idle canonicalization with the same committed-token ledger. Do not duplicate the whole KV cache or start a second model.

Implement prompt/ngram lookup first (match lengths 1–4, maximum draft 8), greedy-only initially. Test no match, zero/partial/full acceptance and chunk boundaries. Select block length using measured accepted tokens per **total** draft + verification cycle time and expert-union read volume. Turn it off for workloads where it loses. It is a targeted copy/code/summarization optimization, not evidence of 4 tokens/s on arbitrary conversation.

Next prototype training-free layer-skipped self-drafting with shared immutable weights, separate bounded draft suffix state, and full target verification. Search skip patterns on calibration only. Do not assume LayerSkip-style early exits work on untrained Sarvam. Discard the prototype if acceptance cannot pay for its expert reads and state handling. A trained Sarvam draft head is a later option requiring an explicit dataset, separate training compute/artifact and held-out validation; it is not a dependency of the initial pack.

Retain chunk 128 and exact warm-prefix reuse by default. Sweep 32/64/128/256 using complete prompts; enlarge beyond 256 only after binding limits, causal masks, scratch and memory tests change together. Keep additional inference scratch within 96 MiB. Preserve interruptible idle preparation and account for its work separately.

### E. Explicit speed-first derivatives

Build derivatives as separate `.particle` packs with new content IDs and `approximate` provenance; never overwrite the native pack. Existing Q4_K_M is already quantized, so further conversion is lossy. No full-precision download or claim of recovered original accuracy is part of this plan.

Start with GGML-compatible **Q3_K and Q2_K expert encodings**, using the independent [GGML quantizer/dequantizer reference](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) and new Metal decode/batched kernels. Process bounded row/block tiles. Keep routers, selection bias, norms, attention, shared/dense branches and output head at their current precision initially.

Compare uniform Q3_K, uniform Q2_K, and a sensitivity-calibrated mixed 2/3/native expert profile. Collect activation error and routing-frequency sensitivity from the native runner on a fixed 16K-token calibration set spanning English, Hindi, Tamil, Telugu and code, disjoint from evaluation. Use these data to minimize measured activation error under requested pack-byte budgets; do not infer sensitivity from weight magnitude alone. Candidate ranks/bit allocations are frozen before held-out testing. The standalone native-to-Q4-to-Q2 lineage is explicit in metadata.

After those results, run isolated approximation sweeps: active experts 6/4/3/2 and protected-position thresholds for near-zero SwiGLU neuron blocks. Use the correct selection-only bias, stable expert ranking and normalization over the actually selected set. Never skip a selected expert merely because it is absent from cache. Dynamic expert/neuron reduction changes the model; measure it as such.

If neither quantization nor verified blocks reaches the targets, conduct a bounded shared-basis/low-rank expert prototype: one MoE layer, one matrix family at a time, ranks 32/64/128/256, activation-error and GPU latency comparisons including basis residency. Expand to a full TD-MoE-style derivative only after that prototype wins against mixed quantization at equal memory and measured quality. Do not implement a new graph for an unverified hypothesis or describe it as exact Sarvam.

Lossless compression is a separate small sample experiment (LZ4 and Zstd): promote only when saved read time exceeds decompression/checksum cost for actual native expert blobs. Do not add a codec just because its offline file is smaller. ANE/Core ML, full expert mmap, `mlock`, M5-only NAX, beam batching and server-scale CPU/GPU recipes stay out of the critical path unless a bounded M2 experiment identifies a concrete benefit.

## 4. Validation, release gates and delivery

### Correctness and recovery

- Unit/fixture tests: canonical format round-trip, every K-quant encoding, per-layer stride differences, alignment/overflow, truncated or overlapping tensors, unknown mandatory fields, unsafe paths, incomplete receipts, source changes, corruption and read failures.
- Installation tests: source-order copies split across tensor/block boundaries, checkpoint durability, kill/resume, missing SD, disk full, checksum mismatch, completed destination protection, final atomic promotion, and independent operation without the GGUF.
- Cache tests: all payload and reservations under cap across namespaces, persistent read-through hits without rewrites, mirror completeness, corrupt tier fallback, partial-file recovery, concurrent readers/eviction and buffer ownership. At steady state a complete mirror must not issue SD expert/common reads or victim writes.
- Runtime tests: causal/GQA/chunk/split-softmax correctness; exact route selection; six-rank reduction; deferred-command dependencies; zero/partial/all speculative acceptance; `/new`, context exhaustion, idle interruption and foreground cancellation; Unicode combining marks, byte fallback, reasoning markers and official-template equivalence.
- Reference packing/storage changes must preserve bitwise tensor bytes and reference full logits/continuations. Numerically different kernels require the existing 0.0002 full-logit/continuation gate and matching selected IDs before default promotion. Approximate packs have independent numerical oracles and measured quality comparisons; they cannot pass as lossless.

### Benchmark protocol

Use one release build and one model process, on AC power with power mode/thermal state and background load recorded. Check model lease, macOS/Swift, free disk and memory pressure before inference; stop if process/preflight inspection cannot be completed. Keep build outputs on SD. Never purge system caches or run a second inference engine to manufacture a cold run.

Test storage separately: current GGUF + 4 GiB victim cache; native pack on SD with no tier; SD with 4/8/12/18 GiB tiers; and SSD-primary pack with no disk cache. Use a trace-only and bounded I/O sweep to prune weak settings before expensive model runs. For storage A/B, hold kernels and numerical profile fixed; for kernel A/B, hold storage, RAM budget and pack fixed.

Benchmark history lengths 512/2048/4096/7168 and suffixes 1/16/64/128, reserving up to 512 output positions inside the existing 8192 cap. Include English/Hindi explanations, code, extraction/JSON, multilingual translation, multi-turn chat and unrelated topic switches. Use at least 256 actual post-prefill decode forwards for throughput qualification, three alternating control/candidate pairs, and a 512-token or ten-minute sustained run (whichever is longer) for thermal/memory stability. Early-stop answers are reported naturally and are not padded or replayed to qualify throughput; choose additional prompts when they do not provide enough steps.

Report these conditions distinctly: empty secondary tier (first-use provisioning), fresh process with prepared tier, warm conversation, and unfamiliar routes after a topic switch. Fresh process is not physical-cold media; disclose OS-cache state. Initial mirror provisioning and pack installation are one-time operations reported separately, never hidden in a warm benchmark. A first-use empty-tier run is permitted to miss the SD target; its latency remains part of the product report.

Measure template time, new/replayed/retained KV tokens, TTFT, prefill, committed decode tokens, all target/draft forwards, acceptance, p50/p95 token or verification-cycle delay, SD/SSD requested bytes, read count, checksum time, cache writes, memory footprint, planned allocations, memory pressure and swap-out deltas. Report reasoning and final-answer token counts separately. Exclude the first prefill-produced token and non-text stop IDs from decode-rate accounting; speculative candidates count only once accepted and emitted. Requested reads are not physical-device traffic; publish device counters only when actually available.

**SSD acceptance:** each primary varied-chat/code/multilingual workload at the tested history lengths reaches a median of at least **4.0 committed tokens/s** in fresh-process and warm conditions, with no growing swap-out trend or repetition/garbage counted as successful output. Report native, numerically optimized and approximate profiles independently. A four-token warm-RAM diagnostic, repetitive raw completion or speculative copy-only win cannot satisfy this gate.

**SD acceptance:** primary SD installation with the explicitly disclosed prepared tier reaches a median of at least **0.6 committed tokens/s**, aiming for 0.9+. Report the smallest qualifying SSD budget; do not label complete-mirror performance SD-only. Record fresh-process/unfamiliar-route and warm results separately.

Before speed-first experiments, freeze a small quality suite: 100 arithmetic/instruction questions; 50 executable code/JSON checks; 20 copy/translation examples each in Hindi, Tamil, Telugu and Bengali; and 15 long-answer prompts at two seeds with up to 512 generated tokens. Report native-baseline and candidate scores, syntax validity, repetition and degeneration by task/language. Because the user selected speed-first exploration, no arbitrary small-loss ceiling is imposed on experimental artifacts. **Default profiles remain reference-quality**; an inaccurate derivative is explicitly labeled, and incoherent output is not advertised as usable 4-token/s inference.

Promote a candidate only when it improves median complete latency by at least 10% in its intended regime, with no regression over 5% in other tested regimes, and passes the applicable correctness/quality designation. Keep a reversible reference selector. If a target is missed, publish the remaining time/byte budget and the best measured profile rather than relaxing accounting or asserting success.

### Deliverables and order

1. Baseline/profiling report and reproducible workload fixtures; new source abstraction and native `.particle` shared format/packer/loader/verification.
2. Verified SD pack; explicit storage profiles; bounded read-through and complete mirror provisioning; direct SSD-primary baseline.
3. Independently measured GPU routing, dispatch/deferred-tail, persistent MoE, head and memory/I/O candidates; promote only qualifying paths.
4. Verified-block generation plus adaptive prompt lookup/self-draft prototypes; long-context and multi-turn regression results.
5. Optional lower-bit and structural derivative packs, calibration provenance and honest quality/speed comparisons; a target-qualification report and launch commands for the best validated configurations.

No implementation checkpoint is complete solely because a microbenchmark improved. The central success criterion is sustained, correctly accounted generation on this actual laptop, with storage placement and any model-quality trade-off visible.
