# Particle optimization audit

23 October 2026. This maps **all 104 indexed experiment IDs** in the inherited TurboFieldfare inventory to the Sarvam implementation. The upstream heading says 103; its actual tables include 104 (including KV-15). This is an implementation audit, not a claim that every upstream speedup has been reproduced.

Sources: the local [experiment inventory](experiments/EXPERIMENT_INVENTORY.md), its ten detailed summaries and corresponding Swift/Metal code; [upstream inventory](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/EXPERIMENT_INVENTORY.md); [Sarvam’s model/configuration](https://huggingface.co/sarvamai/sarvam-30b). Gemma affine group-64 quantization and Sarvam GGML K-quants are different binary and mathematical contracts.

## What changed in this pass

- Actual batched FP32 GPU projections, including attention QKV/output, dense/shared experts, router and grouped routed experts. Four prompt rows reuse each decoded weight. The previous grouped path still issued one GEMV per row.
- 128-token prompt chunks by default. Row scratch remains bounded (under 31 MiB at 128 rows for this model); causal attention and original top-six reduction order remain intact.
- Fused decode and batched-prefill gate/up plus SwiGLU, removing separate activation dispatches and unused up scratch. Native K-quant bytes and FP32 accumulation remain unchanged.
- User-approved lazy internal SSD expert cache, at most 4096 MiB of payload for the installed model, an exclusive SSD tier holding recent RAM evictions, one contiguous read per hit, source-identity isolation, SHA-256 validation, corruption fallback and independent source/SSD metrics. No complete checkpoint copy or conversion.
- Warm interactive CLI with persistent weights/cache, complete official chat-template rendering, exact KV-prefix matching and suffix rewind. `/new` resets conversation while keeping weights warm. Capped final tokens are not incorrectly treated as already processed.
- Bounded deterministic sampling candidates and an isolated paired GPU benchmark command.

The model still has 128 routed experts, executes the selected top six plus the shared branch, uses the same stored Q4_K/Q5_K/Q6_K weights and retains FP32 KV. There is no expert dropping, cache-dependent router, additional weight quantization or hidden reduction of attention context.

## Method-by-method decisions

“Partial” and “deferred” identify real remaining work. The [24 October implementation](PARTICLE_PREFILL_OPTIMIZATION.md) adds experimental native K-quant staged matrices, FP32 TensorOps/full-history grouped and split attention, and exact block-metadata reuse for standalone heads. Persistent multi-expert reduction and FP16 KV remain unimplemented. Those methods cannot be counted as implemented just because the Gemma engine contains them. Upstream rejected candidates are not automatic requirements to reproduce.

| Experiment | Particle decision |
| --- | --- |
| [IO-01: `mmap` versus `pread`](experiments/summaries/01-model-install-and-expert-io.md#io-01) | Implemented: bounded parallel pread into fixed Metal-visible slots; no full expert mmap. |
| [IO-02: Parallel reads and coalescing](experiments/summaries/01-model-install-and-expert-io.md#io-02) | Implemented parallel miss reads; new SSD tier stores gate/up/down in one contiguous entry. Cold GGUF misses still need three ranges. |
| [IO-03: Darwin file hints](experiments/summaries/01-model-install-and-expert-io.md#io-03) | Source GGUF hints unchanged. SSD tier uses F_NOCACHE to avoid a redundant file-page copy of hot RAM slots; no isolated hint speed claim. |
| [IO-04: Dedicated I/O executor](experiments/summaries/01-model-install-and-expert-io.md#io-04) | Not adopted: upstream dedicated executor lost its comparison. |
| [IO-05: Custom I/O worker pool](experiments/summaries/01-model-install-and-expert-io.md#io-05) | Not adopted: concurrentPerform remains; custom pool had no stable gain. |
| [IO-06: `mlock`](experiments/summaries/01-model-install-and-expert-io.md#io-06) | Not adopted: pinning pages adds pressure and did not fix upstream contention. |
| [IO-07: Expert compression](experiments/summaries/01-model-install-and-expert-io.md#io-07) | Not adopted: native K-quant bytes retained; compression does not remove the bandwidth gap. |
| [IO-08: Speculative reads](experiments/summaries/01-model-install-and-expert-io.md#io-08) | Not adopted: no speculative reads or unproven route predictions. |
| [IO-09: MTLIO](experiments/summaries/01-model-install-and-expert-io.md#io-09) | Not adopted: source is slow removable storage, not a reliably warm NVMe working set. |
| [IO-10: Remote streaming repack](experiments/summaries/01-model-install-and-expert-io.md#io-10) | Different contract: consume the installed GGUF in place. New lazy partial SSD packing avoids a second full checkpoint. |
| [DEC-01: One SIMDgroup per row](experiments/summaries/02-decode-moe-int4-and-router.md#dec-01) | Implemented: each SIMDgroup computes one output row; four groups per threadgroup. |
| [DEC-02: Cooperative MoE row](experiments/summaries/02-decode-moe-int4-and-router.md#dec-02) | Not adopted: preserve independent row parallelism. |
| [DEC-03: Persistent multi-TG MoE](experiments/summaries/02-decode-moe-int4-and-router.md#dec-03) | Partial: fused gate/up kernel added. Persistent multi-expert down/reduction needs a separate K-quant implementation and real-model gate. |
| [DEC-04: Initial 16-slot cache](experiments/summaries/02-decode-moe-int4-and-router.md#dec-04) | Configurable 6–16 RAM slots; keep 12 default until a longer memory-pressure sweep validates 16. Add bounded SSD capacity instead. |
| [DEC-05: Shared-MLP/I/O overlap](experiments/summaries/02-decode-moe-int4-and-router.md#dec-05) | Implemented: shared-expert GPU command executes while demand expert reads complete. |
| [DEC-06: MoE INT4 vectorization](experiments/summaries/02-decode-moe-int4-and-router.md#dec-06) | Partial: fused gate/up and four-token weight reuse implemented. Gemma affine INT4 vector loads cannot decode K-quants unchanged. |
| [DEC-07: Standalone INT4 vectorization](experiments/summaries/02-decode-moe-int4-and-router.md#dec-07) | Deferred: format-specific wide-load K-quant head kernel; test all block and live offsets before promotion. |
| [DEC-08: Multi-TG router](experiments/summaries/02-decode-moe-int4-and-router.md#dec-08) | Implemented equivalent row-parallel FP32 router projection. Sarvam router is sigmoid with selection-only bias, not Gemma softmax. |
| [DEC-09: Parallel top-k selector](experiments/summaries/02-decode-moe-int4-and-router.md#dec-09) | Not adopted: deterministic 128-way CPU selector is small; retain exact routing. |
| [DEC-10: Gate/up fusion](experiments/summaries/02-decode-moe-int4-and-router.md#dec-10) | Implemented: gate/up reductions plus SwiGLU in one decode kernel, tested bitwise against separate kernels. |
| [DEC-11: Phase 2 plus reduce](experiments/summaries/02-decode-moe-int4-and-router.md#dec-11) | Deferred: fuse down plus weighted reduction without changing the original top-six addition order. |
| [DEC-12: Phase-1 paired rows](experiments/summaries/02-decode-moe-int4-and-router.md#dec-12) | Not adopted: upstream paired phase-1 rows regressed. |
| [DEC-13: Phase-1 `u16` loads](experiments/summaries/02-decode-moe-int4-and-router.md#dec-13) | Deferred: native K-quant block alignment and scale encoding differ from affine group-64. |
| [DEC-14: Phase-2 d2/d2p/d4](experiments/summaries/02-decode-moe-int4-and-router.md#dec-14) | Not adopted without a Sarvam-specific shape sweep; upstream results were mixed. |
| [DEC-15: Group-sum reuse](experiments/summaries/02-decode-moe-int4-and-router.md#dec-15) | Not adopted: the stronger upstream wrapper gate reversed the isolated win. |
| [DEC-16: Adaptive geometry](experiments/summaries/02-decode-moe-int4-and-router.md#dec-16) | Not adopted: no trace-specific adaptive geometry. |
| [DEC-17: Progressive execution](experiments/summaries/02-decode-moe-int4-and-router.md#dec-17) | Not adopted: no progressive scheduling that multiplies dispatches. |
| [DEC-18: Hit-first split](experiments/summaries/02-decode-moe-int4-and-router.md#dec-18) | Implemented reservation/pin/read/complete transactions for prompt and decode hits while misses load; separate rank outputs retain original reduction order. Whole-model performance gates still apply. |
| [CACHE-01: LRU to LFU](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-01) | Implemented LFU with recency tie-breaking, row-weighted prompt uses. SSD uses recency for evicted RAM experts, avoiding redundant active RAM copies. At most two saved victim buffers, entry pinning and reserved write capacity preserve bytes and the payload cap. |
| [CACHE-02: Windowed LFU](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-02) | RAM keeps cumulative LFU; window variants lacked a stable upstream win. The new SSD victim tier has a different role and uses recency. |
| [CACHE-03: 24/32 slots](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-03) | Bounded 6–16 RAM slots and explicit SSD byte cap; no cache expansion that forces swap on 8 GB. |
| [CACHE-04: Heterogeneous allocation](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-04) | Not adopted: uniform per-layer RAM capacity; no training-trace slot allocation. |
| [CACHE-05: Cross-layer predictor](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-05) | Not adopted: no weak cross-layer route predictor. |
| [CACHE-06: Markov prefetch](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-06) | Not adopted: no Markov speculative overwrite or cache-conditional routing. |
| [CACHE-07: Offset read order](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-07) | Not adopted: keep rank semantics; offset reordering lacked a repeated win. |
| [CACHE-08: Packed layout](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-08) | Adapted only to the approved SSD tier: exact gate/up/down bytes in one partial entry; source GGUF unchanged. No full-model permutation. |
| [CACHE-09: APFS preallocation](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-09) | Not claimed: lazy entries, no full preallocated packed model. |
| [RAD-01: Initial miss-only signal](experiments/summaries/04-rdadvise.md#rad-01) | Off: preliminary upstream short-run result is insufficient to promote on SD. |
| [RAD-02: Repeated short gate](experiments/summaries/04-rdadvise.md#rad-02) | Off: later upstream long-context results supersede the short win. |
| [RAD-03: Advice cap](experiments/summaries/04-rdadvise.md#rad-03) | Not adopted: capped advice increased demand I/O. |
| [RAD-04: Advice in I/O path](experiments/summaries/04-rdadvise.md#rad-04) | Not adopted: advice in the read path regressed. |
| [RAD-05: Long-context policy](experiments/summaries/04-rdadvise.md#rad-05) | Not adopted: long-context policy failed another length. |
| [RAD-06: Adaptive policy](experiments/summaries/04-rdadvise.md#rad-06) | Not adopted: no stable adaptive rule. |
| [RAD-07: Async overlap](experiments/summaries/04-rdadvise.md#rad-07) | Deferred optional experiment only; no async advice enabled by default. |
| [KV-01: Split-KV attention](experiments/summaries/05-attention-and-kv-cache.md#kv-01) | Implemented experimentally: 256/512-position blocks and stable FP32 softmax merge, tested through 8192 positions. Reference remains available. |
| [KV-02: GQA-aware SWA](experiments/summaries/05-attention-and-kv-cache.md#kv-02) | Sarvam has full attention on every layer; Gemma sliding-window specialization is inapplicable. |
| [KV-03: Full-GQA A3](experiments/summaries/05-attention-and-kv-cache.md#kv-03) | Not adopted: geometry lost at a larger context. |
| [KV-04: A4 local variants](experiments/summaries/05-attention-and-kv-cache.md#kv-04) | Not adopted: local attention variants did not survive long rows. |
| [KV-05: MLX geometry v1](experiments/summaries/05-attention-and-kv-cache.md#kv-05) | Not adopted: early fast geometry failed forced-prefix validation. |
| [KV-06: MLX geometry v2](experiments/summaries/05-attention-and-kv-cache.md#kv-06) | Not adopted: instruction-checkpoint quality failed. |
| [KV-07: Packed K4/V4 attention](experiments/summaries/05-attention-and-kv-cache.md#kv-07) | Not adopted: retain uncompressed KV and exact attention. |
| [KV-08: Alternative codecs](experiments/summaries/05-attention-and-kv-cache.md#kv-08) | Not adopted: alternative KV codecs slowed reads. |
| [KV-09: K4/V4 quality](experiments/summaries/05-attention-and-kv-cache.md#kv-09) | Not adopted: K4/V4 changed quality; no silent KV requantization. |
| [KV-10: FP16 full-attention island](experiments/summaries/05-attention-and-kv-cache.md#kv-10) | Not adopted: reduced-precision attention failed the stronger gate. |
| [KV-11: Packed chunk 32](experiments/summaries/05-attention-and-kv-cache.md#kv-11) | Not adopted: codec holdout quality failed. |
| [KV-12: FP16 KV ring](experiments/summaries/05-attention-and-kv-cache.md#kv-12) | Deferred, explicitly: FP32 KV retained. Sarvam already uses compact GQA (4 KV heads × 64), so the memory saving is 38 MiB at context 2048, not Gemma’s ~0.6 GB. |
| [KV-13: Ring kernel follow-up](experiments/summaries/05-attention-and-kv-cache.md#kv-13) | Not adopted: below the upstream whole-step action threshold. |
| [KV-14: Prefill attention race](experiments/summaries/05-attention-and-kv-cache.md#kv-14) | Applied lifetime rule: separate prompt-row projections, causal KV positions, queue-ordered scratch, wait before expert slot refill. |
| [KV-15: Grouped full attention](experiments/summaries/05-attention-and-kv-cache.md#kv-15) | Deferred: group all 16 Sarvam Q heads sharing a KV head; requires a Sarvam full-attention kernel and long-context distributional gate. |
| [PF-01: Mixed QMM/GEMV](experiments/summaries/06-prefill.md#pf-01) | Implemented scalar path for one row, four-token native K-quant batch path for multiple rows. Benchmark actual shapes before any further dispatch policy. |
| [PF-02: Chunk 128](experiments/summaries/06-prefill.md#pf-02) | Implemented 128-token default and configurable 1–256 (above 128 experimental); partial tiles and chunks retain causality. |
| [PF-03: Deeper lookahead](experiments/summaries/06-prefill.md#pf-03) | Not adopted: avoid doubling expert tiles to increase lookahead. |
| [PF-04: Routed SIMD](experiments/summaries/06-prefill.md#pf-04) | Not adopted: no cross-lane cooperation that removes useful expert/row parallelism. |
| [PF-05: Shared INT8 QMM](experiments/summaries/06-prefill.md#pf-05) | Inapplicable as written: Sarvam shared experts are K-quants, not INT8. Use native batched K-quant projections. |
| [PF-06: Metadata reduction](experiments/summaries/06-prefill.md#pf-06) | Implemented one expert→(row,rank) use plan; rows keep separate rank outputs to preserve exact summation. |
| [PF-07: Argument-buffer reuse v1](experiments/summaries/06-prefill.md#pf-07) | Not adopted: per-dispatch immutable argument buffers are retained until completion; no unsafe in-flight reuse. |
| [PF-08: Argument-buffer rings](experiments/summaries/06-prefill.md#pf-08) | Not adopted: argument-buffer rings regressed M2 despite fewer allocations. |
| [PF-09: Shared/fetch overlap M2](experiments/summaries/06-prefill.md#pf-09) | Basic shared/read overlap exists in Particle; deeper upstream scheduler v3 is not ported. Longer Sarvam prompt comparisons still needed. |
| [PF-10: Shared/fetch overlap M5](experiments/summaries/06-prefill.md#pf-10) | Not adopted: an M5-specific small win did not survive M2 validation. |
| [PF-11: Trusted receipt](experiments/summaries/06-prefill.md#pf-11) | Different integrity contract: original GGUF metadata validation plus source identity and per-entry SHA-256 for cached bytes; no unverified conversion receipt. |
| [PF-12: Staged affine MPP](experiments/summaries/06-prefill.md#pf-12) | Implemented experimentally: bounded native Q4_K/Q5_K/Q6_K half tiles with FP32 matrix accumulation and fused gate/up activation. Half-rounded CPU primitive oracle passes; full-model quality/promotion gates remain required. |
| [PF-13: Direct UInt4](experiments/summaries/06-prefill.md#pf-13) | Not adopted: direct UInt4 is neither the current upstream winner nor a GGML K-quant decoder. |
| [PF-14: QMM TG reuse](experiments/summaries/06-prefill.md#pf-14) | Not adopted without a significant whole-step opportunity. |
| [PF-15: Batched routed MoE](experiments/summaries/06-prefill.md#pf-15) | Implemented native batched routed MoE: each loaded expert serves all its rows; four token accumulators reuse a decoded weight, original rank reduction preserved. |
| [PF-16: Long endpoint gate](experiments/summaries/06-prefill.md#pf-16) | Adopted validation principle; current short logits/continuation gate is not a 512/1017-token quality evaluation. |
| [PF-17: TensorOps full attention](experiments/summaries/06-prefill.md#pf-17) | Implemented experimentally: Sarvam FP32 TensorOps QK/PV, 16-head GQA sharing and full causal history; independent primitive checks and resident M2 context/suffix benchmark. No end-to-end speedup inferred from Gemma. |
| [ORCH-01: QKV epilogue](experiments/summaries/07-fusions-head-and-orchestration.md#orch-01) | Partial: Sarvam Q/K norm+RoPE+KV writes are one epilogue; fused linear/epilogue candidate remains. |
| [ORCH-02: Layer tail](experiments/summaries/07-fusions-head-and-orchestration.md#orch-02) | Partial: residual operations remain simple ordered kernels; future fused residual+RMS requires direct parity and timings. |
| [ORCH-03: Monolithic middle fusion](experiments/summaries/07-fusions-head-and-orchestration.md#orch-03) | Not adopted: monolithic fusion removed parallelism upstream. |
| [ORCH-04: Fused QKV GEMV](experiments/summaries/07-fusions-head-and-orchestration.md#orch-04) | Implemented Sarvam’s fused QKV weight tensor and one projection, followed by its own per-head normalization/RoPE. |
| [ORCH-05: Fusion D head family](experiments/summaries/07-fusions-head-and-orchestration.md#orch-05) | Current head is a row-SIMD direct Q6_K projection; Gemma affine row-head code is incompatible. Further vectorization remains. |
| [ORCH-06: Shader sweep](experiments/summaries/07-fusions-head-and-orchestration.md#orch-06) | Not adopted: no first-run shader sweep winners promoted without balanced repeat timings. |
| [ORCH-07: RoPE tables](experiments/summaries/07-fusions-head-and-orchestration.md#orch-07) | Not adopted: direct RoPE calculation; tables did not help upstream. |
| [ORCH-08: LM-head width/tiles](experiments/summaries/07-fusions-head-and-orchestration.md#orch-08) | Deferred head shape study; do not infer a full-step win from isolated bandwidth. |
| [ORCH-09: Single-wait pipeline](experiments/summaries/07-fusions-head-and-orchestration.md#orch-09) | Partial: one ordered queue, router CPU boundary and final routed wait; fully single-wait multi-expert pipeline remains. |
| [ORCH-10: Second Metal queue](experiments/summaries/07-fusions-head-and-orchestration.md#orch-10) | Not adopted: second queue lost upstream; one queue owns scratch ordering. |
| [ORCH-11: O3 token chain](experiments/summaries/07-fusions-head-and-orchestration.md#orch-11) | Not adopted: token-chain schedule failed longer controls. |
| [ORCH-12: CB1 encode-ahead](experiments/summaries/07-fusions-head-and-orchestration.md#orch-12) | Not adopted: no encode-ahead while the same scratch is owned by pending work. |
| [ORCH-13: Decode arg-buffer reuse](experiments/summaries/07-fusions-head-and-orchestration.md#orch-13) | Not adopted: no mutable decode argument-buffer reuse across in-flight commands. |
| [ORCH-14: All-hit fast path](experiments/summaries/07-fusions-head-and-orchestration.md#orch-14) | Not adopted: synchronization opportunity is too small relative to SD demand reads. |
| [ORCH-15: Merge shared/hit CB](experiments/summaries/07-fusions-head-and-orchestration.md#orch-15) | Not adopted: fewer submissions did not improve upstream wall time. |
| [OUT-01: One-pass sampling and Top-64](experiments/summaries/08-sampling-tokenization-and-output.md#out-01) | Adapted: deterministic bounded top-k heap replaces sorting 262,144 candidates. Greedy retains exact tie behavior. An opt-in compact GPU greedy argmax preserves ties and finite-value checks; positive-temperature sampling remains on CPU. |
| [OUT-02: Fused Gumbel head](experiments/summaries/08-sampling-tokenization-and-output.md#out-02) | Not adopted: fused Gumbel would change the seeded sampling algorithm and needs separate contract/quality tests. |
| [OUT-03: Tokenizer process cache](experiments/summaries/08-sampling-tokenization-and-output.md#out-03) | Implemented one tokenizer and one runner per interactive process; reuse common weights, expert caches and verified KV prefixes. |
| [OUT-04: Bounded detokenizer tail](experiments/summaries/08-sampling-tokenization-and-output.md#out-04) | Implemented incremental native detokenizer and byte-fallback runs; no whole-response re-decode for each token. |
| [METH-01: Reordered floating-point work needs a distributional quality oracle.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-01) | Applied: independent FP64 CPU primitive oracle plus full-logit/continuation gate. Approximate MPP/FP16 changes stay deferred until distributional validation. |
| [METH-02: Claimed-lossless work still needs exact identity.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-02) | Applied exact identity for lossless byte cache and reordered dispatch. |
| [METH-03: Test production-shaped offsets.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-03) | Applied production 4096-column native blocks, nonzero offsets and incomplete row tiles. |
| [METH-04: Prove synchronization scope.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-04) | Applied explicit argument-buffer residency, immutable per-dispatch bindings, separate row scratch, wait before slot overwrite. |
| [METH-05: Profiler tok/s is diagnostic.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-05) | Applied: GPU benchmark excludes storage; real-model footer separates prefill, decode, source, SSD, and blocking waits. |
| [METH-06: Interleave warm, thermally balanced rounds.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-06) | Applied alternating isolated kernel rounds; real model repeated-cache diagnostics label order and cache state. |
| [METH-07: Mechanism counts are not outcomes.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-07) | Applied: no claim that fewer dispatches/reads alone proves faster generation. |
| [METH-08: Trace-trained policies need holdouts.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-08) | Applied: repeat-prompt SSD cache results are explicitly not unseen-prompt generalization. |
| [METH-09: Detect greedy repetition loops.](experiments/summaries/09-validation-and-measurement-lessons.md#meth-09) | Applied: bounded output counts, stop handling, no claimed speed from long repetition loops. |

## Additional scope boundaries

Draft-model speculative decoding never reached an upstream runtime candidate and would need an additional validated model and memory budget. ANE/Core ML offload is a separate backend, not a flag for the streamed GGUF kernels. Sub-four-bit weights, dropped experts, compressed KV and cache-dependent routing are excluded from this lossless optimization pass. Sarvam has full attention on every layer, so Gemma sliding-window retention and ring eviction cannot be copied blindly.

The SSD victim tier is Particle-specific work for the slow SD setup, not an upstream TurboFieldfare feature. It improved the observed repeated eight-token diagnostic from 0.220 to 0.793 tokens/s while retaining the same RAM slots and 4 GiB SSD cap; see the performance record for cache conditions and limits.

## Next useful gates

1. Cache-cold, cache-warm, and different-prompt decode; report source MiB per forward step and total wall time. A repeated short prompt can fit its entire requested expert set in 4 GiB, which is not evidence that arbitrary long conversations will.
2. Short/medium/long Sarvam prompt rows with scalar/batch/scalar comparisons. The scalar path is `--prefill-chunk 1`; keep model ownership serial and do not flush the OS page cache.
3. When faster storage reduces the I/O share, profile native K-quant head/vector loads and persistent top-six down/reduction before adding scheduler complexity.
4. Long-context full GQA kernels and staged K-quant MPP need independent primitive checks plus a multi-prompt distributional quality gate before becoming defaults. FP16 KV also changes precision and must be evaluated explicitly.

See [current validation and performance](PARTICLE_PERFORMANCE.md), [CLI usage](PARTICLE_CLI.md), and [historical SD measurements](GTURBO_OPTIMIZATION.md).
