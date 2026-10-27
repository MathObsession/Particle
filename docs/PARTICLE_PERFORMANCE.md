# Particle performance and validation

This is the **23 October baseline**. See [the 24 October implementation and validation](PARTICLE_PREFILL_OPTIMIZATION.md) for current execution paths and their promotion status.

23 October 2026. Particle now has native batched K-quant prefill, fused gate/up activation, a user-approved bounded internal SSD expert cache, and warm interactive sessions with exact KV-prefix reuse. The original 19.58 GB GGUF remains on SD. Expert routing and stored quantization are unchanged; KV remains FP32.

The final release reached **0.421 tokens/s** on a partially cached eight-token completion and **0.719 tokens/s** on its warm repeat. An earlier exclusive-cache run reached 0.793 tokens/s; the initial inclusive cache stalled around 0.220 on the same eight-token repeat. These are capped diagnostics with seven post-prefill forward steps, not sustained arbitrary-prompt guarantees. Cold starts and unfamiliar routes remain slower.


The [complete method audit](PARTICLE_OPTIMIZATION_AUDIT.md) maps all 104 inherited experiment IDs, including rejected candidates and unfinished Sarvam ports. At this baseline, staged MPP, TensorOps full attention, split attention, FP16 KV, persistent multi-expert reduction and native K-quant head metadata reuse had not been implemented. The new implementation report supersedes this historical status.

## Environment and identity

- Apple M2, 8,589,934,592 bytes RAM.
- macOS 27.0.1, build 26A434; Xcode 27.1 on SD; Swift 6.4 (`swiftlang-6.4.0.34.1 clang-2100.3.34.1`), arm64 macOS 27 target; Metal 4 safe math.
- Original GGUF: `/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf`, 19,582,216,032 bytes; no model conversion or full copy.
- Pinned tokenizer revision `071ae95e933605ca1104a6b4524a36a98488efa4`; all five sidecar SHA-256 values verified locally.
- SSD expert cache: `/Users/lakshyaprajapati/Library/Caches/Particle/Experts/kquant-v1/`, maximum 4096 MiB of native expert payload plus small metadata. Source changes invalidate cached entries within this budget. SHA-256 validates each loaded entry; corruption falls back to the original GGUF.
- No Git commit is available: this is a downloaded snapshot. Final package/Particle Swift and Metal source fingerprint: `f3564614c9669066b0a3bffda04c4452ed7d54fc2c0240aeee1be21045303a60`. See [file list and fingerprint scheme](validation/particle-source-fingerprint.json).
- All real-model runs were serial, with one model lease. Disk, memory, toolchain, model and tokenizer were checked. No user processes were stopped and no OS caches were purged. Initial free disk was 108 GiB internally and 33 GiB on SD; sampled free memory remained acceptable. Sampled process footprints near 1.75 GiB exclude the total system/file-cache working set.

The earlier four-token and inclusive eight-token runs are historical controls. The final runtime uses the exclusive SSD tier; final tests and full-logit/KV-replay checks include it, along with raw-buffer checksums and bounded temporary lifetimes. Historical controls have the same quantization and decode math. The final release completion record is identified separately below.

## Correctness gates

```sh
Scripts/particle test -c release --filter 'ParticleTests|OptimizationTests'
```

Exit 0: **15 tests passed** (final release build 22.26 sec; tests 6.013 seconds; the raw log is authoritative). The focused suites cover all native quantization formats, independent CPU primitive references, 4096-column blocks and nonzero offsets, incomplete and 128-row batches, distinct gate/up weights, bitwise scalar/batched/fused agreement, router bias semantics, exact top-k ordering, tokenizer fixtures and historical reasoning formatting, expert-cache replacement, SSD capacity/restart/source invalidation/corruption recovery, and prefix-ledger boundaries. This is not a run of every inherited Gemma test.

[Full test output](validation/particle-optimization-tests.log).

```sh
Scripts/particle run --compare-prefill --raw --prompt 'The capital of France is' \
  --prefill-chunk 4 --max-context 64 --expert-cache-slots 12 --verbose
```

Exit 0. The final command runs scalar/batched/scalar, then rewinds from an existing continuation to a two-token prefix and replays the last three prompt tokens. **All four passes have zero bitwise logit mismatches, zero maximum logit difference and zero continuation difference** over the 262,144-token vocabulary. Every pass selects `Paris` (9079), then `.` (236761). The continuation also validates KV state after differently grouped processing.

| Final pass | Reused KV tokens | Prompt processing | Source expert reads | Continuation |
| --- | ---: | ---: | ---: | ---: |
| Scalar first, common mappings cold | 0 | 52.592 s | 0.0 MiB | 1.520 s |
| Batched, 4+1 prompt split | 0 | 6.460 s | 0.0 MiB | 1.360 s |
| Scalar repeat, warmed | 0 | 6.836 s | 0.0 MiB | 1.153 s |
| Rewind and replay final three tokens | 2 | 1.910 s | 0.0 MiB | 1.059 s |
The SSD tier persists experts from RAM before clearing RAM slots for the first three passes; parking time is outside each timed pass. The final replay retains RAM entries. OS caches were not flushed. The final warm batch was about 5.5% quicker than the scalar repeat on this five-token prompt; this is a tiny diagnostic, not a statistically established long-prompt speedup. Cold common-weight faults make the first scalar pass unsuitable as a causal batching baseline. An earlier inclusive-tier comparison measured about 8–10% lower warm prefill wall time; the final exclusive tier trades extra fast SSD writes for greater effective expert coverage.

The initial exclusive-cache comparison exposed temporary allocations surviving RAM clears (about 1.7 → 4.8 GiB). Per-operation/per-park autorelease pools, raw-buffer SHA-256 and one explicit swap allocation fix that path. Final sampled footprints for the four passes were 1752.0, 1754.3, 1754.4, 1755.1 MiB: no per-pass accumulation. Normal eight-token interactive runs also remained near 1.75 GiB.

[Final bounded four-pass JSON](validation/particle-exclusive-bounded-parity.json), [full progress log](validation/particle-exclusive-bounded-parity.log), [earlier inclusive-tier JSON](validation/particle-reuse-parity.json).

These are internal scheduling-equivalence gates against the scalar Sarvam path. Full-checkpoint comparison with an independent inference engine and longer distributional quality evaluation remain separate work.

## Isolated GPU compute

```sh
Scripts/particle run --benchmark-kernels
```

Exit 0. Resident synthetic 4096-column × 1024-output projections, native K-quants, token rows 4/16/128, one warmup pair and three alternating timed pairs; median Metal GPU timestamps. The four-row kernel decodes a weight once and uses four independent FP32 accumulators with the scalar lane/reduction order.

| Format | 4 tokens | 16 tokens | 128 tokens |
| --- | ---: | ---: | ---: |
| Q4_K | 3.22× | 3.30× | 3.94× |
| Q5_K | 2.09× | 4.38× | 3.61× |
| Q6_K | 1.69× | 2.85× | 2.76× |

These are projection speedups, not model tokens/s, and exclude storage. Final batched gate/up additionally fuses SwiGLU and removes separate activation dispatches and unused up scratch. That additional fusion has exact numerical gates; no separate isolated speedup is claimed for it.

[Raw GPU timings](validation/particle-kernel-benchmark.json).

## Eight-token cache-policy gate

Two identical France prompts followed by `/quit` (reproducible input: [prompt lines](validation/particle-completion-prompts.txt)):

```sh
Scripts/particle run --interactive --raw --max-new 8 --max-context 64 \
  --expert-cache-slots 12 --prefill-chunk 128
```

All runs exited 0 and produced byte-for-byte identical output, including the final release. The [cache-policy patch](validation/particle-cache-policy.patch) preserves the inclusive-to-final source change for the historical control; it is not a runtime flag. The first pass populated missing experts; the second reused the warm process.

| Policy / turn | Prefill | Seven decode steps | Decode rate | Decode source expert reads |
| --- | ---: | ---: | ---: | ---: |
| Initial inclusive cache, first | 49.99 s | 31.57 s | 0.222 tokens/s | 835.0 MiB |
| Initial inclusive cache, repeated | 3.07 s | 31.78 s | 0.220 tokens/s | 806.0 MiB |
| Exclusive cache, first | 51.42 s | 31.74 s | 0.221 tokens/s | 697.5 MiB |
| Exclusive cache, repeated | 4.55 s | 8.82 s | 0.793 tokens/s | 0 MiB |
| Final release, partially cached first | 50.79 s | 16.62 s | **0.421 tokens/s** | 242.3 MiB |
| Final release, warm repeat | 4.99 s | 9.73 s | **0.719 tokens/s** | **0 MiB** |

The promoted exclusive tier swaps verified experts into RAM and stores recent RAM evictions on SSD. At most one saved expert buffer is live per serialized swap, under 8 MiB for this model; weight checksums use raw buffers. Current incoming batch entries are protected from eviction. RAM keeps LFU; SSD uses recency, adapting to new routes instead of pinning old high-frequency copies. Explicit RAM clears and shutdown park resident entries back on SSD within the same cap.

During the repeated exclusive turn, SSD read/write traffic was 4227.6 MiB each for the whole turn, with 2687.7 MiB of decode reads. Active SSD payload was 3232.7 MiB plus the existing RAM slots, and sampled footprint was 1753.0 MiB. Shutdown parks RAM entries and restores the disk working set up to 4096 MiB. Extra SSD writes trade fast backing-store traffic for fewer slow SD misses. First-use source reads still dominate when a working set is not cached.

This is one sequential control/candidate diagnostic with different historical cache contents, not a thermally balanced statistical benchmark. The similar first-pass rates and elimination of repeated source reads support the mechanism, but the exact ratio is not a general speedup guarantee. All eight-token runs stop at the output cap, not end of turn.

[Inclusive baseline footers](validation/particle-final-warm-completions.log), [exclusive candidate footers](validation/particle-exclusive-warm-completions.log), [candidate text](validation/particle-exclusive-warm-completions.txt). The final release repeats the check after the temporary-memory fix: it reads 242.3 MiB from the source during its first decode and zero on the repeat, with 1752.9 MiB largest sampled footprint. Active SSD payload is 2775.5 MiB; shutdown parking leaves **4089.5625 MiB in 564 expert files**, independently checked below the 4096 MiB cap. The cache’s historical contents differ between runs, so first-run rates are especially sensitive to prior use.

[Final release footers](validation/particle-release-completions.log), [verbatim release output](validation/particle-release-completions.txt). Final validation also checks that repeated cache clears release temporary memory promptly.

## Earlier short warm completions and changed prompt

One process received these lines on stdin: `The capital of France is`, the same line again, `The capital of Japan is`, `/quit`.

```sh
Scripts/particle run --interactive --raw --max-new 4 --max-context 64 \
  --expert-cache-slots 12 --prefill-chunk 128
```

Exit 0. The SSD cache was already partially populated by the earlier Paris parity run. Raw interactive mode resets KV per prompt while preserving common weights and both expert caches.

| Turn | Prefill | Actual decode steps / wall time | Decode rate | Source expert reads for whole turn | SSD expert reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| France; new process, common weights cold | 49.64 s | 3 / 23.06 s | 0.130 tokens/s | 605.6 MiB | 3433.1 MiB |
| France again; warm process and expert set | 3.42 s | 3 / 2.72 s | **1.103 tokens/s** | 0 MiB | 3182.9 MiB |
| Japan; related changed prompt | 10.54 s | 3 / 10.23 s | **0.293 tokens/s** | 456.2 MiB | 2893.8 MiB |

Outputs were ` Paris.\n\nThe` twice and ` Tokyo.\n\nThe`. All finish at the four-token limit, not end of turn. The footer computes decode rate using the **three actual post-prefill forward passes**, not four sampled outputs. The final sampled token has no KV row until subsequently processed.

The first turn included a one-second stack sample while investigating the remaining SD wait; it is diagnostic, not a clean promotion benchmark. The repeated and changed-prompt turns followed it. SSD payload grew from about 3065 MiB to 3671 MiB after the first turn and 4090 MiB after Japan. Largest sampled footprint: 1752.7 MiB. These short turns used the initial inclusive SSD policy.

[Complete timing footers and progress](validation/particle-warm-completions.log), [verbatim output](validation/particle-warm-completions.txt).

## Warm multi-turn chat

One process received `Hi`, `Hi`, `/new`, `/quit`:

```sh
Scripts/particle run --interactive --max-new 1 --max-context 64 \
  --expert-cache-slots 12 --verbose
```

Exit 0. First prompt processed 12 tokens in 112.59 s; the second reused **all 12 matching KV rows** and processed **16 new/reformatted tokens** in 55.89 s. Source expert reads were 1960.8 MiB and 1605.3 MiB respectively; these unfamiliar chat routes show why four-token repeated completion speed does not generalize automatically. The one-token caps yielded reasoning markers, not completed answers. `/new` succeeded and retained warm model/cache state. No post-prefill decode-rate claim is made for one-token runs.

[Chat log](validation/particle-warm-chat.log), [streamed output](validation/particle-warm-chat.txt).

## Memory and storage interpretation

- Prompt row scratch at the maximum 128-token chunk is **30.59375 MiB** for this installed embedding format, plus transient GPU bindings and ordinary runtime overhead. It is bounded across prompt length; it is separate from KV and expert buffers.
- FP32 KV at context 2048 is about 76 MiB. FP16 could halve that but changes precision and is not enabled silently.
- Twelve RAM expert slots across all MoE layers hold about 1557 MiB. The exclusive tier can use up to 4096 MiB of additional distinct expert payload while the process is warm, instead of duplicating those RAM slots on SSD. Common mappings exclude the 576 MiB embedding table, which is read one packed 2304-byte row at a time.
- Expert-read counters distinguish original GGUF from SSD entries. They exclude embedding rows and common mapped-weight faults, so zero source **expert** bytes does not mean zero total physical SD traffic.
- The expert-load wait includes SSD validation and cache writes as well as demand reads. GPU blocking waits can include mapped-weight faults; use the isolated benchmark for compute timing.
- Interactive aggregate hit/read counters are cumulative; per-turn source/SSD and prefill/decode lines are deltas.

## Performance limits and next work

All runs deviate from the inherited Gemma community benchmark: different model and architecture, short raw/chat prompts, capped outputs, small contexts, explicit warm SSD-tier state, no OS-cache flush, no long steady-state decode or thermal balancing. The user’s SD estimate is 17 MB/s; requested MiB and wall time are not a physical bandwidth measurement. Historical SD-only measurements remain in [the first optimization study](GTURBO_OPTIMIZATION.md).

The 0.3–0.5 tokens/s target is exceeded by the warmed eight-token completion with the exclusive tier; first-use routes and the related changed prompt remain constrained by source reads. It is not guaranteed for arbitrary text, long responses or cache-cold runs. The 6–8 tokens/s future target is unproven. A 4 GiB cache holds only part of the expert pool; unseen routes still need the slow source volume.

Priorities after this pass: longer and unrelated prompt/output gates; larger-cache memory sweeps; native K-quant head/vector loads and persistent routed down/reduction when storage waits shrink; then staged MPP and long-context GQA/TensorOps with independent quality gates. Do not reinstate upstream rejected speculative prefetch, compression, cache-dependent routing or KV codecs simply because they appear in the experiment history.
