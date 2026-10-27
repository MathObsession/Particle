# Particle prefill and generation: M2 implementation

24 October 2026. The original Sarvam GGUF stays on SD. Conversation turns are preserved within the configured context limit; no automatic summarization, truncation, sliding window, expert pruning, or KV quantization is introduced. The official Sarvam template still determines which historical reasoning text enters the model.

## Run

The existing command keeps validated math and warm weights/KV:

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/particle run --interactive --max-context 2048
```

Explicitly select the new experimental acceleration paths:

```sh
Scripts/particle run --interactive --max-context 2048 \
  --attention fast --projection vector --gpu-greedy
```

`fast` chooses split-KV attention for up to 16 new rows beyond one KV block; FP32 TensorOps attention for at least 32 new rows and 512 total positions; exact grouped attention for multiple rows at 128 or more total positions; otherwise the scalar reference. It compiles optional matrix pipelines once at startup. `--attention auto` retains the reference until whole-model promotion gates are satisfied; it is intentionally distinct from experimental `fast`.

Other attention choices are `reference`, `tiled`, `split`, and `tensor`. `--split-kv-block 256|512` controls splitting. Explicit `tensor` applies it to decode too, which is slower than splitting on the measured M2; use `fast` to select by shape.

Projection choices are `auto`, `reference`, `vector`, and `staged`. `vector` reuses native K-quant block metadata while retaining FP32 accumulation order. `staged` uses 8 KiB of threadgroup tiles and FP32 matrix accumulation, with half-rounded inputs and weights. The gate/up matrix path fuses SwiGLU without allocating a full expanded weight or activation matrix. Outside eligible matrix batches, staged mode uses the exact vector metadata path for scalar and small-row projections. Half staging is a distinct numerical path and remains experimental, even when its primitive tests pass. Unsupported explicit matrix pipelines fail clearly; ordinary reference execution does not require them.

`--gpu-greedy` returns a compact GPU argmax at temperature zero, preserves lowest-token-ID ties, and rejects any non-finite vocabulary logit. Positive-temperature sampling retains the existing deterministic CPU sampler.

## Session ownership and idle preparation

One model worker owns the runner, caches and KV lineage. A separate bounded input mailbox reads lines and signals idle cancellation when a line arrives or stdin ends. It does not cancel foreground generation.

After a reply, Particle renders only that assistant reply in its canonical historical form using the official template with a future-user boundary. It joins this to the already verified prompt prefix, avoiding another render/tokenization of older turns during idle work too. It keeps the exact matching token prefix and evaluates only the changed tail, without computing an unused output head. The installed sidecar is not edited. The tokenizer/template/config fingerprints must match the pinned official assets before incremental encoding is allowed. With that verified boundary, the cached historical token prefix is joined to just the newly rendered user turn; earlier conversation text is not tokenized again on the foreground path. Unknown asset revisions fall back to full rendering and disable idle canonicalization; exact model-KV prefix matching still applies. Exact-token equivalence tests include Unicode and reasoning markers. If input is already queued or idle work is disabled, the next foreground turn canonicalizes only the pending last reply from its saved prompt prefix; it still avoids re-tokenizing older turns.

Idle work uses checkpoints of eight tokens by default (`--idle-prefill-chunk 1...128`), capped by the foreground chunk setting. Smaller checkpoints retain useful progress when input interrupts a long replay. Every completed chunk commits its token ledger. An interrupted chunk may have written future KV positions, but those positions are masked and overwritten on subsequent use. Cancellation recovery preserves only the last fully completed position. GPU/I/O failures require reset. No second KV copy is created. The next message's exact prefix check remains authoritative even after idle preparation.

Idle preparation is enabled in chat mode by default; `--no-idle-preparation` disables it. Work moved to idle time is reported separately, not claimed as eliminated computation. `/new` clears transcript/KV and keeps warm expert caches; `/quit` exits. A rejected oversized prompt preserves the previous conversation and valid KV. Raw interactive mode continues to represent independent completions.

## GPU and expert I/O

- Batched prefill writes all chunk K/V before attention; each query masks its absolute future positions. Tracked resource ordering between encoders makes writes visible to attention.
- Exact grouped attention shares a 32-position K/V tile across four heads and two query rows, retaining the scalar per-position FP32 order.
- Split attention evaluates independent 256/512-position KV blocks and merges max/sum/weighted-value statistics. Splitting is limited to at most 16 query rows, keeping scratch below 8.3 MiB at the maximum context.
- FP32 TensorOps attention groups four query rows and all 16 Q heads sharing one KV head. It performs QK/PV matrix operations and stable online softmax over the complete history. Reused threadgroup workspace is approximately 25 KiB; there is no full attention-score matrix.
- A fetch transaction reserves all requested expert slots, exposes ready hits, and reads missing experts asynchronously. Ready outputs and the shared branch may run while reads proceed. All selected weights remain pinned until their consumers finish; top-six rank outputs are combined in their original order.
- SSD reads and hashes run outside the metadata lock. Entries are pinned against eviction, and writes reserve physical payload space before creating temporary files. At most two saved victims are live, under 16 MiB for this model. The existing 4096 MiB payload cap, source identity, SHA verification, corruption fallback, and exclusive victim-cache semantics remain.
- `--source-read-workers 1|2|4` and `--ssd-read-workers 1|2|4` expose independent limits. Default limits are four source workers and two SSD workers. Two live SSD exchanges remain the memory bound even with four admission workers.
- `--coalesce-reads` groups only known demand ranges, with at most 10% extra bytes and one staging allocation of at most 8 MiB. The coalesced source path is serial; additional workers do not multiply staging buffers. Individual source reads use the selected worker count.
- `--serial-expert-loads` is a comparison control that waits for misses before starting resident expert work.
- Layer and expert-batch autorelease pools release completed command/binding objects promptly, without reusing argument buffers still owned by GPU commands. Prompt scratch persists across calls. The default chunk remains 128 (about 30.6 MiB of row scratch); 256 is experimental (about 61.2 MiB). Split scratch is released for larger query batches. Packed expert buffers, shared/common mappings, and KV allocations retain their existing bounds. Additional execution scratch is designed to stay below 96 MiB for the installed model; this is a logical allocation budget, not a measured whole-system memory guarantee.

`--hybrid-prefill` enables the experimental two-worker CPU/GPU expert-row queue. GPU consumers claim up to four independent rows, CPU consumers claim one and may claim more after completion. Native weights are shared and pinned; CPU expansion is bounded to 32 output rows per worker. Inputs must already be complete. A layer cannot advance until all consumers finish. The CPU path uses Accelerate and changes reduction order, so it needs model parity and end-to-end timing gates before promotion. Future decode tokens remain autoregressive.

## Timings and reproducible checks

```sh
Scripts/particle run --interactive --attention fast \
  --timings-json /tmp/particle-timings.jsonl
Scripts/particle run --compare-prefill --raw --prompt 'The capital of France is' \
  --attention tensor --projection vector --prefill-chunk 4 --max-context 64
Scripts/particle test -c release --filter 'ParticleTests|OptimizationTests|AttentionOptimizationTests|ProjectionOptimizationTests'
```

JSONL reports distinguish reused KV, history replay, new prompt tokens, template time, prompt/decode time, first-token latency, source/SSD bytes, cache payload, and sampled footprint. Attention profiling adds explicit GPU command boundaries, so it can perturb throughput; benchmark without profiling for promotion. First-token latency in interactive mode starts when the input line is received, including any wait for an interrupted idle operation. Read-wait counters measure exposed waiting at transaction completion, not total background service time. Source-expert byte counts exclude common mapped-weight page faults and embedding-row reads.

```sh
Scripts/particle-interactive-check
Scripts/particle-performance attention
Scripts/particle-performance kernels --projection vector
Scripts/particle-performance kernels --projection staged
Scripts/particle-performance hybrid
Scripts/particle-performance io
Scripts/particle-performance smoke --attention tensor --projection vector
Scripts/particle-performance session --attention fast --projection vector
Scripts/particle-performance full --attention fast --projection vector
```

The interactive regression uses short replies to exercise idle cancellation, an oversized message, retained KV, Control-C and `/new` without a second model process. `PARTICLE_INTERACTIVE_MAX_NEW`, `PARTICLE_INTERACTIVE_PROMPT`, `PARTICLE_INTERACTIVE_DELAY` and `PARTICLE_INTERACTIVE_LOG_BASE` customize its diagnostics.

The harness records commands, environment, source SHA-256 values, raw output, cache-policy limitations and exit status. Attention benchmarks are resident synthetic KV tests. I/O benchmarks compare 1/2/4 workers and bounded coalescing with six synthetic demanded experts; they exclude checksum/scatter/GPU overhead, preserve OS caches and cannot establish cold-media bandwidth.

Session benchmarks extend one real KV lineage, rewind only the suffix for alternating control/candidate passes, and compare full first/final logits plus generated token IDs. Default full configuration: histories 512/2048/4096/8000, suffixes 1/16/64/128, three AB/BA rounds, and 64 decode forward passes. The longest case fills 8192 positions. Stop IDs are deliberately processed to measure a fixed number of forward passes; this is not a natural-response completion benchmark. Long cases read substantial model data from SD and are not replaced by synthetic KV when reporting model results.

Promotion requires strict scalar parity, at least 10% median end-to-end improvement on target workloads, and no tested regression over 5%. Primitive speedups alone do not promote defaults. New numerical paths remain explicitly experimental when those gates are incomplete or fail. FP16 KV, cache-dependent routing, speculative future-layer reads and upstream argument-buffer rings are not enabled.

## Evidence

The [24 October validation report and raw evidence](validation/particle-prefill-20261024/README.md) records 31 focused tests, actual model parity, interaction checks, and the measured limits. A repetitive 64-token completion reached 0.495 tokens/s; two varied chat replies reached 0.200 and 0.221. Resident attention kernels improved by up to about 5.3×, while CPU/GPU sharing was slower and stays disabled. Full real-model long-context promotion gates remain open. Historical eight-token numbers remain in [the previous performance report](PARTICLE_PERFORMANCE.md); they are not measurements of this implementation.

Primary design sources: [TurboFieldfare prefill experiments](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/06-prefill.md), [system design](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md), [FlashAttention](https://arxiv.org/abs/2205.14135), and [FlexInfer](https://proceedings.mlsys.org/paper_files/paper/2025/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf). Server/discrete-GPU results are not M2 measurements.
