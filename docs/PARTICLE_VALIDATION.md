# Particle CLI validation

23 October 2026. This is functional acceptance of the first Sarvam CLI and short SD-card smoke measurements, not a throughput benchmark or full-checkpoint numerical parity claim.

The results below record the original scalar CLI. The current implementation adds grouped prompt processing, embedding-row reads, and a revised cache budget; see the [current optimization study](GTURBO_OPTIMIZATION.md) for numerical parity checks and updated measurements.

## Environment

- Source: local Particle snapshot at `/Users/lakshyaprajapati/Repositories/Particle`. No Git checkout/commit identity is available.
- Hardware: Apple M2, 8,589,934,592 bytes RAM (8 GiB).
- macOS: 27.0.1, build 26A434.
- Xcode: 27.1, build 27A9275, `/Volumes/Particle/Xcode.app`.
- Swift: Apple Swift 6.4, swiftlang-6.4.0.34.1, clang-2100.3.34.1; arm64-apple-macosx27.0.0.
- Metal Toolchain: 27A266a.
- Model: `/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf`, 19,582,216,032 bytes. GGUF v3, 264 tensors: 114 Q4_K, 19 Q5_K, 18 Q6_K, 113 F32.
- Tokenizer: official `sarvamai/sarvam-30b`, revision `071ae95e933605ca1104a6b4524a36a98488efa4`. All five sidecar SHA-256 checks passed.
- Storage: `/dev/disk5s1`, `/Volumes/Particle`, 59 GiB capacity, approximately 33 GiB free before inference. User-reported SD speed is approximately 17 MB/s; no standalone disk-speed benchmark was performed.
- Before the first run, `memory_pressure -Q` reported 45% system-wide free memory. Before the sequential completion test, it reported 65%. No other real-model owner or test helper was running at either launch.

## Build and focused tests

Release compilation succeeded. Final focused command:

```sh
Scripts/particle test -c release --filter ParticleTests
```

Exit 0. Final incremental build: 26.63 seconds. All eight focused tests passed in 4.379 seconds. The inherited targets compiled as part of the test build; the full inherited test suite was not executed.

Coverage includes bounded GGUF parsing/rejection, quantization reference fixtures, all installed quantized GEMVs and embedding gathers, RMS normalization, Q/K normalization plus NeoX RoPE, unmodified V, causal grouped-query attention, router bias/weighting, expert gather/replacement, sampling, and streaming byte fallback. The tokenizer test checks every GGUF vocabulary entry and 22 independent Rust-tokenizers fixtures, including Indic combining-mark edge cases. Chat IDs also match an independent Python Jinja/Rust-tokenizers rendering.

`Scripts/particle run --validate-kernels` exited 0. Maximum absolute GEMV error against the independently unpacked CPU reference ranged from 3.58e-6 to 1.53e-5 across F32/F16/Q4_K/Q5_K/Q6_K fixtures. Norm, rotary, and attention checks passed their explicit tolerances.

Validation caught and fixed a Metal half-bitcast compilation issue and Unicode replacement issues at the tokenizer integration boundary. The adapter preserves binary-distinct vocabulary keys and uses scalar-compatible space normalization without modifying the official sidecars or dependency checkout.

## Chat smoke test

```sh
Scripts/particle run --prompt Hi --max-new 8 --max-context 64 --expert-cache-slots 6 --verbose
```

Exit 0. Greedy sampling (temperature 0), official chat template, no-thinking marker requested. Generated text:

```text
<think>The user just said "Hi"
```

This verifies live chat framing and generation, but the eight-token limit ended during reasoning. It is not a completed greeting. The runtime streams generated text verbatim; the no-thinking marker did not suppress reasoning in this run.

Complete timing footer:

```text
Particle: prompt=12 output=8 finish=length load=3.77s prefill=268.77s first_token=268.78s decode=104.76s
Particle: decode_steps=7 decode_rate=0.067 tok/s
Particle: expert_reads=8698.8 MiB cache_hits=849 cache_misses=1203 peak_sampled_footprint=922.1 MiB
```

The complete progress/output log is [sarvam-sd-first-run.log](validation/sarvam-sd-first-run.log).

## Direct completion smoke test

```sh
Scripts/particle run --raw --prompt 'The capital of France is' --max-new 4 --max-context 64 --expert-cache-slots 6 --verbose
```

Exit 0. Generated text (leading space preserved):

```text
 Paris.

The
```

The first two tokens complete the prompt as “The capital of France is Paris.” The four-token limit then ends during subsequent continuation. Complete timing footer:

```text
Particle: prompt=5 output=4 finish=length load=3.96s prefill=176.68s first_token=176.69s decode=54.23s
Particle: decode_steps=3 decode_rate=0.055 tok/s
Particle: expert_reads=4614.8 MiB cache_hits=226 cache_misses=638 peak_sampled_footprint=954.9 MiB
```

The complete progress/output log is [sarvam-sd-completion.log](validation/sarvam-sd-completion.log).

## Interpretation and limits

The CLI integrates the GGUF, tokenizer, Metal graph, expert storage/cache, sampling, streaming output, and resource lifecycle against the actual installed model. These short runs establish a working baseline, not equivalence to an independent full-model engine. Per-layer and final-logit comparisons with such an engine remain future work.

The first chat test was slower than the temporary 0.5–1 token/s aspiration. The 6–8 token/s internal-SSD goal is still unverified. No throughput ceiling can be inferred from these short SD runs.

Protocol differences from the inherited Gemma community benchmark: this uses the authorized Sarvam GGUF and Particle preflight rules, tiny fixed prompts, six cache slots, context 64, verbose progress, and no warmup/repeated performance sample. There was no cache flush; the second process may benefit from filesystem caching. One model-owning process ran at a time. No full checkpoint was downloaded, duplicated, or requantized.

Rates exclude prompt processing and count actual post-prefill forward steps. First-token latency is measured after runner initialization; add `load` for time from CLI start. Expert bytes are requested GGUF ranges, not measured physical disk traffic. Process footprint is sampled at token boundaries and is neither a continuous peak nor the sum of all OS file-cache/GPU/system allocations.

See the [implementation plan](IMPLEMENTATION_PLAN.md) for remaining optimization work and the [CLI guide](PARTICLE_CLI.md) for usage.
