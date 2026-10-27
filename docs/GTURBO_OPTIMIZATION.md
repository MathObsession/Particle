# What `.gturbo` does, and the first Particle optimization

> Historical first optimization pass. Current implementation and results: [Particle performance](PARTICLE_PERFORMANCE.md) and [complete method audit](PARTICLE_OPTIMIZATION_AUDIT.md).

23 October 2026. This study traces the local TurboFieldfare implementation and applies its prompt-weight reuse design to Sarvam. It does not convert the installed model to Gemma's format.

## The installer prepares a layout

`.gturbo` is a model directory. The installer reads configuration and safetensors headers, plans the destination, and streams bounded HTTP byte ranges into it. It does not need a second complete source checkpoint. The main pieces are a manifest, indexed common weights in `model_weights.bin`, and page-aligned files in `packed_experts/`.

The expert writer rearranges tensors into expert-major order: an expert's gate/up/down weights and quantization companions occupy one contiguous blob. Each blob is padded to 16 KiB. The runtime can therefore fill one Metal-visible cache slot with one `pread` per missing expert. Parallel reads load the selected misses while shared-expert GPU work runs. LFU retains frequently selected experts.

The current source is already quantized MLX affine Gemma. The writer copies its packed values and scale/bias components; it is not a generic GGUF-to-affine quantizer. The current Gemma loader, manifest checks, tokenizer, and kernels cannot execute Sarvam merely by changing the directory suffix.

Relevant source:

- `Sources/TurboFieldfareRepack/Core/Remote/RemoteStreamingRepacker.swift`: resumable, bounded download/repack transaction.
- `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift`: resident ordering, expert-major components, and 16 KiB strides.
- `Sources/TurboFieldfareRepack/Core/Writing/WriterCore.swift`: bounded byte copies and hashing.
- `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift`: fixed Metal slots, LFU, parallel whole-expert reads.
- `Sources/TurboFieldfare/Infrastructure/ModelIO/ResidentBuffer.swift`: resident mapping and random-access advice.
- `Sources/TurboFieldfare/Runtime/Prefill/`: layer-major prompt processing and grouped expert consumers.

## Where Particle was losing reuse

The first Particle CLI ran every prompt token through all 19 layers before starting the next token. With only a few expert slots per layer, the same expert could be read repeatedly while processing one prompt. Its GGUF stores gate, up, and down expert tensors separately, so one miss needs three reads.

Sarvam's common tensors total approximately 2050.8 MiB. Across its 18 MoE layers, loading six previously uncached experts per layer requires approximately 778.5 MiB. Contiguous packing can reduce the number of reads and seeks, but the same selected quantized weights still have to cross the storage connection. At the user-reported 17 MB/s, that cold-miss volume alone is roughly 48 seconds. Sustainable speed depends on cache reuse as well as layout.

TurboFieldfare also has optimized affine matrix kernels and grouped/chunked prompt processing. Its throughput is not a property of the filename alone. The local experiment records report failed or mixed results for speculative expert reads, larger caches under 8 GB pressure, compression, and pinning resident pages; those controls were not copied blindly.

## Implemented changes

| Area | First Particle CLI | Updated Particle |
| --- | --- | --- |
| Prompt schedule | Complete one token, then the next | Complete one layer over a small token chunk |
| Routed expert use | Request each token's selected set | Load each distinct expert once per layer/chunk and serve its rows |
| Numerical accumulation | Original top-six rank order | Same rank order, using separate bounded result buffers |
| Cache accounting | One use per request | Frequency counts include every row served by a grouped expert |
| Common mapping | Full embedding matrix and other common tensors | Read embedding rows into tiny buffers; retain random-access advice on the other mappings |
| Temporary GPU objects | Function lifetime | Autorelease pool per forward step or prompt chunk |
| Diagnostics | Total requested bytes and total times | Separate prompt/decode requested bytes and blocking read/GPU waits |

The default prompt chunk is 16 tokens; `--prefill-chunk 1` retains the scalar reference. Chunks up to 128 use under 19 MiB of extra activation/embedding scratch for the installed Q4_K table. Expert capacity remains 6–16 slots, with a new default of 12. The embedding-row change funds the increased slot capacity within a similar planned weight budget; the measured private footprint and planned mapped weight sizes are reported below.

The GGUF and quantized kernels remain unchanged. Expert misses still use three GGUF ranges. No duplicate model or packed checkpoint was created. This first change targets repeated reads and first-token latency; steady-state decode still has one newly generated token at a time and needs separate measurement.

## Validation and measurements

Release compilation and all eleven focused tests pass (final build 30.44 seconds; tests 4.646 seconds; exit 0), including a multi-block 4096-column embedding-row fixture and separate per-row GPU buffers. Host: M2 / 8 GiB, macOS 27.0.1 (26A434), Xcode 27.1, Swift 6.4. There is no Git commit identity for this downloaded snapshot. The final package/Particle Swift and Metal source fingerprint is `695e4189a78d1599c5a9b6101e0d6f5dd6478c062956cec096f6a951ec8dbc1d` (SHA-256 of sorted relative paths and file bytes, NUL separated).

### Prefill-only numerical and reuse comparison

This phase preceded the embedding-row change. One process ran scalar/grouped/scalar, clearing expert slots each time while retaining ordinary filesystem caching:

```sh
Scripts/particle run --compare-prefill --raw --prompt 'The capital of France is' --prefill-chunk 4 --max-context 64 --expert-cache-slots 6 --verbose
```

Exit 0. All 262,144 final logits were bitwise identical, and a subsequent forward step also had zero maximum logit difference. All passes selected `Paris` (ID 9079) and then `.` (ID 236761). The 4+1 row split exercises a partial chunk and continuation across its boundary.

| Pass | Prefill wall time | Expert bytes requested | Blocking expert reads | Blocking GPU waits |
| --- | ---: | ---: | ---: | ---: |
| Scalar first | 170.03 s | 3065.7 MiB | 101.35 s | 68.53 s |
| Grouped, four rows | 109.32 s | 2850.8 MiB | 94.08 s | 14.89 s |
| Scalar repeat | 109.04 s | 3065.7 MiB | 99.54 s | 9.28 s |

Grouping reduced expert bytes by 7.01%. It did **not** beat the warmed scalar repeat on this tiny prompt. The cold-versus-warm time difference cannot be attributed to grouping. [Full measurements](validation/prefill-comparison.json) and [progress/parity log](validation/prefill-comparison.log) are retained.

### Final row-reader and cache configuration

Sarvam has an untied output head. The full 576 MiB Q4_K embedding tensor was mapped as a GPU resource for row lookups. The new reader loads only the requested 2304-byte packed row, preserving its quantized bits. Each prompt row has its own buffer, avoiding overwrite before GPU consumption.

The default expert cache changes from eight to twelve slots per MoE layer. Planned payload sizes, excluding page alignment, KV, activations, and runtime overhead:

| Weight budget | Previous default: eight slots | Updated default: twelve slots |
| --- | ---: | ---: |
| Mapped common tensors | 2050.8 MiB | 1474.8 MiB |
| Expert-slot capacity | 1038.0 MiB | 1557.0 MiB |
| Combined | 3088.8 MiB | 3031.8 MiB |

This is a planned working-set comparison, not a claim that every mapped byte was physically resident. Private expert allocations increase while mapped embedding resources shrink. The final smoke test used the new defaults explicitly:

```sh
Scripts/particle run --raw --prompt 'The capital of France is' --max-new 4 --max-context 64 --prefill-chunk 16 --expert-cache-slots 12 --verbose
```

Exit 0. Output matched the original four-token completion byte for byte: ` Paris.`, two newlines, then `The`. Complete footer:

```text
Particle: prompt=5 output=4 finish=length load=5.00s prefill=138.58s first_token=138.59s decode=51.49s
Particle: decode_steps=3 decode_rate=0.058 tok/s
Particle: expert_reads=4038.8 MiB cache_hits=146 cache_misses=558 peak_sampled_footprint=1749.5 MiB
Particle: prefill_chunk=16 prefill_expert_reads=2744.4 MiB prefill_read_wait=92.35s prefill_gpu_wait=46.14s
Particle: decode_expert_reads=1294.3 MiB decode_read_wait=42.38s decode_gpu_wait=8.69s
```

[Full optimized run](validation/sarvam-sd-optimized.log).

Compared with the [original scalar/six-slot smoke run](validation/sarvam-sd-completion.log), first-token latency after runner creation changed from 176.69 to 138.59 seconds (21.56% lower), and total requested expert bytes fell from 4614.8 to 4038.8 MiB (12.48% lower). Decode changed from 0.055 to 0.058 token/s, which is essentially unchanged at this three-step sample size. These are short runs with different cache settings and uncontrolled filesystem state; they do not establish a repeatable throughput improvement.

Memory remained acceptable during this M2 run: the observed system-wide free-memory report was 34% during prompt processing, with no other model process running. Sampled process footprint was 1749.5 MiB; it excludes some file-backed/system memory and must not be read as total machine RAM use. Before this run the free-memory report was 39%; after process exit it was 70%.

Expert cache filling accounted for 42.38 / 51.49 seconds (82.3%) of decode time, with 1294.3 MiB requested for three steps, or roughly 431 MiB/step. GPU wait measures CPU blocking until completion and can include resident-file page faults; it is not isolated kernel execution time. This identifies weight I/O as the immediate limit. The directory suffix cannot remove those bytes.

These are authorized Sarvam diagnostics, with tiny completion prompts and bounded caches, rather than the inherited Gemma community end-of-turn benchmark. Power/thermal conditions and other desktop activity were not controlled, caches were not purged, and the full inherited test suite was not run. All model processes were sequential. No weights were downloaded, duplicated, or requantized for this work.

## Next decisions

Use the new read/wait attribution to decide the next change. A Sarvam-specific lossless expert pack could reduce three reads per miss to one, with checksums, explicit GGML K-quant layouts, and transactional installation. It must preserve Q4_K/Q5_K/Q6_K bytes and Sarvam's graph; the Gemma v1 affine loader is not suitable.

For future downloads, stream directly into that prepared layout instead of retaining both a full GGUF and a full packed copy. For the already installed artifact, compare bounded representative gather/packed I/O and then full-model behavior before spending the time and storage on a conversion. Repacking cannot remove the physical bandwidth required for cold weights.

Other candidates are larger prompt chunks for long inputs, shape-specific quantized matrix kernels, and a memory-budgeted cache capacity study. Promote them only after full-logit/continuation checks and measured end-to-end gains. The 6–8 token/s internal-SSD target remains unverified.
