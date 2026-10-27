# Particle CLI

ParticleCLI runs the installed Sarvam 30B Q4_K_M GGUF directly with Swift and Metal on Apple Silicon. It keeps common tensors mapped and gathers only selected experts into bounded Metal buffers. It keeps the original model in place; an optional bounded SSD cache copies only requested expert bytes.

## Build and run

From the Particle project folder:

```sh
Scripts/particle doctor
Scripts/particle build -c release --product ParticleCLI
Scripts/particle run --prompt "Say hello." --max-new 64 --max-context 128 --expert-cache-slots 12
```

The helper selects Xcode, model, build outputs, and compiler caches on `/Volumes/Particle`. The approved expert data cache uses the internal SSD by default. Keep the card mounted. The CLI prints generated text to stdout and progress/timings to stderr. `--verbose` shows per-layer progress when storage is slow. Control-C cancels after current I/O and GPU work complete. One Particle inference process can hold the model lease at a time.

```sh
Scripts/particle run --inspect
Scripts/particle run --tokenize --prompt "नमस्ते"
Scripts/particle run --validate-kernels
Scripts/particle test -c release --filter 'ParticleTests|OptimizationTests'
Scripts/particle run --benchmark-kernels
```

Inspection reads metadata and validates the model configuration; it does not execute the graph. Tokenization checks the full GGUF vocabulary against the official tokenizer before emitting IDs. Kernel validation compares Metal results with independent CPU calculations without loading model weights.

## Keep the model warm

```sh
Scripts/particle run --interactive --max-new 64 --max-context 512
```

Enter another message at `You>`. The process keeps common weights, expert slots and the SSD cache warm. Chat history uses Sarvam’s official template. Particle compares the exact new prompt IDs with completed KV rows, keeps their common prefix, and recomputes only the changed suffix. Historical reasoning formatting can change, so reuse does not assume every turn is a simple append. The last sampled token at a length limit is processed on the next turn if needed.

`/new` clears the conversation and KV state while retaining warm weights and experts. `/quit` exits. `/help` lists these commands. Control-C during generation cancels the current turn and invalidates partial KV state; the next message replays previously completed history. EOF also exits. At the context limit use `/new` or restart with a larger `--max-context`; old history is not silently discarded.

`--interactive --raw` keeps the model warm for **independent** completions. Each prompt starts at position zero and keeps both expert caches. An optional `--prompt` supplies the first message. Interactive mode cannot combine with `--prompt -`, `--tokenize`, or `--compare-prefill`.

## Choose the RAM expert cache

`--expert-cache-slots <n>` accepts every integer from **6 to 16**, with **12** as the default. The value applies separately to each of Sarvam's 18 MoE layers. Set it at launch; restart the process to change it.

| Slots per MoE layer | Approximate total expert-buffer capacity | Choice |
| --- | ---: | --- |
| 6 | 0.76 GiB | Smallest supported RAM cache |
| 8 | 1.01 GiB | Lower RAM usage |
| 12 | 1.52 GiB | Default |
| 16 | 2.03 GiB | More room for reused experts |

```sh
Scripts/particle run --interactive --expert-cache-slots 6
Scripts/particle run --interactive --expert-cache-slots 12
Scripts/particle run --interactive --expert-cache-slots 16
```

These capacities describe only expert buffers for the installed GGUF. Common weights, KV, prompt scratch, file caches and system memory are additional. More slots may reduce disk reads but do not guarantee higher token speed, especially under memory pressure. Six is the minimum because the current implementation pins all six selected experts together. This setting changes neither conversation retention nor the separate SSD-cache cap. Compare settings on the same prompts and cache conditions before choosing a larger value.

## Bounded internal SSD cache

The default expert cache is `~/Library/Caches/Particle/Experts` with a 4096 MiB (4 GiB) payload cap. It is a demand-filled cache, so the first use still reads missing experts from SD. Later RAM misses can read a single contiguous gate/up/down entry from SSD. RAM keeps its LFU policy. The SSD tier stores recent RAM evictions, and a verified SSD hit moves into RAM instead of keeping a duplicate copy. Incoming batch entries are protected during swaps. RAM entries are parked on SSD before deliberate cache clears or process exit, within the same disk cap. This never changes the router’s choice. Entries preserve native quantized bytes, bind to source file identity, and verify SHA-256 before use. Invalid entries fall back to the GGUF. A cache write failure is reported and subsequent inference continues from the model.

```sh
# Pure original-volume reads; no SSD entry creation or loading.
Scripts/particle run --interactive --no-ssd-cache --max-context 512

# Smaller cache, or an explicitly chosen fast cache location.
Scripts/particle run --interactive --ssd-cache-mib 2048
Scripts/particle run --interactive --ssd-cache /absolute/path/ParticleExperts
```

The payload cap excludes small metadata. A model identity change invalidates old entries in its cache namespace. The original GGUF remains authoritative. This is not a full-model conversion or a `.gturbo` Gemma install. The directory may be removed while Particle is stopped; doing so loses warm-start benefits. Do not put this cache on SD when measuring SSD-tier gains.

## Inputs and settings

Use `--prompt-file /absolute/path/prompt.txt` or `--prompt -` for UTF-8 file/stdin input (at most 1 MiB). `--system` supplies optional system guidance. Chat uses the official Sarvam Jinja template and requests no thinking by default; `--thinking` changes that request. The tested checkpoint still emitted reasoning with the no-thinking marker. Output is shown verbatim, and reasoning counts against `--max-new`, so a short limit can end before the answer.

`--raw` provides unframed completion and cannot be combined with system/thinking options. A shorter smoke test is:

```sh
Scripts/particle run --raw --prompt "The capital of France is" --max-new 4 --max-context 64 --expert-cache-slots 12
```

Greedy sampling is the default (`--temperature 0`). Positive temperatures enable seeded sampling with `--top-k`, `--top-p`, and `--seed`. EOS and end-of-turn tokens stop generation. `--max-new` limits output tokens, including a stop token if selected.

The model path can be overridden using `--model` or `PARTICLE_MODEL_PATH`. `--tokenizer` selects a directory containing the official tokenizer sidecars. Otherwise it uses a `tokenizer/` directory beside the GGUF. To fetch just the pinned tokenizer metadata again:

```sh
python3 Scripts/fetch_sarvam_tokenizer.py --help
```

That script never downloads model weights. Model tokens must agree with the official vocabulary. A different model, split GGUF, unsupported quantization, or incompatible tensor layout fails with an error.

## Compare prompt processing

```sh
Scripts/particle run --compare-prefill --raw --prompt "The capital of France is" \
  --prefill-chunk 4 --max-context 64 --expert-cache-slots 6
```

This runs scalar/grouped/scalar in one process, clears RAM expert slots for the first three passes, then rewinds and replays a suffix in a fourth pass. It compares full logits and one continuation step for every pass, and prints JSON measurements. It uses one extra context position and ignores the output-token allowance. It can take several minutes on SD. The SSD cache is retained and filesystem caches are not flushed, so read the ordering note before interpreting timing differences.

## Memory and speed

Prompt processing reuses expert weights over chunks of 128 tokens by default and batches GPU projections across rows. `--prefill-chunk` accepts 1–256; sizes above 128 are experimental. A chunk of 1 selects scalar processing. Row scratch persists across calls and uses about 30.6 MiB at the default 128 rows, or 61.2 MiB at 256. Embeddings are read one packed row at a time. Activations and KV cache remain FP32.

Each MoE layer keeps 12 expert slots by default; `--expert-cache-slots` accepts 6–16. The smaller mapped embedding footprint funds the larger default cache within a similar planned weight budget. Context is 2048 by default and accepts 32–8192; prompt plus output allowance must fit. These are allocation limits, not a promise that every combination fits an 8 GB machine. Start with context 128; six expert slots remain available under memory pressure.

The footer reports load time, prompt processing time, first-token latency from input submission in interactive mode, post-prefill decode rate, expert bytes read, cache hits/misses, and sampled process footprint. Extra lines separate prompt/decode bytes and blocking read/GPU waits. Expert-load wait includes SSD validation/writes as well as demand reads. GPU wait can include file-backed page faults; it is not an isolated kernel-time measurement. Grouped cache counters count physical expert requests, with each request potentially serving several token rows. A one-token run has no decode forward step and reports its rate as unavailable. Footprint samples are not a continuous peak-memory trace or the sum of all OS file-cache/system allocations. Expert read bytes count all requested storage ranges; separate counters distinguish source GGUF and SSD entry reads/writes. Embedding rows and common tensor faults are outside the expert-source counters, so these bytes are not all physical SD traffic; filesystem caching can change physical SD traffic.

Cold expert misses and resident-weight page faults can dominate the 17 MB/s SD card. Actual speed must be measured; neither 0.5–1 token/s on SD nor 6–8 token/s on an internal SSD is guaranteed. Batched native K-quant projections and lazy SSD expert packing are implemented. Staged MPP, TensorOps full attention, persistent multi-expert reduction, split attention and vectorized K-quant heads remain candidates; they are not claimed as completed. The [optimization study](GTURBO_OPTIMIZATION.md) records why the SD-card bandwidth still dominates.

This version supports single prompts and warm multi-turn text sessions. The inherited Gemma GUI/server still use the legacy model format. They do not serve Sarvam.

See the [validation record](PARTICLE_VALIDATION.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [codebase analysis](CODEBASE_ANALYSIS.md).

See the [complete upstream-method audit](PARTICLE_OPTIMIZATION_AUDIT.md) and [current performance record](PARTICLE_PERFORMANCE.md) for measured gains and remaining work.

## New prefill and generation controls

Idle historical-prefix preparation is enabled in chat mode; `--no-idle-preparation` disables it. New input interrupts idle model work safely. All conversation turns are preserved within the context limit. See [the current implementation, experimental acceleration controls, timing fields and benchmark commands](PARTICLE_PREFILL_OPTIMIZATION.md).
