# Upstream codebase analysis and Sarvam port context

Reviewed 23 October 2026 against the upstream snapshot. The implementation now lives directly in the Particle project root (`Sources/`, `Tests/`, and `Package.swift`); the former `turbo-fieldfare-main/` and `engine/` nesting has been removed. This review describes the original Gemma implementation, before the Sarvam port.

The Sarvam integration is now implemented separately in `ParticleGGUF`, `ParticleCore`, and `ParticleCLI`. The historical findings below explain why the Gemma graph was kept separate. See the [implementation plan](IMPLEMENTATION_PLAN.md) and [CLI guide](PARTICLE_CLI.md) for the current implementation and its limits.

## Scope and evidence

This is a repository-wide structural review with targeted implementation tracing through installation, format validation, model loading, decode, prefill, GPU kernels, tokenization, generation, app/service integration, server, tests, and experiment records. It is not a line-by-line correctness audit or a measured performance assessment.

The source tree contains 275 Swift files and 14 Metal files, roughly 62,500 lines including source-side text resources. Tests contain 272 Swift files, roughly 54,200 Swift lines. There are 25 documentation files/assets and nine script files. No `.build` directory or completed `scratch/gemma4.gturbo` model was present. Neither the workspace root nor the nested project was recognized as a Git checkout, so there is no verified commit identity for this snapshot.

No source, defaults, dependencies, or model artifacts were changed. No build, package tests, model download, or inference was run. Numerical correctness and documented throughput were not independently verified. Conclusions below describe the local implementation; they do not assume an architecture for the future target model.

## Main conclusion

This is a purpose-built Gemma 4 26B-A4B IT inference engine for Apple Silicon, not a general MoE engine and not an existing Sarvam implementation. Searches found no Sarvam implementation in the source. A new model requires a port of its computation and checkpoint contract, not just different dimensions or a repository URL.

The strongest reusable foundation is the bounded-memory system: mapped common weights, SSD-backed expert files, Metal-visible cache slots, asynchronous reads, bounded prompt processing, and explicit GPU ownership. The least reusable parts are Gemma's layer graph, routing semantics, fused operations, checkpoint naming, and tokenizer/output grammar.

## Architecture and execution map

| Area | Responsibility | Main implementation |
| --- | --- | --- |
| Format | Foundation-only v1 manifest, resident index, packed expert layout, vision contract | `Sources/TurboFieldfareFormat/` |
| Repack | Pinned remote source, safetensors metadata, byte-range planning, resumable writes, receipts | `Sources/TurboFieldfareRepack/Core/` |
| Model I/O | Validate manifest and tensors, map resident bytes, verify streamed layers | `Sources/TurboFieldfare/Infrastructure/ModelIO/` |
| Expert I/O | Fixed slots, LFU/LRU planning, parallel `pread`, optional read advice | `Sources/TurboFieldfare/Infrastructure/Streaming/` |
| Inference | Layer graph and coordination of resident and streamed computation | `Sources/TurboFieldfare/Runtime/Inference/` |
| Prefill / KV | Bounded chunk scratch, tiled expert work, FP16 cache and continuation | `Sources/TurboFieldfare/Runtime/Prefill/`, `Runtime/KVCache/` |
| GPU | Swift dispatch wrappers and runtime-compiled Metal shaders | `Sources/TurboFieldfare/Kernels/`, `Metal/` |
| Generation | Completion loop, sampling, stops, cancellation, conversation lineage | `Sources/TurboFieldfare/Runtime/Generation/` |
| Tokenization | Hugging Face tokenizer wrapper, Gemma chat framing, lossless decode, tool parsing | `Sources/TurboFieldfare/Tokenization/` |
| CLI | Installation-independent command-line generation | `Sources/TurboFieldfareCLI/` |
| Mac app | Install, history, settings, presentation, service client | `Sources/TurboFieldfareApp/` |
| Decode service | Separate model-owning process and serialized command protocol | `Sources/TurboFieldfareDecodeService/`, `TurboFieldfareDecodeProtocol/` |
| Server | Loopback OpenAI-compatible API, streaming, serialized generation, prefix reuse | `Sources/TurboFieldfareServer/` |
| Validation | Independent CPU primitive references, tolerances, deterministic fixtures | `Sources/TurboFieldfareValidation/Support/`, `Tests/` |

The package declares Swift 6.2 and macOS/iOS 26. Its shipped user interfaces and documented validation target Macs. Dependencies supply tokenizers/Hub access, SwiftNIO networking, and SwiftMath presentation; the transformer runs through custom Swift/Metal code rather than MLX or llama.cpp.

### Install and load

`SupportedModelSource.swift` pins an MLX affine 4-bit Gemma checkpoint, revision, index hash, and size estimates. `SourceFingerprint.swift` recognizes that source. `ArchInfo.swift` reads Gemma-specific `text_config` fields. `RepackPlanner.swift` recognizes the `language_model.` namespace and `.experts.switch_glu.` expert tensors.

The repacker copies existing quantized values, scales, and biases into a common resident file and page-aligned per-layer expert blobs. It does not provide a general checkpoint quantizer. Remote range downloads and transactional resume avoid staging full source shards. A completed manifest and verification receipt gate installation promotion.

`Model.load` validates architecture and quantization, maps resident weights without copying them into Swift arrays, and opens expert streamers lazily. Common files are verified eagerly; expert-layer verification occurs on first access, with a distinct trusted-receipt policy available. Preserve these integrity checks when adding a source.

### Decode

`RealForwardRunner.swift` is the central forward implementation, approximately 1,900 lines. Each layer executes attention and routing, reads the selected expert IDs back to the CPU, plans cache hits/misses, fills missing slots from SSD, and combines shared/routed expert outputs. Shared-expert GPU work can overlap expert reads. Queue ordering and waits protect slots until GPU consumers finish.

After the final layer, a tied INT4 embedding/head produces either logits or a direct greedy token. `runRawCompletion` handles the outer loop. `LogitProducer`, `ContinuableLogitProducer`, and the chunked-prefill interface are useful existing boundaries for introducing another forward implementation.

### Prefill

Prefill is layer-major over bounded chunks, production default 128 text tokens. It groups token/expert pairs so fetched weights can serve multiple rows. Expert tiles and a slot-aware scheduler permit bounded prefetch without overwriting in-flight buffers. Eligible projections use staged affine MPP matrix multiplication; decode uses custom GEMV paths. A port must update and validate both paths independently.

## Current model contract

`Infrastructure/ModelIO/ModelTypes.swift` defines the canonical configuration:

| Property | Current value |
| --- | --- |
| Hidden width / layers | 2,816 / 30 |
| Routed experts / selected per token | 128 / 8 |
| Routed / shared FFN width | 704 / 2,112 |
| Query heads | 16 |
| Sliding-attention KV heads / head dimension | 8 / 256 |
| Full-attention KV heads / head dimension | 2 / 512 |
| Layer pattern | Every sixth layer full; 25 sliding and five full layers |
| Sliding window | 1,024 tokens |
| Vocabulary | 262,144 |
| Activation | Tanh-approximate GELU / GeGLU |
| Head | Tied to embedding |
| Quantization | Group-64 unsigned affine INT4, BF16 scales and biases; INT8 router |
| Logit softcap | 30 |
| Position ceiling | 262,144 in canonical runtime config |

These fields do not describe the complete computation. The forward runner also applies embedding scaling by `sqrt(hidden_size)`, attention scaling of 1.0, per-head Q/K and unweighted V normalization, NeoX RoPE variants, separate shared/routed branch norms, and a learned layer scalar. Full-attention layers reuse the raw K projection for V, then transform K and V differently. Their KV buffers must remain distinct.

Router computation normalizes hidden state, applies learned feature scaling folded with `1/sqrt(hidden_size)`, projects with affine INT8 weights, selects eight logits, softmaxes the selected scores, and applies learned per-expert scales. This cannot be assumed to match another model's routing.

## Concrete porting constraints

1. **Architecture settings do not switch layer semantics.** `Model.load(expecting:)` accepts a configuration, but the resident accessors, runner, and fusions still implement Gemma's graph. Several app/server/identity paths explicitly reference `.gemma4_26B_A4B`.
2. **Decode requires exactly eight experts.** `Kernels/MoE/MoE.swift` checks `topK == 8`; `Metal/MoE/moe.metal` has eight-entry arrays, argument bindings, and a K8 selection/reduction path. Prefill router checks allow K in 1...64, but that does not establish support through the complete prefill pipeline or decode. Expert counts above 256 also exceed current router limits.
3. **Fused layer operations have a hidden-width ceiling.** `FusedLayerTail` checks D <= 4096, and `fused.metal` allocates threadgroup arrays of 4096 elements. Larger widths need another implementation or fallback, not just resized model buffers.
4. **Activation is encoded in shaders.** Routed and shared FFN paths implement GELU-based gating. If the target uses SiLU/SwiGLU or another activation, both decode and prefill need matching implementations and references.
5. **Normalization and attention differ beyond dimensions.** Epsilon is 1e-6 in the runner. Norm placement, learned weights, head normalization, attention scaling, rotary layout/scaling, projection sharing, and residual ordering all need comparison against the target implementation. Decode attention wrappers currently bound head dimensions to <=512.
6. **Quantized storage is a specific contract.** `Quantization.groupSize` is 64 and manifest validation constrains schemes, precisions, scale/bias dtypes, and bit widths. An unquantized, GGUF, GPTQ, AWQ, or differently quantized checkpoint is not directly interchangeable. A conversion decision is required after selecting the target artifact.
7. **Tensor naming and layout are model-specific.** Resident accessors assume `language_model.model.*`; routed planning assumes stacked rank-3 gate/up/down tensors. Separate per-expert tensors, fused projections, missing shared experts, or dense-only layers require mapping/planning changes and corresponding runtime support.
8. **The output head is always tied in the implementation.** `Model.lmHead` returns the embedding. Merely setting `tieWordEmbeddings=false` will not load or execute a separate head.
9. **Tokenizer compatibility is deliberately strict.** `GFTokenizer` requires Gemma's decoder sequence, control tokens, image markers, and a fixed vocabulary size. Batch and streaming decode bypass generic cleanup behavior to preserve output. A new tokenizer needs a deliberate decoding and streaming contract.
10. **Prompt and tool protocols are Gemma-specific.** App/CLI framing is handwritten; server tool mode uses the installed Jinja template and Gemma parsers. The server's model-ID option changes API identity, not architecture. Stops, channels, tools, and image support must match the target's actual capabilities.
11. **Conversation identity must change with the model.** Stored token IDs are bound to model, checkpoint, template identity, and image processing version. Retain refusal of incompatible history replay during a port.
12. **Fast-path specialization is not uniformly a blocker.** Many primitive wrappers already select generic kernels when shapes differ. Distinguish those optimization constants from hard limits and semantic assumptions before rewriting code.

## Memory and performance implications

Reusable design does not imply reusable memory or speed numbers. For an ordinary gated expert with dimensions D and F, three projection matrices contain about `3*D*F` weights. At the current INT4/group-64 affine representation, bytes before alignment are approximately `3*D*F*(0.5 + 4/64)`, including BF16 scale and bias metadata. Slot allocation is page-rounded.

Expert-slot capacity is approximately the sum over layers of `slots * aligned_expert_bytes`. FP16 KV storage is the sum over layers of `2 * token_capacity * KV_heads * head_dim * 2` bytes. Add resident weights, scratch, projected image features when applicable, allocator overhead, and OS file-cache effects. These are capacity estimates, not measured physical footprint.

The default is 16 LFU slots per layer. Chunked prefill currently requires at least 16; eight-expert tiles use disjoint slots while pending work completes. Different expert widths, layer counts, routing locality, shared branches, and attention types change both memory and I/O economics. Do not carry the documented approximately 2 GB or throughput claims over to the new model.

Experiment records argue against automatically repeating speculative expert prediction, Markov prefetch, trace-trained layouts, or larger caches. Some lost on holdouts or under memory pressure. Use fresh target-model traces and end-to-end measurements before selecting such policies.

## Reuse recommendation and implementation order

Preserve the current Gemma path as a reference. Introduce only the model boundaries needed for the chosen target, rather than generalizing every subsystem upfront.

1. **Establish the target contract:** exact model/revision, configuration, reference forward implementation, tensor names/shapes, tokenizer/template, quantization, hardware/RAM budget, and required interfaces.
2. **Define explicit model behavior:** checkpoint mapping, per-layer kinds, routing/activation/norm/attention semantics, tied or separate head, tokenizer and capability declarations. Existing `ArchConfig` needs semantic companions; dimensions alone are insufficient.
3. **Implement bounded installation and loading:** source fingerprint, safe tensor mapping, quantized representation, manifests, and layout validation. Assess whether the v1 wire contract can express the target before deciding on an extension/version.
4. **Get scalar decode numerically correct:** implement the target's layer graph with simpler primitives first, preserving streaming ownership. Compare intermediate tensors and logits against a trusted target reference.
5. **Add chunked prefill:** reuse grouping/scheduling where compatible, then validate scalar-versus-chunked behavior, KV boundaries, and continuation.
6. **Integrate user-facing behavior:** correct text encoding, prompts/stops, streaming, history identity, CLI/app/server model selection. Enable tools or vision only when supported and validated.
7. **Optimize measured bottlenecks:** specialize target shapes and fuse only after correctness, then benchmark representative short/medium/long prompts and held-out routing workloads under the actual RAM budget.

## Validation plan

The existing suite provides substantial infrastructure coverage: format compatibility, resume/install transactions, corrupt inputs, affine references, nonzero/alignment-sensitive offsets, cache planning/reads, MoE and attention, chunk scheduling, ring KV, cancellation/continuation, Unicode output, app history/state, service lifecycle, and HTTP/tool ingress. Presence of tests is not proof they currently pass; they were inspected, not executed.

For the port, add target-specific checks for routed IDs and weights, each projection/activation/norm, intermediate layer outputs, final logits, and scalar/chunked consistency. Include real tensor dimensions and offsets, cache hits/misses and eviction, context/ring boundaries, cancellation recovery, and exact template/tokenizer fixtures. Include multilingual text if required by the target use case.

Use exact byte identity for lossless repack and exact outputs for changes claiming unchanged math. For reordered floating-point computation, use justified numerical/distributional checks against an independent reference; generated text alone is a weak oracle. Preserve original-model regressions if both models remain supported.

Run package tests through `Scripts/test.sh` serially. Follow `AGENTS.md` preflight requirements before any real-model run, and keep a single model-owning process. CI currently declares build/toolchain checks, serial tests, packaging-related checks, and documentation checks; this review did not verify remote CI status.

## Documentation cautions and next input

Some documentation/comments lag implementation: the system-design conversation section says conversations are not saved, while app history storage and README describe persistence; `SharedExpertInt8` naming/comments do not describe the current INT4 shared-expert production route by themselves. Prefer actual dispatch and dataflow over names when adapting the code.

At the time of this upstream review, the exact target artifact and hardware were still required. Those inputs are now established: the installed Sarvam 30B Q4_K_M GGUF, official pinned tokenizer, M2, and 8 GB RAM. The current implementation plan records the validated Sarvam contract; the upstream discussion above remains historical.
