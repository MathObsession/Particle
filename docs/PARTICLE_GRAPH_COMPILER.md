> Product scope changed in 0.6.0: the compiler remains internal, but only the
> [curated model collection](MODEL_LIBRARY.md) can be installed or selected.

# Automatic MoE graph compiler

Particle 0.5.0 introduced operation-level adaptation across the trusted bundled MLX backend. It no longer needs a Particle-specific family recipe for every compatible MoE graph. Sarvam remains the default Churo experience, and existing v1/v2 installations remain supported.

## How adaptation works

1. Resolve the public repository to a revision. Read configuration, tokenizer/template and bounded safetensors headers before downloading weights.
2. Instantiate the matching bundled MLX graph lazily. Repository Python and remote code are never loaded. Discover routed `SwitchLinear` projections and native routing settings from the actual module graph.
3. Trace the backend's checkpoint normalization using named lazy source arrays. Prove byte-preserving rename, reshape and leading-axis concatenation operations; turn them into ordered tensor assembly instructions. Unsupported transformations receive a compatibility explanation.
4. Validate every parameter shape, dtype and quantization tuple. Preserve mixed precision, affine offsets, projection biases and native MXFP4/MXFP8/NVFP4 layouts. No entire expert matrix is expanded to floating point.
5. Write a v3 `.particle` pack containing the generated execution plan, provenance, original configuration/template and checksum-protected payload. Stream original source ranges directly into resident tensors and contiguous expert banks. Preserve imported originals and existing resumable installation behavior.
6. Rebuild the same bundled graph and replace only expert projection leaves with paged operators. The backend retains routing, activation, shared experts, attention and reduction semantics. Selected experts use bounded parallel demand reads, LFU reuse and leases across their consumers. There is one inference worker.

The plan records its backend version, configuration and tensor inventory digests, original-to-canonical assemblies, projection bindings, bank counts, native top-k, cache classes and memory estimates. Native packs are re-inspected against this executable contract before import. Cache sizing uses the graph's resident and expert bytes; staging and read concurrency are bounded separately. KV accounting remains configuration-derived and conservative for unusual cache representations.

## User experience

Search a public Hugging Face name or paste a repository link in Models. Recommendations are starting points, not the compatibility list. Install is offered after configuration, full header and tokenizer/template inspection succeeds. A model must also fit this Mac's resident/staging requirements; successful conversion alone does not establish speed or trained-model quality.

Each installed model has its own expert count, cache, chunk and context profile. Native expert selection is the reference; reducing it is explicitly experimental. Unsupported repository code, graph operations or normalization transformations are reported before large transfers.

BPE, Unigram and WordPiece tokenizers use the original chat template. Templates without a system role receive the hidden instructions folded into the first rendered user message without changing the saved transcript. Channel-framed tokenizers keep private analysis and addressed tool payloads out of public chat. Their Instant mode completes a native final-channel prefix; bounded Thinking transitions through evaluated control tokens. Generic models are chat-only until tool-specific qualification.

## Focused evidence, 4 November 2026

On the M2/8 GB development Mac, three small synthetic checkpoints passed exact conversion-byte and two-step alternating full-logit/continuation checks against the bundled reference math:

| Layout | Exact tensor checks | Inferred assemblies | Maximum logit / continuation difference |
| --- | ---: | ---: | ---: |
| Mixtral, separate original experts | 31 | 9 | 0 / 0 |
| OLMoE | 33 | 0 | 0 / 0 |
| Qwen3 native MXFP4 | 24 | 0 | 0 / 0 |

Distinct Mixtral expert values exercised assembly ordering. Source files were unchanged. Real public Mixtral-8x7B-Instruct-v0.1-4bit and gpt-oss-20b-MXFP4-Q8 metadata passed inspection without weight downloads. These checks establish bounded integration coverage, not a trained-model quality or throughput qualification. Long benchmarks were intentionally not repeated at the user's request.

## Scope and next boundary

This is a backend-bound graph compiler, not arbitrary mathematical architecture synthesis or a portable ONNX runtime. A graph must exist in the bundled MLX implementation, expose supported routed projection operations, use supported native layouts and pass complete normalization/parameter checks. Some registered architectures will still be incompatible. Dense, vision/audio, custom repository code and unproven tensor transformations remain unsupported. A future backend update can expose additional graphs to the same compiler without adding Particle family-name mappings.

There is no universal 2–6 tokens/s promise. Storage bandwidth, active parameter bytes, common weights, memory pressure, context and backend kernels still determine performance. CPU expert sharing and speculative prefetch are not enabled without a relevant measurement.

## Research and implementation references

- [Pinned MLX graph registry](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Libraries/MLXLLM/LLMModelFactory.swift): trusted graph construction and architecture implementations.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361): sparsity-aware expert caching; its reported hardware results are not Particle/M2 predictions.
- [ONNX graph concepts](https://onnx.ai/onnx/intro/concepts.html): graph/operator separation; Particle's present plan remains backend-specific.

Core source: `AutomaticGraphCompiler`, `SourceTensorNormalizer`, `ParticleGraphPlan`, `CompiledGraphModel`, `GraphExpertStore` and `GraphResponseFramer`. The CLI can force this path for diagnostics with `ParticleLibraryCLI convert --source <folder> --output <pack.particle> --graph`.
