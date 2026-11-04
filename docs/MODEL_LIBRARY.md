# Automatic MoE conversion and model library

Particle 0.5.0 extends configuration-driven recipes with an operation-level MoE graph compiler. A repository name is not a compatibility allowlist.

Search a model name or paste its public Hugging Face URL. Particle resolves an immutable revision, reads configuration and bounded tensor headers, detects the graph and native mixed quantization, then offers Install when its bundled runtime can execute that graph. Unsupported search results remain visible with the specific missing adapter or format. Model cards do not execute code.

Install streams source ranges into a `.particle` pack, groups routed tensors into contiguous layer/expert blobs, keeps common tensors separate, preserves the source quantized bytes, and verifies source, payload and expert hashes. It supports indexed shards and single-file safetensors, and extracts an embedded template from tokenizer_config.json when no separate template is supplied. Originals are retained. Existing v1/v2 packs and resumable transfers remain supported.

## Automatic graph execution

Existing Sarvam and Qwen3/Qwen3.5 v1/v2 paths remain intact. New models can use the
v3 graph compiler across the trusted bundled MLX graph registry. It discovers routed
projection operations, infers checkpoint assembly from the backend's lazy normalization
graph, and preserves native affine or supported floating-point quantized bytes.
Mixtral's unstacked experts, OLMoE's graph and native MXFP4 have focused synthetic
conversion/reference coverage. Real Mixtral and GPT-OSS metadata inspection passed.

A repository name is not an allowlist. Conversely, membership in the backend registry
alone is insufficient: the complete graph, parameter inventory, normalization,
tokenizer/template and memory checks must pass. Unknown operators and custom remote
code remain unsupported. This does not synthesize arbitrary new mathematics from
weight names. See [the compiler design and limits](PARTICLE_GRAPH_COMPILER.md).

## Experts and profiles

Every installed model has an Active experts selector directly in Installed, also available in Settings. Choices run from one to native top-k for the MLX adapters; the existing Sarvam worker retains its supported 2–6 range. The native count is labelled. Reduced routing is explicitly experimental and can change answer quality. It never changes or requantizes stored weights.

Preferences are saved independently per model. Selecting a different model opens a model-bound conversation; returning restores its model. Changing routing for the active worker reloads its context while preserving messages. Cache slots, read workers, prompt chunks and conversation limits remain separate controls. Large resident/cache allocations are checked against this Mac’s physical memory before weight loading. Prompt subchunks are automatically bounded when the worst-case selected-expert staging set would exceed the temporary buffer budget.

## State, installation and distribution

Discover / Installed / Downloads share one persistent resumable queue. Public ungated sources only; no publisher token is stored. Paused jobs retain verified progress. New resumptions reconstruct completed source chunks from local payloads and a bounded on-disk header/padding archive; old checkpoints retain their verified re-read path.

Only one worker or converter owns the model lease. Recurrent models keep a stable-prefix recurrent checkpoint plus full-attention KV; Qwen3 uses its full-attention cache. Original templates, Instant/Thinking controls, input-byte acknowledgements, Unicode streaming and context exhaustion remain in the desktop protocol. Chronos and Qwen3 Code mode remain disabled pending tool-specific qualification.

The app bundles conversion, runtime and shaders; testers need neither Xcode nor Python. Model weights download separately. Removing models deletes only app-managed payloads after confirmation; imported originals and conversations remain.

## Narrow validation for 0.4.1

- Real public Qwen3-30B-A3B-4bit and Qwen3-30B-A3B-Instruct-2507-4bit metadata recognized automatically without downloading weights. The latter preserves 8-bit router overrides.
- A tiny synthetic Qwen3 checkpoint converted losslessly from one safetensors file, with 4-bit/group-32 weights, raw router and an embedded template. Its full-logit/continuation comparison against upstream selected-expert execution had zero differences over two steps. This is a graph integration check, not trained-model quality or speed qualification.
- Chronos metadata regression and model-specific expert-range checks are retained. No exhaustive benchmark matrix is rerun for this update, per user instruction.

The generic compiler can adapt additional bundled graphs without Particle family recipes. Unsupported operators still need an executable backend implementation; successful repacking alone never qualifies runtime correctness or speed.
