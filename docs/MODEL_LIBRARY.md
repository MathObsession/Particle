# Automatic MoE conversion and model library

Particle 0.4.1 replaces the Chronos-only conversion gate with a configuration-driven recipe. A repository name is not a compatibility allowlist.

Search a model name or paste its public Hugging Face URL. Particle resolves an immutable revision, reads configuration and bounded tensor headers, detects the graph and native mixed quantization, then offers Install when its bundled runtime can execute that graph. Unsupported search results remain visible with the specific missing adapter or format. Model cards do not execute code.

Install streams source ranges into a `.particle` pack, groups routed tensors into contiguous layer/expert blobs, keeps common tensors separate, preserves the source quantized bytes, and verifies source, payload and expert hashes. It supports indexed shards and single-file safetensors, and extracts an embedded template from tokenizer_config.json when no separate template is supplied. Originals are retained. Existing v1/v2 packs and resumable transfers remain supported.

## Current runtime families

| Family | Accepted input | Execution |
| --- | --- | --- |
| Sarvam | Existing supported GGUF / v1 Particle | Existing Churo Chat and Code default |
| Qwen3 MoE | MLX affine safetensors, packed switch_mlp banks | Configuration-driven upstream graph, streamed experts; early-access Chat |
| Qwen3.5 MoE | Text-only MLX affine safetensors, packed switch_mlp banks | Configuration-driven upstream hybrid-attention graph, streamed experts; early-access Chat |

The MLX recipes accept affine 2/3/4/5/6/8-bit weights and group sizes 32/64/128, preserving module-specific precision, raw routing tensors and quantization overrides. Dimensions, vocabulary, layer count, expert count, native top-k, attention heads, context accounting and recurrent schedule come from configuration and are checked against the complete tensor graph. The source BPE/ByteLevel tokenizer and Jinja chat template are used; the tokenizer is not pinned to a single checkpoint hash. No whole expert matrix is expanded into floating point.

The generic runtime keeps the original upstream router, shared branch and reduction, replacing the expert-bank leaf with bounded demand reads and LFU reuse. Cache eviction cannot overwrite arrays held by outstanding GPU operations. The bundled MLX dependencies remain pinned to Swift LM 3.32.3 / MLX Swift 0.32.3, with FoundationModels disabled.

**This is not arbitrary architecture synthesis.** A new attention/routing architecture needs an executable graph adapter; repacking bytes cannot infer its mathematical operations. Mixtral, DeepSeek, GPT-OSS, unknown GGUF families, vision/audio and repository-defined code are not enabled by changing a filename. Search explains these limits instead of silently hiding results. No universal speed or quality guarantee is made.

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

Further families can reuse the conversion scheduler and pack layout after adding their graph/schema adapter. They must not be marked runnable simply because a converter can write their tensors.
