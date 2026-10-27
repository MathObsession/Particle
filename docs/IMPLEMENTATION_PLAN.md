# Particle implementation plan

Target: Sarvam 30B Q4_K_M on an M2 with 8 GB RAM. First deliverable: one local text CLI using the existing GGUF, with bounded expert memory and measurable behavior. SD-card testing comes before internal-SSD optimization.

## 1. Establish the model contract — implemented

`ParticleGGUF` parses bounded GGUF v3 metadata and validates tensor shapes, types, offsets, alignment, and file ranges. `SarvamConfiguration` checks the installed architecture: hidden size 4096, 19 layers, a first dense layer, 18 MoE layers, 128 experts/top-6, one shared expert, 64 Q heads/4 KV heads, head dimension 64, Q/K normalization, and full causal attention. The graph uses Sarvam's norms, routing, and NeoX rotary positions.

Official tokenizer metadata is pinned by revision and checksum. The runtime checks vocabulary agreement and applies the actual chat template. Independent tokenizer fixtures cover English, Indic languages, Urdu, emoji, whitespace, byte fallback, and control markers.

## 2. Integrate the runtime — implemented

`ParticleCore` owns direct Q4_K/Q5_K/Q6_K/F32/F16 Metal decoding, embedding gather, normalization, QKV processing, attention, SwiGLU, residuals, sigmoid expert selection, and output logits. It reuses the upstream Metal device factory while keeping the Sarvam graph separate from Gemma assumptions.

Common weights use read-only mappings. Per-layer caches gather gate/up/down ranges for the six selected experts into fixed Metal-visible slots. LFU eviction protects all experts needed by the current token. A single command queue orders scratch reuse, and reads finish before GPU consumption. No second model pack is required.

## 3. Deliver CLI and validate — first CLI accepted

`ParticleCLI` connects model validation, tokenizer/template, forward passes, sampling, streaming output, cancellation, single-process ownership, and timing/memory/cache counters. `Scripts/particle run` launches its release binary with the SD-card paths.

Release compilation, all eight focused tests, and two sequential real-model smoke runs passed. The direct completion produced “Paris.” See the [validation record](PARTICLE_VALIDATION.md) for exact commands, full footers, observed reasoning behavior, and the measured SD decode rates of 0.055–0.067 token/s.

Acceptance gates:

- Release build and focused serial tests.
- Independent CPU/Metal comparisons for every installed quantization, embedding gather, RMS normalization, Q/K norm and rotary positions, unmodified V, and causal grouped-query attention.
- Router selection/weighting, expert cache gathering/replacement, tokenizer parity, parser rejection, sampling, and byte-stream checks.
- Real installed-model output on a short prompt, with exact command, complete footer, environment, and limits recorded in the validation notes.

Passing numerical unit checks does not establish full-checkpoint logit parity with an independent engine. That comparison remains a separate gate before making parity claims.

## 4. Improve throughput — native batched compute and tiered expert I/O implemented

Implemented: layer-major grouped prompt processing, actual four-token weight reuse inside native K-quant GPU projections, 128-token default chunks, original-order expert reduction, fused decode gate/up plus SwiGLU, bounded embedding rows, fixed LFU RAM slots, optional lazy SSD expert packing capped at 4 GiB, source identity/checksums, bounded sampling candidates, and detailed source/SSD/prefill/decode counters.

The [current performance record](PARTICLE_PERFORMANCE.md) separates isolated GPU timing from model throughput. The [complete optimization audit](PARTICLE_OPTIMIZATION_AUDIT.md) maps all 104 upstream experiment IDs to implemented, partial, deferred or deliberately excluded work. The [first SD study](GTURBO_OPTIMIZATION.md) remains historical evidence, not the current implementation description.

Remaining: native K-quant staged MPP, persistent multi-expert down/reduction, standalone head vectorization, split/grouped full attention and TensorOps long-context prefill. Reduced-precision KV changes quality and requires an explicit distributional gate. Do not promote upstream rejected prefetch/compression/scheduler variants without a Sarvam-specific measured win.

The 0.3–0.5 token/s goal must distinguish cache-warm repeated prompts from fresh SD reads. The 6–8 token/s internal-SSD goal remains experimental.

## 5. Expand product surfaces — warm CLI implemented

Multi-turn CLI retains one runner/tokenizer, renders the full official template, reuses only identical committed KV rows and rewinds changed suffixes. Cancellation invalidates partial KV state; completed history and warm expert caches survive. `/new` starts a fresh chat. Raw interactive mode provides independent warm completions.

A Sarvam server and app adapter remain later work. Existing upstream GUI/server behavior stays tied to Gemma. Preserve upstream licensing and attribution.
