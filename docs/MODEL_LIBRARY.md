# Particle model library

The library is implemented in `ParticleModelLibrary`, with native conversion
in `ParticleLibraryCLI`, the existing Sarvam worker in `ParticleCLI`, and the
new Qwen3.5 MoE worker in `ParticleMLXCLI`. All installation and inference stays
on the user's Mac. No publisher credentials are used.

The desktop adds Models → Discover / Installed / Downloads. Existing chats
and code sessions bind to the default Sarvam installation without rewriting
their text. New conversations retain their selected installation ID. Model
settings include expert slots, routing count, read workers and prompt chunks.
Sarvam keeps the Churo Chat / Churo Code identities; other models keep their
source names and provenance.

Discover starts with **Recommended**, including the default assistant and
one-click Install / Use model actions. Recommendations remain available from
cached metadata offline. They use the same qualification checks as search;
unvalidated model names cannot become installable recommendations. Additional
curated recommendations appear when their adapters qualify.

## Qualification

Sarvam's existing adapter remains qualified. The Chronos converter and native
worker are experimental and hidden from normal installation until numerical,
conversation-state and performance gates pass. Conversion success is not a
runtime qualification. No general 2–6 tokens/s guarantee is made.

The first additional layout is the text-only 40-layer, 256-expert Qwen3.5 MoE
35B-A3B family with affine 3-bit/group-64 expert weights and 8-bit routing
overrides. The initial tokenizer and template contract is restricted to the
checksum-pinned Chronos assets; another tokenizer requires separate validation.
Other architectures, dense or multimodal models, arbitrary remote
code and incompatible quantizations receive an explanation before weights
are downloaded. This is an adapter system, not universal MoE conversion.

## Pack and installation

The existing Sarvam v1 reader is unchanged. V2 records architecture, immutable
source revision, configuration, tensor ranges, quantization, tokenizer,
template, licenses, complete source-file SHA-256 hashes, destination hashes
and individual expert hashes. Conversion copies native quantized bytes;
it does not requantize or reduce routing.

Remote conversion reads sequential bounded ranges directly into a partial
destination. It never stages a second full source checkpoint. New transfers
use at most 16 MiB per range and durable checkpoints every 64 MiB. Resume
rechecks committed source and destination bytes. Complete source hashes and
payload/expert hashes are verified before atomic publication. Git metadata
files also verify their pinned Git blob IDs when available. Shard lengths use
64-bit accounting. Inference and conversion share an exclusive model lease.
Critical system memory pressure pauses conversion with verified progress
retained.

Local imports preserve originals. GGUF imports use the native packer and
prepare Sarvam's checksum-pinned public tokenizer when needed. Indexed MLX
folders use the v2 converter. Existing local packs are fully verified in place. Public repositories with a root v2
`manifest.json` download the native pack directly, preserving the exact manifest,
original provenance and quantized payload bytes; they are not reconverted.
The transfer remains pinned to the hosting repository revision, including when
that repository is a mirror of the original checkpoint.
App-managed removals retain conversations, memories and user-owned sources.
Imported entries can also be removed from the library while preserving their
original files.
Local imports and remote transfers share the persistent pause/resume queue.
Conversations whose model was removed remain readable; sending a new turn
requires reinstalling their bound model rather than silently using another.

## Chronos worker

The worker uses MLX Swift LM **3.32.3** and MLX Swift **0.32.3**, with
FoundationModels integration disabled. Building these dependencies requires
Swift **6.3 or newer**. Testers need neither a compiler nor Python: app
packaging includes the native converters, both workers and the compiled MLX
shader bundle.

The upstream Qwen3.5 model graph owns attention and recurrent computation.
Common tensors stay resident. Routed expert arrays remain quantized in a
bounded per-layer LFU cache; selected misses load concurrently. Grouped
prompt rows reuse each expert. No full expert matrix bank is expanded into floating point.

Full-attention KV and recurrent/convolution state have different rewind rules.
One recurrent-only stable-prefix checkpoint is retained; full KV is trimmed
to that position when valid. Otherwise the worker rebuilds state. Failed or
cancelled turns do not commit a partial conversation. Instant and Thinking
use the original template and evaluated reasoning delimiters. Tools remain
disabled for Chronos until native round-trip tests qualify them.

## Development commands

Use the configured SD toolchain wrapper. Metadata inspection does not download
model payloads:

```bash
Scripts/particle build -c release --product ParticleLibraryCLI
/Volumes/Particle/Build/Particle/SwiftPM/release/ParticleLibraryCLI inspect \
  --repo Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit
```

An explicit development conversion remains unavailable for normal library
selection while its adapter is unqualified:

```bash
/Volumes/Particle/Build/Particle/SwiftPM/release/ParticleLibraryCLI convert \
  --repo Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit \
  --revision a982bb2ed776bf0db64375de480a1d0a00475a99 \
  --output /absolute/path/chronos.particle --resume \
  --allow-unqualified-conversion
```

`ParticleMLXCLI --validate-reference --validation-steps 64` compares all
vocabulary logits and a greedy continuation in ABBA order using one model
process. The reference retains the upstream router/shared branch and streams
selected expert weights into its public SwitchGLU operations. Reference and
candidate never load a second whole model.

## Optimize for this Mac

Optimization is optional and can take a long time. It runs one worker at a
time, alternates control/candidate trials, and evaluates safe slot budgets,
one/two/four readers, and 32/64/128-token chunks. Each workload executes at
least 64 decode steps. Routing count and context allowance remain unchanged.
Cancellation keeps the current settings.
Storage tuning runs the qualified candidate under each profile, avoiding a
redundant upstream-reference run inside every trial. Independent adapter
qualification performs upstream comparisons. Tuning still compares complete
logit and continuation digests across alternating saved/candidate profiles.

Promotion requires repeated complete measurements, unchanged full-logit and
continuation digests, at least 10% lower median latency on a target workload,
and no regression greater than 5% on other workloads. Results bind to model
manifest digest, worker digest, hardware, OS, memory and model storage path.
Calibration uses forced continuations; it is not a natural-completion speed
or quality claim. Logical read bytes do not measure physical SSD traffic.
Power source and Low Power Mode are recorded; a change aborts calibration
without replacing settings.

## Sources

- [Hugging Face public Hub API](https://huggingface.co/docs/hub/en/api)
- [Pinned MLX Swift LM package](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Package.swift)
- [Upstream Qwen3.5 graph](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Libraries/MLXLLM/Models/Qwen35.swift)
- [Upstream quantized switch layers](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Libraries/MLXLMCommon/SwitchLayers.swift)

Current work and measurements are recorded in
[`validation/model-library/STATUS.md`](validation/model-library/STATUS.md).
