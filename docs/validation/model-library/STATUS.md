# Particle 0.4.0 model library

Updated 4 November 2026.

## Delivered source

- Discover / Installed / Downloads, recommended models, public Hugging Face
  metadata, pinned revisions, compatibility filtering and local imports.
- Sarvam remains the Churo Chat / Churo Code default. Chronos is an early-access
  chat alternative; it has no coding-tool capability.
- Lossless streamed v2 conversion, full source/payload/expert checksums,
  interruption/resume, direct native-pack downloads and original-file preservation.
- New resumptions verify completed source bytes locally instead of downloading
  those bytes again. Legacy checkpoints remain supported.
- Per-model profiles, one-worker ownership, model-bound conversations, safe
  removal, offline cached recommendations and optional Mac-specific optimization.
- Bundled native conversion/inference executables and MLX shaders. Testers do
  not need Xcode or Python.

## Validation and limits

- 82 focused checks passed before release preparation, including actual native
  pack metadata inspection without reading model payloads.
- Two subsequent targeted checks passed: quantized reference/kernel equivalence
  and legacy checkpoint compatibility. The eight-expert workspace persistence
  regression also passed after the app check exposed Sarvam-only database limits.
- The isolated model-library UI passed 14 checks. Earlier native worker state
  checks passed 13 cases covering retained/edited context, reasoning modes,
  Unicode, cancellation and shutdown.
- Fresh short ABBA comparison: zero full-vocabulary and continuation differences.
- Six 64-step ABBA workloads passed with zero differences: histories 512 with
  suffixes 1/16/64/128 and 2,048 with suffixes 1/16. Native candidate times were
  diagnostic, approximately 1.2–1.7 tokens/s on this M2/8 GB Mac with internal SSD.
  These are not a universal speed claim or a controlled comparison across all
  hardware and profiles.
- The long comparison aborted with the 499,000-buffer limit. A debugger isolated
  growth to reference construction (~600 buffers per reference step); sampled
  production streaming steps had flat counts. The reference now replaces unused
  quantization graphs with evaluated bounded placeholders.
- At the user's explicit request, exhaustive benchmarks were stopped and release
  work continued on battery. The complete 4K/8K matrix, maximum-context endurance
  and full optional optimization runs are not release claims.
- Chronos remains visibly early access. No native Chronos tool round-trip is
  claimed; Code is available through the existing Sarvam adapter.

The final packaged app passed its native import/chat check (11 checks), including
full verification in place, a real sampled reply, input-byte acknowledgement and
clean worker shutdown with developer tools unavailable. This caught and fixed
the desktop top-k sampler’s missing batch dimension. The result and installation
receipts are kept with local release evidence. Models, transcripts and memories are preserved
when older application releases are pruned.

## Published release

Particle 0.4.0, build `20261104130529`, is installed locally and published at
https://github.com/MathObsession/Particle/releases/tag/v0.4.0. The archive SHA-256 is
`65fbba2de03a46988e4cc2b80b6474a211caf5614b19fa35752a8f4b45ec76c1`,
matching GitHub’s uploaded asset digest. The existing internal Chronos pack was
registered without copying its weights; older application releases were removed
after verified installation. Chat data and models were preserved.

## 0.4.1: automatic recipes and expert selection

The converter now derives Qwen3 MoE and text-only Qwen3.5 MoE layouts from
configuration and complete tensor headers, with mixed native affine quantization.
Checkpoint dimensions, 3-bit precision and tokenizer/template hashes are no longer
fixed to Chronos. Single-file checkpoints and embedded templates are supported.
Search accepts names or public repository links; other architectures/formats show
specific compatibility reasons. Expert selections persist per model, with reduced
routing marked experimental. Search appears above the starter recommendations.

Five narrow regression checks passed. A tiny synthetic Qwen3 single-file pack
(4-bit/group-32, raw router, embedded template) converted successfully and had zero
full-logit/continuation differences across a two-step ABBA check. The installed
Chronos pack still loaded and replied “Hello.” Real Qwen3-30B-A3B-4bit and
Qwen3-30B-A3B-Instruct-2507-4bit metadata were accepted without weight downloads.
No trained Qwen3 quality/speed or exhaustive quantization-grid benchmark is claimed.
Unknown architectures still require an executable graph adapter.

Release build `20261104140233`; archive SHA-256 `1aeef69d7077371cce03d29075247cbbaa5a365ba80c46a0beea349193872d53`.

## 0.5.0: operation-level graph adaptation

Implemented backend-bound graph compilation, inferred byte-preserving tensor
normalization/assembly, v3 plans and paged projection execution across trusted
MLX graphs. Native mixed quantization, biased projections, per-bank expert counts,
bounded reads/leases, graph-derived profiles and channel-framed chat are included.
Sarvam and existing v1/v2 paths remain intact. Mixtral and GPT-OSS starter cards
are metadata-inspected through the same compiler, not a family allowlist.

Three synthetic layouts passed exact payload checks (31 Mixtral, 33 OLMoE,
24 MXFP4 tensors), unchanged-source checks and two-step alternating full-logit /
continuation checks with zero differences. Real Mixtral and GPT-OSS metadata
passed without weight downloads. Full trained-model quality, long-context
endurance and universal throughput are not claimed. Long benchmarks were not
repeated, following the user's request. See `docs/PARTICLE_GRAPH_COMPILER.md`.

Release build `20261104161646`; archive SHA-256 `b152adf62d75068bfa4314d2837239d3db5a0c31cb7f6c6c62fb870cda9a18a8`.
The packaged compiler and a tiny sampled chat passed with developer paths
removed. Six focused channel-framing checks and the SwiftUI source/probe
guards passed. No full model weights were downloaded for these checks.

## 0.6.0: curated collection, 5 November 2026

The product catalog now contains exactly ten main entries and three Fast entries.
Open-ended discovery/admission has been removed. Sarvam, Chronos, Mixtral and
other old entries retain their model files and conversations but are no longer
offered/selectable. Startup opens the curated library instead of downloading Sarvam.

All thirteen preferred real checkpoints passed configuration, tensor/archive,
tokenizer/template and runtime-graph metadata inspection without downloading
full pretrained weights. Added native Qwen2-MoE, DeepSeek-MoE and MiniCPM-MoE
graphs, Nemotron block-list normalization, tied MiniCPM output embeddings and
original w1/w2/w3 expert mappings. Preserved unused tensors in a separate
checksummed archive and fixed sparse module-array replacement across dense layers.

Three tiny original PyTorch checkpoints converted with 87 exact tensor checks,
preserved source-only bytes and zero full-logit/continuation differences over
two-step alternating streamed/reference execution. A real tiny torch.save
archive matched exact BF16/F32 bytes; executable pickle was rejected without
execution. Eleven focused catalog boundary assertions and SwiftUI source/probe
guards passed. No long benchmark matrix or trained-model quality/speed claim
is made. MiniCPM remains native BF16 with larger memory/disk requirements.

Evidence: `curated-0.6.0/catalog-metadata-checks.json` and
`curated-0.6.0/fast-checks.json`.

Packaged compiler/PyTorch inspection and Metal reference checks passed without
developer paths. The thirteen-model cached library rendered successfully.
Release build `20261105145326`; archive SHA-256 `01d891fc00d43fd7eb3f5d31230ab11118984b547d9dbea7be9db4ab79a90c29`.


## 0.6.1: Sarvam Default and website, 5 November 2026

Restored the original Sarvam 30B adapter and pinned public v1 model pack as Default, alongside the thirteen curated alternatives. Existing Sarvam paths register without a payload copy; existing selected models, expert preferences and transcripts remain intact. The default uses the established installer and qualified Churo Chat/Code workflow, not the new graph converter. Fresh installations require an explicit Install selection.

Release build `20261105162306`; archive SHA-256 `127a8e01d9019afbde792d1c48ab1bdad64a2f8b3bcfdab35ba0c979117f961e`. Release build, fourteen focused catalog boundary assertions and both SwiftUI source/probe guards passed. Package code signatures and installed version were verified. No new full-model download or performance benchmark was required for restoring the existing backend.

The ParticleDesign website includes the current install/update command, curated model cards and accurate system requirements. Desktop/mobile browser checks passed: filters, search, default badge, clipboard, preview tabs, FAQ, mobile navigation, theme persistence, no horizontal overflow and no JavaScript errors. Artwork and fonts are self-hosted.
