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
