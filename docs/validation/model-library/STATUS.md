# Model library implementation status

Updated 4 November 2026. This is a development checkpoint, not a released or
qualified Chronos adapter.

Completed source components:

- Shared descriptors, compatibility registry and installed-model records.
- Public anonymous Hub search, pinned metadata, bounded header inspection,
  cached browsing, pagination and rate-limit handling.
- Lossless resumable v2 conversion with source/payload/expert verification.
- Native streamed Chronos worker, mixed quantization and bounded caches.
- Desktop model library, imports, queue, installed-model selection and
  conversation bindings; per-model cache, reader and chunk controls.
- Optional calibration harness and conservative profile promotion rules.
- Native converter/worker/MLX shader packaging, with upstream licenses.
- Recommended models at the top of Discover, with immediate default-model
  installation and the same qualification filtering as search.
- Local imports share the persistent installation queue; missing-model chats
  remain readable without being sent to a different model.

Validation evidence so far:

- Earlier shared-library checks: 10 new compatibility/conversion/migration
  checks plus 10 existing context/mode/agent-recovery checks passed.
- Earlier synthetic 3-bit GPU expert, cache-hit and eviction checks passed.
- SwiftUI macro source guard and compiler/plugin-removal probes passed.
- Revised focused checks: **35 passed**, including the native calibration
  orchestrator with explicit mock workers and cancellation/worker cleanup.
- Chronos conversion completed and verified 1,757 tensors from pinned revision
  `a982bb2ed776bf0db64375de480a1d0a00475a99`. Common payload is
  1,086,153,216 bytes; each selected layer-expert occupies 1,376,256 bytes.
- An unqualified 2 November prototype generated 96 capped tokens with native
  eight-expert routing: 95 actual decode steps in 44.0423 seconds, **2.16
  tokens/s**, and 8.3479 seconds prefill. M2/8 GB, internal SSD pack, 16 slots,
  four readers, 4,096 context, greedy sampling. Logical source reads were
  37,307,547,648 bytes. This working-tree diagnostic predates numerical fixes;
  it is not evidence that the current adapter meets all release gates.
- The 3 November initial ABBA comparison selected the same eight continuation
  tokens but failed the full-logit gate (maximum difference 1.03125). The
  adapter remains experimental. The reference and matching gathered-operator
  changes are being checked; this failure must not be hidden or waived.
- The revised upstream-math candidate passed an initial eight-step ABBA test
  with **zero** full-logit/continuation difference and identical token IDs in
  every pass. Candidate decode times were 3.9003 and 3.8212 seconds for eight
  actual steps (2.05–2.09 tokens/s). This short diagnostic does not qualify
  sustained speed or the adapter; broader checks are pending.
- The packaged worker subsequently passed the **64-step ABBA comparison**:
  zero full-vocabulary and final-continuation difference, with identical token
  IDs across all four passes. It ran with no developer tools in its environment
  and loaded the bundled MLX Metal library. Candidate decode took 35.5364 and
  36.7424 seconds for 64 actual steps (1.74–1.80 tokens/s). Power changed during
  the run, so these are diagnostics, not a controlled speed comparison.
- The isolated desktop library smoke passed 10 checks covering model-bound
  chats, profile restoration, persistent jobs and safe removal. Updated checks
  additionally cover local queued imports and offline recommendations.
- Final focused regression: **74 tests passed** (48 existing app/harness,
  five GPU/text and 21 library/conversion/calibration checks). Both SwiftUI
  macro compiler probes and the source guard passed.
- Updated isolated desktop smoke: **12 checks passed**, including persistent
  local imports and offline recommendations. Actual light/dark library
  previews were rendered and visually inspected using bundled PNG materials.
- Testing caught a blocking Foundation Process wait during cancellation.
  Shutdown now waits asynchronously for owned workers; cancellation regression
  passed, and app updates wait for installation/optimization ownership too.
- Packaged real-worker state checks: **12 passed**, covering handshake, warm
  reuse, retained earlier context, Unicode input/stream equality, over-limit
  rejection without replay loss, edited recurrent state, bounded Thinking,
  Instant, prefill/generation cancellation, reset and shutdown. These used
  synthetic prompts, one native process, 4,096 context, 32-token output cap,
  native eight experts, 16 slots and four readers on the M2/8 GB Mac. No
  developer tools were available in the worker environment.
- Full warm-history comparison started on AC power: 512/2,048/4,096/8,192
  retained tokens, suffixes 1/16/64/128, 64 actual decode steps, ABBA upstream
  selected-expert reference/candidate passes. The run hit the Metal resource
  limit (499,000) during the final 2,048-token workload and exited 133. It did
  not finish or qualify the adapter; the final aggregate report was not written.
- The streaming loop omitted upstream MLX's per-step autorelease boundary and
  joint evaluation of functional cache updates. Fixes are prepared, including
  random-key evaluation for the validation-only module constructors, periodic
  unused-buffer cleanup, incremental workload reports and MLX memory counters.
  They still need a rebuild and fresh short/long comparisons.
- Source implementation and recommendations are pushed in `06a7fac`; the
  first Swift 6.3 test-compatibility adjustment in `455cb21` was insufficient.
  A fixture that owns its AppKit state on the main actor is prepared for the
  next clean CI run. All five native products built on that runner.
- Further prepared changes preserve matching restored prefixes, reject
  unsupported configurations before fetching tokenizer/header data, validate
  expert-layer ownership/overlap, and allow forgetting an imported model
  without deleting its original files. These changes are not yet validated.

Still required before publishing/qualifying this feature:

- Run short/long histories and natural completions with at least 64 tokens,
  recording memory pressure, cache state and storage traffic.
- Exercise actual calibration/cancellation and desktop installation/import
  transactions beyond the passing synthetic orchestration/UI checks.
- Verify standalone app packaging, install the tested release, update the
  public installer, and apply release retention after successful installation.

The published installer still serves the existing 0.3.3 release. No new
application release has been published for this work yet. Chronos Code mode
remains disabled until its separate native tool round-trip tests pass.
