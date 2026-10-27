# Particle Code 0.3.1 validation

The native Mac workbench uses the same local model and one serialized worker as chat. Its source is an application/agent harness, not a newly trained coding model or a claim of Codex/OpenCode feature parity.

## Functional checks

38 focused tests in nine suites passed locally. Coverage includes official-template token-prefix stability, native tool parsing, transport validation, project boundaries, file conflicts/undo, cancellation, command timeouts, URL restrictions, persisted sessions and chat compatibility. The optional tokenizer test used the installed official sidecars; model-free CI does not download model weights.

95 isolated desktop checks passed. They exercise real temporary file edits and shell execution with a mock model transport, including approval/denial, stale approval IDs, verification follow-ups, session deletion/project restoration, composer drafts, manual edit verification state, and preserved chat. Mock checks do not establish real-model quality or speed.

A separate real six-expert run passed: identity, read, approved edit, approved exact verification command with exit code zero, completion, same warm worker and conflict-checked undo. It used only a temporary test project and isolated app data. No user project or conversation was changed for testing.

Light/dark new-session and editor views were rendered and inspected. The green/grain materials are actual deterministic PNG assets produced by Scripts/particleone/render_code_materials.py. Dark paper remains #20241F. Preview conversations are illustrative fixtures, not fabricated performance measurements.

## Performance scope

For a fixed read-only startup task with the same default user preferences, the native prompt decreased from 1296 to 1031 tokens (20.45%). Template-only verification confirmed an exact 1047-token cached prefix for the next tool turn, with 25 new template tokens plus the normal three-token no-thinking prefix. The official chat template and model weights are unchanged.

In the four alternating trials, median prefill was **157.27 s → 128.28 s** (18.44% less time, 1.226× throughput). Median submission-to-first-token was **162.01 s → 132.77 s**. These are empty-KV trials with warm resident weights, not full model boot time. The validation uses a single M2/8 GiB model process, alternating reference/candidate runs, the internal model pack, six experts, 12 RAM slots and reset KV for each startup trial. OS caches are not purged. This is a short startup diagnostic; it does not qualify a sustained token rate or broad coding-quality target. Repeated template rendering and model/expert I/O remain material costs.

Machine-specific logs, private receipts, app archives, temporary chats and model weights remain local and are excluded from the source repository. The shared source guide records the outcome without uploading that private runtime state.
