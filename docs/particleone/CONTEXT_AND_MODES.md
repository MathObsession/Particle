# Context and response modes — Particle 0.3.2

Both Chat and Code use the native model’s full causal attention. The model’s official configuration specifies 131,072 positions, four KV heads, 19 layers and a 64-dimensional head ([official configuration](https://huggingface.co/sarvamai/sarvam-30b/blob/main/config.json)). The installed pack checks those same fields. This update lifts Particle’s earlier 8,192-position implementation limit; it does not change RoPE or claim support beyond the trained position range.

## Capacity and memory

| Installed unified memory | Default / maximum selectable context | FP32 KV capacity budget |
| --- | ---: | ---: |
| 8 GB | 32,768 | 1.1875 GiB |
| 12–16 GB | 65,536 | 2.375 GiB |
| 24 GB and above | 131,072 | 4.75 GiB |
| Below 8 GB | 8,192 | 0.296875 GiB |

These are conservative admission tiers, not a guarantee under competing workloads. Resident weights, existing expert slots, prompt scratch, the app and macOS need additional memory. KV remains FP32. Larger contexts are rejected on smaller-memory Macs before allocating the model; saved transcripts are retained. Settings show the available capacities and exact KV estimate. Smaller capacities remain selectable. Long-context attention takes more work; this is a capacity increase, not a decode speedup.

A context is **input plus output**. The default output allowance is **16,384 tokens**, exceeding the requested 16,286. It applies to both Chat and each Code model turn, replacing Code’s hidden 1,024-token cap. The model can finish earlier. On a 32K context, reserving a 16K reply leaves approximately 16K for the formatted prompt, including system/tool instructions; Instant also needs three prefix positions. An oversized prompt fails explicitly without discarding or summarizing history. Reduce the response allowance or select a larger supported context to fit more input. Lowering capacity in Settings also lowers an output allowance that no longer fits.

The backend accepts 32…131,072 context positions within its memory tier and up to 131,072 output tokens subject to exact prompt admission. The desktop exposes discrete supported capacities and output choices. The output setting is not silently halved. A small-context fallback leaves at least 32 positions; exact tokenized admission can still require a smaller output allowance for the actual prompt.

An existing workspace gets a one-time upgrade to its machine’s recommended context and at least 16,384 output tokens. Later deliberate small settings are preserved. Model path, active experts, chats, memory, appearance and custom instructions are preserved. Chat and Code default to Instant unless an existing Chat preference says otherwise.

## Instant and Thinking

Each composer has a mode control. Chat and Code save independent preferences in each workspace; changing modes does not load a second model or change expert count. Mode changes are disabled during a response or tool operation. Settings also expose Thinking · extended.

- **Instant:** official `enable_thinking=false`, evaluated `<think></think>\n` prefix and zero-token reasoning budget. Reopened reasoning is forced closed through a normal forward pass; repeated reopening stops the turn. This is a generation constraint, not merely hidden UI text.
- **Thinking:** official `enable_thinking=true`, up to 64 reasoning tokens, then the evaluated closing token. The answer/tool call continues within the configured total output allowance.
- **Thinking · extended:** native, unrestricted reasoning within the total output allowance.

Code sends its selected budget on every agent request. It no longer overrides the mode inside the worker. Thinking is kept separate from executable tool calls and can be expanded on completed prose replies. Partial or reasoning-only calls remain inert; permissions, Plan mode, cancellation, conflict checks and verification follow-ups are unchanged. Switching modes uses exact token-prefix comparison before reusing KV, because formatting can change the prefix.

## Verification scope

Run focused checks with the SD toolchain:

```sh
Scripts/particle test -c release --filter 'ContextAndModeTests|LongContextAttentionTests|AgentHarnessTests|AgentPromptRenderingTests|ParticleOneTests|MessageEditingTests|NoThinkingTests|ThinkingControlTests|WorkspacePolishTests'
```

`ContextAndModeTests` cover tiers, byte budgets, one-time migration, output forwarding, protocol validation and exact context/output admission. `LongContextAttentionTests` run actual Metal attention at 32K, 64K and 128K against an analytic sparse-value oracle, including a poisoned future row, the last valid positions and the 16:1 grouped-head mapping. These GPU-only checks do not load model weights, and are not a full long-context model-quality benchmark. CI includes the model-free policy/protocol tests; the Metal test is run locally on Apple Silicon.

Local release evidence, app captures, runtime results and user-state preservation records remain excluded from source publication. No full 16K completion or 64K/128K end-to-end model run is implied by a successful capacity or kernel check on an 8 GB Mac.

Release checks passed locally on M2/8 GiB, macOS 27.0.1, Swift 6.4: **44 focused tests in 11 suites**, the SwiftUI compatibility guards, four publisher/retention tests, and **100 isolated desktop checks**. Light/dark composer captures were inspected. A single native six-expert worker completed Chat Instant, Code Thinking and Code Instant at configured 32,768-token capacity and 16,384-token output allowance, with three/zero/three evaluated assistant-prefix tokens respectively. Replies ended naturally; an attempted 64K allocation on this 8 GB Mac was rejected. The installed app/worker match the validated package, and both saved workspace payloads were unchanged by installation. These checks do not claim full-length answer quality or sustained performance at maximum capacity.
