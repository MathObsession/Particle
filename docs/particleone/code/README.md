# Particle Code

Particle is one native Mac app with two modes. Churo Chat is its chat assistant; Churo Code powers the Particle Code workbench. Both use the same installed local weights and one model worker. The names, creator attribution to Lakshya Prajapati and truthful base-model provenance are built-in system instructions, separate from the editable user preferences and visible transcript. Original upstream licensing and technical format identifiers remain intact. This is assistant/runtime branding, not a claim that new weights were trained.

## Workbench

Switch to Code in the sidebar and open a specific project folder. Load Churo Code, enter a task and press Return (Shift-Return adds a line; Command-Return also runs). Build mode can use tools; Plan mode denies edits and terminal execution. The centered new-session composer has a project-folder menu directly beneath it, including recent project folders and Change project folder. The optional inspector has Files, Editor, Changes, Activity and Terminal tabs. The editor supports lexical syntax colors, local undo, search, explicit Save/Discard, and private draft restoration with external-change detection. Coding sessions and undo snapshots persist in the same private local SQLite workspace as chat, with separate histories. Each workspace can keep independent coding sessions and projects.

The coding context must be at least 4096 tokens; 8192 is recommended. The inline Enable coding context button changes that setting through the ordinary model reload flow. It does not change the expert count. Six experts is recommended for tool reliability; reduced-expert profiles remain experimental, with no fixed quality guarantee. The current transcript is retained when the context fills; start a new coding session rather than silently discarding earlier user requests.

## Actual tool loop

The engine has a separate `agent` desktop command. It validates a user/assistant/tool history and a bounded tool catalog, renders declarations and tool observations using the unchanged official model template, and evaluates a direct-answer assistant prefix. Chat mode retains its existing conversation path. Every subsequent request carries the real completed tool history. Exact token-prefix comparison reuses valid KV; tool declaration JSON has stable sorted keys. One model stays resident. Coding uses a zero reasoning budget and at most 1024 output tokens per turn, without changing chat thinking preferences.

Only complete, stopped native tool-call blocks invoke tools. Fenced examples and ordinary prose remain inert. Invalid names, arguments, duplicate fields and truncated calls do not execute; the model gets at most two format repairs. Up to four calls per response execute sequentially, with a configurable 24/48/96-model-turn run budget (48 by default) and repeated-call detection. Tool output is bounded, marked as untrusted data and escaped against control-token injection before being returned in the native observation role. The UI keeps the actual output and status.

Tools:

- `list_files`: project tree, excluding generated folders and credential locations.
- `read_file`: UTF-8 text, numbered lines and file fingerprint.
- `search_files`: literal source search with paths and line numbers.
- `write_file` and `edit_file`: reviewed project changes, with prior-read and conflict checks, atomic replacement, snapshots and undo.
- `run_command`: explicitly approved shell command, streamed terminal output, exit code, timeout and process-group cancellation.
- `web_search` and `fetch_url`: public Exa MCP tools (with Jina Reader fallback for pages Exa cannot fetch), discovered using the provider's advertised schemas. Only the approved query or URL is sent; no API key is bundled. The client supports the advertised `web_fetch_exa` interface and the older `crawling_exa` interface. Provider limits/errors remain visible, never fabricated as search results.

## Permissions and boundaries

Reads and direct edits use a selected project directory and descriptor-relative traversal with `O_NOFOLLOW`; symlink escapes, `..`, absolute paths and common credential locations are rejected. The selected root cannot be the entire filesystem or home directory. Existing files must have been read before a model edit. Updates compare the current fingerprint against the reviewed content before replacing it. Undo also checks that the applied result has not been changed externally. The full before/after snapshots can be reviewed before approval. These are optimistic conflict checks, not a filesystem-wide transaction against every possible concurrent writer.

Edits ask by default. The user may allow direct project edits for the current session; commands and web requests still ask individually. Approvals carry an operation ID, so a stale or double click cannot approve a different operation. Plan restrictions are enforced outside the model. A tool result or project instruction cannot grant permission.

Terminal commands run with the Mac user's ordinary permissions after approval; this is not an OS sandbox. They use a minimal environment without inherited API-key variables, their own process group, a 1–300 second deadline and 64 KiB captured-output limit. Background children are stopped at command completion; persistent servers and interactive PTY sessions are not provided in this release. Public page requests use the fixed Exa/Jina services rather than contacting arbitrary model-selected local addresses from the Mac. Web approval identifies the external provider and exact query/URL.

Cancellation waits for a running command to finish stopping before unlocking another turn. On restart, pending/running operations are marked interrupted and never replay automatically. A partially generated tool call is never committed as an action. Manually edited project files and separately saved memories are not deleted by deleting a coding transcript; deleting a session removes that session's undo history only.

Limits keep the 8 GB Mac bounded: text files up to 1 MiB, up to 200 numbered lines per read, 2000 listed files / 10,000 traversal entries / 16 directory levels, 500 searched files / 60 matches, 6000 characters per model observation, and 32 MiB of change snapshots per coding session. Narrow a tool request when output is truncated. Larger projects can still be addressed by selecting a narrower subfolder or explicit paths. No multiple-model or speculative background agent process is introduced.

This release implements the requested native file/terminal/web coding loop. It does not claim parity with every OpenCode feature: there is no LSP, debugger, interactive PTY, arbitrary MCP/plugin manager or multi-agent execution UI.

References: [OpenCode tools](https://docs.opencode.ai/docs/tools/), [OpenCode agents](https://opencode.ai/docs/agents), [Exa MCP](https://exa.ai/mcp), [Jina Reader](https://jina.ai/reader/).


## Coding startup and verification

Coding uses short native tool schemas and a dedicated compact identity/workflow prompt. A bounded local startup index includes at most 36 root entries (1100 characters) and 1800 characters of root AGENTS.md instructions; it does not scan entire source files or grant edit access to files that were not read. The model must inspect any remaining or nested instructions. Root context is saved with its turn, stays out of the visible user message, and remains stable during that tool loop. Four independent reads can be requested together, so they share one model response.

The status reports actual new and cached tokens and prefill progress. Compact prompts reduce cold input work; they do not eliminate the first model evaluation. Subsequent turns use exact KV-prefix matching through the unchanged official template.

In Build mode, after a successful edit the host follows up if the assistant tries to finish with no passing command after the last edit. It permits at most two verification follow-ups, respects denial, and displays actual command success, failure, denial or missing execution separately from the assistant's prose. A zero exit code is evidence about that command, not proof of every behavior. Run checks explicitly requests another verification pass. HTML files can be opened from the editor in your browser. Persistent servers and interactive terminals remain outside this version.

Changed source is shown in the editor and diff review; repeated fenced code is collapsed in the conversation. Exact native history is retained. The background and composer use real, reproducibly rendered ParticleDesign PNGs, with dark paper retained at #20241F.

[Validation scope and results](../CODE_VALIDATION.md)
