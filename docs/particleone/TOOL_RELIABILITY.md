# File-tool reliability — Particle 0.3.3

A saved `write_file` request supplied both a complete HTML document and an absolute path inside the selected project. Particle 0.3.2 rejected that path with “Use a relative project path.” The model then claimed the file existed despite the failed operation. Absolute `list_files` requests for the project root failed for the same reason. Missing content was not the cause.

## Paths and approvals

File tools now accept either a project-relative path or an absolute path strictly inside the selected canonical project root. A component boundary check rejects sibling prefixes. Only the selected root is resolved to recognize macOS aliases such as `/tmp` and `/private/tmp`; supplied child paths are never resolved through symlinks. Raw parent traversal, NULs and backslashes are rejected before normalization. Credential/private paths remain blocked, and every file access continues through the existing descriptor-relative operations with `O_NOFOLLOW`. Child symlinks remain disallowed; normalization does not resolve them into extra access.

All remembered read hashes use one project-relative key. Reading an absolute path and then editing its relative alias therefore retains the read-before-edit protection. Proposed changes store that same canonical path, remain subject to approval, and check the original file hash again before the atomic commit. No tool takes code from a prose reply and writes it automatically.

## Failed file changes

The host derives outstanding mutation failures from actual tool events. A successful write/edit to the same canonical target resolves its earlier failure. Reads, checks and writes to other targets cannot erase it. Per-run accounting excludes old failures when the user starts a new instruction; the session can still show its unresolved historical failures.

A failed file response explicitly includes `file_applied: false` and a host instruction explaining that the file was not created or overwritten. If the model tries to finish while that run has unresolved file failures, the host hides the unsupported final reply, supplies the exact errors and requests correction, at most twice. A continuing failure ends with a host error rather than “Ready” plus an unsupported success summary. The raw model transcript remains in local history for continuity; it is not treated as proof of a successful operation.

Denied and Plan-blocked file operations are not automatic recovery candidates. The host also blocks another attempt at the same denied target in that run, including relative/absolute aliases and write/edit substitutions. A new user instruction can authorize a fresh attempt. Existing terminal/web approvals and verification after the last successful edit remain in force. Correcting a path does not bypass an approval or execute a denied operation.

While a failed file operation is being repaired, unsupported prose/code streaming is withheld. Failed sessions expose the actual host failure; code blocks in those sessions remain collapsible. Historical user messages and model replies are preserved rather than rewritten.

## Checks

`AgentRecoveryTests` reproduce an absolute HTML write, approval-before-apply, root listing, Unicode reads, relative/absolute edit aliases, undo, sibling-root rejection, traversal/private-path rejection and symlink escapes. They also verify that unrelated successful operations cannot clear a failed mutation and that denials retain their retry restriction through persistence.

The isolated desktop fixture deliberately emits a false success summary after an unread-file write fails. It checks that the file remains unchanged, the unsupported summary is hidden, retries stop at the bound, and a separate corrected read/write/check flow can recover successfully. A separate fixture exercises the screenshot’s full-path HTML creation through actual temporary filesystem operations. Model-free fixtures do not establish model intelligence or speed; native checks use one real six-expert model and a temporary project.

Local machine-specific logs and runtime state are excluded from Git. Models, user workspaces, memories and expert preferences are preserved by the app update.

The 0.3.3 release passed **48 focused tests in 11 suites**, **115 isolated desktop checks**, the SwiftUI compatibility guards and four update/retention tests. A single real six-expert model passed 15 native checks, including absolute-path read/write calls, an approved verification command, the same warm worker and conflict-checked undo. The installed application and worker match the tested package; both saved workspace payloads and the model/expert settings were unchanged by installation. This is functional validation, not a coding-quality or speed qualification for reduced-expert profiles.
