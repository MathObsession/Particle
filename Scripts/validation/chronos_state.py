#!/usr/bin/env python3
"""Maintainer-only native worker check; Python is not an app dependency.

Runs exactly one caller-selected worker. No model download, mutation or mock
inference. Stores synthetic prompts/results only in the supplied evidence dir.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not str(output).startswith("/private/tmp/particle-agent-native-"):
        parser.error("Use an isolated /private/tmp/particle-agent-native-* evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    events = queue.Queue()
    checks = {}
    command = [args.worker, "--model", args.model, "--interactive", "--desktop-events",
               "--max-context", "4096", "--max-new", "32", "--temperature", "0",
               "--expert-cache-slots", "16", "--read-workers", "4", "--allow-unqualified-run"]
    environment = dict(os.environ, PATH="/usr/bin:/bin", DEVELOPER_DIR="/particle-no-developer-tools")
    with (output / "stderr.log").open("w") as errors, (output / "events.jsonl").open("w") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=errors, text=True, encoding="utf-8", env=environment, bufsize=1)

        def reader():
            for line in child.stdout:
                log.write(line)
                log.flush()
                try:
                    events.put(json.loads(line))
                except json.JSONDecodeError:
                    events.put({"event": "invalid_json", "line": line})
            events.put({"event": "eof"})

        threading.Thread(target=reader, daemon=True).start()

        def until(wanted, timeout=600):
            deadline = time.monotonic() + timeout
            received = []
            while time.monotonic() < deadline:
                item = events.get(timeout=max(.01, deadline - time.monotonic()))
                received.append(item)
                if item.get("event") in wanted:
                    return received
                if item.get("event") in {"eof", "invalid_json", "error"}:
                    raise RuntimeError(item)
            raise TimeoutError(wanted)

        def send(payload):
            child.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            child.stdin.flush()

        def restore(messages=None, budget=0):
            send({"type": "restore", "messages": messages or [], "reasoningBudget": budget,
                  "system": "Respond in English. Follow the user's instructions precisely and briefly."})
            until({"ready"})

        def turn(text):
            send({"type": "message", "text": text})
            received = until({"turn"})
            result = received[-1]
            stream = "".join(item.get("text", "") for item in received if item.get("event") == "text")
            assert stream == result["text"], "Stream differs from committed response"
            assert any(item.get("event") == "input" and item.get("utf8_sha256") == hashlib.sha256(text.encode()).hexdigest() for item in received), "Input bytes changed"
            until({"ready"})
            return result

        try:
            startup = until({"ready"})
            checks["capabilities"] = any(row.get("event") == "capabilities" and row.get("architecture") == "qwen3_5_moe" and not row.get("tools") for row in startup)
            restore()
            first = turn("Remember that the secret word is violet. Reply only: noted")
            second = turn("What is the secret word? Reply with only the word.")
            checks["warm_history_reused"] = second["reused"] > 0
            checks["earlier_context_preserved"] = "violet" in second["text"].lower()
            unicode_turn = turn("Reply with exactly this text, with no explanation: café 🪷 नमस्ते")
            checks["unicode_input_and_stream"] = unicode_turn["finish"] != "unicode_boundary"
            # An over-limit request is rejected before touching the live lineage.
            send({"type": "message", "text": "memory " * 5000})
            rejected = until({"error"})[-1]
            until({"ready"})
            after_limit = turn("What is the secret word? Reply with only the word.")
            checks["context_exhaustion_retains_history"] = "retained" in rejected.get("message", "").lower() and after_limit["reused"] > 0 and "violet" in after_limit["text"].lower()
            # Edits arrive as an explicit restored transcript, never a KV-only rewind.
            restore([{"role": "user", "content": "Remember: the secret word is amber."},
                     {"role": "assistant", "content": "Noted."}])
            edited = turn("What is the secret word? Reply with only the word.")
            checks["edited_history_replaces_recurrent_state"] = "amber" in edited["text"].lower() and "violet" not in edited["text"].lower()
            restore(budget=8)
            thinking = turn("What is 7 times 8? Give a short answer.")
            checks["thinking_budget_closes_prefilled_span"] = thinking["text"].startswith("<think>") and "</think>" in thinking["text"] and thinking["assistant_prefix_tokens"] == 1
            restore(budget=0)
            instant = turn("Reply only: ready")
            checks["instant_mode_has_no_prefilled_reasoning"] = instant["assistant_prefix_tokens"] == 0 and not instant["text"].startswith("<think>")
            restore()
            send({"type": "message", "text": "Explain in detail: " + "a small language model " * 160})
            until({"prefill"})
            child.send_signal(signal.SIGINT)
            until({"cancelled"})
            until({"ready"})
            fresh = turn("Reply only: clean")
            checks["cancelled_prefill_discards_partial_state"] = fresh["reused"] == 0 and fresh["finish"] != "reasoning_limit"
            send({"type": "message", "text": "Write a long story about a lighthouse."})
            until({"text"})
            child.send_signal(signal.SIGINT)
            until({"cancelled"})
            until({"ready"})
            recovered = turn("Reply only: recovered")
            checks["cancelled_generation_recovers"] = recovered["reused"] == 0 and "recovered" in recovered["text"].lower()
            send({"type": "reset", "messages": []})
            until({"ready"})
            new = turn("Reply only: new")
            checks["new_conversation_resets_state"] = new["reused"] == 0
            send({"type": "quit"})
            child.wait(timeout=30)
            checks["clean_shutdown"] = child.returncode == 0
        except Exception as error:
            checks["completed"] = False
            checks["error"] = str(error)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    send({"type": "quit"})
                    child.wait(timeout=30)
                except (subprocess.TimeoutExpired, BrokenPipeError):
                    child.terminate()
                    child.wait(timeout=30)
    report = {"passed": all(value is True for value in checks.values()), "checks": checks,
              "command": command, "kind": "Synthetic native state checks; not speed or quality qualification"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
