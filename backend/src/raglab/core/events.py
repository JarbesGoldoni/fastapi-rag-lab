"""Parsing CLI output lines and formatting server-sent events. Pure."""

from __future__ import annotations

import json
from typing import Any

from raglab.domain import RESTART, Restart


def _loads(line: bytes | str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def claude_text_delta(line: bytes | str) -> str | Restart | None:
    """Text from a `claude -p --output-format stream-json --include-partial-messages` line.

    A new message_start means the CLI (re)started the answer, so earlier text is void.
    """
    data = _loads(line)
    if not data or data.get("type") != "stream_event":
        return None
    event = data.get("event") or {}
    if event.get("type") == "message_start":
        return RESTART
    delta = event.get("delta") or {}
    if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return str(delta.get("text", ""))
    return None


def claude_error(line: bytes | str) -> str | None:
    """The error message from the final `result` line, when the run failed."""
    data = _loads(line)
    if data and data.get("type") == "result" and data.get("is_error"):
        return str(data.get("result") or data.get("subtype") or "claude CLI failed")
    return None


def codex_message(line: bytes | str) -> str | None:
    """Agent message text from a `codex exec --json` line (one completed message per event)."""
    data = _loads(line)
    if not data or data.get("type") != "item.completed":
        return None
    item = data.get("item") or {}
    if item.get("type") in {"agent_message", "assistant_message"}:
        return str(item.get("text", ""))
    return None


def sse(event: str, data: object) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()
