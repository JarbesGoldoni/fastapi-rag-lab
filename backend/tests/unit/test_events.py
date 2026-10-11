import json

from raglab.core.events import claude_error, claude_text_delta, codex_message, sse
from raglab.domain import RESTART


def test_claude_text_delta_reads_partial_message_events() -> None:
    line = json.dumps(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        }
    ).encode()
    assert claude_text_delta(line) == "hello"
    assert claude_text_delta(b'{"type":"system","subtype":"init"}') is None
    assert claude_text_delta(b"not json") is None
    start = b'{"type":"stream_event","event":{"type":"message_start","message":{}}}'
    assert claude_text_delta(start) is RESTART


def test_claude_error_only_on_failed_result() -> None:
    assert claude_error(b'{"type":"result","is_error":true,"result":"Not logged in"}') == (
        "Not logged in"
    )
    assert claude_error(b'{"type":"result","is_error":false,"result":"ok"}') is None


def test_codex_message_reads_completed_agent_messages() -> None:
    line = b'{"type":"item.completed","item":{"id":"1","type":"agent_message","text":"hi"}}'
    assert codex_message(line) == "hi"
    assert codex_message(b'{"type":"item.started","item":{"type":"agent_message"}}') is None


def test_sse_format() -> None:
    assert sse("delta", {"text": "olá"}) == b'event: delta\ndata: {"text":"ol\xc3\xa1"}\n\n'
