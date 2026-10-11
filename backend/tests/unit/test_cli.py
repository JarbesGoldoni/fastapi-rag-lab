import sys

import pytest

from raglab.adapters.cli import CliError, claude_args, stream_process

pytestmark = pytest.mark.anyio

ECHO_LINES = "import sys\nfor line in sys.stdin: print(line.strip().upper(), flush=True)"


async def test_stream_process_feeds_stdin_and_parses_lines(tmp_path: str) -> None:
    out = [
        text
        async for text in stream_process(
            [sys.executable, "-c", ECHO_LINES],
            stdin="one\ntwo\n",
            cwd=str(tmp_path),
            parse=lambda line: line.decode().strip() or None,
            time_limit=10,
        )
    ]
    assert out == ["ONE", "TWO"]


async def test_stream_process_raises_with_stderr_on_failure(tmp_path: str) -> None:
    with pytest.raises(CliError, match="bad things"):
        async for _ in stream_process(
            [sys.executable, "-c", "import sys; sys.stderr.write('bad things'); sys.exit(3)"],
            stdin="",
            cwd=str(tmp_path),
            parse=lambda _: None,
            time_limit=10,
        ):
            pass


async def test_stream_process_kills_on_timeout(tmp_path: str) -> None:
    with pytest.raises(TimeoutError):
        async for _ in stream_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin="",
            cwd=str(tmp_path),
            parse=lambda _: None,
            time_limit=0.5,
        ):
            pass


def test_claude_args_disable_tools_and_settings() -> None:
    from raglab.domain import Prompt

    args = claude_args("claude", Prompt(system="sys", user="u"))
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--system-prompt") + 1] == "sys"
    assert "--no-session-persistence" in args
