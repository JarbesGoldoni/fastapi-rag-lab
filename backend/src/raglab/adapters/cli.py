"""Local agent CLIs as generators: the tester's own `claude` or `codex` login, no API key.

Each call runs in a fresh temporary directory with tools off (the transcription extractor
enables only Read), so the CLI behaves like a plain model call.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

from raglab.core.events import claude_error, claude_text_delta, codex_message
from raglab.core.prompts import parse_transcription, transcribe_prompt
from raglab.domain import Page, Progress, Prompt, RawDocument, Restart, Stage
from raglab.ports import Extractor, Generator, Report

STREAM_LIMIT = 16 * 1024 * 1024  # stream-json lines can be large


class CliError(RuntimeError):
    pass


async def stream_process[T](
    args: Sequence[str],
    *,
    stdin: str,
    cwd: str,
    parse: Callable[[bytes], T | None],
    error_of: Callable[[bytes], str | None] = lambda _: None,
    time_limit: float,
) -> AsyncIterator[T]:
    """Run a process, feed stdin, and yield whatever `parse` extracts from each stdout line.

    The process is killed on timeout, on error, or when the consumer stops early
    (for example, the browser closed the connection).
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_LIMIT,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    # Drain stderr concurrently: a full pipe would block the process.
    stderr_task = asyncio.create_task(proc.stderr.read())
    try:
        proc.stdin.write(stdin.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        error: str | None = None
        async with asyncio.timeout(time_limit):
            async for line in proc.stdout:
                if (text := parse(line)) is not None:
                    yield text
                error = error or error_of(line)
            code = await proc.wait()
            stderr = (await stderr_task).decode(errors="replace").strip()
        if code != 0 or error:
            raise CliError(error or stderr[-2000:] or f"{args[0]} exited with code {code}")
    finally:
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
        stderr_task.cancel()


def claude_args(binary: str, prompt: Prompt, *, tools: str = "") -> list[str]:
    return [
        binary,
        "-p",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--tools", tools,
        "--no-session-persistence",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--system-prompt", prompt.system,
    ]  # fmt: skip


def codex_args(binary: str) -> list[str]:
    return [binary, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only", "-"]


def claude_generator(binary: str, *, timeout: float) -> Generator:
    async def stream(prompt: Prompt) -> AsyncIterator[str | Restart]:
        with tempfile.TemporaryDirectory(prefix="raglab-") as cwd:
            async for delta in stream_process(
                claude_args(binary, prompt),
                stdin=prompt.user,
                cwd=cwd,
                parse=claude_text_delta,
                error_of=claude_error,
                time_limit=timeout,
            ):
                yield delta

    return Generator(id="claude-cli", label="Claude Code (local CLI)", kind="cli", stream=stream)


def codex_generator(binary: str, *, timeout: float) -> Generator:
    async def stream(prompt: Prompt) -> AsyncIterator[str]:
        # codex has no system-prompt flag, so the instructions lead the message.
        message = f"{prompt.system}\n\n---\n\n{prompt.user}"
        with tempfile.TemporaryDirectory(prefix="raglab-") as cwd:
            async for text in stream_process(
                codex_args(binary), stdin=message, cwd=cwd, parse=codex_message, time_limit=timeout
            ):
                yield text

    return Generator(id="codex-cli", label="Codex (local CLI)", kind="cli", stream=stream)


def claude_transcriber(binary: str, *, timeout: float) -> Extractor:
    """For scanned PDFs: let Claude Code read the file (Read tool only) and transcribe it."""

    async def extract(doc: RawDocument, report: Report) -> tuple[Page, ...]:
        # One model call: the page count is unknown until it returns (total=0 = indeterminate).
        await report(Progress(Stage.READING, 0, 0))
        prompt = transcribe_prompt("document.pdf")
        with tempfile.TemporaryDirectory(prefix="raglab-") as cwd:
            await asyncio.to_thread(Path(cwd, "document.pdf").write_bytes, doc.content)
            parts: list[str] = []
            async for delta in stream_process(
                claude_args(binary, prompt, tools="Read"),
                stdin=prompt.user,
                cwd=cwd,
                parse=claude_text_delta,
                error_of=claude_error,
                time_limit=timeout,
            ):
                if isinstance(delta, Restart):
                    parts.clear()
                else:
                    parts.append(delta)
        pages = parse_transcription("".join(parts))
        await report(Progress(Stage.READING, len(pages), len(pages)))
        return pages

    return Extractor(id="claude-cli", extract=extract)
