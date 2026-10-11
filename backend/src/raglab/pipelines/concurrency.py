"""Small async combinators used by the pipelines."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any

from raglab.domain import Progress, Prompt, Restart
from raglab.ports import Report, Stream


def batches[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def map_batched[T, R](
    fn: Callable[[Sequence[T]], Awaitable[list[R]]],
    items: Sequence[T],
    *,
    batch: int,
    concurrency: int,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[R]:
    """Run fn over batches with at most `concurrency` in flight; results keep input order.

    `on_progress(done, total)` is called with item counts as each batch finishes.
    """
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def run(part: Sequence[T]) -> list[R]:
        nonlocal done
        async with semaphore:
            result = await fn(part)
        done += len(part)
        if on_progress:
            await on_progress(done, len(items))
        return result

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(run(part)) for part in batches(items, batch)]
    return [item for task in tasks for item in task.result()]


async def complete(stream: Stream, prompt: Prompt) -> str:
    """Collect a streamed generation into one string, honoring restarts."""
    parts: list[str] = []
    async for delta in stream(prompt):
        if isinstance(delta, Restart):
            parts.clear()
        else:
            parts.append(delta)
    return "".join(parts)


async def ignore_progress(_: Progress) -> None:
    return None


def throttled(
    report: Report, *, interval: float, clock: Callable[[], float] = time.monotonic
) -> Report:
    """Forward at most one update per `interval`, but never drop a stage's first or last
    update, so the stored progress always ends on a real, complete count."""
    last_at = -float("inf")
    last_stage: object = None

    async def send(progress: Progress) -> None:
        nonlocal last_at, last_stage
        now = clock()
        edge = progress.stage != last_stage or progress.done in (0, progress.total)
        if edge or now - last_at >= interval:
            last_at, last_stage = now, progress.stage
            await report(progress)

    return send


async def guarded[T](
    work: Coroutine[Any, Any, T],
    guard: Coroutine[Any, Any, None],
    *,
    lost: Callable[[], Exception],
) -> T:
    """Run `work` while `guard` keeps running. If the guard returns first, the work is
    cancelled and `lost()` is raised; if the guard fails, the work is cancelled and the
    guard's error is raised. Either way, nothing is left running."""
    work_task = asyncio.create_task(work)
    guard_task = asyncio.create_task(guard)
    try:
        await asyncio.wait({work_task, guard_task}, return_when=asyncio.FIRST_COMPLETED)
        if work_task.done():
            return work_task.result()
        guard_task.result()
        raise lost()
    finally:
        for task in (work_task, guard_task):
            task.cancel()
        await asyncio.gather(work_task, guard_task, return_exceptions=True)
