"""Decide which providers exist on this machine. Pure given `which` and settings."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

from raglab.adapters.apis import anthropic_generator, anthropic_transcriber, openai_generator
from raglab.adapters.cli import claude_generator, claude_transcriber, codex_generator
from raglab.ports import Extractor, Generator
from raglab.settings import Settings


@dataclass(frozen=True, slots=True)
class Providers:
    generators: dict[str, Generator]
    transcribers: dict[str, Extractor]

    @property
    def default_generator(self) -> str | None:
        return next(iter(self.generators), None)


def discover(settings: Settings, which: Callable[[str], str | None] = shutil.which) -> Providers:
    generators: dict[str, Generator] = {}
    transcribers: dict[str, Extractor] = {}
    timeout = settings.cli_timeout_seconds

    if claude := which(settings.claude_cli):
        generators["claude-cli"] = claude_generator(claude, timeout=timeout)
        transcribers["claude-cli"] = claude_transcriber(claude, timeout=timeout)
    if codex := which(settings.codex_cli):
        generators["codex-cli"] = codex_generator(codex, timeout=timeout)
    if settings.anthropic_api_key:
        key = settings.anthropic_api_key.get_secret_value()
        generators["anthropic-api"] = anthropic_generator(key, settings.anthropic_model)
        transcribers["anthropic-api"] = anthropic_transcriber(key, settings.anthropic_model)
    if settings.openai_api_key:
        key = settings.openai_api_key.get_secret_value()
        generators["openai-api"] = openai_generator(key, settings.openai_model)

    return Providers(generators=generators, transcribers=transcribers)
