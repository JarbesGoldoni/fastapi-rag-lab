"""API-key generators: Anthropic and OpenAI. Optional; the CLIs cover the no-key path."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator

from raglab.core.prompts import parse_transcription, transcribe_prompt
from raglab.domain import Page, Progress, Prompt, RawDocument, Stage
from raglab.ports import Extractor, Generator, Report

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def anthropic_generator(api_key: str, model: str) -> Generator:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)

    async def stream(prompt: Prompt) -> AsyncIterator[str]:
        # Server-side fallback: if the model declines on a safety classifier, the API
        # retries on a suitable model inside the same call.
        async with client.beta.messages.stream(
            model=model,
            max_tokens=8000,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            betas=[FALLBACK_BETA],
            extra_body={"fallbacks": "default"},
        ) as response:
            async for text in response.text_stream:
                yield text

    return Generator(
        id="anthropic-api", label=f"Anthropic API ({model})", kind="api", stream=stream
    )


def anthropic_transcriber(api_key: str, model: str) -> Extractor:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)

    async def extract(doc: RawDocument, report: Report) -> tuple[Page, ...]:
        await report(Progress(Stage.READING, 0, 0))  # page count unknown until the reply
        prompt = transcribe_prompt(doc.filename)
        async with client.messages.stream(
            model=model,
            max_tokens=64000,
            system=prompt.system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64.b64encode(doc.content).decode(),
                            },
                        },
                        {"type": "text", "text": "Transcribe every page of this PDF."},
                    ],
                }
            ],
        ) as response:
            message = await response.get_final_message()
        text = "".join(block.text for block in message.content if block.type == "text")
        pages = parse_transcription(text)
        await report(Progress(Stage.READING, len(pages), len(pages)))
        return pages

    return Extractor(id="anthropic-api", extract=extract)


def openai_generator(api_key: str, model: str) -> Generator:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    async def stream(prompt: Prompt) -> AsyncIterator[str]:
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        async for event in response:
            if event.choices and (text := event.choices[0].delta.content):
                yield text

    return Generator(id="openai-api", label=f"OpenAI API ({model})", kind="api", stream=stream)
