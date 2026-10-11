"""Browser-facing protection for a server that runs your own CLI login.

CORS only stops a foreign page from *reading* responses. A form POST is sent without a
preflight, so without this check any page you visit could upload documents and make the
backend run your Claude or Codex CLI.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def write_allowed(origin: str | None, *, host: str, allowed: Sequence[str]) -> bool:
    """Browsers send Origin on every cross-origin write. No Origin means a non-browser
    client (curl, scripts, tests); the backend's own origin and listed origins are fine."""
    if origin is None:
        return True
    return origin in allowed or urlsplit(origin).netloc == host


class SameOriginWrites:
    def __init__(self, app: ASGIApp, *, allowed: Sequence[str]) -> None:
        self.app = app
        self.allowed = tuple(allowed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in UNSAFE_METHODS:
            headers = Headers(scope=scope)
            if not write_allowed(
                headers.get("origin"), host=headers.get("host", ""), allowed=self.allowed
            ):
                response = PlainTextResponse("Cross-origin request refused", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
