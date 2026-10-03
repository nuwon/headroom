"""Loopback decision gateway: one stable API for Python and Rust (plan §4.9).

Endpoints::

    GET  /health
    POST /v1/systemone   {"state": "...", "questions": {"id": {"type", "instructions", "criteria"}}}

Responses follow the TypeSafe/JevK5 answer shape plus aggregate
``usage.input_tokens`` and ``latency_ms``. The gateway delegates to a
:class:`~headroom.intelligence.jevk5_client.DecisionClient` (normally the GGUF
client pointed at the managed ``llama-server``). It never logs ``state``.

Two ways to serve it:

* standalone: ``headroom intelligence gateway`` (stdlib ``ThreadingHTTPServer``
  bound to 127.0.0.1);
* in-process: :func:`register_routes` mounts ``/v1/intelligence/systemone`` and
  ``/v1/intelligence/status`` on the proxy's FastAPI app (loopback-only).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .jevk5_client import DecisionClient, DecisionClientError
from .jevk5_protocol import validate_question

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_QUESTIONS = 16


def answer_batch(
    client: DecisionClient, body: Any, *, lock: threading.Lock | None = None
) -> tuple[int, dict[str, Any]]:
    """Validate + answer a /v1/systemone body. Returns ``(status, payload)``."""
    started = time.perf_counter()
    if (
        not isinstance(body, dict)
        or "state" not in body
        or not isinstance(body.get("questions"), dict)
    ):
        return 400, {"error": "body must be {state, questions:{id: question}}"}
    questions = body["questions"]
    if not questions or len(questions) > MAX_QUESTIONS:
        return 400, {"error": f"between 1 and {MAX_QUESTIONS} questions required"}
    try:
        normalized = {
            str(qid): validate_question(q) for qid, q in questions.items() if isinstance(q, dict)
        }
    except (ValueError, KeyError, TypeError) as exc:
        return 400, {"error": str(exc)}
    if len(normalized) != len(questions):
        return 400, {"error": "every question must be an object"}
    answers: dict[str, Any] = {}
    try:
        ctx = lock if lock is not None else _NullLock()
        with ctx:
            for qid, q in normalized.items():
                answers[qid] = client.decide(body["state"], q)
    except DecisionClientError as exc:
        return 503, {"error": f"decision model unavailable: {exc}"}
    tokens = 0
    for a in answers.values():
        if isinstance(a, dict):
            tokens += int(a.pop("input_tokens", 0) or 0)
    return 200, {
        "model": body.get("model") or "jevk5",
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "latency_ms": round((time.perf_counter() - started) * 1e3, 2),
    }


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def make_handler(client: DecisionClient, name: str = "jevk5") -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "headroom-decision-gateway/1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413 if length > MAX_BODY_BYTES else 400, {"error": "bad body size"})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                self._send(400, {"error": "invalid JSON"})
                return
            status, payload = answer_batch(client, body, lock=lock)
            self._send(status, payload)

        def log_message(self, *args: Any) -> None:  # never log request bodies/state
            pass

    return Handler


def serve(client: DecisionClient, *, port: int = 0, name: str = "jevk5") -> ThreadingHTTPServer:
    """Start a loopback gateway in a daemon thread; returns the server."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(client, name))
    thread = threading.Thread(
        target=server.serve_forever, name="headroom-decision-gateway", daemon=True
    )
    thread.start()
    return server


def register_routes(app: Any, get_advisor: Any, *, dependencies: list[Any] | None = None) -> None:
    """Mount the gateway on a FastAPI app (loopback dependencies supplied by caller)."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    deps = dependencies or []

    @app.get("/v1/intelligence/status", dependencies=deps)
    async def intelligence_status() -> dict[str, Any]:
        advisor = get_advisor()
        return {"advisor": advisor.stats() if advisor is not None else None}

    @app.post("/v1/intelligence/systemone", dependencies=deps)
    async def intelligence_systemone(request: Request) -> JSONResponse:
        advisor = get_advisor()
        client = advisor._resolve_client() if advisor is not None else None  # noqa: SLF001
        if client is None:
            return JSONResponse({"error": "decision model unavailable"}, status_code=503)
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse({"error": "body too large"}, status_code=413)
        try:
            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        import asyncio

        status, payload = await asyncio.to_thread(answer_batch, client, body)
        return JSONResponse(payload, status_code=status)
