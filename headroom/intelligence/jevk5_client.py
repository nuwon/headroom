"""Typed JevK5 decision clients.

* :class:`GGUFDecisionClient` reads JevK5 through a running ``llama-server``
  exactly like upstream ``jevk5.JevK5GGUF``: tokenize the chat-templated prompt
  with ``parse_special=true``, evaluate it with ``n_predict=1``,
  ``n_probs=top_k``, ``temperature=0``, ``cache_prompt=false``, read the
  option letters' log-probabilities and renormalize under the file's
  calibration temperature. Questions with >16 options use the knockout
  combination (``jevk5.prompt.spread``). Nothing is ever generated.
* :class:`SystemOneClient` talks to an operator-managed TypeSafe-style
  ``/v1/systemone`` endpoint (``jevk5-serve`` or Headroom's own gateway).

Both return the upstream answer shape: ``{"type", "confidence",
"input_tokens", "noul"|"choice"+"probabilities"|"score"+"probabilities"}``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from . import jevk5_protocol as proto


class DecisionClient(Protocol):
    def decide(self, state: Any, question: dict[str, Any]) -> dict[str, Any]: ...


class DecisionClientError(RuntimeError):
    pass


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise DecisionClientError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DecisionClientError(f"{type(exc).__name__} contacting {url}") from exc
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise DecisionClientError(f"invalid JSON from {url}") from exc
    if not isinstance(data, dict):
        raise DecisionClientError(f"unexpected payload from {url}")
    return data


class GGUFDecisionClient:
    def __init__(
        self,
        url: str,
        *,
        temperature: float,
        knockout_temperature: float | None,
        top_k: int = 40,
        timeout_s: float = 1.2,
        method: str = "knockout",
    ) -> None:
        if method not in proto.METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.url = url.rstrip("/")
        self.temperature = float(temperature)
        self.knockout_temperature = knockout_temperature
        self.top_k = max(16, int(top_k))
        self.timeout_s = float(timeout_s)
        self.method = method
        self.missing = 0
        self.last_seconds = 0.0

    def _logprobs(self, prompt: str) -> tuple[dict[str, float], int]:
        tokens = _post(
            f"{self.url}/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
            self.timeout_s,
        ).get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise DecisionClientError("llama-server /tokenize returned no tokens")
        out = _post(
            f"{self.url}/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": self.top_k,
                "temperature": 0,
                "cache_prompt": False,
            },
            self.timeout_s,
        )
        try:
            top = out["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecisionClientError(
                "llama-server /completion returned no completion_probabilities[0].top_logprobs "
                "(llama.cpp too old for n_probs log-probabilities?)"
            ) from exc
        seen: dict[str, float] = {}
        for entry in top:
            if isinstance(entry, dict) and "token" in entry and "logprob" in entry:
                seen[str(entry["token"])] = float(entry["logprob"])
        return seen, int(out.get("tokens_evaluated", len(tokens)) or 0)

    def probabilities(self, state: Any, question: dict[str, Any]) -> tuple[dict[str, float], int]:
        question = proto.validate_question(question)
        options = proto.decision_options(question)
        tokens = 0
        started = time.perf_counter()

        def read(texts: list[str]) -> list[float]:
            nonlocal tokens
            prompt = proto.prompt_text(state, question["instructions"], texts)
            seen, count = self._logprobs(prompt)
            tokens += count
            probs, missing = proto.letter_distribution(seen, len(texts), self.temperature)
            if missing:
                self.missing += 1
            return probs

        second = self.knockout_temperature if self.method == "knockout" else None
        probs = proto.spread(read, [text for _, text in options], self.method, second)
        self.last_seconds = time.perf_counter() - started
        return {key: v for (key, _), v in zip(options, probs, strict=True)}, tokens

    def decide(self, state: Any, question: dict[str, Any]) -> dict[str, Any]:
        probs, tokens = self.probabilities(state, question)
        return proto.answer(question, probs, tokens)


class SystemOneClient:
    """Client for an operator-managed ``/v1/systemone`` endpoint."""

    def __init__(self, url: str, *, timeout_s: float = 1.2) -> None:
        base = url.rstrip("/")
        self.endpoint = base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"
        self.timeout_s = float(timeout_s)

    def decide(self, state: Any, question: dict[str, Any]) -> dict[str, Any]:
        question = proto.validate_question(question)
        data = _post(self.endpoint, {"state": state, "questions": {"q": question}}, self.timeout_s)
        answers = data.get("answers")
        if not isinstance(answers, dict) or "q" not in answers:
            raise DecisionClientError("systemone response missing answers.q")
        answer = dict(answers["q"])
        usage = data.get("usage") or {}
        answer.setdefault("input_tokens", int(usage.get("input_tokens", 0) or 0))
        return answer
