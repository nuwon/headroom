"""Shared secret redaction for everything Headroom logs or persists.

This is the one shared redactor. Wire-debug capture
(:mod:`headroom.proxy.wire_debug_redaction_policy`), CCR retrieval logging
(:mod:`headroom.cache.compression_store`) and the agent-state store
(:mod:`headroom.intelligence.agent_state`) all call it.

Two layers:

* **Key-based** (:func:`should_redact_key`). A field whose *name* marks it as a
  credential (``authorization``, ``cookie``, ``*_api_key``, ``password`` and
  similar) has its whole value replaced.
* **Text-based** (:func:`redact_text`). A secret embedded in free text, such
  as ``API_KEY=…``, ``Authorization: Bearer …``, provider key prefixes, private
  key blocks, JWTs or a URL with ``user:password@``, is replaced in place and
  the surrounding text is kept.

Redaction is lossy by design and errs on the side of over-redaction.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

SECRET_KEYS = (
    "authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "x-api-key",
    "openai-api-key",
    "anthropic-api-key",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer",
    "password",
    "secret",
    "token",
    "credential",
)
_SECRET_KEY_SET = frozenset(k.replace("-", "_") for k in SECRET_KEYS)


def should_redact_key(key: str) -> bool:
    """Return whether a field *name* marks its value as a secret."""
    normalized = str(key).lower().replace("-", "_")
    if normalized in _SECRET_KEY_SET:
        return True
    return (
        normalized.endswith("_api_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_password")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
        or normalized.endswith("_token")
        or normalized in {"apikey", "passwd", "private_key", "client_secret"}
    )


# ``NAME=value`` / ``"name": "value"`` where NAME looks like a credential.
SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_-]*)"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]+)"
)
AUTH_VALUE_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")
API_KEY_VALUE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_HEADER_LINE_RE = re.compile(r"(?im)^(\s*(?:authorization|cookie|set-cookie|x-api-key)\s*:\s*).+$")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_PROVIDER_TOKEN_RE = re.compile(
    r"\b(?:"
    r"AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abposr]-[A-Za-z0-9-]{10,}"  # Slack
    r"|AIza[0-9A-Za-z_-]{30,}"  # Google API key
    r"|glpat-[A-Za-z0-9_-]{16,}"  # GitLab
    r"|hf_[A-Za-z0-9]{24,}"  # Hugging Face
    r")\b"
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_URL_CREDENTIALS_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@")


def redact_text(text: str) -> str:
    """Replace secrets embedded in free text and keep the rest unchanged."""
    if not text:
        return text
    out = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    out = _HEADER_LINE_RE.sub(lambda m: m.group(1) + REDACTED, out)
    out = AUTH_VALUE_RE.sub(r"\1 [REDACTED]", out)
    out = SECRET_KEY_VALUE_RE.sub(r"\1\2\3[REDACTED]", out)
    out = API_KEY_VALUE_RE.sub("sk-[REDACTED]", out)
    out = _PROVIDER_TOKEN_RE.sub(REDACTED, out)
    out = _JWT_RE.sub(REDACTED, out)
    return _URL_CREDENTIALS_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}:{REDACTED}@", out)


def redact_value(value: Any, *, text: bool = True) -> Any:
    """Redact a JSON-like structure, key-based plus text-based on strings.

    The shape is preserved. With ``text=False`` only the key-based layer runs,
    which is the wire-debug contract: it captures bodies verbatim apart from
    credential fields.
    """
    if isinstance(value, dict):
        return {
            key: (REDACTED if should_redact_key(str(key)) else redact_value(item, text=text))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, text=text) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, text=text) for item in value)
    if text and isinstance(value, str):
        return redact_text(value)
    return value


_ENV_FILE_RE = re.compile(r"(?i)(?:^|[\\/])\.env(?:\.[A-Za-z0-9_-]+)?$")
_SECRET_FILE_RE = re.compile(
    r"(?i)(?:^|[\\/])(?:id_(?:rsa|dsa|ecdsa|ed25519)(?!\.pub)|[^\\/]*\.(?:pem|key|p12|pfx|jks)"
    r"|credentials(?:\.json)?|\.netrc|\.pgpass|\.npmrc|\.pypirc|secrets?\.(?:json|ya?ml|toml))$"
)


def is_secret_path(path: str) -> bool:
    """True for files whose contents are credentials (``.env``, keys, …)."""
    p = str(path or "").strip()
    if not p:
        return False
    if p.lower().endswith((".env.example", ".env.sample", ".env.template")):
        return False
    return bool(_ENV_FILE_RE.search(p) or _SECRET_FILE_RE.search(p))
