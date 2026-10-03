"""Tool-result interceptors.

An interceptor rewrites a single tool_result's text before it reaches the
model. Each interceptor is self-contained: declare a `matches()` predicate
and a `transform()` function, register it in the `INTERCEPTORS` list, and
the proxy pipeline will call it automatically.

Adding a new interceptor later is one file plus one `register()` call — no
proxy or metrics changes required.
"""

# Side-effect: register the built-in interceptors.
from . import astgrep  # noqa: F401
from .base import (
    INTERCEPTORS,
    InterceptionResult,
    ToolResultInterceptor,
    ToolResultInterceptorTransform,
    TransformSpan,
    apply_to_messages,
    interceptor_failure_counts,
    register,
)


def enable_rich_interception(enabled: bool = True) -> None:
    """Turn on rich interception (Optimization 14) at the composition root.

    Adds the test-runner interceptor and makes the ast-grep Read outline store
    the exact original in CCR (verified) with a retrieval marker.
    """
    from . import astgrep as _astgrep
    from . import test_runner as _test_runner

    _astgrep.set_rich_mode(enabled)
    if enabled:
        _test_runner.enable()


__all__ = [
    "enable_rich_interception",
    "INTERCEPTORS",
    "InterceptionResult",
    "ToolResultInterceptor",
    "ToolResultInterceptorTransform",
    "TransformSpan",
    "apply_to_messages",
    "interceptor_failure_counts",
    "register",
]
