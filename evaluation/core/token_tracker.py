"""Per-system LLM token accounting via a global litellm CustomLogger.

The evaluate harness runs systems sequentially (one full system to completion
before the next starts), so a single module-level "active system" string is
enough to attribute every LiteLLM call to the system currently running. Both
BT-GraphRAG and vanilla GraphRAG route their completions through
`graphrag_llm` -> LiteLLM, so a single callback captures both worlds.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


@dataclass
class _ModelTokens:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _SystemTokens:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, _ModelTokens] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_model": {
                m: {
                    "calls": t.calls,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "total_tokens": t.total_tokens,
                }
                for m, t in self.by_model.items()
            },
        }


class TokenTracker(CustomLogger):
    """Aggregates LiteLLM token usage per active system."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._totals: dict[str, _SystemTokens] = {}
        self._active_system: str = "_unattributed"

    def set_active_system(self, system: str) -> None:
        with self._lock:
            self._active_system = system or "_unattributed"

    def _record(self, system: str, model: str, prompt_t: int, comp_t: int) -> None:
        sys_t = self._totals.setdefault(system, _SystemTokens())
        sys_t.calls += 1
        sys_t.prompt_tokens += prompt_t
        sys_t.completion_tokens += comp_t
        sys_t.total_tokens += prompt_t + comp_t
        m = sys_t.by_model.setdefault(model, _ModelTokens())
        m.calls += 1
        m.prompt_tokens += prompt_t
        m.completion_tokens += comp_t
        m.total_tokens += prompt_t + comp_t

    def _extract(self, kwargs: dict, response_obj: Any) -> tuple[str, int, int]:
        model = kwargs.get("model") or getattr(response_obj, "model", "") or "unknown"
        usage = getattr(response_obj, "usage", None)
        if usage is None and isinstance(response_obj, dict):
            usage = response_obj.get("usage")
        prompt_t = 0
        comp_t = 0
        if usage is not None:
            prompt_t = (
                getattr(usage, "prompt_tokens", None)
                if not isinstance(usage, dict)
                else usage.get("prompt_tokens", 0)
            ) or 0
            comp_t = (
                getattr(usage, "completion_tokens", None)
                if not isinstance(usage, dict)
                else usage.get("completion_tokens", 0)
            ) or 0
        return str(model), int(prompt_t), int(comp_t)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D401
        model, prompt_t, comp_t = self._extract(kwargs, response_obj)
        with self._lock:
            self._record(self._active_system, model, prompt_t, comp_t)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        model, prompt_t, comp_t = self._extract(kwargs, response_obj)
        with self._lock:
            self._record(self._active_system, model, prompt_t, comp_t)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {sys: t.to_dict() for sys, t in self._totals.items()}

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()


_TRACKER: TokenTracker | None = None
_INSTALL_LOCK = threading.Lock()


def get_tracker() -> TokenTracker:
    """Return the global TokenTracker, installing the LiteLLM callback once."""
    global _TRACKER
    with _INSTALL_LOCK:
        if _TRACKER is None:
            _TRACKER = TokenTracker()
            # Register on both sync + async callback lists; litellm fans out to
            # whichever matches the call path.
            if _TRACKER not in litellm.callbacks:
                litellm.callbacks.append(_TRACKER)
            if _TRACKER not in litellm.success_callback:
                litellm.success_callback.append(_TRACKER)
        return _TRACKER


@contextmanager
def active_system(system: str):
    """Mark all LLM calls made inside the block as belonging to `system`."""
    tracker = get_tracker()
    prev = tracker._active_system  # noqa: SLF001 — internal swap
    tracker.set_active_system(system)
    try:
        yield tracker
    finally:
        tracker.set_active_system(prev)
