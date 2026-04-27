"""
Gemini API Key Pool
====================
Thread-safe round-robin pool for multiple Gemini API keys.

On a 429 (rate limit) the caller should call `rotate()` before retrying —
the next `current_key()` will return a different key, giving the request
a fresh quota window.

Key discovery order (first non-empty wins):
  1. GOOGLE_API_KEY_1 … GOOGLE_API_KEY_10  (env vars)
  2. GOOGLE_API_KEY                         (env var, for backward compat)
  3. Same names from st.secrets              (Streamlit Cloud)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional


class GeminiKeyPool:
    """Round-robin API key pool with 429-triggered rotation."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError(
                "GeminiKeyPool requires at least one API key. "
                "Set GOOGLE_API_KEY or GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, … in your environment."
            )
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()
        # per-key: monotonic time after which it's back in rotation
        self._backoff_until: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "GeminiKeyPool":
        """Load all GOOGLE_API_KEY* entries from env (and st.secrets as fallback)."""
        keys = _collect_keys_from_env()

        if not keys:
            keys = _collect_keys_from_streamlit_secrets()

        if not keys:
            raise ValueError(
                "No Gemini API keys found. "
                "Set GOOGLE_API_KEY (or GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, …) "
                "in your .env / Streamlit secrets."
            )

        unique_keys = list(dict.fromkeys(keys))  # deduplicate, preserve order
        if len(unique_keys) > 1:
            print(f"GeminiKeyPool: loaded {len(unique_keys)} API keys (round-robin rotation enabled)")
        return cls(unique_keys)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def key_count(self) -> int:
        return len(self._keys)

    def current_key(self) -> str:
        """Return the currently active key."""
        with self._lock:
            return self._keys[self._index % len(self._keys)]

    def rotate(self) -> str:
        """
        Advance to the next key that is not in backoff.
        Returns the new current key.
        """
        with self._lock:
            now = time.monotonic()
            # Try each key once; if all in backoff, just advance anyway
            for _ in range(len(self._keys)):
                self._index = (self._index + 1) % len(self._keys)
                key = self._keys[self._index]
                if now >= self._backoff_until.get(key, 0.0):
                    return key
            # All keys appear exhausted — still return the next one
            return self._keys[self._index]

    def mark_rate_limited(self, key: str, backoff_seconds: float = 62.0) -> None:
        """Mark a key as temporarily exhausted for backoff_seconds."""
        with self._lock:
            self._backoff_until[key] = time.monotonic() + backoff_seconds

    def all_exhausted(self) -> bool:
        """Return True if every key is currently in a backoff window."""
        now = time.monotonic()
        with self._lock:
            return all(
                now < self._backoff_until.get(k, 0.0) for k in self._keys
            )

    def status(self) -> list[dict]:
        """Return per-key status for debugging."""
        now = time.monotonic()
        result = []
        with self._lock:
            for i, key in enumerate(self._keys):
                backoff = self._backoff_until.get(key, 0.0)
                result.append({
                    "index": i,
                    "key_suffix": f"…{key[-6:]}",
                    "active": i == self._index % len(self._keys),
                    "available": now >= backoff,
                    "backoff_remaining_s": max(0.0, round(backoff - now, 1)),
                })
        return result


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------
def _collect_keys_from_env() -> list[str]:
    keys: list[str] = []
    # Numbered keys take priority: GOOGLE_API_KEY_1 … GOOGLE_API_KEY_10
    for i in range(1, 11):
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    # Plain fallback
    base = os.environ.get("GOOGLE_API_KEY", "").strip()
    if base:
        keys.append(base)
    return keys


def _collect_keys_from_streamlit_secrets() -> list[str]:
    keys: list[str] = []
    try:
        import streamlit as st  # noqa: PLC0415
        secrets = st.secrets
        for i in range(1, 11):
            k = (secrets.get(f"GOOGLE_API_KEY_{i}") or "").strip()
            if k:
                keys.append(k)
        base = (secrets.get("GOOGLE_API_KEY") or "").strip()
        if base:
            keys.append(base)
    except Exception:
        pass
    return keys
