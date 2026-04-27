"""Shared Gemini quota management for the project.

This keeps embedding and Flash requests inside the configured RPM / TPM / RPD
windows even across repeated runs of the same process.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - macOS/Linux path is used in this workspace
    fcntl = None


PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE = Path(
    os.environ.get("GEMINI_RATE_STATE_FILE", str(PROJECT_ROOT / ".gemini_rate_state.json"))
)

WINDOW_SECONDS = 60
DAY_SECONDS = 24 * 60 * 60

QUOTAS = {
    "embedding": {
        "rpm": int(os.environ.get("GEMINI_EMBEDDING_RPM", "100")),
        "rpm_headroom": float(os.environ.get("GEMINI_EMBEDDING_RPM_HEADROOM", "0.75")),
        "tpm": int(os.environ.get("GEMINI_EMBEDDING_TPM", "30000")),
        "rpd": int(os.environ.get("GEMINI_EMBEDDING_RPD", "1000")),
        "window": WINDOW_SECONDS,
        "day_window": DAY_SECONDS,
    },
    "flash": {
        "rpm": int(os.environ.get("GEMINI_FLASH_RPM", "5")),
        "rpm_headroom": float(os.environ.get("GEMINI_FLASH_RPM_HEADROOM", "0.9")),
        "tpm": int(os.environ.get("GEMINI_FLASH_TPM", "250000")),
        "rpd": int(os.environ.get("GEMINI_FLASH_RPD", "20")),
        "window": WINDOW_SECONDS,
        "day_window": DAY_SECONDS,
    },
}


def estimate_tokens(text: str) -> int:
    """Roughly estimate tokens from text length."""
    if not text:
        return 1
    return max(1, (len(text) + 3) // 4)


def estimate_texts_tokens(texts: list[str]) -> int:
    """Estimate total tokens for a batch of texts."""
    return sum(estimate_tokens(text) for text in texts)


def _load_state(handle) -> dict:
    handle.seek(0)
    raw = handle.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _write_state(handle, state: dict) -> None:
    handle.seek(0)
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def _locked_state():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle, _load_state(handle)
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _prune(events: list[list[float]], cutoff: float) -> list[list[float]]:
    return [event for event in events if event[0] >= cutoff]


def _bucket_state(state: dict, bucket: str) -> dict:
    return state.setdefault(bucket, {"minute": [], "day": [], "last_request_ts": 0.0})


def _fits(events: list[list[float]], tokens: int, rpm: int, tpm: int) -> bool:
    return len(events) < rpm and (sum(event[1] for event in events) + tokens) <= tpm


def _wait_until(events: list[list[float]], tokens: int, rpm: int, tpm: int, window_seconds: int) -> float:
    """Return the earliest timestamp when the next request can fit in the window."""
    running_count = len(events)
    running_tokens = sum(event[1] for event in events)
    wait_until = 0.0

    for timestamp, token_count in events:
        if running_count + 1 <= rpm and running_tokens + tokens <= tpm:
            break
        running_count -= 1
        running_tokens -= token_count
        wait_until = max(wait_until, timestamp + window_seconds)

    if wait_until == 0.0 and events:
        wait_until = events[0][0] + window_seconds

    return wait_until


def acquire(bucket: str, tokens: int, operation: str = "request") -> None:
    """Block until the requested Gemini bucket has enough capacity."""
    if bucket not in QUOTAS:
        raise ValueError(f"Unknown Gemini quota bucket: {bucket}")

    quota = QUOTAS[bucket]
    effective_rpm = max(1, int(quota["rpm"] * quota["rpm_headroom"]))
    min_request_gap = quota["window"] / effective_rpm

    if tokens > quota["tpm"]:
        raise ValueError(
            f"{operation} needs about {tokens} tokens, which exceeds the {bucket} TPM limit of {quota['tpm']}."
        )

    while True:
        with _locked_state() as (handle, state):
            now = time.time()
            bucket_state = _bucket_state(state, bucket)

            minute_events = _prune(bucket_state.get("minute", []), now - quota["window"])
            day_events = _prune(bucket_state.get("day", []), now - quota["day_window"])
            bucket_state["minute"] = minute_events
            bucket_state["day"] = day_events
            last_request_ts = float(bucket_state.get("last_request_ts", 0.0))

            minute_ok = _fits(minute_events, tokens, effective_rpm, quota["tpm"])
            day_ok = _fits(day_events, tokens, quota["rpd"], quota["tpm"] * quota["rpd"])
            gap_ok = (now - last_request_ts) >= min_request_gap

            if minute_ok and day_ok and gap_ok:
                event = [now, tokens]
                minute_events.append(event)
                day_events.append(event)
                bucket_state["last_request_ts"] = now
                _write_state(handle, state)
                return

            minute_wait = _wait_until(minute_events, tokens, effective_rpm, quota["tpm"], quota["window"])
            day_wait = _wait_until(day_events, tokens, quota["rpd"], quota["tpm"] * quota["rpd"], quota["day_window"])
            gap_wait = last_request_ts + min_request_gap
            wait_until = max(minute_wait, day_wait, gap_wait)

        sleep_for = max(1.0, wait_until - time.time())
        print(
            f"  ⚠ Gemini {bucket} quota reached for {operation}; sleeping {sleep_for:.0f}s to stay within limits..."
        )
        time.sleep(sleep_for)