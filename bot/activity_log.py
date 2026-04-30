import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "activity_log.jsonl"
_lock = threading.Lock()
_events: deque = deque(maxlen=200)


def _append_file(event: dict) -> None:
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_file() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    events = []
    for line in lines[-500:]:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def push(symbol: str, event_type: str, message: str, data: dict = None):
    with _lock:
        event = {
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "symbol": symbol,
            "type": event_type,
            "message": message,
            "data": data or {},
        }
        _events.appendleft(event)
        _append_file(event)


def get_recent(n: int = 50, event_type: str | None = None) -> list:
    with _lock:
        memory = list(_events)
    merged = {}
    for event in _read_file() + list(reversed(memory)):
        if not isinstance(event, dict):
            continue
        if event_type and str(event.get("type") or "").lower() != str(event_type).lower():
            continue
        key = (
            str(event.get("ts") or ""),
            str(event.get("symbol") or ""),
            str(event.get("type") or ""),
            str(event.get("message") or ""),
        )
        merged[key] = event
    return list(reversed(list(merged.values())))[:n][::-1]
