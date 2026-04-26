import threading
from collections import deque
from datetime import datetime, timezone

_lock = threading.Lock()
_events: deque = deque(maxlen=200)


def push(symbol: str, event_type: str, message: str, data: dict = None):
    """
    event_type: one of:
      signal, trade_open, trade_close, ml, risk, regime, error, info
    """
    with _lock:
        _events.appendleft(
            {
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "symbol": symbol,
                "type": event_type,
                "message": message,
                "data": data or {},
            }
        )


def get_recent(n: int = 50) -> list:
    with _lock:
        return list(_events)[:n]
