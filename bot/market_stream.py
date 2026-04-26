from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from typing import Dict, Iterable, List, Tuple

import websockets

import activity_log
import binance_broker
from bot_config import normalize_symbol

logger = logging.getLogger("market_stream")

_lock = threading.Lock()
_thread: threading.Thread | None = None
_started = threading.Event()
_stop = threading.Event()
_specs: set[Tuple[str, int]] = set()
_candles: Dict[Tuple[str, int], deque] = {}
_max_candles = 600
_seed_limit = 250


def _key(symbol: str, timeframe: int) -> Tuple[str, int]:
    return normalize_symbol(symbol), int(timeframe)


def _to_candle(payload: dict) -> dict | None:
    k = payload.get("k") or {}
    try:
        ts = int(k.get("t") or 0)
        if ts <= 0:
            return None
        return {
            "ts": ts,
            "open": float(k.get("o") or 0.0),
            "high": float(k.get("h") or 0.0),
            "low": float(k.get("l") or 0.0),
            "close": float(k.get("c") or 0.0),
            "volume": float(k.get("v") or 0.0),
        }
    except Exception:
        return None


def _seed(symbol: str, timeframe: int) -> None:
    key = _key(symbol, timeframe)
    with _lock:
        if key in _candles and len(_candles[key]) > 0:
            return
    try:
        rows = binance_broker.get_kline(symbol, timeframe, _seed_limit)
    except Exception as exc:
        logger.warning("seed failed for %s %sm: %s", symbol, timeframe, exc)
        return
    dq = deque(maxlen=_max_candles)
    for row in rows[-_seed_limit:]:
        try:
            dq.append(
                {
                    "ts": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        except Exception:
            continue
    with _lock:
        _candles[key] = dq


def _upsert(symbol: str, timeframe: int, candle: dict) -> None:
    key = _key(symbol, timeframe)
    with _lock:
        dq = _candles.setdefault(key, deque(maxlen=_max_candles))
        for idx in range(len(dq) - 1, -1, -1):
            if dq[idx]["ts"] == candle["ts"]:
                dq[idx] = candle
                return
        if dq and candle["ts"] < dq[-1]["ts"]:
            return
        dq.append(candle)


async def _worker(symbol: str, timeframe: int) -> None:
    stream = f"{normalize_symbol(symbol).lower()}@kline_{int(timeframe)}m"
    uri = f"wss://fstream.binance.com/ws/{stream}"
    _seed(symbol, timeframe)
    while not _stop.is_set():
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                activity_log.push(symbol, "info", f"Binance WS conectado {symbol} {timeframe}m")
                async for raw in ws:
                    payload = json.loads(raw)
                    if payload.get("e") != "kline":
                        continue
                    candle = _to_candle(payload)
                    if candle:
                        _upsert(symbol, timeframe, candle)
        except Exception as exc:
            activity_log.push(symbol, "error", f"WS {symbol} {timeframe}m reconexión: {exc}")
            await asyncio.sleep(3)


async def _runner(specs: Iterable[Tuple[str, int]]) -> None:
    tasks = [asyncio.create_task(_worker(symbol, timeframe)) for symbol, timeframe in specs]
    if not tasks:
        return
    await asyncio.gather(*tasks)


def start(symbol_timeframes: Iterable[Tuple[str, int]]) -> None:
    global _thread
    specs = {_key(symbol, timeframe) for symbol, timeframe in symbol_timeframes}
    if not specs:
        return
    with _lock:
        _specs.update(specs)
        if _thread and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=lambda: asyncio.run(_runner(sorted(_specs))), daemon=True)
        _thread.start()
        _started.set()


def stop() -> None:
    _stop.set()


def get_candles(symbol: str, timeframe: int, limit: int = 250) -> List[dict]:
    key = _key(symbol, timeframe)
    with _lock:
        dq = _candles.get(key)
        if dq:
            return list(dq)[-limit:]
    _seed(symbol, timeframe)
    with _lock:
        dq = _candles.get(key)
        if dq:
            return list(dq)[-limit:]
    return []
