from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "historical"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
RATE_SLEEP_SECONDS = 0.12
MAX_LIMIT = 1000
HISTORY_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_vol",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _timeframe_label(tf_min: int) -> str:
    tf = int(tf_min)
    if tf < 60:
        return f"{tf}m"
    if tf % 1440 == 0:
        return f"{tf // 1440}d"
    if tf % 60 == 0:
        return f"{tf // 60}h"
    return f"{tf}m"


def _timeframe_ms(tf_min: int) -> int:
    return int(tf_min) * 60 * 1000


def _parquet_path(symbol: str, tf_min: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{str(symbol).upper()}_{int(tf_min)}m.parquet"


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"])


def _request_klines(symbol: str, tf_min: int, start_ms: int, end_ms: int, limit: int = MAX_LIMIT, retries: int = 5) -> List[list]:
    params = {
        "symbol": str(symbol).upper(),
        "interval": _timeframe_label(tf_min),
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": int(limit),
    }
    backoff = RATE_SLEEP_SECONDS
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            with urlopen(f"{BINANCE_KLINES_URL}?{urlencode(params)}", timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 3.0)
    if last_error is not None:
        raise last_error
    return []


def fetch_klines(symbol: str, tf_min: int, start_ms: int, end_ms: int) -> List[list]:
    rows: List[list] = []
    cursor = int(start_ms)
    step_ms = _timeframe_ms(tf_min)
    while cursor < int(end_ms):
        batch = _request_klines(symbol, tf_min, cursor, end_ms, limit=MAX_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(RATE_SLEEP_SECONDS)
        if len(batch) < MAX_LIMIT:
            break
    return rows


def _rows_to_frame(rows: Iterable[list]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=HISTORY_COLUMNS)
    if frame.empty:
        return _empty_history()
    frame = frame[["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)


def load_history(symbol: str, tf_min: int) -> pd.DataFrame:
    path = _parquet_path(symbol, tf_min)
    if not path.exists():
        return _empty_history()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return _empty_history()
    if frame.empty:
        return _empty_history()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)


def history_metadata(symbol: str, tf_min: int) -> Dict[str, object]:
    path = _parquet_path(symbol, tf_min)
    frame = load_history(symbol, tf_min)
    step_ms = _timeframe_ms(tf_min)
    if frame.empty:
        return {
            "symbol": str(symbol).upper(),
            "timeframe": int(tf_min),
            "rows": 0,
            "from": None,
            "to": None,
            "gaps": 0,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0.0,
            "path": str(path),
        }
    diffs = frame["ts"].diff().fillna(step_ms)
    gaps = int((diffs != step_ms).sum())
    return {
        "symbol": str(symbol).upper(),
        "timeframe": int(tf_min),
        "rows": int(len(frame)),
        "from": datetime.fromtimestamp(int(frame.iloc[0]["ts"]) / 1000, tz=timezone.utc).isoformat(),
        "to": datetime.fromtimestamp(int(frame.iloc[-1]["ts"]) / 1000, tz=timezone.utc).isoformat(),
        "gaps": gaps,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0.0,
        "path": str(path),
    }


def download_history(symbol: str, tf_min: int, days: int, force: bool = False) -> Dict[str, object]:
    symbol = str(symbol).upper()
    tf_min = int(tf_min)
    now_ms = _utc_now_ms()
    target_start = now_ms - int(days) * 24 * 60 * 60 * 1000
    step_ms = _timeframe_ms(tf_min)
    path = _parquet_path(symbol, tf_min)

    existing = _empty_history() if force else load_history(symbol, tf_min)
    parts: List[pd.DataFrame] = []
    fetched_batches = 0

    if existing.empty:
        fetched = _rows_to_frame(fetch_klines(symbol, tf_min, target_start, now_ms))
        fetched_batches += 1 if not fetched.empty else 0
        if not fetched.empty:
            parts.append(fetched)
    else:
        parts.append(existing)
        first_ts = int(existing.iloc[0]["ts"])
        last_ts = int(existing.iloc[-1]["ts"])
        if target_start < first_ts:
            older = _rows_to_frame(fetch_klines(symbol, tf_min, target_start, max(target_start, first_ts - step_ms)))
            fetched_batches += 1 if not older.empty else 0
            if not older.empty:
                parts.append(older)
        if last_ts + step_ms < now_ms:
            newer = _rows_to_frame(fetch_klines(symbol, tf_min, last_ts + step_ms, now_ms))
            fetched_batches += 1 if not newer.empty else 0
            if not newer.empty:
                parts.append(newer)

    merged = pd.concat(parts, ignore_index=True) if parts else _empty_history()
    if merged.empty:
        merged = _empty_history()
    else:
        for column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged = merged.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    merged.to_parquet(path, index=False, compression="zstd")
    meta = history_metadata(symbol, tf_min)
    meta["fetched_batches"] = int(fetched_batches)
    meta["days_requested"] = int(days)
    meta["force"] = bool(force)
    return meta


def _parse_timeframes(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [60]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download incremental Binance BTC history to parquet")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--timeframes", default="5,15,60,240")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = []
    for tf_min in _parse_timeframes(args.timeframes):
        meta = download_history(args.symbol, tf_min, args.days, force=bool(args.force))
        results.append(meta)
        print(
            f"{args.symbol} {tf_min}m rows={meta['rows']} from={meta['from']} to={meta['to']} "
            f"gaps={meta['gaps']} size_mb={meta['size_mb']}"
        )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
