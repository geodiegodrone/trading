from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import historical_data


ROOT = Path(__file__).resolve().parent
SWING_DATA_DIR = ROOT / "data" / "swingvolume"
SWING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, label: str) -> Path:
    return SWING_DATA_DIR / f"{str(symbol).upper()}_{label}.parquet"


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _timeframe_ms(tf_min: int) -> int:
    return int(tf_min) * 60 * 1000


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return frame


def _save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _is_stale(frame: pd.DataFrame, tf_min: int) -> bool:
    if frame.empty:
        return True
    last_ts = int(frame.iloc[-1]["ts"])
    return (_utc_now_ms() - last_ts) > (_timeframe_ms(tf_min) * 2)


def _sync(symbol: str, tf_min: int, days: int, label: str, force: bool = False) -> tuple[pd.DataFrame, Dict[str, Any]]:
    path = _cache_path(symbol, label)
    cached = _load_cache(path)
    if force or cached.empty or _is_stale(cached, tf_min):
        historical_data.download_history(symbol, tf_min, days, force=False)
        frame = historical_data.load_history(symbol, tf_min)
        if not frame.empty:
            _save_cache(path, frame)
        cached = frame
    meta = historical_data.history_metadata(symbol, tf_min)
    meta.update(
        {
            "cache_path": str(path),
            "cache_rows": int(len(cached)),
            "cache_from": datetime.fromtimestamp(int(cached.iloc[0]["ts"]) / 1000, tz=timezone.utc).isoformat() if not cached.empty else None,
            "cache_to": datetime.fromtimestamp(int(cached.iloc[-1]["ts"]) / 1000, tz=timezone.utc).isoformat() if not cached.empty else None,
        }
    )
    return cached.copy(), meta


def refresh_swingvolume_data(symbol: str = "BTCUSDT", force: bool = False, daily_days: int = 730, h4_days: int = 730) -> Dict[str, Dict[str, Any]]:
    d1, d1_meta = _sync(symbol, 1440, daily_days, "D1", force=force)
    h4, h4_meta = _sync(symbol, 240, h4_days, "H4", force=force)
    return {"D1": d1_meta, "H4": h4_meta, "rows": {"D1": len(d1), "H4": len(h4)}}


def get_daily_data(symbol: str = "BTCUSDT", days: int = 730, force: bool = False) -> pd.DataFrame:
    frame, _ = _sync(symbol, 1440, days, "D1", force=force)
    return frame


def get_hourly4_data(symbol: str = "BTCUSDT", days: int = 730, force: bool = False) -> pd.DataFrame:
    frame, _ = _sync(symbol, 240, days, "H4", force=force)
    return frame


def metadata(symbol: str = "BTCUSDT") -> Dict[str, Dict[str, Any]]:
    return {
        "D1": historical_data.history_metadata(symbol, 1440),
        "H4": historical_data.history_metadata(symbol, 240),
    }
