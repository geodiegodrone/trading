from __future__ import annotations

import os
from typing import Dict, List


DEFAULT_SYMBOLS = ["BTCUSDT"]

DEFAULT_CONFIG: Dict[str, float | int] = {
    "timeframe": int(os.getenv("TIMEFRAME", "5")),
    "leverage": int(os.getenv("LEVERAGE", "2")),
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 200,
    "rsi_period": 14,
    "adx_period": 14,
    "atr_period": 14,
    "atr_mult": 1.5,
    "supertrend_period": 14,
    "supertrend_mult": 3.5,
    "adx_threshold": 25,
    "volume_mult": 1.2,
    "rsi_min": 30,
    "rsi_max": 70,
    "ml_threshold": 0.55,
    "daily_risk_cap": 5.0,
    "breakeven_r": 1.0,
    "breakeven_buffer_pct": 0.0,
    "trail_start_r": 1.5,
    "trail_atr_mult": 1.25,
}

SYMBOL_CONFIG: Dict[str, Dict[str, float | int]] = {}


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper().strip()


def parse_symbols(raw: str | None = None) -> List[str]:
    return ["BTCUSDT"]


def get_symbol_config(symbol: str) -> Dict[str, float | int]:
    key = normalize_symbol(symbol)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(SYMBOL_CONFIG.get(key, {}))
    return cfg
