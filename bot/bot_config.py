from __future__ import annotations

import os
from typing import Dict, List


DEFAULT_SYMBOLS = ["BTCUSDT"]

DEFAULT_CONFIG: Dict[str, float | int] = {
    "timeframe": int(os.getenv("TIMEFRAME", "60")),
    "strategy_mode": os.getenv("STRATEGY_MODE", "regime"),
    "primary_timeframe": int(os.getenv("PRIMARY_TIMEFRAME", "60")),
    "confirmation_timeframe": int(os.getenv("CONFIRMATION_TIMEFRAME", "240")),
    "leverage": int(os.getenv("LEVERAGE", "2")),
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_confirm": 50,
    "ema_trend": 200,
    "rsi_period": 14,
    "adx_period": 14,
    "atr_period": 14,
    "atr_mult": 1.5,
    "supertrend_period": 14,
    "supertrend_mult": 3.5,
    "adx_threshold": 25,
    "volume_mult": 1.2,
    "adx_threshold_1h": float(os.getenv("ADX_THRESHOLD_1H", "18")),
    "volume_mult_1h": float(os.getenv("VOLUME_MULT_1H", "1.0")),
    "rsi_min": 30,
    "rsi_max": 70,
    "ml_threshold": 0.55,
    "ml_auto_bootstrap_days": int(os.getenv("ML_BOOTSTRAP_DAYS", "120")),
    "ml_watchdog_hours": float(os.getenv("ML_WATCHDOG_HOURS", "24")),
    "ml_force_bootstrap_after_failures": int(os.getenv("ML_FORCE_BOOTSTRAP_FAILS", "3")),
    "daily_risk_cap": 5.0,
    "circuit_breaker_enabled": os.getenv("CIRCUIT_BREAKER", "1") == "1",
    "cb_daily_loss_pct": float(os.getenv("CB_DAILY_LOSS_PCT", "3.0")),
    "cb_weekly_loss_pct": float(os.getenv("CB_WEEKLY_LOSS_PCT", "7.0")),
    "cb_consecutive_losses": int(os.getenv("CB_CONSECUTIVE_LOSSES", "4")),
    "cb_rolling_window": int(os.getenv("CB_ROLLING_WINDOW", "20")),
    "cb_rolling_winrate_min": float(os.getenv("CB_ROLLING_WINRATE_MIN", "0.30")),
    "cb_volatility_spike_pct": float(os.getenv("CB_VOL_SPIKE_PCT", "8.0")),
    "cb_cooldown_hours": float(os.getenv("CB_COOLDOWN_HOURS", "12")),
    "breakeven_r": 1.0,
    "breakeven_buffer_pct": 0.0,
    "trail_start_r": 1.5,
    "trail_atr_mult": 1.25,
    "meanrev_atr_mult": 0.5,
    "meanrev_tp_at_sma": True,
    "breakout_atr_mult": 1.0,
    "breakout_tp_atr_mult": 3.0,
    "volatile_atr_mult": 2.0,
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
