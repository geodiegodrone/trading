from __future__ import annotations

import os
from typing import Dict, List


DEFAULT_SYMBOLS = ["BTCUSDT"]

DEFAULT_CONFIG: Dict[str, float | int] = {
    "timeframe": int(os.getenv("TIMEFRAME", "240")),
    "strategy_mode": os.getenv("STRATEGY_MODE", "swingvolume"),
    "primary_timeframe": int(os.getenv("PRIMARY_TIMEFRAME", "240")),
    "confirmation_timeframe": int(os.getenv("CONFIRMATION_TIMEFRAME", "1440")),
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
    "ml_auto_bootstrap_days": int(os.getenv("ML_BOOTSTRAP_DAYS", "365")),
    "ml_watchdog_hours": float(os.getenv("ML_WATCHDOG_HOURS", "0")),
    "ml_force_bootstrap_after_failures": int(os.getenv("ML_FORCE_BOOTSTRAP_FAILS", "3")),
    "daily_risk_cap": float(os.getenv("DAILY_RISK_CAP", "0.5")),
    "risk_pct_per_trade": float(os.getenv("RISK_PCT_PER_TRADE", "0.15")),
    "max_trades_per_day": int(os.getenv("MAX_TRADES_PER_DAY", "1")),
    "max_open_trades": int(os.getenv("MAX_OPEN_TRADES", "1")),
    "circuit_breaker_enabled": os.getenv("CIRCUIT_BREAKER", "1") == "1",
    "cb_daily_loss_pct": float(os.getenv("CB_DAILY_LOSS_PCT", "0.5")),
    "cb_weekly_loss_pct": float(os.getenv("CB_WEEKLY_LOSS_PCT", "7.0")),
    "cb_consecutive_losses": int(os.getenv("CB_CONSECUTIVE_LOSSES", "3")),
    "cb_rolling_window": int(os.getenv("CB_ROLLING_WINDOW", "20")),
    "cb_rolling_winrate_min": float(os.getenv("CB_ROLLING_WINRATE_MIN", "0.35")),
    "cb_volatility_spike_pct": float(os.getenv("CB_VOL_SPIKE_PCT", "8.0")),
    "cb_cooldown_hours": float(os.getenv("CB_COOLDOWN_HOURS", "48")),
    "cb_rolling_pause_hours": float(os.getenv("CB_ROLLING_PAUSE_HOURS", "168")),
    "breakeven_r": 1.0,
    "breakeven_buffer_pct": 0.0,
    "trail_start_r": 1.5,
    "trail_atr_mult": 1.25,
    "swing_macd_floor_pct": float(os.getenv("SWING_MACD_FLOOR_PCT", "-0.0005")),
    "swing_macd_recovery_low": float(os.getenv("SWING_MACD_RECOVERY_LOW", "-0.0001")),
    "swing_macd_recovery_high": float(os.getenv("SWING_MACD_RECOVERY_HIGH", "0.0005")),
    "swing_volume_mult": float(os.getenv("SWING_VOLUME_MULT", "1.3")),
    "swing_volume_z": float(os.getenv("SWING_VOLUME_Z", "1.5")),
    "swing_body_frac": float(os.getenv("SWING_BODY_FRAC", "0.6")),
    "swing_stop_atr_fallback": float(os.getenv("SWING_STOP_ATR_FALLBACK", "1.2")),
    "swing_tp_r": float(os.getenv("SWING_TP_R", "3.0")),
    "swing_breakeven_r": float(os.getenv("SWING_BREAKEVEN_R", "0.3")),
    "swing_lock_r": float(os.getenv("SWING_LOCK_R", "0.7")),
    "swing_lock_stop_r": float(os.getenv("SWING_LOCK_STOP_R", "0.2")),
    "swing_time_stop_bars": int(os.getenv("SWING_TIME_STOP_BARS", "5")),
    "swing_d1_days": int(os.getenv("SWING_D1_DAYS", "730")),
    "swing_h4_days": int(os.getenv("SWING_H4_DAYS", "730")),
    "meanrev_atr_mult": 0.5,
    "meanrev_tp_at_sma": True,
    "breakout_atr_mult": 1.0,
    "breakout_tp_atr_mult": 3.0,
    "volatile_atr_mult": 2.0,
    "vol_meanrev_sigma": float(os.getenv("VOL_MEANREV_SIGMA", "2.5")),
    "vol_meanrev_rsi_low": float(os.getenv("VOL_MEANREV_RSI_LOW", "25")),
    "vol_meanrev_rsi_high": float(os.getenv("VOL_MEANREV_RSI_HIGH", "75")),
    "vol_meanrev_atr_z": float(os.getenv("VOL_MEANREV_ATR_Z", "1.5")),
    "vol_meanrev_ema_floor": float(os.getenv("VOL_MEANREV_EMA_FLOOR", "0.85")),
    "vol_meanrev_ema_ceiling": float(os.getenv("VOL_MEANREV_EMA_CEILING", "1.15")),
    "vol_meanrev_sl_atr": float(os.getenv("VOL_MEANREV_SL_ATR", "1.0")),
    "vol_meanrev_tp_atr": float(os.getenv("VOL_MEANREV_TP_ATR", "0.6")),
    "vol_meanrev_time_stop_bars": int(os.getenv("VOL_MEANREV_TIME_STOP_BARS", "12")),
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
    cfg["strategy_mode"] = "swingvolume"
    cfg["timeframe"] = 240
    cfg["primary_timeframe"] = 240
    cfg["confirmation_timeframe"] = 1440
    return cfg
