from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pandas_ta as ta

from features import build_features


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = {"atr_pct_zscore_50", "bb_width_20", "bb_pos_20", "adx", "bb_lower_20", "bb_upper_20"}
    if required.issubset(df.columns):
        return df
    features = build_features(df)
    merged = df.copy()
    for column in features.columns:
        merged[column] = features[column]
    bb = ta.bbands(merged["close"], length=20, std=2.0)
    if bb is not None:
        merged["bb_lower_20"] = bb.get("BBL_20_2.0", 0.0)
        merged["bb_mid_20"] = bb.get("BBM_20_2.0", 0.0)
        merged["bb_upper_20"] = bb.get("BBU_20_2.0", 0.0)
    return merged


def signal_trend(last: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> str:
    if last is None or df.empty:
        return "NEUTRAL"
    cols = df.attrs.get("indicator_cols", {})
    ema_fast = _safe_float(last.get(cols.get("ema_fast", "ema9")))
    ema_slow = _safe_float(last.get(cols.get("ema_slow", "ema21")))
    ema_trend = _safe_float(last.get(cols.get("ema_trend", "ema200")))
    close = _safe_float(last.get("close"))
    rsi = _safe_float(last.get("rsi"))
    adx = _safe_float(last.get("adx"))
    volume = _safe_float(last.get("volume"))
    vol_ma20 = _safe_float(last.get("vol_ma20"))
    supertrend_dir = int(_safe_float(last.get(cols.get("supertrend_dir", "supertrend_direction"))))
    long_ok = (
        ema_fast > ema_slow
        and close > ema_trend
        and supertrend_dir == 1
        and adx > float(cfg.get("adx_threshold", 25))
        and volume > (vol_ma20 * float(cfg.get("volume_mult", 1.2)) if vol_ma20 else 0.0)
        and float(cfg.get("rsi_min", 30)) < rsi < float(cfg.get("rsi_max", 70))
    )
    short_ok = (
        ema_fast < ema_slow
        and close < ema_trend
        and supertrend_dir == -1
        and adx > float(cfg.get("adx_threshold", 25))
        and volume > (vol_ma20 * float(cfg.get("volume_mult", 1.2)) if vol_ma20 else 0.0)
        and float(cfg.get("rsi_min", 30)) < rsi < float(cfg.get("rsi_max", 70))
    )
    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return "NEUTRAL"


def signal_meanrev(last: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> str:
    if last is None or df.empty:
        return "NEUTRAL"
    frame = _feature_frame(df)
    row = frame.iloc[-1]
    close = _safe_float(row.get("close"))
    rsi = _safe_float(row.get("rsi"))
    adx = _safe_float(row.get("adx"))
    atr_z = _safe_float(row.get("atr_pct_zscore_50"))
    bb_width = _safe_float(row.get("bb_width_20"))
    bb_width_z = _safe_float(row.get("bb_width_zscore_50"))
    lower = _safe_float(row.get("bb_lower_20"))
    upper = _safe_float(row.get("bb_upper_20"))
    if not (adx < 20.0 and -1.0 <= atr_z <= 0.5 and bb_width_z < 0.5 and bb_width > 0):
        return "NEUTRAL"
    if close <= lower and rsi <= 30.0:
        return "LONG"
    if close >= upper and rsi >= 70.0:
        return "SHORT"
    return "NEUTRAL"


def signal_breakout(last: Dict[str, Any], df: pd.DataFrame, cfg: Dict[str, Any]) -> str:
    if last is None or df.empty or len(df) < 25:
        return "NEUTRAL"
    frame = df.copy() if {"atr_pct_zscore_50", "volume"}.issubset(df.columns) else _feature_frame(df)
    row = frame.iloc[-1]
    ranges = (frame["high"] - frame["low"]).tail(7)
    if len(ranges) < 7 or ranges.iloc[-1] != ranges.min():
        return "NEUTRAL"
    volume = _safe_float(row.get("volume"))
    vol_ma20 = _safe_float(frame["volume"].rolling(20).mean().iloc[-1])
    atr_z = _safe_float(row.get("atr_pct_zscore_50"))
    donchian_high = _safe_float(frame["high"].rolling(20).max().shift(1).iloc[-1])
    donchian_low = _safe_float(frame["low"].rolling(20).min().shift(1).iloc[-1])
    close = _safe_float(row.get("close"))
    volume_ok = volume > vol_ma20 * 1.5 if vol_ma20 > 0 else False
    if close > donchian_high and volume_ok and atr_z > 0.5:
        return "LONG"
    if close < donchian_low and volume_ok and atr_z > 0.5:
        return "SHORT"
    return "NEUTRAL"


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    bb = ta.bbands(frame["close"], length=20, std=2.0)
    if bb is not None:
        frame["bb_lower_20"] = bb.get("BBL_20_2.0", 0.0)
        frame["bb_mid_20"] = bb.get("BBM_20_2.0", 0.0)
        frame["bb_upper_20"] = bb.get("BBU_20_2.0", 0.0)
    frame["donchian_high_20"] = frame["high"].rolling(20).max()
    frame["donchian_low_20"] = frame["low"].rolling(20).min()
    frame["nr7_range"] = (frame["high"] - frame["low"]).rolling(7).apply(lambda values: values[-1] if len(values) == 7 else 0.0, raw=True)
    return _feature_frame(frame)
