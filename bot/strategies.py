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


def _volume_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    if df.empty or "volume" not in df:
        return 0.0, 0.0, 0.0
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    window = volume.tail(50)
    if window.empty:
        return 0.0, 0.0, 0.0
    current = _safe_float(window.iloc[-1])
    mean = _safe_float(window.mean())
    std = _safe_float(window.std(ddof=0))
    zscore = (current - mean) / std if std > 0 else 0.0
    quantile = float((window <= current).mean()) if len(window) else 0.0
    ratio = current / mean if mean > 0 else 0.0
    return zscore, quantile, ratio


def volume_filter_passes(df: pd.DataFrame, regime: str, cfg: Dict[str, Any]) -> tuple[bool, str]:
    if df.empty:
        return False, "sin velas para filtro de volumen"
    zscore, quantile, ratio = _volume_stats(df)
    regime_name = str(regime or "DEFAULT").upper()
    if regime_name in {"TREND", "TRENDING"}:
        passed = quantile >= 0.30 or zscore >= -0.5
        reason = f"vol q={quantile:.2f} {'ok' if passed else '< 0.30'} (TRENDING), z={zscore:.2f}"
        return passed, reason
    if regime_name == "BREAKOUT":
        passed = quantile >= 0.65 and zscore >= 0.8
        reason = f"vol q={quantile:.2f} / z={zscore:.2f} {'ok' if passed else 'bloquea'} (BREAKOUT)"
        return passed, reason
    if regime_name in {"RANGE", "MEANREV"}:
        passed = quantile <= 0.40
        reason = f"vol q={quantile:.2f} {'ok' if passed else '> 0.40'} (MEANREV)"
        return passed, reason
    if regime_name == "VOLATILE":
        passed = zscore >= 1.0
        reason = f"vol z={zscore:.2f} {'ok' if passed else '< 1.00'} (VOLATILE)"
        return passed, reason
    volume_mult = float(cfg.get("volume_mult", 1.2))
    passed = ratio >= volume_mult
    reason = f"vol ratio={ratio:.2f} {'ok' if passed else f'< {volume_mult:.2f}'} (LEGACY)"
    return passed, reason


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
    supertrend_dir = int(_safe_float(last.get(cols.get("supertrend_dir", "supertrend_direction"))))
    primary_tf = int(cfg.get("primary_timeframe", cfg.get("timeframe", 60)))
    adx_threshold = float(cfg.get("adx_threshold_1h", 18)) if primary_tf >= 60 else float(cfg.get("adx_threshold", 25))
    long_ok = (
        ema_fast > ema_slow
        and close > ema_trend
        and supertrend_dir == 1
        and adx > adx_threshold
        and float(cfg.get("rsi_min", 30)) < rsi < float(cfg.get("rsi_max", 70))
    )
    short_ok = (
        ema_fast < ema_slow
        and close < ema_trend
        and supertrend_dir == -1
        and adx > adx_threshold
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
    atr_z = _safe_float(row.get("atr_pct_zscore_50"))
    donchian_high = _safe_float(frame["high"].rolling(20).max().shift(1).iloc[-1])
    donchian_low = _safe_float(frame["low"].rolling(20).min().shift(1).iloc[-1])
    close = _safe_float(row.get("close"))
    if close > donchian_high and atr_z > 0.5:
        return "LONG"
    if close < donchian_low and atr_z > 0.5:
        return "SHORT"
    return "NEUTRAL"


def signal_vol_meanrev(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> str:
    if df.empty or len(df) < 60:
        return "NEUTRAL"
    runtime_cfg = dict(cfg or {})
    required = {"close", "ema200", "rsi", "atr_pct_zscore_50"}
    frame = df.copy() if required.issubset(df.columns) else _feature_frame(df)
    row = frame.iloc[-1]
    close = _safe_float(row.get("close"))
    ema200 = _safe_float(row.get("ema200"))
    rsi = _safe_float(row.get("rsi"))
    atr_z = _safe_float(row.get("atr_pct_zscore_50"))
    return_1h = _safe_float(frame["close"].pct_change().iloc[-1] * 100.0)
    return_mean = _safe_float(frame["close"].pct_change().mul(100.0).rolling(50).mean().iloc[-1])
    return_std = _safe_float(frame["close"].pct_change().mul(100.0).rolling(50).std(ddof=0).iloc[-1])
    return_z = (return_1h - return_mean) / return_std if return_std > 0 else 0.0
    sigma = float(runtime_cfg.get("vol_meanrev_sigma", 2.5))
    rsi_low = float(runtime_cfg.get("vol_meanrev_rsi_low", 25.0))
    rsi_high = float(runtime_cfg.get("vol_meanrev_rsi_high", 75.0))
    atr_z_min = float(runtime_cfg.get("vol_meanrev_atr_z", 1.5))
    ema_floor = float(runtime_cfg.get("vol_meanrev_ema_floor", 0.85))
    ema_ceiling = float(runtime_cfg.get("vol_meanrev_ema_ceiling", 1.15))
    if return_z <= -sigma and rsi <= rsi_low and atr_z >= atr_z_min and ema200 > 0 and close > ema200 * ema_floor:
        return "LONG"
    if return_z >= sigma and rsi >= rsi_high and atr_z >= atr_z_min and ema200 > 0 and close < ema200 * ema_ceiling:
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
