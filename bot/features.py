from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta


FEATURE_COLUMNS = [
    "ema9_minus_ema21_pct",
    "ema21_minus_ema50_pct",
    "price_vs_ema200_pct",
    "rsi",
    "rsi_dist_50",
    "rsi_slope_3",
    "macd_hist",
    "macd_signal_diff",
    "stoch_k",
    "stoch_d",
    "cci_20",
    "williams_r_14",
    "atr_pct",
    "atr_pct_zscore_50",
    "bb_width_20",
    "bb_width_zscore_50",
    "bb_pos_20",
    "true_range_pct",
    "range_5_pct",
    "range_10_pct",
    "return_1_pct",
    "return_3_pct",
    "return_5_pct",
    "return_10_pct",
    "return_20_pct",
    "log_return_skew_20",
    "log_return_kurt_20",
    "volume_vs_ma20_ratio",
    "volume_zscore_50",
    "obv_slope_10",
    "volume_trend_3_ratio",
    "cvd_20_zscore",
    "taker_buy_ratio",
    "price_vs_poc_pct",
    "is_in_value_area",
    "order_flow_imbalance",
    "liquidity_sweep_flag",
    "candle_body_pct",
    "body_avg_3_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "close_pos_5",
    "close_pos_20",
    "adx",
    "adx_strong",
    "di_plus_minus_diff",
    "supertrend_direction",
    "supertrend_aligned_side",
    "hurst_50",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "regime_code",
    "side_buy",
]


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _series_or_empty(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(index=index, dtype=float)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def _rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def _slope(values: np.ndarray) -> float:
        if len(values) != window or np.isnan(values).any():
            return np.nan
        slope, _ = np.polyfit(x, values.astype(float), 1)
        return float(slope)

    return series.rolling(window).apply(_slope, raw=True)


def _hurst_exponent(values: np.ndarray) -> float:
    if len(values) < 20 or np.isnan(values).any():
        return np.nan
    lags = [2, 5, 10, 20]
    tau = []
    valid_lags = []
    for lag in lags:
        if lag >= len(values):
            continue
        diff = values[lag:] - values[:-lag]
        std = np.std(diff)
        if std <= 0:
            continue
        tau.append(math.sqrt(std))
        valid_lags.append(lag)
    if len(valid_lags) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    return float(slope * 2.0)


def _rolling_hurst(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(_hurst_exponent, raw=True)


def _volume_profile_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 100,
    bins: int = 24,
) -> tuple[pd.Series, pd.Series]:
    poc = np.full(len(close), np.nan, dtype=float)
    value_area = np.zeros(len(close), dtype=float)
    price = ((high + low + close) / 3.0).to_numpy(dtype=float)
    high_arr = high.to_numpy(dtype=float)
    low_arr = low.to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    vol_arr = volume.to_numpy(dtype=float)
    for idx in range(window - 1, len(close)):
        start = idx - window + 1
        lo = np.nanmin(low_arr[start: idx + 1])
        hi = np.nanmax(high_arr[start: idx + 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        hist = np.zeros(bins, dtype=float)
        for j in range(start, idx + 1):
            if not np.isfinite(price[j]) or not np.isfinite(vol_arr[j]):
                continue
            bucket = int(np.clip(np.searchsorted(edges, price[j], side="right") - 1, 0, bins - 1))
            hist[bucket] += max(0.0, vol_arr[j])
        if hist.sum() <= 0:
            continue
        poc_idx = int(np.argmax(hist))
        poc[idx] = float(centers[poc_idx])
        target = hist.sum() * 0.70
        order = np.argsort(hist)[::-1]
        chosen = []
        acc = 0.0
        for bucket in order:
            chosen.append(bucket)
            acc += hist[bucket]
            if acc >= target:
                break
        if chosen:
            val = edges[min(chosen)]
            vah = edges[max(chosen) + 1]
            value_area[idx] = 1.0 if val <= close_arr[idx] <= vah else 0.0
    return pd.Series(poc, index=close.index), pd.Series(value_area, index=close.index)


def _coerce_side(value: Any) -> float:
    text = str(value or "").strip().upper()
    if text in {"LONG", "BUY", "1", "TRUE"}:
        return 1.0
    if text in {"SHORT", "SELL", "0", "FALSE", "-1"}:
        return 0.0
    try:
        return 1.0 if float(value) > 0 else 0.0
    except Exception:
        return 0.0


def build_features(df: pd.DataFrame, history_lookback: int = 50) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    for col in ("ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"):
        if col in frame:
            frame[col] = _safe_float_series(frame[col])

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    volume = frame["volume"]

    ema9 = _series_or_empty(ta.ema(close, length=9), frame.index)
    ema21 = _series_or_empty(ta.ema(close, length=21), frame.index)
    ema50 = _series_or_empty(ta.ema(close, length=50), frame.index)
    ema200 = _series_or_empty(ta.ema(close, length=200), frame.index)
    rsi = _series_or_empty(ta.rsi(close, length=14), frame.index)
    macd = ta.macd(close)
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    cci = _series_or_empty(ta.cci(high, low, close, length=20), frame.index)
    willr = _series_or_empty(ta.willr(high, low, close, length=14), frame.index)
    atr = _series_or_empty(ta.atr(high, low, close, length=14), frame.index)
    bb = ta.bbands(close, length=20, std=2.0)
    adx = ta.adx(high, low, close, length=14)
    obv = _series_or_empty(ta.obv(close, volume), frame.index)
    st = ta.supertrend(high, low, close, length=14, multiplier=3.5)

    macd_hist = macd["MACDh_12_26_9"] if macd is not None and "MACDh_12_26_9" in macd else pd.Series(index=frame.index, dtype=float)
    macd_signal = macd["MACDs_12_26_9"] if macd is not None and "MACDs_12_26_9" in macd else pd.Series(index=frame.index, dtype=float)
    macd_line = macd["MACD_12_26_9"] if macd is not None and "MACD_12_26_9" in macd else pd.Series(index=frame.index, dtype=float)
    stoch_k = stoch["STOCHk_14_3_3"] if stoch is not None and "STOCHk_14_3_3" in stoch else pd.Series(index=frame.index, dtype=float)
    stoch_d = stoch["STOCHd_14_3_3"] if stoch is not None and "STOCHd_14_3_3" in stoch else pd.Series(index=frame.index, dtype=float)
    adx_col = "ADX_14"
    dmp_col = "DMP_14"
    dmn_col = "DMN_14"
    adx_values = adx[adx_col] if adx is not None and adx_col in adx else pd.Series(index=frame.index, dtype=float)
    di_plus = adx[dmp_col] if adx is not None and dmp_col in adx else pd.Series(index=frame.index, dtype=float)
    di_minus = adx[dmn_col] if adx is not None and dmn_col in adx else pd.Series(index=frame.index, dtype=float)
    bb_low = bb["BBL_20_2.0"] if bb is not None and "BBL_20_2.0" in bb else pd.Series(index=frame.index, dtype=float)
    bb_mid = bb["BBM_20_2.0"] if bb is not None and "BBM_20_2.0" in bb else pd.Series(index=frame.index, dtype=float)
    bb_high = bb["BBU_20_2.0"] if bb is not None and "BBU_20_2.0" in bb else pd.Series(index=frame.index, dtype=float)
    supertrend_direction = st["SUPERTd_14_3.5"] if st is not None and "SUPERTd_14_3.5" in st else pd.Series(index=frame.index, dtype=float)
    quote_vol = frame["quote_vol"] if "quote_vol" in frame.columns else pd.Series(index=frame.index, dtype=float)
    taker_quote = frame["taker_quote"] if "taker_quote" in frame.columns else pd.Series(index=frame.index, dtype=float)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    candle_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low

    log_returns = np.log(close.replace(0.0, np.nan)).diff()
    highest_5 = high.rolling(5).max()
    lowest_5 = low.rolling(5).min()
    highest_20 = high.rolling(20).max()
    lowest_20 = low.rolling(20).min()
    volume_ma20 = volume.rolling(20).mean()
    obv_slope = _rolling_linear_slope(obv, 10)
    cvd_step = np.where(close >= open_, volume, -volume)
    cvd_roll = pd.Series(cvd_step, index=frame.index).rolling(20).sum()
    poc_100, value_area_100 = _volume_profile_features(high, low, close, volume, window=100, bins=24)

    ts = pd.to_datetime(frame["ts"], unit="ms", utc=True, errors="coerce")
    hours = ts.dt.hour.fillna(0).astype(float)
    weekdays = ts.dt.dayofweek.fillna(0).astype(float)

    features = pd.DataFrame(index=frame.index)
    features["ema9_minus_ema21_pct"] = ((ema9 - ema21) / ema21.replace(0.0, np.nan)) * 100.0
    features["ema21_minus_ema50_pct"] = ((ema21 - ema50) / ema50.replace(0.0, np.nan)) * 100.0
    features["price_vs_ema200_pct"] = ((close - ema200) / ema200.replace(0.0, np.nan)) * 100.0
    features["rsi"] = rsi
    features["rsi_dist_50"] = rsi - 50.0
    features["rsi_slope_3"] = rsi - rsi.shift(3)
    features["macd_hist"] = macd_hist
    features["macd_signal_diff"] = macd_line - macd_signal
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    features["cci_20"] = cci
    features["williams_r_14"] = willr
    features["atr_pct"] = (atr / close.replace(0.0, np.nan)) * 100.0
    features["atr_pct_zscore_50"] = _rolling_zscore(features["atr_pct"], history_lookback)
    features["bb_width_20"] = ((bb_high - bb_low) / bb_mid.replace(0.0, np.nan)) * 100.0
    features["bb_width_zscore_50"] = _rolling_zscore(features["bb_width_20"], history_lookback)
    features["bb_pos_20"] = (close - bb_low) / (bb_high - bb_low).replace(0.0, np.nan)
    features["true_range_pct"] = (true_range / close.replace(0.0, np.nan)) * 100.0
    features["range_5_pct"] = ((highest_5 - lowest_5) / close.replace(0.0, np.nan)) * 100.0
    features["range_10_pct"] = ((high.rolling(10).max() - low.rolling(10).min()) / close.replace(0.0, np.nan)) * 100.0
    features["return_1_pct"] = close.pct_change(1) * 100.0
    features["return_3_pct"] = close.pct_change(3) * 100.0
    features["return_5_pct"] = close.pct_change(5) * 100.0
    features["return_10_pct"] = close.pct_change(10) * 100.0
    features["return_20_pct"] = close.pct_change(20) * 100.0
    features["log_return_skew_20"] = log_returns.rolling(20).skew()
    features["log_return_kurt_20"] = log_returns.rolling(20).kurt()
    features["volume_vs_ma20_ratio"] = volume / volume_ma20.replace(0.0, np.nan)
    features["volume_zscore_50"] = _rolling_zscore(volume, history_lookback)
    features["obv_slope_10"] = obv_slope
    features["volume_trend_3_ratio"] = volume.rolling(3).mean() / volume.rolling(20).mean().replace(0.0, np.nan)
    features["cvd_20_zscore"] = _rolling_zscore(cvd_roll, 100)
    features["taker_buy_ratio"] = taker_quote / quote_vol.replace(0.0, np.nan)
    features["price_vs_poc_pct"] = ((close - poc_100) / poc_100.replace(0.0, np.nan)) * 100.0
    features["is_in_value_area"] = value_area_100
    features["order_flow_imbalance"] = (close - low) / candle_range
    features["liquidity_sweep_flag"] = np.where(
        ((high > highest_20.shift(1)) & (close < highest_20.shift(1)))
        | ((low < lowest_20.shift(1)) & (close > lowest_20.shift(1))),
        1.0,
        0.0,
    )
    features["candle_body_pct"] = (body / open_.replace(0.0, np.nan)) * 100.0
    features["body_avg_3_pct"] = features["candle_body_pct"].rolling(3).mean()
    features["upper_shadow_pct"] = (upper_shadow / open_.replace(0.0, np.nan)) * 100.0
    features["lower_shadow_pct"] = (lower_shadow / open_.replace(0.0, np.nan)) * 100.0
    features["close_pos_5"] = (close - lowest_5) / (highest_5 - lowest_5).replace(0.0, np.nan)
    features["close_pos_20"] = (close - lowest_20) / (highest_20 - lowest_20).replace(0.0, np.nan)
    features["adx"] = adx_values
    features["adx_strong"] = (adx_values > 25.0).astype(float)
    features["di_plus_minus_diff"] = di_plus - di_minus
    features["supertrend_direction"] = supertrend_direction
    features["supertrend_aligned_side"] = 0.0
    features["hurst_50"] = _rolling_hurst(close, history_lookback)
    features["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    features["dayofweek_sin"] = np.sin(2.0 * np.pi * weekdays / 7.0)
    features["dayofweek_cos"] = np.cos(2.0 * np.pi * weekdays / 7.0)
    features["regime_code"] = 0.0
    features["side_buy"] = 0.0

    if "side" in frame.columns:
        features["side_buy"] = frame["side"].map(_coerce_side).astype(float)
        features["supertrend_aligned_side"] = np.where(
            ((features["side_buy"] >= 0.5) & (features["supertrend_direction"] > 0))
            | ((features["side_buy"] < 0.5) & (features["supertrend_direction"] < 0)),
            1.0,
            0.0,
        )

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in FEATURE_COLUMNS:
        if column not in features:
            features[column] = 0.0
    return features[FEATURE_COLUMNS]
