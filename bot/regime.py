from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta

from features import _rolling_hurst

BASE_DIR = Path(__file__).resolve().parent
HMM_PATH = BASE_DIR / "hmm_model.pkl"

try:
    from hmmlearn.hmm import GaussianHMM

    USING_HMM = True
except Exception:  # pragma: no cover
    USING_HMM = False
    GaussianHMM = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rule_based_regime(df: pd.DataFrame) -> Dict[str, Any]:
    frame = df.copy()
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    returns = np.log(close.replace(0.0, np.nan)).diff()
    hurst = _rolling_hurst(close, 100).iloc[-1] if len(frame) >= 100 else np.nan
    adx_df = ta.adx(frame["high"], frame["low"], frame["close"], length=14)
    adx = _safe_float(adx_df["ADX_14"].iloc[-1] if adx_df is not None and "ADX_14" in adx_df else 0.0)
    realized_vol = _safe_float(returns.tail(50).std(ddof=0), 0.0) * np.sqrt(365.0 * 24.0) * 100.0
    vol_history = returns.rolling(50).std(ddof=0) * np.sqrt(365.0 * 24.0) * 100.0
    vol_q33 = _safe_float(vol_history.tail(200).quantile(0.33), realized_vol)
    vol_q66 = _safe_float(vol_history.tail(200).quantile(0.66), realized_vol)
    if adx > 30.0 and _safe_float(hurst, 0.0) > 0.55:
        trend_strength = "strong"
    elif adx < 20.0 and _safe_float(hurst, 1.0) < 0.5:
        trend_strength = "weak"
    else:
        trend_strength = "neutral"
    if realized_vol < vol_q33:
        vol_regime = "low"
    elif realized_vol > vol_q66:
        vol_regime = "high"
    else:
        vol_regime = "normal"
    if trend_strength == "strong" and vol_regime in {"low", "normal"}:
        regime = "TREND"
    elif trend_strength == "weak" and vol_regime == "low":
        regime = "RANGE"
    elif vol_regime == "high":
        regime = "VOLATILE"
    else:
        regime = "MIXED"
    return {
        "regime": regime,
        "trend_strength": trend_strength,
        "vol_regime": vol_regime,
        "adx": adx,
        "hurst": _safe_float(hurst, 0.0),
        "realized_volatility_pct": realized_vol,
        "source": "rule",
    }


def _load_hmm() -> Any | None:
    if not USING_HMM or not HMM_PATH.exists():
        return None
    try:
        with HMM_PATH.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _fit_hmm(df: pd.DataFrame) -> Any | None:
    if not USING_HMM or len(df) < 200:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    returns = np.log(close.replace(0.0, np.nan)).diff().fillna(0.0)
    abs_returns = returns.abs()
    sample = pd.DataFrame({"log_returns": returns, "abs_returns": abs_returns}).dropna()
    if len(sample) < 150:
        return None
    try:
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)
        model.fit(sample[["log_returns", "abs_returns"]].to_numpy())
        with HMM_PATH.open("wb") as handle:
            pickle.dump(model, handle)
        return model
    except Exception:
        return None


def _hmm_regime(df: pd.DataFrame) -> Dict[str, Any] | None:
    if not USING_HMM:
        return None
    model = _load_hmm() or _fit_hmm(df)
    if model is None:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    returns = np.log(close.replace(0.0, np.nan)).diff().fillna(0.0)
    abs_returns = returns.abs()
    sample = pd.DataFrame({"log_returns": returns, "abs_returns": abs_returns}).dropna()
    if len(sample) < 50:
        return None
    try:
        states = model.predict(sample[["log_returns", "abs_returns"]].to_numpy())
    except Exception:
        return None
    state_means = []
    for state_id in range(model.n_components):
        mask = states == state_id
        subset = sample.loc[mask]
        if subset.empty:
            state_means.append((state_id, 0.0, 0.0))
        else:
            state_means.append((state_id, float(subset["log_returns"].abs().mean()), float(subset["abs_returns"].mean())))
    sorted_states = sorted(state_means, key=lambda item: (item[2], item[1]))
    quiet_state = sorted_states[0][0]
    volatile_state = sorted_states[-1][0]
    trend_state = sorted_states[1][0] if len(sorted_states) > 2 else volatile_state
    current_state = int(states[-1])
    if current_state == volatile_state:
        regime = "VOLATILE"
        trend_strength = "neutral"
        vol_regime = "high"
    elif current_state == trend_state:
        regime = "TREND"
        trend_strength = "strong"
        vol_regime = "normal"
    else:
        regime = "RANGE"
        trend_strength = "weak"
        vol_regime = "low"
    return {
        "regime": regime,
        "trend_strength": trend_strength,
        "vol_regime": vol_regime,
        "adx": 0.0,
        "hurst": 0.0,
        "realized_volatility_pct": float(abs_returns.tail(50).std(ddof=0) * np.sqrt(365.0 * 24.0) * 100.0),
        "source": "hmm",
        "state": current_state,
    }


def classify_regime(df: pd.DataFrame, lookback: int = 200) -> Dict[str, Any]:
    if df.empty:
        return {
            "regime": "MIXED",
            "trend_strength": "neutral",
            "vol_regime": "normal",
            "adx": 0.0,
            "hurst": 0.0,
            "realized_volatility_pct": 0.0,
            "source": "rule",
        }
    window = df.tail(max(lookback, 200)).copy()
    hmm_result = _hmm_regime(window)
    if hmm_result:
        return hmm_result
    return _rule_based_regime(window)
