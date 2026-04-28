from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

import regime


def _frame(volatility: float = 0.2, trend: float = 0.0, spike: float = 0.0) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=260, freq="h", tz="UTC")
    base = 100 + np.cumsum(np.full(len(ts), trend) + np.random.default_rng(42).normal(0, volatility, len(ts)))
    open_ = pd.Series(base).shift(1).fillna(base[0])
    close = pd.Series(base)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    if spike:
        high.iloc[-1] = close.iloc[-1] * (1 + spike / 100.0)
        low.iloc[-1] = close.iloc[-1] * (1 - spike / 100.0)
    return pd.DataFrame({"ts": (ts.view("int64") // 10**6), "open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


def test_trend_regime() -> None:
    df = _frame(volatility=0.05, trend=0.35)
    fake_adx = pd.DataFrame({"ADX_14": [35.0] * len(df)})
    with patch("regime.ta.adx", return_value=fake_adx), patch("regime._rolling_hurst", return_value=pd.Series([0.60] * len(df), index=df.index)):
        result = regime.classify_regime(df)
    assert result["regime"] == "TREND", result


def test_volatile_regime() -> None:
    df = _frame(volatility=0.05, trend=0.0, spike=12.0)
    fake_adx = pd.DataFrame({"ADX_14": [18.0] * len(df)})
    with patch("regime.ta.adx", return_value=fake_adx), patch("regime._rolling_hurst", return_value=pd.Series([0.48] * len(df), index=df.index)):
        result = regime.classify_regime(df)
    assert result["regime"] == "VOLATILE", result


if __name__ == "__main__":
    test_trend_regime()
    test_volatile_regime()
    print("test_regime_ok")
