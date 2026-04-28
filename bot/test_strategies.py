from __future__ import annotations

import pandas as pd

from bot_config import DEFAULT_CONFIG
from strategies import signal_breakout, signal_meanrev, signal_trend


def test_trend() -> None:
    df = pd.DataFrame([{"close": 110, "rsi": 55, "adx": 32, "volume": 150, "vol_ma20": 100, "ema9": 108, "ema21": 104, "ema200": 95, "supertrend_direction": 1}])
    df.attrs["indicator_cols"] = {"ema_fast": "ema9", "ema_slow": "ema21", "ema_trend": "ema200", "supertrend_dir": "supertrend_direction"}
    assert signal_trend(df.iloc[-1].to_dict(), df, DEFAULT_CONFIG) == "LONG"


def test_meanrev() -> None:
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 100} for _ in range(30)]
    rows[-1].update({
        "close": 94.0,
        "low": 93.5,
        "high": 95.0,
        "rsi": 28.0,
        "adx": 15.0,
        "atr_pct_zscore_50": 0.0,
        "bb_width_20": 2.0,
        "bb_width_zscore_50": 0.0,
        "bb_lower_20": 94.5,
        "bb_upper_20": 105.0,
        "bb_pos_20": 0.05,
    })
    df = pd.DataFrame(rows)
    assert signal_meanrev(df.iloc[-1].to_dict(), df, DEFAULT_CONFIG) == "LONG"


def test_breakout() -> None:
    rows = []
    for i in range(25):
        rows.append({"ts": i * 3600000, "open": 100 + i * 0.2, "high": 101 + i * 0.2, "low": 99 + i * 0.2, "close": 100 + i * 0.2, "volume": 100})
    for idx in range(18, 24):
        rows[idx]["high"] = rows[idx]["close"] + 3.5
        rows[idx]["low"] = rows[idx]["close"] - 2.5
    rows[-1].update({"open": 108.6, "close": 110.4, "high": 110.7, "low": 110.2, "volume": 260})
    df = pd.DataFrame(rows)
    df["atr_pct_zscore_50"] = 0.8
    assert signal_breakout(df.iloc[-1].to_dict(), df, DEFAULT_CONFIG) == "LONG"


if __name__ == "__main__":
    test_trend()
    test_meanrev()
    test_breakout()
    print("test_strategies_ok")
