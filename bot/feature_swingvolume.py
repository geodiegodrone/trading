from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pandas_ta as ta


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class DivergenceState:
    side: str
    first_idx: int
    second_idx: int
    first_hist_pct: float
    second_hist_pct: float
    first_price: float
    second_price: float
    bars_since: int


class SwingVolumeAnalyzer:
    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = str(symbol or "BTCUSDT").upper()
        self.last_divergence: DivergenceState | None = None
        self.divergence_expiry_bars = 4

    def prepare_h4(self, df_h4: pd.DataFrame) -> pd.DataFrame:
        frame = df_h4.copy()
        if frame.empty:
            return frame
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ema200"] = ta.ema(frame["close"], length=200)
        frame["rsi_14"] = ta.rsi(frame["close"], length=14)
        frame["atr_14"] = ta.atr(frame["high"], frame["low"], frame["close"], length=14)
        frame["atr_pct"] = (frame["atr_14"] / frame["close"]).replace([pd.NA], 0.0) * 100.0
        atr_mean = frame["atr_pct"].rolling(50).mean()
        atr_std = frame["atr_pct"].rolling(50).std(ddof=0)
        frame["atr_pct_zscore_50"] = ((frame["atr_pct"] - atr_mean) / atr_std.replace(0.0, pd.NA)).fillna(0.0)
        macd = ta.macd(frame["close"], fast=12, slow=26, signal=9)
        hist_col = next((column for column in list(macd.columns) if "MACDh" in column), None) if macd is not None else None
        frame["macd_hist"] = macd[hist_col] if hist_col else 0.0
        frame["macd_hist_pct"] = (frame["macd_hist"] / frame["close"]).replace([pd.NA], 0.0).fillna(0.0)
        frame["vol_ma20"] = frame["volume"].rolling(20).mean()
        vol_std = frame["volume"].rolling(20).std(ddof=0)
        frame["vol_zscore_20"] = ((frame["volume"] - frame["vol_ma20"]) / vol_std.replace(0.0, pd.NA)).fillna(0.0)
        candle_range = (frame["high"] - frame["low"]).replace(0.0, pd.NA)
        frame["body_frac"] = ((frame["close"] - frame["open"]).abs() / candle_range).fillna(0.0)
        return frame

    def prepare_d1(self, df_d1: pd.DataFrame) -> pd.DataFrame:
        frame = df_d1.copy()
        if frame.empty:
            return frame
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ema20"] = ta.ema(frame["close"], length=20)
        frame["ema200"] = ta.ema(frame["close"], length=200)
        day_range = (frame["high"] - frame["low"]).replace(0.0, pd.NA)
        frame["close_pos"] = ((frame["close"] - frame["low"]) / day_range).fillna(0.5)
        return frame

    def daily_bias(self, df_d1: pd.DataFrame) -> dict[str, Any]:
        frame = self.prepare_d1(df_d1)
        if frame.empty or len(frame) < 220:
            return {"bias": "NEUTRAL", "reason": "D1 insuficiente", "close_pos": 0.5}
        row = frame.iloc[-1]
        ema20 = _safe_float(row.get("ema20"))
        ema200 = _safe_float(row.get("ema200"))
        close_pos = _safe_float(row.get("close_pos"), 0.5)
        if ema20 <= 0 or ema200 <= 0:
            return {"bias": "NEUTRAL", "reason": "EMA D1 incompleta", "close_pos": close_pos}
        if close_pos < 0.02 or close_pos > 0.98:
            return {"bias": "NEUTRAL", "reason": "cierre D1 en extremo", "close_pos": close_pos}
        if ema20 > ema200:
            return {"bias": "ALCISTA", "reason": "EMA20 D1 > EMA200 D1", "close_pos": close_pos}
        if ema20 < ema200:
            return {"bias": "BAJISTA", "reason": "EMA20 D1 < EMA200 D1", "close_pos": close_pos}
        return {"bias": "NEUTRAL", "reason": "sesgo D1 no claro", "close_pos": close_pos}

    def _pivot_indices(self, values: pd.Series, side: str) -> list[int]:
        pivots: list[int] = []
        if len(values) < 5:
            return pivots
        for idx in range(1, len(values) - 1):
            left = _safe_float(values.iloc[idx - 1])
            mid = _safe_float(values.iloc[idx])
            right = _safe_float(values.iloc[idx + 1])
            if side == "LONG" and mid < left and mid <= right:
                pivots.append(idx)
            if side == "SHORT" and mid > left and mid >= right:
                pivots.append(idx)
        return pivots

    def detect_macd_divergence(self, df_h4: pd.DataFrame, side: str = "LONG", last_n_bars: int = 40) -> DivergenceState | None:
        frame = self.prepare_h4(df_h4).tail(max(20, int(last_n_bars))).reset_index(drop=True)
        if frame.empty or len(frame) < 12:
            self.last_divergence = None
            return None
        bias_side = str(side or "LONG").upper()
        hist = frame["macd_hist_pct"]
        pivots = self._pivot_indices(hist, bias_side)
        if not pivots:
            self.last_divergence = None
            return None
        current_idx = len(frame) - 1
        threshold = -0.0005 if bias_side == "LONG" else 0.0005
        for second in reversed(pivots):
            bars_since = current_idx - second
            if bars_since < 0 or bars_since > self.divergence_expiry_bars:
                continue
            second_hist = _safe_float(hist.iloc[second])
            second_price = _safe_float(frame.iloc[second]["low" if bias_side == "LONG" else "high"])
            if bias_side == "LONG" and second_hist > -0.0001:
                continue
            if bias_side == "SHORT" and second_hist < 0.0001:
                continue
            for first in reversed([idx for idx in pivots if 2 <= second - idx <= 8]):
                first_hist = _safe_float(hist.iloc[first])
                first_price = _safe_float(frame.iloc[first]["low" if bias_side == "LONG" else "high"])
                if bias_side == "LONG":
                    if not (first_hist <= threshold and second_price < first_price and second_hist > first_hist):
                        continue
                else:
                    if not (first_hist >= threshold and second_price > first_price and second_hist < first_hist):
                        continue
                state = DivergenceState(
                    side=bias_side,
                    first_idx=first,
                    second_idx=second,
                    first_hist_pct=first_hist,
                    second_hist_pct=second_hist,
                    first_price=first_price,
                    second_price=second_price,
                    bars_since=bars_since,
                )
                self.last_divergence = state
                return state
        self.last_divergence = None
        return None

    def validate_volume_signal(self, df_h4: pd.DataFrame, side: str = "LONG") -> tuple[bool, str]:
        frame = self.prepare_h4(df_h4)
        if frame.empty or len(frame) < 22:
            return False, "H4 insuficiente"
        row = frame.iloc[-1]
        vol = _safe_float(row.get("volume"))
        vol_ma = _safe_float(row.get("vol_ma20"))
        vol_z = _safe_float(row.get("vol_zscore_20"))
        prev_one = _safe_float(frame.iloc[-2]["volume"])
        prev_two = _safe_float(frame.iloc[-3]["volume"])
        body_frac = _safe_float(row.get("body_frac"))
        open_price = _safe_float(row.get("open"))
        close_price = _safe_float(row.get("close"))
        direction_ok = close_price > open_price if str(side).upper() == "LONG" else close_price < open_price
        if vol_ma <= 0:
            return False, "vol_ma20 invalida"
        if vol <= vol_ma * 1.3:
            return False, f"vol {vol:.0f} <= 1.3*ma20 {vol_ma * 1.3:.0f}"
        if vol_z < 1.5:
            return False, f"vol_z {vol_z:.2f} < 1.50"
        if prev_one >= vol_ma * 0.9 or prev_two >= vol_ma * 0.9:
            return False, "prev_vol no decreciente"
        if body_frac < 0.60:
            return False, f"body {body_frac:.2f} < 0.60"
        if not direction_ok:
            return False, "vela sin direccionalidad"
        return True, f"vol ok z={vol_z:.2f} body={body_frac:.2f}"

    def check_macd_recovery(self, df_h4: pd.DataFrame, side: str = "LONG") -> tuple[bool, str]:
        frame = self.prepare_h4(df_h4)
        if frame.empty or len(frame) < 3:
            return False, "H4 insuficiente"
        current = _safe_float(frame.iloc[-1]["macd_hist_pct"])
        previous = _safe_float(frame.iloc[-2]["macd_hist_pct"])
        bias_side = str(side or "LONG").upper()
        if bias_side == "LONG":
            if -0.0001 <= current <= 0.0005:
                return True, f"macd_hist_pct {current:.5f} en rango"
            if previous < 0.0 < current:
                return True, f"macd cruzo positivo {previous:.5f}->{current:.5f}"
            return False, f"macd_hist_pct {current:.5f} fuera de rango"
        if -0.0005 <= current <= 0.0001:
            return True, f"macd_hist_pct {current:.5f} en rango"
        if previous > 0.0 > current:
            return True, f"macd cruzo negativo {previous:.5f}->{current:.5f}"
        return False, f"macd_hist_pct {current:.5f} fuera de rango"
