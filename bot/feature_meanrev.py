from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from features import build_features


@dataclass
class MeanRevState:
    side: str
    return_z: float
    rsi: float
    atr_z: float
    entry_price: float
    stop_price: float
    tp_price: float
    reason: str


class MeanRevAnalyzer:
    def __init__(self) -> None:
        self.last_trade_ts = None

    def frame(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        if df_1h.empty:
            return df_1h.copy()
        frame = df_1h.copy().reset_index(drop=True)
        features = build_features(frame)
        for column in features.columns:
            frame[column] = features[column]
        return frame

    def check_overshoot(self, df_1h: pd.DataFrame, sigma: float = 2.5) -> Dict[str, Any]:
        frame = self.frame(df_1h)
        if frame.empty or len(frame) < 52:
            return {"long_ok": False, "short_ok": False, "reason": "sin datos", "return_z": 0.0}
        returns = pd.to_numeric(frame["close"], errors="coerce").pct_change(1).mul(100.0)
        rolling = returns.rolling(50)
        mean = rolling.mean().iloc[-1]
        std = rolling.std(ddof=0).iloc[-1]
        value = returns.iloc[-1]
        return_z = float((value - mean) / std) if std and std > 0 else 0.0
        return {
            "long_ok": return_z <= -float(sigma),
            "short_ok": return_z >= float(sigma),
            "return_z": return_z,
            "reason": f"return_z={return_z:.2f}",
        }

    def check_rsi_extreme(self, df_1h: pd.DataFrame, low: float = 25.0, high: float = 75.0) -> Dict[str, Any]:
        frame = self.frame(df_1h)
        if frame.empty:
            return {"long_ok": False, "short_ok": False, "rsi": 0.0, "reason": "sin datos"}
        rsi = float(pd.to_numeric(frame.get("rsi"), errors="coerce").iloc[-1] if "rsi" in frame else 0.0)
        return {
            "long_ok": rsi <= float(low),
            "short_ok": rsi >= float(high),
            "rsi": rsi,
            "reason": f"rsi={rsi:.2f}",
        }

    def check_vol_high(self, df_1h: pd.DataFrame, atr_z_min: float = 1.5) -> Dict[str, Any]:
        frame = self.frame(df_1h)
        if frame.empty:
            return {"ok": False, "atr_z": 0.0, "reason": "sin datos"}
        atr_z = float(pd.to_numeric(frame.get("atr_pct_zscore_50"), errors="coerce").iloc[-1] if "atr_pct_zscore_50" in frame else 0.0)
        return {
            "ok": atr_z >= float(atr_z_min),
            "atr_z": atr_z,
            "reason": f"atr_z={atr_z:.2f}",
        }

    def build_signal(self, df_1h: pd.DataFrame, cfg: Dict[str, Any]) -> MeanRevState | None:
        frame = self.frame(df_1h)
        if frame.empty or len(frame) < 52:
            return None
        sigma = float(cfg.get("meanrev_sigma", 2.5))
        rsi_low = float(cfg.get("meanrev_rsi_low", 25.0))
        rsi_high = float(cfg.get("meanrev_rsi_high", 75.0))
        atr_z_min = float(cfg.get("meanrev_atr_z", 1.5))
        signal = self.check_overshoot(frame, sigma=sigma)
        rsi_state = self.check_rsi_extreme(frame, low=rsi_low, high=rsi_high)
        vol_state = self.check_vol_high(frame, atr_z_min=atr_z_min)
        last = frame.iloc[-1]
        entry_price = float(last.get("close") or 0.0)
        atr = float(pd.to_numeric(frame.get("atr"), errors="coerce").iloc[-1] if "atr" in frame else 0.0)
        if entry_price <= 0 or atr <= 0:
            return None
        side = "NEUTRAL"
        if signal["long_ok"] and rsi_state["long_ok"] and vol_state["ok"]:
            side = "LONG"
        elif signal["short_ok"] and rsi_state["short_ok"] and vol_state["ok"]:
            side = "SHORT"
        if side == "NEUTRAL":
            return None
        sl_atr = float(cfg.get("meanrev_sl_atr", 1.0))
        tp_atr = float(cfg.get("meanrev_tp_atr", 0.6))
        if side == "LONG":
            stop_price = round(entry_price - (atr * sl_atr), 2)
            if stop_price == round(entry_price, 2):
                stop_price = round(entry_price - (atr * 1.2), 2)
            tp_price = round(entry_price + (atr * tp_atr), 2)
        else:
            stop_price = round(entry_price + (atr * sl_atr), 2)
            if stop_price == round(entry_price, 2):
                stop_price = round(entry_price + (atr * 1.2), 2)
            tp_price = round(entry_price - (atr * tp_atr), 2)
        reason = f"overshoot {signal['return_z']:.2f} | {rsi_state['reason']} | {vol_state['reason']}"
        return MeanRevState(
            side=side,
            return_z=float(signal["return_z"]),
            rsi=float(rsi_state["rsi"]),
            atr_z=float(vol_state["atr_z"]),
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            reason=reason,
        )
