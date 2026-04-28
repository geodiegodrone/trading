from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import backtest_walkforward as bwf
import ml_model
import multi_bot


def _primary_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    warmup = max(220, int(cfg.get("ema_trend", 200)) + 5)
    for idx in range(warmup, len(df)):
        last = df.iloc[idx].to_dict()
        signal = multi_bot.evaluate_signal(last, df.iloc[: idx + 1], cfg)
        if signal == "NEUTRAL":
            continue
        rows.append({"signal_idx": idx, "side": signal})
    return pd.DataFrame(rows)


def bootstrap(days: int = 180, symbol: str = "BTCUSDT", quiet: bool = False) -> Dict[str, Any]:
    cfg = multi_bot._get_symbol_config(symbol)
    timeframe = int(cfg.get("timeframe", 5))
    df = bwf.fetch_history(symbol, timeframe, days)
    df = multi_bot._compute_indicators(df, cfg)
    if df.empty:
        raise RuntimeError("no history")

    signals = _primary_signals(df, cfg)
    if signals.empty:
        raise RuntimeError("no primary signals")

    log_returns = np.log(pd.to_numeric(df["close"], errors="coerce").replace(0.0, np.nan)).diff().fillna(0.0)
    h = float(log_returns.tail(60).std(ddof=0) * 2.0)
    cusum_events = set(ml_model.cusum_filter(log_returns.tolist(), h))
    signals = signals[signals["signal_idx"].isin(cusum_events)].reset_index(drop=True)
    if signals.empty:
        raise RuntimeError("no cusum events")

    labels = ml_model.apply_triple_barrier(
        df,
        signals,
        atr_mult=float(cfg.get("atr_mult", 1.5)),
        tp_ratio=2.0,
        t_bars=24,
    )
    if labels.empty:
        raise RuntimeError("no triple-barrier labels")

    ml_model.train(labels, symbol=symbol, df_klines=df)
    info = ml_model.model_info(symbol)
    positives = int((labels["label"] == 1).sum())
    negatives = int((labels["label"] == 0).sum())
    result = {
        "symbol": symbol,
        "events": int(len(signals)),
        "labels": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "val_sharpe": float(info.get("val_sharpe", 0.0)),
        "val_auc": float(info.get("val_auc", 0.0)),
        "suggested_threshold": float(info.get("suggested_threshold", 0.55)),
        "ready": bool(ml_model.is_ready(symbol)),
    }
    if not quiet:
        print(
            f"symbol={symbol} events={result['events']} labels={result['labels']} "
            f"pos={positives} neg={negatives} "
            f"val_sharpe={result['val_sharpe']:.3f} "
            f"val_auc={result['val_auc']:.3f} "
            f"suggested_threshold={result['suggested_threshold']:.3f} "
            f"ready={result['ready']}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap BTC meta-label model from Binance history")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    bootstrap(days=args.days, symbol="BTCUSDT", quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
