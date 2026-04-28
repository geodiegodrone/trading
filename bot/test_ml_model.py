from __future__ import annotations

import os

os.environ["ML_MIN_VAL_SHARPE"] = "0.5"

import numpy as np
import pandas as pd
from pathlib import Path

import backtest_walkforward as bwf
import bootstrap_ml
import ml_model
import multi_bot


def main() -> None:
    symbol = "BTCUSDT"
    cfg = multi_bot._get_symbol_config(symbol)
    df = bwf.fetch_history(symbol, int(cfg.get("primary_timeframe", 60)), days=180)
    confirmation_df = bwf.fetch_history(symbol, int(cfg.get("confirmation_timeframe", 240)), days=180)
    df = multi_bot._compute_indicators(df, cfg)
    confirmation_df = multi_bot._compute_indicators(confirmation_df, cfg)
    signals = bootstrap_ml._primary_signals(df, confirmation_df, cfg)
    if signals.empty:
        cfg = dict(cfg)
        cfg["strategy_mode"] = "trend"
        signals = bootstrap_ml._primary_signals(df, confirmation_df, cfg)
    assert not signals.empty, "expected primary signals"

    log_returns = np.log(pd.to_numeric(df["close"], errors="coerce").replace(0.0, np.nan)).diff().fillna(0.0)
    h = float(log_returns.tail(60).std(ddof=0) * 2.0)
    raw_signals = signals.copy()
    cusum_events = set(ml_model.cusum_filter(log_returns.tolist(), h))
    signals = signals[signals["signal_idx"].isin(cusum_events)].reset_index(drop=True)
    if signals.empty:
        signals = raw_signals.reset_index(drop=True)
    assert not signals.empty, "expected cusum-filtered events"

    labels = ml_model.apply_triple_barrier(
        df,
        signals,
        atr_mult=float(cfg.get("atr_mult", 1.5)),
        tp_ratio=2.0,
        t_bars=24,
    )
    assert not labels.empty, "expected triple-barrier labels"

    for name in ("model_BTCUSDT.pkl", "model.pkl"):
        model_path = Path(__file__).resolve().parent / name
        if model_path.exists():
            model_path.unlink()
    ml_model.train(labels, symbol=symbol, df_klines=df)

    info = ml_model.model_info(symbol)
    if int(info.get("trained_on", 0) or 0) >= 20:
        assert float(info["coverage_pct"]) >= 5.0, info
        assert 0.45 <= float(info["suggested_threshold"]) <= 0.70, info
    else:
        assert not ml_model.is_ready(symbol), info
        assert "faltan muestras" in str(info.get("not_ready_reason", "")).lower(), info
    assert isinstance(info.get("not_ready_reason"), str), info

    feature_frame = ml_model.snapshot_features(df.iloc[-1].to_dict(), df.tail(80).to_dict("records"), df.attrs.get("indicator_cols", {}))
    feature_frame["side_buy"] = 1.0
    feature_frame["supertrend_aligned_side"] = 1.0 if feature_frame.get("supertrend_direction", 0.0) > 0 else 0.0
    score = ml_model.predict(feature_frame, symbol=symbol)
    assert 0.0 <= score <= 1.0, score

    print(
        "test_ok",
        f"ready={ml_model.is_ready(symbol)}",
        f"coverage={info['coverage_pct']:.2f}",
        f"val_sharpe={info['val_sharpe']:.3f}",
        f"val_auc={info['val_auc']:.3f}",
        f"tau={info['suggested_threshold']:.3f}",
        f"score={score:.3f}",
    )


if __name__ == "__main__":
    main()
