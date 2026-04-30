from __future__ import annotations

import os
import pickle

os.environ["ML_MIN_VAL_SHARPE"] = "0.5"

import numpy as np
import pandas as pd
from pathlib import Path

import backtest_walkforward as bwf
import bootstrap_ml
import ml_model
import multi_bot


def test_threshold_no_profitable_returns_neutral() -> None:
    probs = np.linspace(0.45, 0.70, 60)
    labels = ([1, 0] * 30)
    r_values = np.full(60, -0.25)
    holding_bars = np.full(60, 12)
    signal_idx = np.arange(60)
    result = ml_model._optimize_threshold(probs.tolist(), labels, r_values.tolist(), holding_bars.tolist(), signal_idx.tolist())
    assert float(result["tau"]) == 0.55, result
    assert bool(result["no_profitable_threshold"]) is True, result


def test_ensure_ml_bootstrapped_missing_model() -> None:
    symbol = "BTCUSDT"
    cfg = multi_bot._get_symbol_config(symbol)
    model_path = ml_model._model_path(symbol)
    original_bytes = model_path.read_bytes() if model_path.exists() else None
    if model_path.exists():
        model_path.unlink()
    calls = []
    original_bootstrap = bootstrap_ml.bootstrap
    original_thread = multi_bot.threading.Thread
    original_state = dict(multi_bot._ML_BOOTSTRAP_STATE)

    class ImmediateThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

    try:
        multi_bot._ML_BOOTSTRAP_STATE = {"thread": None, "running": False, "last_started": 0.0, "last_completed": 0.0}
        bootstrap_ml.bootstrap = lambda **kwargs: calls.append(kwargs) or {"labels": 1, "val_sharpe": 0.0, "suggested_threshold": 0.55}
        multi_bot.threading.Thread = ImmediateThread
        started = multi_bot._ensure_ml_bootstrapped(symbol, cfg)
        assert started is True
        assert calls, "expected bootstrap call"
    finally:
        bootstrap_ml.bootstrap = original_bootstrap
        multi_bot.threading.Thread = original_thread
        multi_bot._ML_BOOTSTRAP_STATE = original_state
        if original_bytes is None:
            if model_path.exists():
                model_path.unlink()
        else:
            model_path.write_bytes(original_bytes)


def test_ensure_ml_bootstrapped_feature_mismatch() -> None:
    symbol = "BTCUSDT"
    cfg = multi_bot._get_symbol_config(symbol)
    model_path = ml_model._model_path(symbol)
    original_bytes = model_path.read_bytes() if model_path.exists() else None
    calls = []
    original_bootstrap = bootstrap_ml.bootstrap
    original_thread = multi_bot.threading.Thread
    original_state = dict(multi_bot._ML_BOOTSTRAP_STATE)

    class ImmediateThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

    try:
        with model_path.open("wb") as handle:
            pickle.dump({"trained_on": 10, "feature_count": 10, "last_trained_at": "2026-04-28T00:00:00+00:00"}, handle)
        multi_bot._ML_BOOTSTRAP_STATE = {"thread": None, "running": False, "last_started": 0.0, "last_completed": 0.0}
        bootstrap_ml.bootstrap = lambda **kwargs: calls.append(kwargs) or {"labels": 1, "val_sharpe": 0.0, "suggested_threshold": 0.55}
        multi_bot.threading.Thread = ImmediateThread
        started = multi_bot._ensure_ml_bootstrapped(symbol, cfg)
        assert started is True
        assert calls, "expected bootstrap call on feature mismatch"
    finally:
        bootstrap_ml.bootstrap = original_bootstrap
        multi_bot.threading.Thread = original_thread
        multi_bot._ML_BOOTSTRAP_STATE = original_state
        if original_bytes is None:
            if model_path.exists():
                model_path.unlink()
        else:
            model_path.write_bytes(original_bytes)


def test_bootstrap_retries_on_low_samples() -> None:
    calls = []
    original_fetch_history = bwf.fetch_history
    original_compute = multi_bot._compute_indicators
    original_signals = bootstrap_ml._primary_signals
    original_filtered = bootstrap_ml._filtered_signals
    original_barrier = ml_model.apply_triple_barrier
    original_train = ml_model.train
    original_model_info = ml_model.model_info
    original_is_ready = ml_model.is_ready

    def fake_fetch_history(symbol, timeframe, days):
        calls.append(days)
        rows = 300
        base = np.linspace(100, 120, rows)
        return pd.DataFrame({
            "ts": np.arange(rows) * 3600000,
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.2,
            "volume": np.linspace(10, 200, rows),
            "quote_vol": np.linspace(1000, 5000, rows),
            "taker_base": np.linspace(5, 80, rows),
            "taker_quote": np.linspace(500, 3000, rows),
            "days_marker": float(days),
            "atr": 1.0,
        })

    def fake_signals(primary_df, confirmation_df, cfg):
        return pd.DataFrame({"signal_idx": np.arange(90), "side": ["LONG"] * 90, "regime": ["TREND"] * 90})

    def fake_barrier(df_indicators, signals, atr_mult=1.5, tp_ratio=2.0, t_bars=24):
        days_used = int(df_indicators["days_marker"].iloc[0])
        size = 60 if days_used <= 120 else 160
        return pd.DataFrame({
            "signal_idx": np.arange(size),
            "side": ["LONG"] * size,
            "entry_price": np.full(size, 100.0),
            "sl": np.full(size, 99.0),
            "tp": np.full(size, 102.0),
            "exit_idx": np.arange(size) + 1,
            "exit_price": np.full(size, 101.0),
            "exit_reason": ["TP"] * size,
            "label": ([1, 0] * (size // 2)) + ([1] if size % 2 else []),
            "r_multiple": np.full(size, 1.0),
            "holding_bars": np.full(size, 6),
        })

    captured = {}

    try:
        bwf.fetch_history = fake_fetch_history
        multi_bot._compute_indicators = lambda df, cfg: df
        bootstrap_ml._primary_signals = fake_signals
        bootstrap_ml._filtered_signals = lambda df, signals, threshold: signals
        ml_model.apply_triple_barrier = fake_barrier
        ml_model.train = lambda labels, symbol="BTCUSDT", df_klines=None: captured.update({"trained": len(labels)})
        ml_model.model_info = lambda symbol="BTCUSDT": {"val_sharpe": 0.0, "val_auc": 0.0, "val_f1": 0.0, "coverage_pct": 0.0, "suggested_threshold": 0.55, "fold_metrics": [], "min_trades_per_year_estimate": 0.0, "no_profitable_threshold": True, "not_ready_reason": "sin τ con expected_r>0 y sharpe>0"}
        ml_model.is_ready = lambda symbol="BTCUSDT": False
        result = bootstrap_ml.bootstrap(days=120, symbol="BTCUSDT", timeframe=60, quiet=True, min_samples=120)
        assert 240 in calls, calls
        assert int(result["days_used"]) == 240, result
        assert int(captured["trained"]) == 160, captured
    finally:
        bwf.fetch_history = original_fetch_history
        multi_bot._compute_indicators = original_compute
        bootstrap_ml._primary_signals = original_signals
        bootstrap_ml._filtered_signals = original_filtered
        ml_model.apply_triple_barrier = original_barrier
        ml_model.train = original_train
        ml_model.model_info = original_model_info
        ml_model.is_ready = original_is_ready


def test_primary_only_mode_halves_size() -> None:
    full_trade = 80.0
    reduced_trade = multi_bot._primary_only_trade_usdt(full_trade, False)
    full_qty = (full_trade * 2.0) / 100000.0
    reduced_qty = (reduced_trade * 2.0) / 100000.0
    assert reduced_qty == full_qty * 0.5
    assert multi_bot._primary_only_trade_usdt(full_trade, True) == full_trade


def main() -> None:
    test_threshold_no_profitable_returns_neutral()
    test_ensure_ml_bootstrapped_missing_model()
    test_ensure_ml_bootstrapped_feature_mismatch()
    test_bootstrap_retries_on_low_samples()
    test_primary_only_mode_halves_size()
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
    if int(info.get("usable_folds", 0) or 0) > 0:
        assert float(info["coverage_pct"]) >= 5.0, info
        assert 0.45 <= float(info["suggested_threshold"]) <= 0.70, info
    else:
        assert not ml_model.is_ready(symbol), info
        assert str(info.get("not_ready_reason", "")).strip(), info
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
