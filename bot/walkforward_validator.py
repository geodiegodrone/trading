from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score

import ml_model
import regime as regime_detector
from bot_config import DEFAULT_CONFIG
from features import FEATURE_COLUMNS, build_features
from strategies import enrich_indicators, signal_breakout, signal_meanrev, signal_trend


_REGIME_CODE = {"TREND": 1.0, "TRENDING": 1.0, "RANGE": -1.0, "VOLATILE": 2.0, "MIXED": 0.0, "BREAKOUT": 0.5, "HYBRID": 0.25}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _config(base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(base or {})
    cfg["strategy_mode"] = str(cfg.get("strategy_mode", "regime")).lower()
    cfg["primary_timeframe"] = int(cfg.get("primary_timeframe", cfg.get("timeframe", 60)))
    cfg["confirmation_timeframe"] = int(cfg.get("confirmation_timeframe", 240))
    cfg["ema_fast"] = int(cfg.get("ema_fast", 9))
    cfg["ema_slow"] = int(cfg.get("ema_slow", 21))
    cfg["ema_confirm"] = int(cfg.get("ema_confirm", 50))
    cfg["ema_trend"] = int(cfg.get("ema_trend", 200))
    cfg["rsi_period"] = int(cfg.get("rsi_period", 14))
    cfg["adx_period"] = int(cfg.get("adx_period", 14))
    cfg["atr_period"] = int(cfg.get("atr_period", 14))
    cfg["supertrend_period"] = int(cfg.get("supertrend_period", 14))
    cfg["supertrend_mult"] = float(cfg.get("supertrend_mult", 3.5))
    cfg["atr_mult"] = float(cfg.get("atr_mult", 1.5))
    cfg["tp_ratio"] = float(cfg.get("tp_ratio", 2.0))
    cfg["t_bars"] = int(cfg.get("t_bars", 24))
    cfg["feature_fraction"] = float(cfg.get("feature_fraction", 0.8))
    cfg["cusum_mode"] = str(cfg.get("cusum_mode", "2xatr_pct"))
    return cfg


def _compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    if frame.empty or len(frame) < 50:
        return frame
    frame[f"ema{int(cfg['ema_fast'])}"] = ta.ema(frame["close"], length=int(cfg["ema_fast"]))
    frame[f"ema{int(cfg['ema_slow'])}"] = ta.ema(frame["close"], length=int(cfg["ema_slow"]))
    frame[f"ema{int(cfg['ema_confirm'])}"] = ta.ema(frame["close"], length=int(cfg["ema_confirm"]))
    frame[f"ema{int(cfg['ema_trend'])}"] = ta.ema(frame["close"], length=int(cfg["ema_trend"]))
    frame["rsi"] = ta.rsi(frame["close"], length=int(cfg["rsi_period"]))
    adx = ta.adx(frame["high"], frame["low"], frame["close"], length=int(cfg["adx_period"]))
    adx_col = f"ADX_{int(cfg['adx_period'])}"
    if adx is not None and adx_col in adx:
        frame["adx"] = adx[adx_col]
    st = ta.supertrend(frame["high"], frame["low"], frame["close"], length=int(cfg["supertrend_period"]), multiplier=float(cfg["supertrend_mult"]))
    st_dir_col = f"SUPERTd_{int(cfg['supertrend_period'])}_{float(cfg['supertrend_mult'])}"
    if st is not None and st_dir_col in st:
        frame["supertrend_direction"] = st[st_dir_col]
    frame["atr"] = ta.atr(frame["high"], frame["low"], frame["close"], length=int(cfg["atr_period"]))
    frame["vol_ma20"] = frame["volume"].rolling(20).mean()
    frame.attrs["indicator_cols"] = {
        "ema_fast": f"ema{int(cfg['ema_fast'])}",
        "ema_slow": f"ema{int(cfg['ema_slow'])}",
        "ema_trend": f"ema{int(cfg['ema_trend'])}",
        "supertrend_dir": "supertrend_direction",
    }
    return enrich_indicators(frame)


def _confirmation_row(confirmation_df: pd.DataFrame, ts_value: int) -> pd.Series | None:
    subset = confirmation_df[confirmation_df["ts"] <= ts_value]
    if subset.empty:
        return None
    return subset.iloc[-1]


def _htf_confirms(signal: str, confirm_row: pd.Series | None, cfg: Dict[str, Any]) -> bool:
    if signal == "NEUTRAL" or confirm_row is None:
        return False
    ema50 = _safe_float(confirm_row.get(f"ema{int(cfg.get('ema_confirm', 50))}"))
    ema200 = _safe_float(confirm_row.get(f"ema{int(cfg.get('ema_trend', 200))}"))
    close = _safe_float(confirm_row.get("close"))
    if signal == "LONG":
        return close > ema200 and ema50 > ema200
    return close < ema200 and ema50 < ema200


def _pick_signal(mode: str, last: Dict[str, Any], primary_slice: pd.DataFrame, confirm_row: pd.Series | None, cfg: Dict[str, Any]) -> tuple[str, str, str]:
    regime_name = "TRENDING"
    strategy_used = "trend"
    if mode == "trend":
        signal = signal_trend(last, primary_slice, cfg)
    elif mode == "meanrev":
        regime_name = "RANGE"
        strategy_used = "meanrev"
        signal = signal_meanrev(last, primary_slice, cfg)
    elif mode == "breakout":
        regime_name = "BREAKOUT"
        strategy_used = "breakout"
        signal = signal_breakout(last, primary_slice, cfg)
    elif mode == "hybrid":
        signal = signal_trend(last, primary_slice, cfg)
        if signal == "NEUTRAL":
            signal = signal_meanrev(last, primary_slice, cfg)
            strategy_used = "meanrev"
            regime_name = "RANGE"
        else:
            strategy_used = "trend"
            regime_name = "TRENDING"
    else:
        regime_info = regime_detector.classify_regime(primary_slice)
        regime_name = str(regime_info.get("regime", "MIXED")).upper()
        if regime_name in {"TREND", "TRENDING"}:
            strategy_used = "trend"
            signal = signal_trend(last, primary_slice, cfg)
        elif regime_name == "RANGE":
            strategy_used = "meanrev"
            signal = signal_meanrev(last, primary_slice, cfg)
        elif regime_name == "VOLATILE":
            strategy_used = "meanrev"
            signal = signal_meanrev(last, primary_slice, cfg)
        else:
            strategy_used = "breakout"
            signal = signal_breakout(last, primary_slice, cfg)
    if signal != "NEUTRAL" and not _htf_confirms(signal, confirm_row, cfg):
        return "NEUTRAL", regime_name, strategy_used
    return signal, regime_name, strategy_used


def _cusum_threshold(primary_df: pd.DataFrame, cfg: Dict[str, Any]) -> float:
    atr_pct = ((pd.to_numeric(primary_df.get("atr"), errors="coerce") / pd.to_numeric(primary_df["close"], errors="coerce").replace(0.0, np.nan)) * 100.0).dropna()
    mean_atr_pct = float(atr_pct.tail(120).mean()) if not atr_pct.empty else 0.5
    mode = str(cfg.get("cusum_mode", "2xatr_pct")).lower()
    if mode == "atr_pct":
        factor = 1.0
    elif mode == "3xatr_pct":
        factor = 3.0
    else:
        factor = 2.0
    return max((mean_atr_pct / 100.0) * factor, 0.0005)


def _event_dataset(primary_df: pd.DataFrame, confirmation_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    warmup = max(220, int(cfg.get("ema_trend", 200)) + 5)
    mode = str(cfg.get("strategy_mode", "regime")).lower()
    signal_rows: List[Dict[str, Any]] = []
    for idx in range(warmup, len(primary_df)):
        ts_value = int(primary_df.iloc[idx]["ts"])
        signal, regime_name, strategy_used = _pick_signal(mode, primary_df.iloc[idx].to_dict(), primary_df.iloc[: idx + 1], _confirmation_row(confirmation_df, ts_value), cfg)
        if signal == "NEUTRAL":
            continue
        signal_rows.append(
            {
                "signal_idx": idx,
                "side": signal,
                "regime": regime_name,
                "regime_code": _REGIME_CODE.get(regime_name, 0.0),
                "strategy_used": strategy_used,
            }
        )
    if not signal_rows:
        return pd.DataFrame(), pd.DataFrame()
    raw_signals = pd.DataFrame(signal_rows)
    log_returns = np.log(pd.to_numeric(primary_df["close"], errors="coerce").replace(0.0, np.nan)).diff().fillna(0.0)
    h = _cusum_threshold(primary_df, cfg)
    cusum_events = set(ml_model.cusum_filter(log_returns.tolist(), h))
    filtered = raw_signals[raw_signals["signal_idx"].isin(cusum_events)].reset_index(drop=True)
    if filtered.empty:
        filtered = raw_signals.reset_index(drop=True)
    labels = ml_model.apply_triple_barrier(
        primary_df,
        filtered,
        atr_mult=float(cfg.get("atr_mult", 1.5)),
        tp_ratio=float(cfg.get("tp_ratio", 2.0)),
        t_bars=int(cfg.get("t_bars", 24)),
    )
    if labels.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = filtered.merge(labels, on=["signal_idx", "side"], how="inner", suffixes=("", "_label"))
    dataset = ml_model._prepare_training_set(merged, primary_df)
    return dataset, merged


def _build_model(y_train: np.ndarray, cfg: Dict[str, Any]) -> Any:
    positives = int(np.sum(y_train))
    negatives = int(len(y_train) - positives)
    if ml_model.USING_LIGHTGBM:
        return ml_model.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.025,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=30,
            feature_fraction=float(cfg.get("feature_fraction", 0.8)),
            bagging_fraction=0.85,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.5,
            objective="binary",
            metric="auc",
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        )
    return ml_model.XGBClassifier(  # pragma: no cover
        n_estimators=600,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=float(cfg.get("feature_fraction", 0.8)),
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=max(1.0, float(negatives) / float(positives)) if positives > 0 else 1.0,
        random_state=42,
        n_jobs=1,
    )


def _fit_model(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, w_train: np.ndarray, w_valid: np.ndarray, cfg: Dict[str, Any]) -> Any:
    model = _build_model(y_train, cfg)
    train_frame = pd.DataFrame(x_train, columns=FEATURE_COLUMNS)
    valid_frame = pd.DataFrame(x_valid, columns=FEATURE_COLUMNS)
    if ml_model.USING_LIGHTGBM:
        model.fit(
            train_frame,
            y_train,
            sample_weight=w_train,
            eval_set=[(valid_frame, y_valid)],
            eval_sample_weight=[w_valid],
            callbacks=[ml_model.early_stopping(50, verbose=False), ml_model.log_evaluation(period=0)],
        )
        return model
    model.fit(train_frame, y_train, sample_weight=w_train, eval_set=[(valid_frame, y_valid)], verbose=False)  # pragma: no cover
    return model


def _profit_factor(r_values: Sequence[float]) -> float:
    gains = sum(r for r in r_values if r > 0)
    losses = abs(sum(r for r in r_values if r < 0))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _drawdown_pct(r_values: Sequence[float]) -> float:
    if not r_values:
        return 0.0
    current = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in np.asarray(r_values, dtype=float):
        current += float(value)
        peak = max(peak, current)
        if peak > 1e-6:
            max_dd = max(max_dd, ((peak - current) / peak) * 100.0)
    return float(min(max(max_dd, 0.0), 100.0))


def _metrics_fail(event_count: int = 0, label_balance: Dict[str, int] | None = None, reason: str = "insufficient_data") -> Dict[str, Any]:
    return {
        "event_count": int(event_count),
        "label_balance": label_balance or {"positive": 0, "negative": 0},
        "median_fold_sharpe": 0.0,
        "profit_factor": 0.0,
        "folds_positive": 0,
        "folds_total": 0,
        "max_drawdown_pct": 0.0,
        "expectancy_R": 0.0,
        "total_trades": 0,
        "val_auc": 0.0,
        "coverage_pct": 0.0,
        "fold_metrics": [],
        "passes": False,
        "reason": reason,
    }


def run_walkforward(primary_df: pd.DataFrame, config: Dict[str, Any], n_folds: int = 12, confirmation_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    cfg = _config(config)
    primary = _compute_indicators(primary_df, cfg)
    confirmation = _compute_indicators(confirmation_df.copy() if confirmation_df is not None else primary_df.copy(), cfg)
    dataset, events = _event_dataset(primary, confirmation, cfg)
    if dataset.empty or len(dataset) < ml_model.MIN_TRAIN_TRADES or dataset["label"].nunique() < 2:
        labels = {"positive": int((events.get("label", pd.Series(dtype=int)) == 1).sum()) if not events.empty else 0, "negative": int((events.get("label", pd.Series(dtype=int)) == 0).sum()) if not events.empty else 0}
        return _metrics_fail(len(dataset), labels, "insufficient_events")

    x = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["label"].to_numpy(dtype=int)
    signal_idx = dataset["signal_idx"].to_numpy(dtype=int)
    exit_idx = dataset["exit_idx"].to_numpy(dtype=int)
    r_values = dataset["r_multiple"].to_numpy(dtype=float)
    holding_bars = dataset["holding_bars"].to_numpy(dtype=float)
    weights = ml_model._uniqueness_weights(signal_idx.tolist(), exit_idx.tolist())

    splitter = ml_model.PurgedKFold(n_splits=max(2, int(n_folds)), embargo=int(float(cfg.get("t_bars", 24)) * 1.5))
    fold_metrics: List[Dict[str, Any]] = []
    all_selected_r: List[float] = []
    total_selected = 0

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(dataset), start=1):
        if len(train_idx) < 40 or len(test_idx) < 5:
            continue
        train_idx_sorted = np.sort(train_idx)
        fit_idx, calib_idx = ml_model._chronological_validation_split(train_idx_sorted)
        if len(fit_idx) < 20 or len(calib_idx) < 5:
            continue
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[calib_idx])) < 2:
            continue
        model = _fit_model(x[fit_idx], y[fit_idx], x[calib_idx], y[calib_idx], weights[fit_idx], weights[calib_idx], cfg)
        calib_raw = ml_model._positive_proba(model, x[calib_idx])
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_raw, y[calib_idx])
        threshold_info = ml_model._optimize_threshold(
            calibrator.predict(calib_raw).tolist(),
            y[calib_idx].tolist(),
            r_values[calib_idx].tolist(),
            holding_bars[calib_idx].tolist(),
            signal_idx[calib_idx].tolist(),
        )
        tau = float(threshold_info.get("tau", 0.55))
        raw_test = ml_model._positive_proba(model, x[test_idx])
        probs = calibrator.predict(raw_test)
        selected = probs >= tau
        selected_r = r_values[test_idx][selected].tolist()
        selected_h = holding_bars[test_idx][selected].tolist()
        selected_count = int(selected.sum())
        total_selected += selected_count
        all_selected_r.extend(selected_r)
        auc = roc_auc_score(y[test_idx], probs) if len(np.unique(y[test_idx])) > 1 else 0.5
        f1 = f1_score(y[test_idx], selected.astype(int), zero_division=0)
        sharpe = ml_model._annualized_sharpe(selected_r, selected_h, bar_minutes=int(cfg.get("primary_timeframe", 60))) if selected_count else 0.0
        pf = _profit_factor(selected_r)
        expectancy = float(np.mean(selected_r)) if selected_r else 0.0
        fold_metrics.append(
            {
                "fold": fold_id,
                "auc": float(auc),
                "f1": float(f1),
                "sharpe": float(sharpe),
                "profit_factor": float(pf if math.isfinite(pf) else 999.0),
                "max_drawdown_pct": float(_drawdown_pct(selected_r)),
                "expectancy_R": float(expectancy),
                "trades": selected_count,
                "coverage_pct": (selected_count / max(1, len(test_idx))) * 100.0,
                "positive": bool(selected_count >= 1 and sharpe > 0 and expectancy > 0 and pf > 1.0),
            }
        )

    if not fold_metrics:
        labels = {"positive": int((dataset["label"] == 1).sum()), "negative": int((dataset["label"] == 0).sum())}
        return _metrics_fail(len(dataset), labels, "no_valid_folds")

    median_sharpe = float(np.median([row["sharpe"] for row in fold_metrics]))
    mean_auc = float(np.mean([row["auc"] for row in fold_metrics]))
    folds_positive = int(sum(1 for row in fold_metrics if row["positive"]))
    total_trades = int(sum(int(row["trades"]) for row in fold_metrics))
    expectancy_r = float(np.mean(all_selected_r)) if all_selected_r else 0.0
    profit_factor = _profit_factor(all_selected_r)
    coverage_pct = (total_selected / max(1, len(dataset))) * 100.0
    metrics = {
        "event_count": int(len(dataset)),
        "label_balance": {
            "positive": int((dataset["label"] == 1).sum()),
            "negative": int((dataset["label"] == 0).sum()),
        },
        "median_fold_sharpe": median_sharpe,
        "profit_factor": float(profit_factor if math.isfinite(profit_factor) else 999.0),
        "folds_positive": folds_positive,
        "folds_total": len(fold_metrics),
        "max_drawdown_pct": float(_drawdown_pct(all_selected_r)),
        "expectancy_R": expectancy_r,
        "total_trades": total_trades,
        "val_auc": mean_auc,
        "coverage_pct": coverage_pct,
        "fold_metrics": fold_metrics,
    }
    metrics["passes"] = bool(
        metrics["median_fold_sharpe"] >= 0.7
        and metrics["profit_factor"] >= 1.3
        and metrics["folds_positive"] >= 7
        and metrics["max_drawdown_pct"] <= 20.0
        and metrics["expectancy_R"] >= 0.15
        and metrics["total_trades"] >= 50
        and metrics["val_auc"] >= 0.55
    )
    return metrics
