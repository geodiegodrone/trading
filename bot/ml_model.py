from __future__ import annotations

import math
import os
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    USING_LIGHTGBM = True
except Exception:  # pragma: no cover
    from xgboost import XGBClassifier

    USING_LIGHTGBM = False

from features import FEATURE_COLUMNS, build_features


BASE_DIR = Path(__file__).resolve().parent
LEGACY_MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_PATH = BASE_DIR / "model_BTCUSDT.pkl"
_TRAIN_LOCK = threading.Lock()

MIN_TRAIN_TRADES = int(os.getenv("ML_MIN_TRAIN_TRADES", "20"))
MIN_VAL_SHARPE = float(os.getenv("ML_MIN_VAL_SHARPE", "0.5"))
MIN_VAL_AUC = float(os.getenv("ML_MIN_VAL_AUC", "0.55"))
MIN_COVERAGE_PCT = float(os.getenv("ML_MIN_COVERAGE_PCT", "5.0"))
MAX_COVERAGE_PCT = float(os.getenv("ML_MAX_COVERAGE_PCT", "60.0"))
MIN_VALIDATION_TRADES = int(os.getenv("ML_MIN_VALIDATION_TRADES", "30"))
MIN_FOLD_POSITIVES = int(os.getenv("ML_MIN_FOLD_POSITIVES", "15"))
MIN_THRESHOLD = float(os.getenv("ML_MIN_THRESHOLD", "0.45"))
MAX_THRESHOLD = float(os.getenv("ML_MAX_THRESHOLD", "0.70"))
T_BARS = int(os.getenv("ML_T_BARS", "24"))
ATR_MULT = float(os.getenv("ML_ATR_MULT", "1.5"))
TP_RATIO = float(os.getenv("ML_TP_RATIO", "2.0"))


def _model_path(symbol: Optional[str] = "BTCUSDT") -> Path:
    _normalize_symbol(symbol)
    return MODEL_PATH


def _normalize_symbol(symbol: Optional[str]) -> str:
    return "BTCUSDT"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def _coerce_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"BUY", "LONG", "1", "TRUE"}:
        return "LONG"
    if text in {"SELL", "SHORT", "-1", "0", "FALSE"}:
        return "SHORT"
    return "LONG"


def _side_buy(value: Any) -> float:
    return 1.0 if _coerce_side(value) == "LONG" else 0.0


def _positive_proba(model: Any, rows: np.ndarray) -> np.ndarray:
    data = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    probs = model.predict_proba(data)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    return np.asarray(probs).reshape(-1)


def _annualized_sharpe(r_values: Sequence[float], holding_bars: Sequence[float], bar_minutes: int = 5) -> float:
    clean = [float(r) for r in r_values if r is not None]
    if len(clean) < 2:
        return 0.0
    mean_r = float(np.mean(clean))
    std_r = float(np.std(clean, ddof=1))
    if std_r <= 0:
        return 0.0
    avg_hold = max(1.0, float(np.mean([max(1.0, _safe_float(v, 1.0)) for v in holding_bars or [1.0]])))
    bars_per_year = (365.0 * 24.0 * 60.0) / float(bar_minutes)
    trades_per_year = bars_per_year / avg_hold
    return (mean_r / std_r) * math.sqrt(trades_per_year)


def _trades_per_year_estimate(holding_bars: Sequence[float], signal_idx: Sequence[int] | None = None, bar_minutes: int = 5) -> float:
    holds = [max(1.0, _safe_float(v, 1.0)) for v in holding_bars or []]
    if not holds:
        return 0.0
    bars_per_year = (365.0 * 24.0 * 60.0) / float(bar_minutes)
    avg_hold = float(np.mean(holds))
    by_holding = bars_per_year / avg_hold if avg_hold > 0 else 0.0
    if signal_idx:
        span = max(1.0, float(max(signal_idx) - min(signal_idx) + 1))
        empirical = (len(holds) / span) * bars_per_year
        return float(min(by_holding, empirical))
    return float(by_holding)


def _max_drawdown(r_values: Sequence[float]) -> float:
    if not r_values:
        return 0.0
    equity = np.cumsum(np.asarray(r_values, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    return float(drawdowns.max()) if len(drawdowns) else 0.0


def cusum_filter(returns: Sequence[float], h: float) -> List[int]:
    threshold = abs(_safe_float(h, 0.0))
    if threshold <= 0:
        return []
    t_events: List[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for idx, value in enumerate(returns):
        ret = _safe_float(value, 0.0)
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)
        if s_pos > threshold:
            s_pos = 0.0
            t_events.append(idx)
        elif s_neg < -threshold:
            s_neg = 0.0
            t_events.append(idx)
    return t_events


def _resolve_intrabar_touch(row: pd.Series, side: str, sl: float, tp: float) -> str | None:
    high = _safe_float(row.get("high"))
    low = _safe_float(row.get("low"))
    open_ = _safe_float(row.get("open"))
    hit_tp = high >= tp if side == "LONG" else low <= tp
    hit_sl = low <= sl if side == "LONG" else high >= sl
    if hit_tp and hit_sl:
        if side == "LONG":
            tp_dist = abs(tp - open_)
            sl_dist = abs(open_ - sl)
        else:
            tp_dist = abs(open_ - tp)
            sl_dist = abs(sl - open_)
        return "TP" if tp_dist < sl_dist else "SL"
    if hit_tp:
        return "TP"
    if hit_sl:
        return "SL"
    return None


def apply_triple_barrier(
    df_indicators: pd.DataFrame,
    signals: Sequence[Dict[str, Any]] | pd.DataFrame,
    atr_mult: float = 1.5,
    tp_ratio: float = 2.0,
    t_bars: int = 24,
) -> pd.DataFrame:
    if isinstance(signals, pd.DataFrame):
        rows = signals.to_dict("records")
    else:
        rows = list(signals or [])
    events: List[Dict[str, Any]] = []
    if df_indicators.empty:
        return pd.DataFrame(events)
    total = len(df_indicators)
    for row in rows:
        signal_idx = int(row.get("signal_idx"))
        side = _coerce_side(row.get("side"))
        if signal_idx < 0 or signal_idx >= total:
            continue
        entry_price = _safe_float(df_indicators.iloc[signal_idx].get("close"))
        atr = _safe_float(df_indicators.iloc[signal_idx].get("atr"))
        if entry_price <= 0 or atr <= 0:
            continue
        sl_dist = atr * atr_mult
        if side == "LONG":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * tp_ratio
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * tp_ratio
        exit_idx = min(total - 1, signal_idx + int(t_bars))
        exit_price = _safe_float(df_indicators.iloc[exit_idx].get("close"), entry_price)
        exit_reason = "TIME"
        for idx in range(signal_idx + 1, min(total, signal_idx + int(t_bars) + 1)):
            touch = _resolve_intrabar_touch(df_indicators.iloc[idx], side, sl, tp)
            if touch:
                exit_idx = idx
                exit_reason = touch
                exit_price = tp if touch == "TP" else sl
                break
        if exit_reason == "TIME":
            exit_price = _safe_float(df_indicators.iloc[exit_idx].get("close"), entry_price)
        pnl_per_unit = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        r_multiple = pnl_per_unit / sl_dist if sl_dist > 0 else 0.0
        label = 1 if exit_reason == "TP" or (exit_reason == "TIME" and pnl_per_unit > 0) else 0
        events.append(
            {
                "signal_idx": signal_idx,
                "side": side,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "exit_idx": exit_idx,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "label": label,
                "r_multiple": r_multiple,
                "holding_bars": max(1, exit_idx - signal_idx),
            }
        )
    return pd.DataFrame(events)


@dataclass
class FoldMetrics:
    fold: int
    auc: float
    f1: float
    sharpe: float
    max_drawdown: float
    trades: int
    coverage_pct: float
    positives: int
    degenerate: bool


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 36) -> None:
        self.n_splits = max(2, int(n_splits))
        self.embargo = max(0, int(embargo))

    def split(self, samples: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(samples)
        if n_samples < self.n_splits:
            yield np.arange(0, max(0, n_samples - 1)), np.arange(max(0, n_samples - 1), n_samples)
            return
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        intervals = samples[["signal_idx", "exit_idx"]].to_numpy(dtype=int)
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = np.arange(start, stop)
            test_start = intervals[test_idx, 0].min()
            test_end = intervals[test_idx, 1].max()
            embargo_end = test_end + self.embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            overlap = (intervals[:, 0] <= test_end) & (intervals[:, 1] >= test_start)
            embargo = (intervals[:, 0] > test_end) & (intervals[:, 0] <= embargo_end)
            train_mask &= ~overlap
            train_mask &= ~embargo
            train_idx = np.where(train_mask)[0]
            current = stop
            if len(train_idx) < 20 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def _uniqueness_weights(signal_idx: Sequence[int], exit_idx: Sequence[int]) -> np.ndarray:
    if not signal_idx:
        return np.array([], dtype=float)
    max_bar = int(max(exit_idx))
    concurrency = np.zeros(max_bar + 2, dtype=float)
    intervals = list(zip(signal_idx, exit_idx))
    for start, end in intervals:
        concurrency[int(start) : int(end) + 1] += 1.0
    weights = []
    for start, end in intervals:
        segment = concurrency[int(start) : int(end) + 1]
        uniq = np.divide(1.0, segment, out=np.zeros_like(segment), where=segment > 0)
        weights.append(float(np.mean(uniq)) if len(uniq) else 1.0)
    return np.asarray(weights, dtype=float)


def _build_model(y_train: np.ndarray | None = None) -> Any:
    positives = int(np.sum(y_train)) if y_train is not None else 0
    negatives = int(len(y_train) - positives) if y_train is not None else 0
    if USING_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.025,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=30,
            feature_fraction=0.8,
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
    return XGBClassifier(  # pragma: no cover
        n_estimators=600,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=max(1.0, float(negatives) / float(positives)) if positives > 0 else 1.0,
        random_state=42,
        n_jobs=1,
    )


def _fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_train: np.ndarray | None = None,
    w_valid: np.ndarray | None = None,
) -> Any:
    model = _build_model(y_train)
    train_frame = pd.DataFrame(x_train, columns=FEATURE_COLUMNS)
    valid_frame = pd.DataFrame(x_valid, columns=FEATURE_COLUMNS)
    if USING_LIGHTGBM:
        model.fit(
            train_frame,
            y_train,
            sample_weight=w_train,
            eval_set=[(valid_frame, y_valid)],
            eval_sample_weight=[w_valid] if w_valid is not None else None,
            callbacks=[early_stopping(50, verbose=False), log_evaluation(period=0)],
        )
        return model
    model.fit(  # pragma: no cover
        train_frame,
        y_train,
        sample_weight=w_train,
        eval_set=[(valid_frame, y_valid)],
        verbose=False,
    )
    return model


def _chronological_validation_split(indices: np.ndarray, validation_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    split = max(1, int(len(indices) * (1.0 - validation_fraction)))
    split = min(split, len(indices) - 1)
    return indices[:split], indices[split:]


def _optimize_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    r_values: Sequence[float],
    holding_bars: Sequence[float],
    signal_idx: Sequence[int] | None = None,
) -> Dict[str, float]:
    candidates = np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 0.0001, 0.005)
    best: Dict[str, float] | None = None
    for tau in candidates:
        mask = np.asarray(probs) >= tau
        selected_r = np.asarray(r_values, dtype=float)[mask]
        selected_h = np.asarray(holding_bars, dtype=float)[mask]
        selected_idx = np.asarray(signal_idx, dtype=int)[mask].tolist() if signal_idx is not None else []
        trade_count = int(mask.sum())
        coverage_pct = (trade_count / max(1, len(probs))) * 100.0
        preds = np.where(mask, 1, 0)
        f1 = f1_score(labels, preds, zero_division=0)
        if trade_count > 0:
            expected_r = float(np.mean(selected_r))
            std_r = float(np.std(selected_r, ddof=1)) if trade_count > 1 else 0.0
            sharpe = _annualized_sharpe(selected_r.tolist(), selected_h.tolist())
            drawdown = _max_drawdown(selected_r.tolist())
            trades_per_year = _trades_per_year_estimate(selected_h.tolist(), selected_idx)
        else:
            expected_r = 0.0
            std_r = 0.0
            sharpe = 0.0
            drawdown = 0.0
            trades_per_year = 0.0
        row = {
            "tau": float(round(tau, 3)),
            "score": float(expected_r * math.sqrt(max(1, trade_count)) - 0.5 * std_r),
            "sharpe": float(sharpe),
            "expected_r": float(expected_r),
            "trades": float(trade_count),
            "coverage_pct": float(coverage_pct),
            "f1": float(f1),
            "drawdown": float(drawdown),
            "trades_per_year": float(trades_per_year),
        }
        if (
            trade_count >= MIN_VALIDATION_TRADES
            and row["expected_r"] > 0.0
            and row["sharpe"] > 0.0
            and (best is None or row["score"] > best["score"])
        ):
            best = row
    if best is None:
        return {
            "tau": 0.55,
            "val_sharpe": 0.0,
            "validation_trades": 0,
            "coverage_pct": 0.0,
            "min_trades_per_year_estimate": 0.0,
            "fallback_to_f1": True,
            "no_profitable_threshold": True,
        }
    selected = best or {
        "tau": 0.55,
        "score": 0.0,
        "sharpe": 0.0,
        "trades": 0.0,
        "coverage_pct": 0.0,
        "f1": 0.0,
        "drawdown": 0.0,
        "trades_per_year": 0.0,
    }
    return {
        "tau": float(selected["tau"]),
        "val_sharpe": float(selected["sharpe"]),
        "validation_trades": int(selected["trades"]),
        "coverage_pct": float(selected["coverage_pct"]),
        "min_trades_per_year_estimate": float(selected["trades_per_year"]),
        "fallback_to_f1": False,
        "no_profitable_threshold": False,
    }


def _median_or_zero(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.median(clean)) if clean else 0.0


def _mean_or_zero(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def _readiness_reason(state: Dict[str, Any]) -> str:
    reasons: List[str] = []
    if bool(state.get("no_profitable_threshold", False)):
        reasons.append("sin τ con expected_r>0 y sharpe>0")
    if int(state.get("trained_on", 0) or 0) < MIN_TRAIN_TRADES:
        reasons.append(f"faltan muestras ({int(state.get('trained_on', 0) or 0)}/{MIN_TRAIN_TRADES})")
    if int(state.get("usable_folds", 0) or 0) <= 0:
        reasons.append("sin folds válidos no degenerados")
    if float(state.get("val_sharpe", 0.0) or 0.0) < MIN_VAL_SHARPE:
        reasons.append(f"sharpe mediano < {MIN_VAL_SHARPE:.2f}")
    if float(state.get("val_auc", 0.0) or 0.0) < MIN_VAL_AUC:
        reasons.append(f"auc < {MIN_VAL_AUC:.2f}")
    coverage_pct = float(state.get("coverage_pct", 0.0) or 0.0)
    if coverage_pct < MIN_COVERAGE_PCT:
        reasons.append(f"coverage < {MIN_COVERAGE_PCT:.1f}%")
    if coverage_pct > MAX_COVERAGE_PCT:
        reasons.append(f"coverage > {MAX_COVERAGE_PCT:.1f}%")
    return "; ".join(reasons)


def _serialize_state(state: Dict[str, Any]) -> None:
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(state, handle)
    with LEGACY_MODEL_PATH.open("wb") as handle:
        pickle.dump(state, handle)


def _load_state() -> Dict[str, Any]:
    for path in (MODEL_PATH, LEGACY_MODEL_PATH):
        if path.exists():
            try:
                with path.open("rb") as handle:
                    state = pickle.load(handle)
                if isinstance(state, dict):
                    return state
            except Exception:
                continue
    return {}


def _prepare_training_set(events: pd.DataFrame, df_klines: pd.DataFrame) -> pd.DataFrame:
    indicators = df_klines.copy().reset_index(drop=True)
    feature_frame = build_features(indicators)
    rows: List[Dict[str, Any]] = []
    for event in events.to_dict("records"):
        signal_idx = int(event["signal_idx"])
        if signal_idx < 0 or signal_idx >= len(feature_frame):
            continue
        feature_row = feature_frame.iloc[signal_idx].to_dict()
        side = _coerce_side(event["side"])
        side_buy = 1.0 if side == "LONG" else 0.0
        feature_row["side_buy"] = side_buy
        feature_row["regime_code"] = _safe_float(event.get("regime_code"), _safe_float(feature_row.get("regime_code"), 0.0))
        feature_row["supertrend_aligned_side"] = 1.0 if (
            (side_buy >= 0.5 and _safe_float(feature_row.get("supertrend_direction")) > 0)
            or (side_buy < 0.5 and _safe_float(feature_row.get("supertrend_direction")) < 0)
        ) else 0.0
        feature_row.update(
            {
                "signal_idx": signal_idx,
                "exit_idx": int(event["exit_idx"]),
                "label": int(event["label"]),
                "r_multiple": _safe_float(event["r_multiple"]),
                "holding_bars": int(event["holding_bars"]),
                "side": side,
                "entry_price": _safe_float(event["entry_price"]),
                "exit_reason": str(event["exit_reason"]),
            }
        )
        rows.append(feature_row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def _legacy_training_set(trades: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, trade in enumerate(trades or []):
        if not isinstance(trade, dict):
            continue
        result = str(trade.get("result") or "").upper()
        pnl = trade.get("pnl")
        if pnl is None or result == "RECONCILED":
            continue
        label = 1 if _safe_float(trade.get("r_multiple"), _safe_float(pnl)) > 0 else 0
        row = {column: _safe_float(trade.get(column), 0.0) for column in FEATURE_COLUMNS}
        row["side_buy"] = _side_buy(trade.get("side"))
        row["supertrend_aligned_side"] = 1.0 if (
            (row["side_buy"] >= 0.5 and _safe_float(trade.get("supertrend_direction", trade.get("supertrend_dir"))) > 0)
            or (row["side_buy"] < 0.5 and _safe_float(trade.get("supertrend_direction", trade.get("supertrend_dir"))) < 0)
        ) else 0.0
        row.update(
            {
                "signal_idx": idx,
                "exit_idx": idx + int(max(1, _safe_float(trade.get("holding_bars"), 1.0))),
                "label": label,
                "r_multiple": _safe_float(trade.get("r_multiple"), _safe_float(pnl)),
                "holding_bars": max(1, int(_safe_float(trade.get("holding_bars"), 1.0))),
                "side": _coerce_side(trade.get("side")),
                "entry_price": _safe_float(trade.get("entry_price")),
                "exit_reason": str(trade.get("exit_reason") or result or ""),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def label_closed_trades(trades: Iterable[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    labeled: List[Tuple[Dict[str, Any], int]] = []
    for trade in trades or []:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("result") or "").upper() == "RECONCILED":
            continue
        if trade.get("pnl") is None:
            continue
        label = 1 if _safe_float(trade.get("r_multiple"), _safe_float(trade.get("pnl"))) > 0 else 0
        labeled.append((trade, label))
    return labeled


def snapshot_features(
    last: Dict[str, Any],
    history: Optional[Iterable[Dict[str, Any]]] = None,
    cols: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    rows = [dict(row) for row in history or [] if isinstance(row, dict)]
    if isinstance(last, dict):
        if not rows:
            rows.append(dict(last))
        else:
            last_ts = rows[-1].get("ts")
            if last_ts != last.get("ts"):
                rows.append(dict(last))
            else:
                rows[-1] = dict(last)
    if not rows:
        return {column: 0.0 for column in FEATURE_COLUMNS}
    frame = pd.DataFrame(rows)
    feature_frame = build_features(frame)
    if feature_frame.empty:
        return {column: 0.0 for column in FEATURE_COLUMNS}
    result = feature_frame.iloc[-1].to_dict()
    result["regime_code"] = _safe_float(last.get("regime_code") if isinstance(last, dict) else None, _safe_float(result.get("regime_code"), 0.0))
    result["side_buy"] = _side_buy(last.get("side") if isinstance(last, dict) else None)
    result["supertrend_aligned_side"] = 1.0 if (
        (result["side_buy"] >= 0.5 and _safe_float(result.get("supertrend_direction")) > 0)
        or (result["side_buy"] < 0.5 and _safe_float(result.get("supertrend_direction")) < 0)
    ) else 0.0
    return {column: _safe_float(result.get(column), 0.0) for column in FEATURE_COLUMNS}


def features_from_trade_row(trade: Dict[str, Any]) -> Dict[str, float]:
    result = {column: _safe_float(trade.get(column), 0.0) for column in FEATURE_COLUMNS}
    result["side_buy"] = _side_buy(trade.get("side"))
    result["supertrend_aligned_side"] = 1.0 if (
        (result["side_buy"] >= 0.5 and _safe_float(trade.get("supertrend_direction", trade.get("supertrend_dir"))) > 0)
        or (result["side_buy"] < 0.5 and _safe_float(trade.get("supertrend_direction", trade.get("supertrend_dir"))) < 0)
    ) else 0.0
    return result


def _train_bootstrap(events: pd.DataFrame, df_klines: pd.DataFrame) -> Dict[str, Any]:
    dataset = _prepare_training_set(events, df_klines)
    if len(dataset) < MIN_TRAIN_TRADES or dataset["label"].nunique() < 2:
        return {}

    x = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["label"].to_numpy(dtype=int)
    signal_idx = dataset["signal_idx"].to_numpy(dtype=int)
    exit_idx = dataset["exit_idx"].to_numpy(dtype=int)
    r_values = dataset["r_multiple"].to_numpy(dtype=float)
    holding_bars = dataset["holding_bars"].to_numpy(dtype=float)
    weights = _uniqueness_weights(signal_idx.tolist(), exit_idx.tolist())

    splitter = PurgedKFold(n_splits=5, embargo=int(T_BARS * 1.5))
    oof_probs = np.zeros(len(dataset), dtype=float)
    scored_mask = np.zeros(len(dataset), dtype=bool)
    folds: List[FoldMetrics] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(dataset), start=1):
        if len(train_idx) < 40 or len(test_idx) < 5:
            continue
        train_idx_sorted = np.sort(train_idx)
        fit_idx, early_idx = _chronological_validation_split(train_idx_sorted)
        if len(fit_idx) < 20 or len(early_idx) < 5:
            continue
        model = _fit_model(
            x[fit_idx],
            y[fit_idx],
            x[early_idx],
            y[early_idx],
            weights[fit_idx],
            weights[early_idx],
        )
        calib_raw = _positive_proba(model, x[early_idx])
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_raw, y[early_idx])
        raw_test_probs = _positive_proba(model, x[test_idx])
        test_probs = calibrator.predict(raw_test_probs)
        oof_probs[test_idx] = test_probs
        scored_mask[test_idx] = True
        positives = int(y[test_idx].sum())
        degenerate = positives < MIN_FOLD_POSITIVES
        auc = roc_auc_score(y[test_idx], test_probs) if len(np.unique(y[test_idx])) > 1 else 0.5
        preds = (test_probs >= 0.5).astype(int)
        f1 = f1_score(y[test_idx], preds, zero_division=0)
        selected = test_probs >= 0.5
        selected_r = r_values[test_idx][selected]
        selected_h = holding_bars[test_idx][selected]
        selected_count = int(selected.sum())
        sharpe = _annualized_sharpe(selected_r.tolist(), selected_h.tolist()) if len(selected_r) else 0.0
        drawdown = _max_drawdown(selected_r.tolist()) if len(selected_r) else 0.0
        folds.append(
            FoldMetrics(
                fold=fold_id,
                auc=float(auc),
                f1=float(f1),
                sharpe=float(sharpe),
                max_drawdown=float(drawdown),
                trades=selected_count,
                coverage_pct=(selected_count / max(1, len(test_idx))) * 100.0,
                positives=positives,
                degenerate=degenerate,
            )
        )

    if not folds:
        return {}

    scored_probs = oof_probs[scored_mask]
    scored_y = y[scored_mask]
    scored_r = r_values[scored_mask]
    scored_h = holding_bars[scored_mask]
    scored_signal_idx = signal_idx[scored_mask]
    threshold_info = _optimize_threshold(
        scored_probs.tolist(),
        scored_y.tolist(),
        scored_r.tolist(),
        scored_h.tolist(),
        scored_signal_idx.tolist(),
    )
    suggested_threshold = float(threshold_info["tau"])
    preds = (scored_probs >= suggested_threshold).astype(int)
    usable_folds = [fold for fold in folds if not fold.degenerate]
    val_auc = _mean_or_zero([fold.auc for fold in usable_folds])
    val_f1 = _median_or_zero([fold.f1 for fold in usable_folds])
    val_sharpe = _median_or_zero([fold.sharpe for fold in usable_folds])
    val_drawdown = _median_or_zero([fold.max_drawdown for fold in usable_folds])
    selected = scored_probs >= suggested_threshold
    selected_r = scored_r[selected]
    coverage_pct = (float(selected.sum()) / max(1, int(scored_mask.sum()))) * 100.0

    full_idx = np.arange(len(dataset))
    fit_idx, valid_idx = _chronological_validation_split(full_idx)
    final_model = _fit_model(
        x[fit_idx],
        y[fit_idx],
        x[valid_idx],
        y[valid_idx],
        weights[fit_idx],
        weights[valid_idx],
    )
    valid_raw = _positive_proba(final_model, x[valid_idx])
    final_calibrator = IsotonicRegression(out_of_bounds="clip")
    final_calibrator.fit(valid_raw, y[valid_idx])

    state = {
        "model": final_model,
        "calibrator": final_calibrator,
        "trained_on": int(len(dataset)),
        "train_samples": int(len(fit_idx)),
        "validation_samples": int(len(valid_idx)),
        "feature_order": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "suggested_threshold": float(suggested_threshold),
        "val_sharpe": float(val_sharpe),
        "val_auc": float(val_auc),
        "val_f1": float(val_f1),
        "val_drawdown": float(val_drawdown),
        "validation_trades": int(threshold_info["validation_trades"]),
        "coverage_pct": float(coverage_pct),
        "min_trades_per_year_estimate": float(threshold_info["min_trades_per_year_estimate"]),
        "calibrated": True,
        "fold_metrics": [fold.__dict__ for fold in folds],
        "label_balance": {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
        },
        "oof_count": int(scored_mask.sum()),
        "usable_folds": int(len(usable_folds)),
        "fallback_to_f1": bool(threshold_info["fallback_to_f1"]),
        "no_profitable_threshold": bool(threshold_info.get("no_profitable_threshold", False)),
        "model_type": "lightgbm" if USING_LIGHTGBM else "xgboost",
    }
    state["ready"] = False
    state["not_ready_reason"] = _readiness_reason(state)
    state["ready"] = not bool(state["not_ready_reason"])
    return state


def _train_legacy(trades: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    dataset = _legacy_training_set(trades)
    if len(dataset) < MIN_TRAIN_TRADES or dataset["label"].nunique() < 2:
        return {}
    x = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["label"].to_numpy(dtype=int)
    weights = np.ones(len(dataset), dtype=float)
    full_idx = np.arange(len(dataset))
    fit_idx, valid_idx = _chronological_validation_split(full_idx)
    model = _fit_model(x[fit_idx], y[fit_idx], x[valid_idx], y[valid_idx], weights[fit_idx], weights[valid_idx])
    valid_raw = _positive_proba(model, x[valid_idx])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(valid_raw, y[valid_idx])
    valid_probs = calibrator.predict(valid_raw)
    threshold_info = _optimize_threshold(
        valid_probs.tolist(),
        y[valid_idx].tolist(),
        dataset.iloc[valid_idx]["r_multiple"].astype(float).tolist(),
        dataset.iloc[valid_idx]["holding_bars"].astype(float).tolist(),
        dataset.iloc[valid_idx]["signal_idx"].astype(int).tolist(),
    )
    suggested_threshold = float(threshold_info["tau"])
    preds = (np.asarray(valid_probs) >= suggested_threshold).astype(int)
    val_auc = roc_auc_score(y[valid_idx], valid_probs) if len(np.unique(y[valid_idx])) > 1 else 0.5
    val_f1 = f1_score(y[valid_idx], preds, zero_division=0)
    selected = np.asarray(valid_probs) >= suggested_threshold
    val_sharpe = _annualized_sharpe(
        dataset.iloc[valid_idx]["r_multiple"].astype(float)[selected].tolist(),
        dataset.iloc[valid_idx]["holding_bars"].astype(float)[selected].tolist(),
    )
    val_drawdown = _max_drawdown(dataset.iloc[valid_idx]["r_multiple"].astype(float)[selected].tolist())
    coverage_pct = (float(selected.sum()) / max(1, len(valid_idx))) * 100.0
    state = {
        "model": model,
        "calibrator": calibrator,
        "trained_on": int(len(dataset)),
        "train_samples": int(len(fit_idx)),
        "validation_samples": int(len(valid_idx)),
        "feature_order": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "last_trained_at": datetime.now(timezone.utc).isoformat(),
        "suggested_threshold": float(suggested_threshold),
        "val_sharpe": float(val_sharpe),
        "val_auc": float(val_auc),
        "val_f1": float(val_f1),
        "val_drawdown": float(val_drawdown),
        "validation_trades": int(threshold_info["validation_trades"]),
        "coverage_pct": float(coverage_pct),
        "min_trades_per_year_estimate": float(threshold_info["min_trades_per_year_estimate"]),
        "calibrated": True,
        "fold_metrics": [],
        "label_balance": {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
        },
        "oof_count": int(len(valid_idx)),
        "usable_folds": 1,
        "fallback_to_f1": bool(threshold_info["fallback_to_f1"]),
        "no_profitable_threshold": bool(threshold_info.get("no_profitable_threshold", False)),
        "model_type": "lightgbm" if USING_LIGHTGBM else "xgboost",
    }
    state["ready"] = False
    state["not_ready_reason"] = _readiness_reason(state)
    state["ready"] = not bool(state["not_ready_reason"])
    return state


def train(
    trades_df_or_list: Iterable[Dict[str, Any]] | pd.DataFrame,
    symbol: Optional[str] = "BTCUSDT",
    df_klines: Optional[pd.DataFrame] = None,
) -> None:
    _normalize_symbol(symbol)
    with _TRAIN_LOCK:
        if df_klines is not None:
            events = trades_df_or_list if isinstance(trades_df_or_list, pd.DataFrame) else pd.DataFrame(list(trades_df_or_list or []))
            state = _train_bootstrap(events, df_klines)
        else:
            rows = trades_df_or_list.to_dict("records") if isinstance(trades_df_or_list, pd.DataFrame) else list(trades_df_or_list or [])
            state = _train_legacy(rows)
        if not state:
            return
        _serialize_state(state)
        print(
            f"[ml_model] BTCUSDT trained={state['trained_on']} "
            f"val_sharpe={state['val_sharpe']:.3f} "
            f"val_auc={state['val_auc']:.3f} "
            f"suggested_threshold={state['suggested_threshold']:.3f} "
            f"coverage={float(state.get('coverage_pct', 0.0)):.1f}% "
            f"ready={bool(state.get('ready'))}"
        )


def predict(features_dict: Dict[str, Any], symbol: Optional[str] = "BTCUSDT") -> float:
    _normalize_symbol(symbol)
    state = _load_state()
    model = state.get("model")
    if model is None:
        return 0.5
    feature_order = list(state.get("feature_order") or FEATURE_COLUMNS)
    row = np.asarray([[_safe_float(features_dict.get(column), 0.0) for column in feature_order]], dtype=float)
    raw = float(_positive_proba(model, row)[0])
    calibrator = state.get("calibrator")
    if calibrator is not None:
        try:
            raw = float(calibrator.predict([raw])[0])
        except Exception:
            pass
    return max(0.0, min(1.0, raw))


def is_ready(symbol: Optional[str] = "BTCUSDT") -> bool:
    _normalize_symbol(symbol)
    state = _load_state()
    if not state:
        return False
    reason = str(state.get("not_ready_reason") or _readiness_reason(state))
    return not bool(reason)


def model_info(symbol: Optional[str] = "BTCUSDT") -> Dict[str, Any]:
    _normalize_symbol(symbol)
    state = _load_state()
    return {
        "symbol": "BTCUSDT",
        "trained_on": int(state.get("trained_on", 0) or 0),
        "train_samples": int(state.get("train_samples", 0) or 0),
        "validation_samples": int(state.get("validation_samples", 0) or 0),
        "validation_trades": int(state.get("validation_trades", 0) or 0),
        "validation_accuracy": float(state.get("val_auc", 0.0) or 0.0),
        "val_sharpe": float(state.get("val_sharpe", 0.0) or 0.0),
        "val_auc": float(state.get("val_auc", 0.0) or 0.0),
        "val_f1": float(state.get("val_f1", 0.0) or 0.0),
        "val_drawdown": float(state.get("val_drawdown", 0.0) or 0.0),
        "suggested_threshold": float(state.get("suggested_threshold", 0.55) or 0.55),
        "coverage_pct": float(state.get("coverage_pct", 0.0) or 0.0),
        "min_trades_per_year_estimate": float(state.get("min_trades_per_year_estimate", 0.0) or 0.0),
        "calibrated": bool(state.get("calibrated", False)),
        "feature_count": int(state.get("feature_count", len(FEATURE_COLUMNS)) or len(FEATURE_COLUMNS)),
        "feature_order": list(state.get("feature_order") or FEATURE_COLUMNS),
        "last_trained_at": state.get("last_trained_at"),
        "label_balance": state.get("label_balance", {"positive": 0, "negative": 0}),
        "fold_metrics": state.get("fold_metrics", []),
        "model_type": state.get("model_type", "lightgbm" if USING_LIGHTGBM else "xgboost"),
        "ready_threshold": MIN_VAL_SHARPE,
        "ready": bool(is_ready(symbol)),
        "not_ready_reason": str(state.get("not_ready_reason") or _readiness_reason(state)),
        "usable_folds": int(state.get("usable_folds", 0) or 0),
        "fallback_to_f1": bool(state.get("fallback_to_f1", False)),
        "no_profitable_threshold": bool(state.get("no_profitable_threshold", False)),
    }
