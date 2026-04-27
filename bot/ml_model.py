from __future__ import annotations

import pickle
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from xgboost import XGBClassifier
    USING_XGBOOST = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    USING_XGBOOST = False


BASE_DIR = Path(__file__).resolve().parent
LEGACY_MODEL_PATH = BASE_DIR / "model.pkl"
MIN_TRAIN_TRADES = int(os.getenv("ML_MIN_TRAIN_TRADES", "20"))
MIN_VALIDATION_ACCURACY = float(os.getenv("ML_MIN_VALIDATION_ACCURACY", "0.55"))
_TRAIN_LOCK = threading.Lock()

FEATURE_ORDER = [
    "ema9_minus_ema21_pct",
    "rsi",
    "price_vs_ema200_pct",
    "candle_body_pct",
    "adx",
    "supertrend_direction",
    "volume_vs_ma20_ratio",
    "atr_pct",
    "return_3_pct",
    "return_5_pct",
    "range_5_pct",
    "body_avg_3_pct",
    "volume_trend_3_ratio",
    "close_pos_5",
    "ema9_slope_3_pct",
]


def _normalize_symbol(symbol: Optional[str]) -> str:
    return (symbol or "BTCUSDT").upper().strip()


def _model_path(symbol: Optional[str]) -> Path:
    return BASE_DIR / f"model_{_normalize_symbol(symbol)}.pkl"


def _load_state_from_path(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict) and "model" in state:
            if state.get("model_type") in {"xgb", "rf", None}:
                return state
    except Exception:
        return None
    return None


def _load_state(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    symbol = _normalize_symbol(symbol)
    state = _load_state_from_path(_model_path(symbol))
    if state is not None:
        return state
    if symbol == "BTCUSDT":
        legacy = _load_state_from_path(LEGACY_MODEL_PATH)
        if legacy is not None:
            return legacy
    return None


def _feature_row(features: Dict[str, Any]) -> List[float]:
    row: List[float] = []
    for key in FEATURE_ORDER:
        try:
            row.append(float(features.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            row.append(0.0)
    return row


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def _rows_from_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not history:
        return []
    cleaned: List[Dict[str, Any]] = []
    for row in history:
        if isinstance(row, dict):
            cleaned.append(row)
    return cleaned


def snapshot_features(last: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None, cols: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    cols = cols or {}
    rows = _rows_from_history(history)
    if not rows and isinstance(last, dict):
        rows = [last]

    def _row_value(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
        return _safe_float(row.get(key), default)

    def _close_from_end(offset: int) -> float:
        if len(rows) <= offset:
            return 0.0
        return _row_value(rows[-(offset + 1)], "close")

    def _slice_avg(field: str, count: int, start_offset: int = 0) -> float:
        if len(rows) <= start_offset:
            return 0.0
        window = rows[max(0, len(rows) - (start_offset + count)):len(rows) - start_offset]
        if not window:
            return 0.0
        values = [_row_value(r, field) for r in window]
        values = [v for v in values if v != 0.0]
        return sum(values) / len(values) if values else 0.0

    ema_fast_key = cols.get("ema_fast", "ema9")
    ema_slow_key = cols.get("ema_slow", "ema21")
    ema_trend_key = cols.get("ema_trend", "ema200")
    supertrend_key = cols.get("supertrend_dir", "supertrend_direction")

    close = _row_value(last, "close")
    open_ = _row_value(last, "open")
    high = _row_value(last, "high")
    low = _row_value(last, "low")
    ema9 = _row_value(last, ema_fast_key)
    ema21 = _row_value(last, ema_slow_key)
    ema200 = _row_value(last, ema_trend_key)
    volume = _row_value(last, "volume")
    vol_ma20 = _row_value(last, "vol_ma20")
    atr = _row_value(last, "atr")
    rsi = _row_value(last, "rsi")
    adx = _row_value(last, "adx")
    supertrend_direction = _safe_float(last.get(supertrend_key), 0.0)

    close_3 = _close_from_end(3)
    close_5 = _close_from_end(5)
    ema9_3 = _safe_float(rows[-4].get(ema_fast_key), ema9) if len(rows) >= 4 else ema9
    recent_5 = rows[-5:] if len(rows) >= 5 else rows[:]
    recent_3 = rows[-3:] if len(rows) >= 3 else rows[:]
    prior_3 = rows[-6:-3] if len(rows) >= 6 else []
    highest_5 = max((_row_value(row, "high") for row in recent_5), default=0.0)
    lowest_5 = min((_row_value(row, "low") for row in recent_5), default=0.0)
    avg_body_3 = sum(abs(_row_value(row, "close") - _row_value(row, "open")) / _row_value(row, "open") * 100.0
                     for row in recent_3 if _row_value(row, "open")) / len(recent_3) if recent_3 else 0.0
    avg_vol_3 = _slice_avg("volume", 3)
    prev_avg_vol_3 = _slice_avg("volume", 3, start_offset=3)
    vol_trend_3 = (avg_vol_3 / prev_avg_vol_3) if prev_avg_vol_3 else 0.0
    close_pos_5 = ((close - lowest_5) / (highest_5 - lowest_5)) if highest_5 > lowest_5 else 0.0
    ema9_slope_3 = ((ema9 - ema9_3) / ema9_3 * 100.0) if ema9_3 else 0.0

    feature_map = {
        "ema9_minus_ema21_pct": ((ema9 - ema21) / ema21 * 100.0) if ema21 else 0.0,
        "rsi": rsi,
        "price_vs_ema200_pct": ((close - ema200) / ema200 * 100.0) if ema200 else 0.0,
        "candle_body_pct": abs(close - open_) / open_ * 100.0 if open_ else 0.0,
        "adx": adx,
        "supertrend_direction": supertrend_direction,
        "volume_vs_ma20_ratio": (volume / vol_ma20) if vol_ma20 else 0.0,
        "atr_pct": (atr / close * 100.0) if close else 0.0,
        "return_3_pct": ((close - close_3) / close_3 * 100.0) if close_3 else 0.0,
        "return_5_pct": ((close - close_5) / close_5 * 100.0) if close_5 else 0.0,
        "range_5_pct": ((highest_5 - lowest_5) / close * 100.0) if close and highest_5 > lowest_5 else 0.0,
        "body_avg_3_pct": avg_body_3,
        "volume_trend_3_ratio": vol_trend_3,
        "close_pos_5": close_pos_5,
        "ema9_slope_3_pct": ema9_slope_3,
    }
    for key in ("return_3_pct", "return_5_pct", "range_5_pct", "body_avg_3_pct", "volume_trend_3_ratio", "close_pos_5", "ema9_slope_3_pct"):
        if key in last and last.get(key) is not None:
            feature_map[key] = _safe_float(last.get(key), feature_map[key])
    return feature_map


def _balanced_accuracy(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> float:
    if not y_true:
        return 0.0
    preds = [1 if prob >= threshold else 0 for prob in y_prob]
    tp = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 0)
    pos_total = tp + fn
    neg_total = tn + fp
    recall_pos = tp / pos_total if pos_total else 0.0
    recall_neg = tn / neg_total if neg_total else 0.0
    return (recall_pos + recall_neg) / 2.0


def _state_is_ready(state: Optional[Dict[str, Any]]) -> bool:
    if not state:
        return False
    if int(state.get("trained_on", 0)) < MIN_TRAIN_TRADES:
        return False
    if list(state.get("feature_order") or []) != FEATURE_ORDER:
        return False
    try:
        return float(state.get("validation_accuracy", 0.0)) >= MIN_VALIDATION_ACCURACY
    except (TypeError, ValueError):
        return False


def label_closed_trades(trades):
    labeled = []
    for trade in trades:
        pnl = trade.get("pnl")
        if pnl is None:
            continue
        try:
            pnl_value = float(pnl or 0.0)
        except (TypeError, ValueError):
            continue
        if pnl_value > 0:
            labeled.append((trade, 1))
        elif pnl_value < 0:
            labeled.append((trade, 0))
    return labeled


def label_triple_barrier(trades, profit_target: float = 0.02, stop_loss: float = 0.01):
    labeled = []
    for trade in trades:
        pnl = trade.get("pnl")
        entry_price = trade.get("entry_price")
        notional_usdt = trade.get("notional_usdt")
        qty = trade.get("qty")
        if pnl is None or entry_price is None:
            continue
        entry_price = float(entry_price or 0.0)
        pnl = float(pnl or 0.0)
        if entry_price <= 0:
            continue
        notional = float(notional_usdt or 0.0)
        if notional <= 0:
            try:
                notional = abs(float(qty or 0.0)) * entry_price
            except (TypeError, ValueError):
                notional = 0.0
        if notional <= 0:
            continue
        if pnl > profit_target * notional:
            labeled.append((trade, 1))
        elif pnl < -stop_loss * notional:
            labeled.append((trade, -1))
    return labeled


def _trade_features(trade: Dict[str, Any]) -> Dict[str, float]:
    return snapshot_features(trade, [trade])


def train(trades: list, symbol: Optional[str] = "BTCUSDT") -> None:
    symbol = _normalize_symbol(symbol)
    filtered = [trade for trade in trades if symbol is None or str(trade.get("symbol", "BTCUSDT")).upper() == symbol]
    labeled = label_closed_trades(filtered)
    if len(labeled) < MIN_TRAIN_TRADES:
        labeled = label_triple_barrier(filtered)
    rows: List[List[float]] = []
    labels: List[int] = []

    for trade, label in labeled:
        features = _trade_features(trade)
        rows.append(_feature_row(features))
        labels.append(1 if label == 1 else 0)

    if len(rows) < MIN_TRAIN_TRADES or len(set(labels)) < 2:
        return

    validation_size = max(4, int(round(len(rows) * 0.2)))
    validation_size = min(validation_size, max(2, len(rows) - 4))
    if validation_size < 2 or len(rows) - validation_size < 4:
        return

    train_rows = rows[:-validation_size]
    train_labels = labels[:-validation_size]
    val_rows = rows[-validation_size:]
    val_labels = labels[-validation_size:]
    if len(set(train_labels)) < 2 or len(set(val_labels)) < 1:
        return

    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0

    with _TRAIN_LOCK:
        if USING_XGBOOST:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
                scale_pos_weight=scale_pos_weight,
            )
        else:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                random_state=42,
                class_weight="balanced",
            )
        model.fit(train_rows, train_labels)

        validation_accuracy = 0.0
        validation_positive_rate = 0.0
        try:
            proba = model.predict_proba(val_rows)
            classes = list(getattr(model, "classes_", []))
            if 1 in classes:
                index = classes.index(1)
                y_prob = [float(row[index]) for row in proba]
                validation_accuracy = _balanced_accuracy(val_labels, y_prob)
                validation_positive_rate = sum(val_labels) / len(val_labels) if val_labels else 0.0
        except Exception:
            validation_accuracy = 0.0

        state = {
            "model": model,
            "trained_on": int(len(rows)),
            "source_trades": int(len(filtered)),
            "feature_order": FEATURE_ORDER,
            "model_type": "xgb" if USING_XGBOOST else "rf",
            "symbol": symbol,
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "train_samples": int(len(train_rows)),
            "validation_trades": int(len(val_rows)),
            "validation_accuracy": float(validation_accuracy),
            "validation_positive_rate": float(validation_positive_rate),
            "validation_threshold": float(MIN_VALIDATION_ACCURACY),
            "label_balance": {
                "positive": int(sum(labels)),
                "negative": int(len(labels) - sum(labels)),
            },
        }
        path = _model_path(symbol)
        with path.open("wb") as f:
            pickle.dump(state, f)
        if symbol == "BTCUSDT":
            with LEGACY_MODEL_PATH.open("wb") as f:
                pickle.dump(state, f)


def predict(features: dict, symbol: Optional[str] = "BTCUSDT") -> float:
    state = _load_state(symbol)
    if not _state_is_ready(state):
        return 0.5

    model = state.get("model")
    if model is None:
        return 0.5

    try:
        row = [_feature_row(features)]
        proba = model.predict_proba(row)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
            return float(proba[0][index])
    except Exception:
        return 0.5
    return 0.5


def is_ready(symbol: Optional[str] = "BTCUSDT") -> bool:
    state = _load_state(symbol)
    return _state_is_ready(state)


def model_info(symbol: Optional[str] = "BTCUSDT") -> Dict[str, Any]:
    state = _load_state(symbol) or {}
    return {
        "trained_on": int(state.get("trained_on", 0) or 0),
        "train_samples": int(state.get("train_samples", 0) or 0),
        "validation_trades": int(state.get("validation_trades", 0) or 0),
        "validation_accuracy": float(state.get("validation_accuracy", 0.0) or 0.0),
        "validation_positive_rate": float(state.get("validation_positive_rate", 0.0) or 0.0),
        "validation_threshold": float(state.get("validation_threshold", MIN_VALIDATION_ACCURACY) or MIN_VALIDATION_ACCURACY),
        "source_trades": int(state.get("source_trades", 0) or 0),
        "label_balance": state.get("label_balance", {"positive": 0, "negative": 0}),
        "last_trained_at": state.get("last_trained_at"),
        "symbol": _normalize_symbol(symbol),
    }
