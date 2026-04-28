"""Machine-learning utilities for trade scoring and persistence."""

from __future__ import annotations

import os
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from xgboost import XGBClassifier

    USING_XGBOOST = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier

    USING_XGBOOST = False

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
LEGACY_MODEL_PATH = BASE_DIR / "model.pkl"
MIN_TRAIN_TRADES = int(os.getenv("ML_MIN_TRAIN_TRADES", "50"))
MIN_VALIDATION_ACCURACY = float(os.getenv("ML_MIN_VALIDATION_ACCURACY", "0.52"))
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
    "side_buy",
    "rsi_dist_50",
    "adx_strong",
    "supertrend_aligned_side",
    "atr_regime_high",
]


def _normalize_symbol(symbol: Optional[str]) -> str:
    return (symbol or "BTCUSDT").upper().strip()


def _model_path(symbol: Optional[str]) -> Path:
    return BASE_DIR / f"model_{_normalize_symbol(symbol)}.pkl"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def _coerce_side_buy(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        side = value.strip().lower()
        if side in {"buy", "long", "1", "true", "bull", "bullish"}:
            return 1.0
        if side in {"sell", "short", "0", "false", "bear", "bearish", "-1"}:
            return 0.0
    try:
        return 1.0 if float(value) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return default


def _rows_from_history(history: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not history:
        return []
    rows: List[Dict[str, Any]] = []
    for row in history:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_value(row: Dict[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        if name and name in row and row.get(name) is not None:
            return _safe_float(row.get(name), default)
    return default


def _close_from_offset(rows: Sequence[Dict[str, Any]], offset: int) -> float:
    index = len(rows) - offset - 1
    if index < 0 or index >= len(rows):
        return 0.0
    return _safe_float(rows[index].get("close"), 0.0)


def _average_field(rows: Sequence[Dict[str, Any]], field: str) -> float:
    values = [_safe_float(row.get(field), 0.0) for row in rows if row.get(field) is not None]
    values = [value for value in values if value != 0.0]
    return sum(values) / len(values) if values else 0.0


def _first_present_float(trade: Dict[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        if name in trade and trade.get(name) is not None:
            return _safe_float(trade.get(name), default)
    return default


def _build_feature_map(
    *,
    ema9: float,
    ema21: float,
    ema200: float,
    rsi: float,
    candle_body_pct: float,
    adx: float,
    supertrend_direction: float,
    volume_vs_ma20_ratio: float,
    atr_pct: float,
    return_3_pct: float,
    return_5_pct: float,
    range_5_pct: float,
    body_avg_3_pct: float,
    volume_trend_3_ratio: float,
    close_pos_5: float,
    ema9_slope_3_pct: float,
    entry_price: float,
    side_buy: float = 1.0,
) -> Dict[str, float]:
    ema9 = _safe_float(ema9)
    ema21 = _safe_float(ema21)
    ema200 = _safe_float(ema200)
    rsi = _safe_float(rsi)
    candle_body_pct = _safe_float(candle_body_pct)
    adx = _safe_float(adx)
    supertrend_direction = _safe_float(supertrend_direction)
    volume_vs_ma20_ratio = _safe_float(volume_vs_ma20_ratio)
    atr_pct = _safe_float(atr_pct)
    return_3_pct = _safe_float(return_3_pct)
    return_5_pct = _safe_float(return_5_pct)
    range_5_pct = _safe_float(range_5_pct)
    body_avg_3_pct = _safe_float(body_avg_3_pct)
    volume_trend_3_ratio = _safe_float(volume_trend_3_ratio)
    close_pos_5 = _safe_float(close_pos_5)
    ema9_slope_3_pct = _safe_float(ema9_slope_3_pct)
    entry_price = _safe_float(entry_price)
    side_buy = 1.0 if _coerce_side_buy(side_buy, 1.0) >= 0.5 else 0.0

    ema9_minus_ema21_pct = ((ema9 - ema21) / ema21 * 100.0) if ema21 else 0.0
    price_vs_ema200_pct = ((entry_price - ema200) / ema200 * 100.0) if ema200 else 0.0
    rsi_dist_50 = (rsi - 50.0) / 50.0
    adx_strong = 1.0 if adx > 25.0 else 0.0
    supertrend_aligned_side = 1.0 if (
        (side_buy >= 0.5 and supertrend_direction > 0.0)
        or (side_buy < 0.5 and supertrend_direction < 0.0)
    ) else 0.0
    atr_regime_high = 1.0 if atr_pct > 1.0 else 0.0

    return {
        "ema9_minus_ema21_pct": ema9_minus_ema21_pct,
        "rsi": rsi,
        "price_vs_ema200_pct": price_vs_ema200_pct,
        "candle_body_pct": candle_body_pct,
        "adx": adx,
        "supertrend_direction": supertrend_direction,
        "volume_vs_ma20_ratio": volume_vs_ma20_ratio,
        "atr_pct": atr_pct,
        "return_3_pct": return_3_pct,
        "return_5_pct": return_5_pct,
        "range_5_pct": range_5_pct,
        "body_avg_3_pct": body_avg_3_pct,
        "volume_trend_3_ratio": volume_trend_3_ratio,
        "close_pos_5": close_pos_5,
        "ema9_slope_3_pct": ema9_slope_3_pct,
        "side_buy": side_buy,
        "rsi_dist_50": rsi_dist_50,
        "adx_strong": adx_strong,
        "supertrend_aligned_side": supertrend_aligned_side,
        "atr_regime_high": atr_regime_high,
    }


def _load_state_from_path(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict) and "model" in state:
            return state
    except Exception:
        return None
    return None


def _load_state(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    normalized = _normalize_symbol(symbol)
    state = _load_state_from_path(_model_path(normalized))
    if state is not None:
        return state
    if normalized == "BTCUSDT":
        return _load_state_from_path(LEGACY_MODEL_PATH)
    return None


def _positive_class_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", []))
    if not classes:
        return 1
    for candidate in (1, 1.0, True, "1", "True", "true", "WIN", "win"):
        if candidate in classes:
            return classes.index(candidate)
    return len(classes) - 1


def _positive_proba(model: Any, rows: Sequence[Sequence[float]]) -> List[float]:
    raw = model.predict_proba(rows)
    index = _positive_class_index(model)
    probabilities: List[float] = []
    for row in raw:
        try:
            probabilities.append(_safe_float(row[index], 0.5))
        except Exception:
            probabilities.append(0.5)
    return probabilities


def _balanced_accuracy(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> float:
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


def _accuracy(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> float:
    if not y_true:
        return 0.0
    preds = [1 if prob >= threshold else 0 for prob in y_prob]
    return sum(1 for yt, yp in zip(y_true, preds) if yt == yp) / len(y_true)


def _feature_vector(features: Dict[str, Any]) -> List[float]:
    normalized = dict(features or {})
    if "side_buy" not in normalized:
        normalized["side_buy"] = _coerce_side_buy(normalized.get("side"), 1.0)
    row: List[float] = []
    for key in FEATURE_ORDER:
        default = 1.0 if key == "side_buy" else 0.0
        if key == "volume_vs_ma20_ratio" and key not in normalized:
            value = normalized.get("volume_ratio", default)
        elif key == "supertrend_direction" and key not in normalized:
            value = normalized.get("supertrend_dir", default)
        elif key == "side_buy" and key not in normalized:
            value = normalized.get("side", default)
        else:
            value = normalized.get(key, default)
        if key == "side_buy":
            row.append(_coerce_side_buy(value, 1.0))
        else:
            row.append(_safe_float(value, default))
    return row


def _is_reconciled_trade(trade: Dict[str, Any]) -> bool:
    return str(trade.get("result") or "").upper() == "RECONCILED"


def _closed_trade_candidates(trades: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [trade for trade in trades if isinstance(trade, dict)]
    rows.sort(key=lambda trade: (str(trade.get("ts") or ""), str(trade.get("id") or "")))
    return rows


def _trade_side_buy(trade: Dict[str, Any]) -> float:
    return _coerce_side_buy(trade.get("side"), 1.0)


def _trade_features(trade: Dict[str, Any]) -> Dict[str, float]:
    return features_from_trade_row(trade)


def features_from_trade_row(trade: Dict[str, Any]) -> Dict[str, float]:
    ema9 = _safe_float(trade.get("ema9"), 0.0)
    ema21 = _safe_float(trade.get("ema21"), 0.0)
    ema200 = _safe_float(trade.get("ema200"), 0.0)
    rsi = _safe_float(trade.get("rsi"), 0.0)
    candle_body_pct = _safe_float(trade.get("candle_body_pct"), 0.0)
    adx = _safe_float(trade.get("adx"), 0.0)
    supertrend_direction = _safe_float(trade.get("supertrend_dir", trade.get("supertrend_direction")), 0.0)
    volume_vs_ma20_ratio = _safe_float(trade.get("volume_ratio", trade.get("volume_vs_ma20_ratio")), 0.0)
    atr_pct = _safe_float(trade.get("atr_pct"), 0.0)
    return_1_pct = _first_present_float(trade, ["return_1_pct", "return_1"], 0.0)
    return_2_pct = _first_present_float(trade, ["return_2_pct", "return_2"], 0.0)
    return_3_pct = _first_present_float(trade, ["return_3_pct", "return_3"], 0.0)
    return_4_pct = _first_present_float(trade, ["return_4_pct", "return_4"], 0.0)
    return_5_pct = _first_present_float(trade, ["return_5_pct", "return_5"], 0.0)
    range_1_pct = _first_present_float(trade, ["range_1_pct", "range_1"], 0.0)
    range_2_pct = _first_present_float(trade, ["range_2_pct", "range_2"], 0.0)
    range_3_pct = _first_present_float(trade, ["range_3_pct", "range_3"], 0.0)
    range_5_pct = _first_present_float(trade, ["range_5_pct"], 0.0)
    body_1_pct = _first_present_float(trade, ["body_1_pct", "body_1"], 0.0)
    body_2_pct = _first_present_float(trade, ["body_2_pct", "body_2"], 0.0)
    body_3_pct = _first_present_float(trade, ["body_3_pct", "body_3"], 0.0)
    body_avg_3_pct = _first_present_float(trade, ["body_avg_3_pct"], 0.0)
    volume_trend_1 = _first_present_float(trade, ["volume_trend_1"], 0.0)
    volume_trend_2 = _first_present_float(trade, ["volume_trend_2"], 0.0)
    volume_trend_3 = _first_present_float(trade, ["volume_trend_3"], 0.0)
    volume_trend_3_ratio = _first_present_float(trade, ["volume_trend_3_ratio"], 0.0)
    close_pos_5 = _safe_float(trade.get("close_pos_5"), 0.0)
    ema9_slope_3_pct = _safe_float(trade.get("ema9_slope_3_pct"), 0.0)
    entry_price = _safe_float(trade.get("entry_price"), 0.0)
    side_buy = trade.get("side_buy")
    if side_buy is None:
        side_buy = trade.get("side")
    if not body_avg_3_pct and (body_1_pct or body_2_pct or body_3_pct):
        body_avg_3_pct = (body_1_pct + body_2_pct + body_3_pct) / 3.0
    if not volume_trend_3_ratio and (volume_trend_1 or volume_trend_2 or volume_trend_3):
        volume_trend_3_ratio = (volume_trend_1 + volume_trend_2 + volume_trend_3) / 3.0
    if not range_5_pct and (range_1_pct or range_2_pct or range_3_pct):
        range_5_pct = max(range_1_pct, range_2_pct, range_3_pct)
    if not return_5_pct and return_4_pct:
        return_5_pct = return_4_pct
    if not return_3_pct and (return_1_pct or return_2_pct or return_4_pct or return_5_pct):
        return_3_pct = (return_1_pct + return_2_pct + return_4_pct + return_5_pct) / 4.0
    return _build_feature_map(
        ema9=ema9,
        ema21=ema21,
        ema200=ema200,
        rsi=rsi,
        candle_body_pct=candle_body_pct,
        adx=adx,
        supertrend_direction=supertrend_direction,
        volume_vs_ma20_ratio=volume_vs_ma20_ratio,
        atr_pct=atr_pct,
        return_3_pct=return_3_pct,
        return_5_pct=return_5_pct,
        range_5_pct=range_5_pct,
        body_avg_3_pct=body_avg_3_pct,
        volume_trend_3_ratio=volume_trend_3_ratio,
        close_pos_5=close_pos_5,
        ema9_slope_3_pct=ema9_slope_3_pct,
        entry_price=entry_price,
        side_buy=_coerce_side_buy(side_buy, 1.0),
    )


def snapshot_features(
    last: Dict[str, Any],
    history: Optional[Iterable[Dict[str, Any]]] = None,
    cols: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    cols = cols or {}
    rows = _rows_from_history(history)
    if not rows and isinstance(last, dict):
        rows = [last]

    close = _row_value(last, [cols.get("close", "close"), "close"], 0.0)
    open_ = _row_value(last, [cols.get("open", "open"), "open"], 0.0)
    ema9 = _row_value(last, [cols.get("ema_fast", "ema9"), "ema9"], 0.0)
    ema21 = _row_value(last, [cols.get("ema_slow", "ema21"), "ema21"], 0.0)
    ema200 = _row_value(last, [cols.get("ema_trend", "ema200"), "ema200"], 0.0)
    volume = _row_value(last, [cols.get("volume", "volume"), "volume"], 0.0)
    vol_ma20 = _row_value(last, [cols.get("vol_ma20", "vol_ma20"), "vol_ma20"], 0.0)
    atr = _row_value(last, [cols.get("atr", "atr"), "atr"], 0.0)
    rsi = _row_value(last, [cols.get("rsi", "rsi"), "rsi"], 0.0)
    adx = _row_value(last, [cols.get("adx", "adx"), "adx"], 0.0)
    supertrend_direction = _row_value(
        last,
        [cols.get("supertrend_dir", "supertrend_direction"), "supertrend_direction"],
        0.0,
    )
    side_buy = _coerce_side_buy(
        last.get(cols.get("side")) if cols.get("side") and cols.get("side") in last else last.get("side"),
        1.0,
    )

    close_3 = _close_from_offset(rows, 3)
    close_5 = _close_from_offset(rows, 5)
    ema9_3 = _safe_float(rows[-4].get(cols.get("ema_fast", "ema9")) if len(rows) >= 4 else ema9, ema9)
    recent_3 = rows[-3:] if len(rows) >= 3 else rows[:]
    recent_5 = rows[-5:] if len(rows) >= 5 else rows[:]
    prior_3 = rows[-6:-3] if len(rows) >= 6 else []

    highest_5 = max((_safe_float(row.get("high"), 0.0) for row in recent_5), default=0.0)
    lowest_5 = min((_safe_float(row.get("low"), 0.0) for row in recent_5), default=0.0)
    body_terms = []
    for row in recent_3:
        row_open = _safe_float(row.get("open"), 0.0)
        row_close = _safe_float(row.get("close"), 0.0)
        if row_open:
            body_terms.append(abs(row_close - row_open) / row_open * 100.0)
    avg_body_3 = sum(body_terms) / len(body_terms) if body_terms else 0.0
    avg_vol_3 = _average_field(recent_3, "volume")
    prev_avg_vol_3 = _average_field(prior_3, "volume")
    vol_trend_3 = (avg_vol_3 / prev_avg_vol_3) if prev_avg_vol_3 else 0.0
    close_pos_5 = ((close - lowest_5) / (highest_5 - lowest_5)) if highest_5 > lowest_5 else 0.0
    ema9_slope_3 = ((ema9 - ema9_3) / ema9_3 * 100.0) if ema9_3 else 0.0

    return _build_feature_map(
        ema9=ema9,
        ema21=ema21,
        ema200=ema200,
        rsi=rsi,
        candle_body_pct=abs(close - open_) / open_ * 100.0 if open_ else 0.0,
        adx=adx,
        supertrend_direction=supertrend_direction,
        volume_vs_ma20_ratio=(volume / vol_ma20) if vol_ma20 else 0.0,
        atr_pct=(atr / close * 100.0) if close else 0.0,
        return_3_pct=((close - close_3) / close_3 * 100.0) if close_3 else 0.0,
        return_5_pct=((close - close_5) / close_5 * 100.0) if close_5 else 0.0,
        range_5_pct=((highest_5 - lowest_5) / close * 100.0) if close and highest_5 > lowest_5 else 0.0,
        body_avg_3_pct=avg_body_3,
        volume_trend_3_ratio=vol_trend_3,
        close_pos_5=close_pos_5,
        ema9_slope_3_pct=ema9_slope_3,
        entry_price=close,
        side_buy=side_buy,
    )


def label_closed_trades(trades):
    rows = _closed_trade_candidates(trades or [])
    robust: List[Tuple[Dict[str, Any], int]] = []
    fallback: List[Tuple[Dict[str, Any], int]] = []

    for trade in rows:
        if _is_reconciled_trade(trade):
            continue
        r_multiple = trade.get("r_multiple")
        if r_multiple is not None:
            r_value = _safe_float(r_multiple, 0.0)
            if r_value >= 0.2:
                robust.append((trade, 1))
            elif r_value <= -0.2:
                robust.append((trade, 0))
        pnl = trade.get("pnl")
        if pnl is None:
            continue
        pnl_value = _safe_float(pnl, 0.0)
        if pnl_value > 0:
            fallback.append((trade, 1))
        elif pnl_value < 0:
            fallback.append((trade, 0))

    if len(robust) >= MIN_TRAIN_TRADES or not fallback:
        return robust
    return fallback


def label_triple_barrier(trades, profit_target: float = 0.02, stop_loss: float = 0.01):
    labeled = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
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


def _state_is_ready(state: Optional[Dict[str, Any]]) -> bool:
    if not state:
        return False
    if int(state.get("trained_on", 0) or 0) < MIN_TRAIN_TRADES:
        return False
    if list(state.get("feature_order") or []) != FEATURE_ORDER:
        return False
    try:
        return float(state.get("validation_accuracy", 0.0) or 0.0) >= MIN_VALIDATION_ACCURACY
    except (TypeError, ValueError):
        return False


def _best_iteration(model: Any) -> int:
    for attr in ("best_iteration", "best_iteration_"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    value = getattr(model, "n_estimators", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _fit_xgb(train_rows, train_labels, val_rows, val_labels, scale_pos_weight):
    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "gamma": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": 1,
    }
    eval_set = [(val_rows, val_labels)]
    try:
        model = XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(train_rows, train_labels, eval_set=eval_set, verbose=False)
        return model
    except TypeError:
        pass
    except Exception:
        pass
    try:
        model = XGBClassifier(**params)
        try:
            model.fit(train_rows, train_labels, eval_set=eval_set, verbose=False, early_stopping_rounds=30)
        except TypeError:
            model.fit(train_rows, train_labels, eval_set=eval_set, verbose=False)
        return model
    except Exception:
        model = XGBClassifier(**params)
        model.fit(train_rows, train_labels, eval_set=eval_set, verbose=False)
        return model


def _fit_simple_model(train_rows, train_labels):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(train_rows, train_labels)
    return model


def _calibrate_model(val_probs: Sequence[float], val_labels: Sequence[int]):
    if IsotonicRegression is None or len(val_labels) < 30:
        return None
    try:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(list(val_probs), list(val_labels))
        return calibrator
    except Exception:
        return None


def _threshold_search(
    val_probs: Sequence[float],
    val_labels: Sequence[int],
    val_r_multiples: Sequence[Optional[float]],
) -> float:
    thresholds = [round(0.45 + 0.01 * step, 2) for step in range(31)]
    usable_r = [value for value in val_r_multiples if value is not None]
    best_threshold = 0.55
    best_score = None
    if usable_r:
        paired = [(float(prob), float(r)) for prob, r in zip(val_probs, val_r_multiples) if r is not None]
        if not paired:
            return best_threshold
        for threshold in thresholds:
            score = sum(r_multiple for probability, r_multiple in paired if probability >= threshold)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold
    for threshold in thresholds:
        score = _accuracy(val_labels, val_probs, threshold)
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _save_state(symbol: str, state: Dict[str, Any]) -> None:
    path = _model_path(symbol)
    with path.open("wb") as f:
        pickle.dump(state, f)
    if symbol == "BTCUSDT":
        with LEGACY_MODEL_PATH.open("wb") as f:
            pickle.dump(state, f)


def _debug_training_snapshot(feature_rows: Sequence[Sequence[float]], labels: Sequence[int], trades: Sequence[Dict[str, Any]]) -> None:
    if os.getenv("ML_DEBUG", "0") not in {"1", "true", "TRUE", "yes", "YES"}:
        return
    print(f"[ml_model.debug] first3_X={list(feature_rows[:3])}")
    print(f"[ml_model.debug] first3_y={list(labels[:3])}")
    if trades:
        sample = trades[0]
        keys = [
            "ema9",
            "ema21",
            "ema200",
            "rsi",
            "candle_body_pct",
            "adx",
            "supertrend_dir",
            "volume_ratio",
            "atr_pct",
            "return_1_pct",
            "return_2_pct",
            "return_3_pct",
            "return_4_pct",
            "return_5_pct",
            "range_1_pct",
            "range_2_pct",
            "range_3_pct",
            "range_5_pct",
            "body_1_pct",
            "body_2_pct",
            "body_3_pct",
            "body_avg_3_pct",
            "volume_trend_1",
            "volume_trend_2",
            "volume_trend_3",
            "volume_trend_3_ratio",
            "close_pos_5",
            "ema9_slope_3_pct",
            "entry_price",
            "side",
            "side_buy",
        ]
        presence = {key: key in sample for key in keys}
        print(f"[ml_model.debug] key_presence={presence}")


def train(trades: list, symbol: Optional[str] = "BTCUSDT") -> None:
    symbol = _normalize_symbol(symbol)
    rows = [trade for trade in _closed_trade_candidates(trades or []) if _normalize_symbol(trade.get("symbol", symbol)) == symbol]
    labeled = label_closed_trades(rows)
    if len(labeled) < MIN_TRAIN_TRADES:
        return

    feature_rows = [_feature_vector(_trade_features(trade)) for trade, _ in labeled]
    labels = [int(label) for _, label in labeled]
    total_labeled = len(feature_rows)
    if total_labeled < MIN_TRAIN_TRADES or len(set(labels)) < 2:
        return

    validation_count = max(1, int(total_labeled * 0.2))
    split_idx = total_labeled - validation_count
    train_cutoff = split_idx - 1
    if train_cutoff < 2 or split_idx < 2:
        return

    train_rows = feature_rows[:train_cutoff]
    train_labels = labels[:train_cutoff]
    val_rows = feature_rows[split_idx:]
    val_labels = labels[split_idx:]
    val_trades = labeled[split_idx:]

    if len(train_rows) < 2 or len(val_rows) < 2:
        return
    if len(set(train_labels)) < 2 or len(set(val_labels)) < 2:
        return

    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    if positives <= 0 or negatives <= 0:
        return
    scale_pos_weight = negatives / positives
    _debug_training_snapshot(train_rows, train_labels, rows)

    with _TRAIN_LOCK:
        try:
            if USING_XGBOOST:
                model = _fit_xgb(train_rows, train_labels, val_rows, val_labels, scale_pos_weight)
            else:
                model = _fit_simple_model(train_rows, train_labels)
        except Exception:
            model = _fit_simple_model(train_rows, train_labels)

        try:
            raw_val_probs = _positive_proba(model, val_rows)
        except Exception:
            raw_val_probs = [0.5 for _ in val_rows]

        calibrator = _calibrate_model(raw_val_probs, val_labels)
        calibrated_val_probs = raw_val_probs
        if calibrator is not None:
            try:
                calibrated_val_probs = [
                    _safe_float(value, 0.5)
                    for value in calibrator.predict(raw_val_probs)
                ]
            except Exception:
                calibrator = None
                calibrated_val_probs = raw_val_probs

        # Keep validation scoring on raw model output so calibration cannot flatten a
        # separable validation set into a false 0.50.
        suggested_threshold = _threshold_search(raw_val_probs, val_labels, [trade.get("r_multiple") for trade, _ in val_trades])
        validation_accuracy = _balanced_accuracy(val_labels, raw_val_probs, suggested_threshold)
        best_iter = _best_iteration(model)

        state = {
            "model": model,
            "calibrator": calibrator,
            "trained_on": int(total_labeled),
            "train_samples": int(len(train_rows)),
            "validation_samples": int(len(val_rows)),
            "source_trades": int(len(rows)),
            "feature_order": list(FEATURE_ORDER),
            "model_type": "xgb" if USING_XGBOOST else "rf",
            "symbol": symbol,
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "validation_accuracy": float(validation_accuracy),
            "optimal_threshold": float(suggested_threshold),
            "best_iteration": int(best_iter),
            "calibration_samples": int(len(val_rows) if calibrator is not None else 0),
            "label_balance": {
                "positive": int(sum(labels)),
                "negative": int(len(labels) - sum(labels)),
            },
        }
        _save_state(symbol, state)

        print(
            f"[ml_model] symbol={symbol} trained={total_labeled} val={len(val_rows)} "
            f"val_acc={validation_accuracy:.2f} thr={suggested_threshold:.2f} "
            f"best_iter={best_iter} features={len(FEATURE_ORDER)}"
        )


def predict(features: dict, symbol: Optional[str] = "BTCUSDT") -> float:
    state = _load_state(symbol)
    if not _state_is_ready(state):
        return 0.5

    model = state.get("model") if state else None
    if model is None:
        return 0.5

    try:
        row = _feature_vector(features or {})
        probability = _positive_proba(model, [row])[0]
        calibrator = state.get("calibrator")
        if calibrator is not None:
            try:
                probability = _safe_float(calibrator.predict([probability])[0], probability)
            except Exception:
                pass
        return max(0.0, min(1.0, float(probability)))
    except Exception:
        return 0.5


def is_ready(symbol: Optional[str] = "BTCUSDT") -> bool:
    state = _load_state(symbol)
    return _state_is_ready(state)


def model_info(symbol: Optional[str] = "BTCUSDT") -> Dict[str, Any]:
    state = _load_state(symbol) or {}
    best_iteration = state.get("best_iteration", -1)
    try:
        best_iteration = int(best_iteration) if best_iteration is not None else -1
    except (TypeError, ValueError):
        best_iteration = -1
    return {
        "symbol": _normalize_symbol(symbol),
        "trained_on": int(state.get("trained_on", 0) or 0),
        "train_samples": int(state.get("train_samples", 0) or 0),
        "validation_samples": int(state.get("validation_samples", 0) or 0),
        "validation_trades": int(state.get("validation_samples", 0) or 0),
        "validation_accuracy": float(state.get("validation_accuracy", 0.0) or 0.0),
        "suggested_threshold": float(state.get("optimal_threshold", 0.55) or 0.55),
        "optimal_threshold": float(state.get("optimal_threshold", 0.55) or 0.55),
        "validation_threshold": MIN_VALIDATION_ACCURACY,
        "best_iteration": best_iteration,
        "calibrated": bool(state.get("calibrator") is not None),
        "calibration_samples": int(state.get("calibration_samples", 0) or 0),
        "feature_order": list(state.get("feature_order") or FEATURE_ORDER),
        "model_type": state.get("model_type"),
        "source_trades": int(state.get("source_trades", 0) or 0),
        "label_balance": state.get("label_balance", {"positive": 0, "negative": 0}),
        "last_trained_at": state.get("last_trained_at"),
    }
