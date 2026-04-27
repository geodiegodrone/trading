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
    ema9 = float(trade.get("ema9") or 0.0)
    ema21 = float(trade.get("ema21") or 0.0)
    rsi = float(trade.get("rsi") or 0.0)
    ema200 = float(trade.get("ema200") or 0.0)
    entry_price = float(trade.get("entry_price") or 0.0)
    candle_body_pct = float(trade.get("candle_body_pct") or 0.0)
    adx = float(trade.get("adx") or 0.0)
    supertrend_direction = float(trade.get("supertrend_dir") or 0.0)
    volume_ratio = float(trade.get("volume_ratio") or 0.0)
    atr_pct = float(trade.get("atr_pct") or 0.0)
    return {
        "ema9_minus_ema21_pct": ((ema9 - ema21) / ema21 * 100.0) if ema21 else 0.0,
        "rsi": rsi,
        "price_vs_ema200_pct": ((entry_price - ema200) / ema200 * 100.0) if ema200 else 0.0,
        "candle_body_pct": candle_body_pct,
        "adx": adx,
        "supertrend_direction": supertrend_direction,
        "volume_vs_ma20_ratio": volume_ratio,
        "atr_pct": atr_pct,
    }


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
            )
        else:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                random_state=42,
            )
        model.fit(rows, labels)

        state = {
            "model": model,
            "trained_on": int(len(rows)),
            "source_trades": int(len(filtered)),
            "feature_order": FEATURE_ORDER,
            "model_type": "xgb" if USING_XGBOOST else "rf",
            "symbol": symbol,
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
        }
        path = _model_path(symbol)
        with path.open("wb") as f:
            pickle.dump(state, f)
        if symbol == "BTCUSDT":
            with LEGACY_MODEL_PATH.open("wb") as f:
                pickle.dump(state, f)


def predict(features: dict, symbol: Optional[str] = "BTCUSDT") -> float:
    state = _load_state(symbol)
    if not state or int(state.get("trained_on", 0)) < MIN_TRAIN_TRADES:
        return 0.5
    if list(state.get("feature_order") or []) != FEATURE_ORDER:
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
    return bool(state and int(state.get("trained_on", 0)) >= MIN_TRAIN_TRADES and list(state.get("feature_order") or []) == FEATURE_ORDER)
