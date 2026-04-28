from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import backtest_walkforward as bwf
import bot_config
import ml_model
import multi_bot
import pandas as pd


def _utc_iso_from_ms(value: Any) -> str:
    try:
        ts = float(value) / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:
            return default
        return result
    except (TypeError, ValueError):
        return default


def _infer_signal_idx(df, trade: Dict[str, Any]) -> Optional[int]:
    signal_idx = trade.get("signal_idx")
    if signal_idx is not None:
        try:
            return int(signal_idx)
        except (TypeError, ValueError):
            pass
    entry_ts = trade.get("entry_ts")
    if entry_ts is None or df.empty:
        return None
    try:
        entry_ts = int(entry_ts)
    except (TypeError, ValueError):
        return None
    ts_values = list(pd.to_numeric(df["ts"], errors="coerce").fillna(-1).astype(int))
    try:
        entry_idx = ts_values.index(entry_ts)
    except ValueError:
        return None
    if entry_idx <= 0:
        return None
    return entry_idx - 1


def _build_trade_record(symbol: str, df, trade: Dict[str, Any], signal_idx: int, trade_usdt: float) -> Optional[Dict[str, Any]]:
    if signal_idx < 0 or signal_idx + 1 >= len(df):
        return None

    signal_row = df.iloc[signal_idx].to_dict()
    entry_row = df.iloc[signal_idx + 1].to_dict()
    history_window = df.iloc[max(0, signal_idx - 9) : signal_idx + 1].to_dict("records")
    cols = df.attrs.get("indicator_cols", {}) or {}
    if not isinstance(cols, dict):
        cols = {}

    features = ml_model.snapshot_features(entry_row, history_window, cols)
    side = "Buy" if str(trade.get("side") or "").upper() == "LONG" else "Sell"
    entry_price = _safe_float(trade.get("entry_price"))
    exit_price = _safe_float(trade.get("exit_price"))
    pnl = _safe_float(trade.get("pnl"))
    leverage = _safe_float((trade.get("leverage") or multi_bot._get_symbol_config(symbol).get("leverage", 1)), 1.0)
    atr = _safe_float(signal_row.get("atr"))
    atr_mult = _safe_float(multi_bot._get_symbol_config(symbol).get("atr_mult", 1.5), 1.5)
    qty = (trade_usdt * leverage / entry_price) if entry_price > 0 else 0.0
    notional_usdt = qty * entry_price
    risk_usdt = qty * atr * atr_mult if atr > 0 else 0.0
    r_multiple = (pnl / risk_usdt) if risk_usdt else None
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

    return {
        "id": uuid.uuid4().hex,
        "signal_idx": signal_idx,
        "ts": _utc_iso_from_ms(trade.get("entry_ts", entry_row.get("ts"))),
        "side": side,
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "ema9": _safe_float(signal_row.get(cols.get("ema_fast", "ema9")), _safe_float(signal_row.get("ema9"))),
        "ema21": _safe_float(signal_row.get(cols.get("ema_slow", "ema21")), _safe_float(signal_row.get("ema21"))),
        "ema200": _safe_float(signal_row.get(cols.get("ema_trend", "ema200")), _safe_float(signal_row.get("ema200"))),
        "rsi": _safe_float(signal_row.get("rsi")),
        "adx": _safe_float(signal_row.get("adx")),
        "supertrend_dir": _safe_float(signal_row.get(cols.get("supertrend_dir", "supertrend_direction")), _safe_float(signal_row.get("supertrend_direction"))),
        "candle_body_pct": _safe_float(features.get("candle_body_pct")),
        "volume_ratio": _safe_float(features.get("volume_vs_ma20_ratio")),
        "volume_vs_ma20_ratio": _safe_float(features.get("volume_vs_ma20_ratio")),
        "atr_pct": _safe_float(features.get("atr_pct")),
        "return_3_pct": _safe_float(features.get("return_3_pct")),
        "return_5_pct": _safe_float(features.get("return_5_pct")),
        "range_5_pct": _safe_float(features.get("range_5_pct")),
        "body_avg_3_pct": _safe_float(features.get("body_avg_3_pct")),
        "volume_trend_3_ratio": _safe_float(features.get("volume_trend_3_ratio")),
        "close_pos_5": _safe_float(features.get("close_pos_5")),
        "ema9_slope_3_pct": _safe_float(features.get("ema9_slope_3_pct")),
        "qty": qty,
        "notional_usdt": notional_usdt,
        "risk_usdt": risk_usdt,
        "r_multiple": r_multiple,
        "result": result,
        "exit_reason": str(trade.get("reason") or trade.get("exit_reason") or ""),
    }


def _simulate_symbol(symbol: str, days: int, trade_usdt: float) -> Optional[Dict[str, Any]]:
    try:
        cfg = multi_bot._get_symbol_config(symbol)
        timeframe = int(cfg.get("timeframe", 5))
    except Exception as exc:
        print(f"[bootstrap_ml] {symbol}: config error: {exc}", file=sys.stderr)
        return None

    try:
        df = bwf.fetch_history(symbol, timeframe, days)
    except Exception as exc:
        print(f"[bootstrap_ml] {symbol}: fetch_history failed: {exc}", file=sys.stderr)
        return None

    if df is None or getattr(df, "empty", True):
        print(f"[bootstrap_ml] {symbol}: skip empty history", file=sys.stderr)
        return None

    try:
        simulated = bwf.simulate(df, cfg, trade_usdt=trade_usdt)
    except Exception as exc:
        print(f"[bootstrap_ml] {symbol}: simulate failed: {exc}", file=sys.stderr)
        return None

    synthetic_trades: List[Dict[str, Any]] = []
    raw_trades = simulated.get("trades") if isinstance(simulated, dict) else []
    for trade in raw_trades or []:
        if not isinstance(trade, dict):
            continue
        signal_idx = _infer_signal_idx(df, trade)
        if signal_idx is None:
            continue
        try:
            trade_record = _build_trade_record(symbol, df, trade, signal_idx, trade_usdt)
        except Exception as exc:
            print(f"[bootstrap_ml] {symbol}: skip trade at signal_idx={signal_idx}: {exc}", file=sys.stderr)
            continue
        if trade_record is not None:
            synthetic_trades.append(trade_record)

    if not synthetic_trades:
        print(f"[bootstrap_ml] {symbol}: skip no synthetic trades", file=sys.stderr)
        return None

    ml_model.train(synthetic_trades, symbol)
    info = ml_model.model_info(symbol)
    wins = sum(1 for trade in synthetic_trades if _safe_float(trade.get("pnl")) > 0)
    losses = sum(1 for trade in synthetic_trades if _safe_float(trade.get("pnl")) < 0)
    return {
        "symbol": symbol,
        "trades_simulated": len(synthetic_trades),
        "wins": wins,
        "losses": losses,
        "val_acc": _safe_float(info.get("validation_accuracy")),
        "suggested_thr": _safe_float(info.get("suggested_threshold", info.get("optimal_threshold"))),
        "ready": bool(ml_model.is_ready(symbol)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap ML models from simulated backtests")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--trade-usdt", type=float, default=50.0)
    args = parser.parse_args()

    symbols = bot_config.parse_symbols()
    results: List[Dict[str, Any]] = []

    for symbol in symbols:
        try:
            result = _simulate_symbol(symbol, args.days, args.trade_usdt)
            if result is None:
                continue
            results.append(result)
            print(
                f"{result['symbol']}|{result['trades_simulated']}|{result['wins']}|{result['losses']}|"
                f"{result['val_acc']:.2f}|{result['suggested_thr']:.2f}|{result['ready']}"
            )
        except Exception as exc:
            print(f"[bootstrap_ml] {symbol}: unexpected error: {exc}", file=sys.stderr)
            continue

    if not results:
        print("[bootstrap_ml] no symbols trained", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
