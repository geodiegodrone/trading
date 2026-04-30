from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import backtest_walkforward as bwf
import ml_model
import multi_bot


def _primary_signals(primary_df: pd.DataFrame, confirmation_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    warmup = max(220, int(cfg.get("ema_trend", 200)) + 5)
    mode = str(cfg.get("strategy_mode", "regime")).lower()
    for idx in range(warmup, len(primary_df)):
        ts_value = int(primary_df.iloc[idx]["ts"])
        confirm_row = bwf._confirmation_row(confirmation_df, ts_value)
        signal, regime_name = bwf._mode_signal(mode, primary_df.iloc[: idx + 1], confirm_row, cfg)
        if signal == "NEUTRAL":
            continue
        rows.append({"signal_idx": idx, "side": signal, "regime": regime_name})
    return pd.DataFrame(rows)

def _cusum_threshold(primary_df: pd.DataFrame) -> float:
    atr_pct = ((pd.to_numeric(primary_df.get("atr"), errors="coerce") / pd.to_numeric(primary_df["close"], errors="coerce").replace(0.0, np.nan)) * 100.0).dropna()
    mean_atr_pct = float(atr_pct.tail(120).mean()) if not atr_pct.empty else 0.0
    if mean_atr_pct > 0:
        return max(mean_atr_pct / 100.0 * 2.0, 0.001)
    log_returns = np.log(pd.to_numeric(primary_df["close"], errors="coerce").replace(0.0, np.nan)).diff().fillna(0.0)
    return max(float(log_returns.tail(60).std(ddof=0) * 2.0), 0.001)


def _filtered_signals(primary_df: pd.DataFrame, signals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    log_returns = np.log(pd.to_numeric(primary_df["close"], errors="coerce").replace(0.0, np.nan)).diff().fillna(0.0)
    cusum_events = set(ml_model.cusum_filter(log_returns.tolist(), threshold))
    filtered = signals[signals["signal_idx"].isin(cusum_events)].reset_index(drop=True)
    return filtered if not filtered.empty else signals.reset_index(drop=True)


def bootstrap(
    days: int = 180,
    symbol: str = "BTCUSDT",
    timeframe: int | None = None,
    quiet: bool = False,
    trade_usdt: float | None = None,
    min_samples: int = 120,
) -> Dict[str, Any]:
    cfg = multi_bot._get_symbol_config(symbol)
    cfg["primary_timeframe"] = int(timeframe or cfg.get("primary_timeframe", 60))
    days_used = int(days)
    attempts: List[Dict[str, Any]] = []
    primary_df = pd.DataFrame()
    labels = pd.DataFrame()
    signals = pd.DataFrame()
    target_samples = max(int(min_samples), int(ml_model.MIN_VALIDATION_TRADES) * 5)
    while days_used <= 730:
        primary_df = bwf.fetch_history(symbol, int(cfg["primary_timeframe"]), days_used)
        confirmation_df = bwf.fetch_history(symbol, int(cfg.get("confirmation_timeframe", 240)), days_used)
        primary_df = multi_bot._compute_indicators(primary_df, cfg)
        confirmation_df = multi_bot._compute_indicators(confirmation_df, cfg)
        if primary_df.empty or confirmation_df.empty:
            raise RuntimeError("no history")
        signals = _primary_signals(primary_df, confirmation_df, cfg)
        if signals.empty and str(cfg.get("strategy_mode", "regime")).lower() == "regime":
            fallback_cfg = dict(cfg)
            fallback_cfg["strategy_mode"] = "trend"
            signals = _primary_signals(primary_df, confirmation_df, fallback_cfg)
        if signals.empty:
            raise RuntimeError("no primary signals")
        h = _cusum_threshold(primary_df)
        filtered_signals = _filtered_signals(primary_df, signals, h)
        labels = ml_model.apply_triple_barrier(
            primary_df,
            filtered_signals,
            atr_mult=float(cfg.get("atr_mult", 1.5)),
            tp_ratio=2.0,
            t_bars=24,
        )
        attempts.append({"days": days_used, "threshold": h, "events": int(len(filtered_signals)), "labels": int(len(labels)), "threshold_retry": False})
        if len(labels) >= target_samples:
            signals = filtered_signals
            break
        if len(filtered_signals) < 80:
            half_h = max(h / 2.0, 0.0005)
            filtered_half = _filtered_signals(primary_df, signals, half_h)
            labels_half = ml_model.apply_triple_barrier(
                primary_df,
                filtered_half,
                atr_mult=float(cfg.get("atr_mult", 1.5)),
                tp_ratio=2.0,
                t_bars=24,
            )
            attempts.append({"days": days_used, "threshold": half_h, "events": int(len(filtered_half)), "labels": int(len(labels_half)), "threshold_retry": True})
            if len(labels_half) > len(labels):
                signals = filtered_half
                labels = labels_half
                h = half_h
            else:
                signals = filtered_signals
            if len(labels) >= target_samples:
                break
        else:
            signals = filtered_signals
        if days_used >= 730:
            break
        days_used = min(days_used * 2, 730)
    if labels.empty:
        raise RuntimeError("no triple-barrier labels")

    ml_model.train(labels, symbol=symbol, df_klines=primary_df)
    info = ml_model.model_info(symbol)
    positives = int((labels["label"] == 1).sum())
    negatives = int((labels["label"] == 0).sum())
    result = {
        "symbol": symbol,
        "events": int(len(signals)),
        "labels": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "ratio": (positives / max(1, positives + negatives)),
        "val_sharpe": float(info.get("val_sharpe", 0.0)),
        "val_auc": float(info.get("val_auc", 0.0)),
        "val_f1": float(info.get("val_f1", 0.0)),
        "coverage_pct": float(info.get("coverage_pct", 0.0)),
        "suggested_threshold": float(info.get("suggested_threshold", 0.55)),
        "ready": bool(ml_model.is_ready(symbol)),
        "not_ready_reason": str(info.get("not_ready_reason") or ""),
        "no_profitable_threshold": bool(info.get("no_profitable_threshold", False)),
        "fold_metrics": list(info.get("fold_metrics", [])),
        "min_trades_per_year_estimate": float(info.get("min_trades_per_year_estimate", 0.0)),
        "days_used": int(days_used),
        "attempts": attempts,
    }
    if not quiet:
        for attempt in attempts:
            retry_label = " h/2" if attempt.get("threshold_retry") else ""
            print(
                f"bootstrap_attempt days={int(attempt['days'])} threshold={float(attempt['threshold']):.5f}{retry_label} "
                f"events={int(attempt['events'])} labels={int(attempt['labels'])}"
            )
        print(f"events={result['events']} positives={positives} negatives={negatives} ratio={result['ratio']:.3f}")
        print("fold | auc | f1 | sharpe | trades | coverage% | positives | degenerate")
        for fold in result["fold_metrics"]:
            print(
                f"{int(fold.get('fold', 0)):>4} | "
                f"{float(fold.get('auc', 0.0)):>5.3f} | "
                f"{float(fold.get('f1', 0.0)):>5.3f} | "
                f"{float(fold.get('sharpe', 0.0)):>6.3f} | "
                f"{int(fold.get('trades', 0)):>6} | "
                f"{float(fold.get('coverage_pct', 0.0)):>8.2f} | "
                f"{int(fold.get('positives', 0)):>9} | "
                f"{str(bool(fold.get('degenerate', False))).lower()}"
            )
        print(
            f"summary: auc(mean)={result['val_auc']:.3f} "
            f"f1(median)={result['val_f1']:.3f} "
            f"sharpe(median)={result['val_sharpe']:.3f} "
            f"coverage%={result['coverage_pct']:.2f}"
        )
        print(
            f"suggested_threshold={result['suggested_threshold']:.3f} "
            f"ready={result['ready']} "
            f"no_profitable_threshold={result['no_profitable_threshold']} "
            f"not_ready_reason={(result['not_ready_reason'] or '-').replace('τ', 'tau')} "
            f"min_trades_per_year_estimate={result['min_trades_per_year_estimate']:.2f}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap BTC meta-label model from Binance history")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--timeframe", type=int, default=60)
    args = parser.parse_args()
    bootstrap(days=args.days, symbol="BTCUSDT", timeframe=args.timeframe, quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
