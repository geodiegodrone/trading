from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import data_manager
from bot_config import DEFAULT_CONFIG
from feature_swingvolume import SwingVolumeAnalyzer
from strategies import Signal, signal_swingvolume


@dataclass
class SimTrade:
    side: str
    entry_ts: int
    entry_price: float
    stop_price: float
    tp_price: float
    qty: float
    risk_usdt: float
    initial_stop_distance: float
    bars_open: int = 0
    open_index: int = 0
    moved_breakeven: bool = False
    locked_profit: bool = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _slice_range(df: pd.DataFrame, start_ts: int | None, end_ts: int | None) -> pd.DataFrame:
    frame = df.copy()
    if start_ts is not None:
        frame = frame[frame["ts"] >= start_ts]
    if end_ts is not None:
        frame = frame[frame["ts"] <= end_ts]
    return frame.reset_index(drop=True)


def _to_ts(raw: str | None) -> int | None:
    if not raw:
        return None
    dt = datetime.fromisoformat(str(raw))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _performance(trades: list[dict[str, Any]], equity_curve: list[float], start_balance: float) -> dict[str, Any]:
    pnl = np.array([_safe_float(trade["pnl"]) for trade in trades], dtype=float)
    r_values = np.array([_safe_float(trade["r_multiple"]) for trade in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0)
    win_rate = float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0
    sharpe = 0.0
    if len(r_values) >= 2 and float(r_values.std(ddof=0)) > 1e-9:
        sharpe = float((r_values.mean() / r_values.std(ddof=0)) * np.sqrt(len(r_values)))
    max_dd = 0.0
    peak = float("-inf")
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, ((peak - value) / peak) * 100.0)
    pnl_pct = ((equity_curve[-1] - start_balance) / start_balance * 100.0) if equity_curve else 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999.0),
        "sharpe": sharpe,
        "max_drawdown_pct": float(max_dd),
        "pnl_pct": float(pnl_pct),
        "expectancy_R": float(r_values.mean()) if len(r_values) else 0.0,
    }


def run_backtest(start: str | None, end: str | None, risk_per_trade_pct: float = 0.15, symbol: str = "BTCUSDT") -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg["risk_pct_per_trade"] = float(risk_per_trade_pct)
    d1 = data_manager.get_daily_data(symbol, days=1095, force=False)
    h4 = data_manager.get_hourly4_data(symbol, days=1095, force=False)
    start_ts = _to_ts(start)
    end_ts = _to_ts(end)
    if end_ts is not None:
        end_ts += (24 * 60 * 60 * 1000) - 1
    d1 = _slice_range(d1, None, end_ts)
    h4 = _slice_range(h4, start_ts, end_ts)
    analyzer = SwingVolumeAnalyzer(symbol)
    h4 = analyzer.prepare_h4(h4)
    d1 = analyzer.prepare_d1(d1)
    if h4.empty or d1.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "pnl_pct": 0.0, "expectancy_R": 0.0}

    balance = 50000.0
    start_balance = balance
    equity_curve = [balance]
    trades: list[dict[str, Any]] = []
    open_trade: SimTrade | None = None
    last_open_day: str | None = None
    daily_realized = 0.0
    current_day: str | None = None

    for idx in range(220, len(h4)):
        row = h4.iloc[idx]
        bar_day = datetime.fromtimestamp(int(row["ts"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if current_day != bar_day:
            current_day = bar_day
            daily_realized = 0.0
        if open_trade is not None:
            open_trade.bars_open += 1
            current_r = 0.0
            if open_trade.risk_usdt > 0:
                pnl_now = (float(row["close"]) - open_trade.entry_price) * open_trade.qty if open_trade.side == "LONG" else (open_trade.entry_price - float(row["close"])) * open_trade.qty
                current_r = pnl_now / open_trade.risk_usdt
            if open_trade.side == "LONG":
                if not open_trade.locked_profit and current_r >= 0.7:
                    open_trade.stop_price = max(open_trade.stop_price, open_trade.entry_price + open_trade.initial_stop_distance * 0.2)
                    open_trade.locked_profit = True
                elif not open_trade.moved_breakeven and current_r >= 0.3:
                    open_trade.stop_price = max(open_trade.stop_price, open_trade.entry_price)
                    open_trade.moved_breakeven = True
            else:
                if not open_trade.locked_profit and current_r >= 0.7:
                    open_trade.stop_price = min(open_trade.stop_price, open_trade.entry_price - open_trade.initial_stop_distance * 0.2)
                    open_trade.locked_profit = True
                elif not open_trade.moved_breakeven and current_r >= 0.3:
                    open_trade.stop_price = min(open_trade.stop_price, open_trade.entry_price)
                    open_trade.moved_breakeven = True
            exit_reason = None
            exit_price = None
            if open_trade.side == "LONG":
                if float(row["low"]) <= open_trade.stop_price:
                    exit_reason = "SL"
                    exit_price = open_trade.stop_price
                elif float(row["high"]) >= open_trade.tp_price:
                    exit_reason = "TP"
                    exit_price = open_trade.tp_price
            else:
                if float(row["high"]) >= open_trade.stop_price:
                    exit_reason = "SL"
                    exit_price = open_trade.stop_price
                elif float(row["low"]) <= open_trade.tp_price:
                    exit_reason = "TP"
                    exit_price = open_trade.tp_price
            if exit_reason is None and open_trade.bars_open >= 5:
                exit_reason = "TIME"
                exit_price = float(row["close"])
            if exit_reason is not None and exit_price is not None:
                pnl = (exit_price - open_trade.entry_price) * open_trade.qty if open_trade.side == "LONG" else (open_trade.entry_price - exit_price) * open_trade.qty
                balance += pnl
                daily_realized += pnl
                equity_curve.append(balance)
                trades.append(
                    {
                        "side": open_trade.side,
                        "entry_ts": open_trade.entry_ts,
                        "exit_ts": int(row["ts"]),
                        "entry_price": open_trade.entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "r_multiple": pnl / open_trade.risk_usdt if open_trade.risk_usdt > 0 else 0.0,
                        "reason": exit_reason,
                    }
                )
                open_trade = None
            continue

        if last_open_day == bar_day or daily_realized <= -(start_balance * 0.005):
            continue

        h4_slice = h4.iloc[: idx + 1].copy()
        d1_slice = d1[d1["ts"] <= int(row["ts"])].copy()
        signal = signal_swingvolume(d1_slice, h4_slice, last_n_bars_h4=40)
        if signal.side == "NEUTRAL":
            continue
        entry_price = _safe_float(signal.entry_price or row["close"])
        stop_price = _safe_float(signal.stop_price)
        risk_distance = abs(entry_price - stop_price)
        if entry_price <= 0 or risk_distance <= 0:
            continue
        risk_usdt = balance * (float(risk_per_trade_pct) / 100.0)
        qty = risk_usdt / risk_distance
        if qty <= 0:
            continue
        open_trade = SimTrade(
            side=signal.side,
            entry_ts=int(row["ts"]),
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=_safe_float(signal.tp_price),
            qty=qty,
            risk_usdt=risk_usdt,
            initial_stop_distance=risk_distance,
            open_index=idx,
        )
        last_open_day = bar_day

    metrics = _performance(trades, equity_curve, start_balance)
    metrics["start"] = start
    metrics["end"] = end
    metrics["risk_per_trade_pct"] = float(risk_per_trade_pct)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest SWINGVOLUME on BTC futures")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-05-01")
    parser.add_argument("--risk-per-trade", type=float, default=0.15)
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    result = run_backtest(args.start, args.end, risk_per_trade_pct=float(args.risk_per_trade), symbol=str(args.symbol or "BTCUSDT"))
    print(
        f"trades={result['trades']} win_rate={result['win_rate']:.2f}% profit_factor={result['profit_factor']:.2f} "
        f"sharpe={result['sharpe']:.2f} max_dd={result['max_drawdown_pct']:.2f}% pnl={result['pnl_pct']:.2f}% expR={result['expectancy_R']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
