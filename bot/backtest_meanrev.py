from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import historical_data
from features import build_features


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "backtest_meanrev_report.json"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_history(symbol: str, start: str, end: str, force_download: bool = False) -> pd.DataFrame:
    df = historical_data.load_history(symbol, 60)
    if df.empty or force_download:
        historical_data.download_history(symbol, 60, 730, force=force_download)
        df = historical_data.load_history(symbol, 60)
    if df.empty:
        return df
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    return df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].reset_index(drop=True)


def _daily_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _annualized_sharpe(r_values: List[float], holding_bars: List[int], timeframe_minutes: int = 60) -> float:
    if len(r_values) < 2:
        return 0.0
    arr = np.asarray(r_values, dtype=float)
    std = float(arr.std(ddof=1))
    if std <= 0:
        return 0.0
    avg_hold = max(1.0, float(np.mean([max(1, h) for h in holding_bars or [1]])))
    bars_per_year = (365.0 * 24.0 * 60.0) / float(timeframe_minutes)
    trades_per_year = bars_per_year / avg_hold
    return float((arr.mean() / std) * math.sqrt(trades_per_year))


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 1e-6:
            max_dd = max(max_dd, ((peak - value) / peak) * 100.0)
    return float(min(max(max_dd, 0.0), 100.0))


@dataclass
class OpenTrade:
    side: str
    entry_idx: int
    entry_ts: int
    entry_price: float
    stop_price: float
    tp_price: float
    qty: float
    risk_usdt: float
    opened_day: str
    reason: str
    peak_price: float
    trough_price: float


def _close_trade(trade: OpenTrade, exit_price: float) -> Dict[str, float]:
    if trade.side == "LONG":
        move = exit_price - trade.entry_price
    else:
        move = trade.entry_price - exit_price
    pnl = move * trade.qty
    r_multiple = pnl / trade.risk_usdt if trade.risk_usdt > 0 else 0.0
    return {"pnl": float(pnl), "r_multiple": float(r_multiple), "exit_price": float(exit_price)}


def run_backtest(df: pd.DataFrame, cfg: Dict[str, float | int], verbose: bool = False) -> Dict[str, object]:
    if df.empty or len(df) < 120:
        return {"trades": [], "metrics": {"reason": "insufficient_data"}}

    frame = df.copy().reset_index(drop=True)
    features = build_features(frame)
    for column in features.columns:
        frame[column] = features[column]
    returns = pd.to_numeric(frame["close"], errors="coerce").pct_change(1).mul(100.0)
    roll_mean = returns.rolling(50).mean()
    roll_std = returns.rolling(50).std(ddof=0).replace(0.0, np.nan)
    frame["return_1h_zscore_50"] = ((returns - roll_mean) / roll_std).fillna(0.0)
    frame["atr"] = frame["atr_pct"] * pd.to_numeric(frame["close"], errors="coerce") / 100.0

    balance = 10000.0
    day_start_balance = balance
    current_day = None
    trades: List[Dict[str, object]] = []
    open_trade: OpenTrade | None = None
    daily_trades = 0
    consecutive_losses = 0
    pause_until_ts = 0
    recent_results: List[int] = []
    equity_curve: List[float] = [balance]
    risk_pct = float(cfg.get("risk_pct_per_trade", 0.15)) / 100.0
    daily_cap_pct = float(cfg.get("daily_risk_cap", 0.5))
    stop_bars = int(cfg.get("meanrev_time_stop_bars", 12))
    sigma = float(cfg.get("meanrev_sigma", 2.5))
    rsi_low = float(cfg.get("meanrev_rsi_low", 25.0))
    rsi_high = float(cfg.get("meanrev_rsi_high", 75.0))
    atr_z_min = float(cfg.get("meanrev_atr_z", 1.5))
    sl_atr = float(cfg.get("meanrev_sl_atr", 1.0))
    tp_atr = float(cfg.get("meanrev_tp_atr", 0.6))

    for idx in range(60, len(frame)):
        row = frame.iloc[idx]
        ts_ms = int(row["ts"])
        day = _daily_key(ts_ms)
        if current_day != day:
            current_day = day
            day_start_balance = balance
            daily_trades = 0
        if pause_until_ts and ts_ms < pause_until_ts and open_trade is None:
            continue

        if open_trade is not None:
            if open_trade.side == "LONG":
                if float(row["low"]) <= open_trade.stop_price:
                    result = _close_trade(open_trade, open_trade.stop_price)
                    balance += result["pnl"]
                    equity_curve.append(balance)
                    trades.append({**result, "side": open_trade.side, "entry_ts": open_trade.entry_ts, "exit_ts": ts_ms, "holding_bars": idx - open_trade.entry_idx, "reason": "SL", "opened_day": open_trade.opened_day})
                    recent_results.append(1 if result["pnl"] > 0 else 0)
                    consecutive_losses = 0 if result["pnl"] > 0 else consecutive_losses + 1
                    open_trade = None
                    if result["pnl"] < 0 and consecutive_losses >= 3:
                        pause_until_ts = ts_ms + 24 * 3600 * 1000
                    continue
                if float(row["high"]) >= open_trade.tp_price:
                    result = _close_trade(open_trade, open_trade.tp_price)
                    balance += result["pnl"]
                    equity_curve.append(balance)
                    trades.append({**result, "side": open_trade.side, "entry_ts": open_trade.entry_ts, "exit_ts": ts_ms, "holding_bars": idx - open_trade.entry_idx, "reason": "TP", "opened_day": open_trade.opened_day})
                    recent_results.append(1 if result["pnl"] > 0 else 0)
                    consecutive_losses = 0 if result["pnl"] > 0 else consecutive_losses + 1
                    open_trade = None
                    continue
            else:
                if float(row["high"]) >= open_trade.stop_price:
                    result = _close_trade(open_trade, open_trade.stop_price)
                    balance += result["pnl"]
                    equity_curve.append(balance)
                    trades.append({**result, "side": open_trade.side, "entry_ts": open_trade.entry_ts, "exit_ts": ts_ms, "holding_bars": idx - open_trade.entry_idx, "reason": "SL", "opened_day": open_trade.opened_day})
                    recent_results.append(1 if result["pnl"] > 0 else 0)
                    consecutive_losses = 0 if result["pnl"] > 0 else consecutive_losses + 1
                    open_trade = None
                    if result["pnl"] < 0 and consecutive_losses >= 3:
                        pause_until_ts = ts_ms + 24 * 3600 * 1000
                    continue
                if float(row["low"]) <= open_trade.tp_price:
                    result = _close_trade(open_trade, open_trade.tp_price)
                    balance += result["pnl"]
                    equity_curve.append(balance)
                    trades.append({**result, "side": open_trade.side, "entry_ts": open_trade.entry_ts, "exit_ts": ts_ms, "holding_bars": idx - open_trade.entry_idx, "reason": "TP", "opened_day": open_trade.opened_day})
                    recent_results.append(1 if result["pnl"] > 0 else 0)
                    consecutive_losses = 0 if result["pnl"] > 0 else consecutive_losses + 1
                    open_trade = None
                    continue
            if open_trade is not None and (idx - open_trade.entry_idx) >= stop_bars:
                result = _close_trade(open_trade, float(row["close"]))
                balance += result["pnl"]
                equity_curve.append(balance)
                trades.append({**result, "side": open_trade.side, "entry_ts": open_trade.entry_ts, "exit_ts": ts_ms, "holding_bars": idx - open_trade.entry_idx, "reason": "TIME", "opened_day": open_trade.opened_day})
                recent_results.append(1 if result["pnl"] > 0 else 0)
                consecutive_losses = 0 if result["pnl"] > 0 else consecutive_losses + 1
                open_trade = None
                if result["pnl"] < 0 and consecutive_losses >= 3:
                    pause_until_ts = ts_ms + 24 * 3600 * 1000
                continue

        if open_trade is not None:
            continue
        if daily_trades >= 1:
            continue
        if balance <= day_start_balance * (1 - daily_cap_pct / 100.0):
            continue
        if len(recent_results) >= 15 and sum(recent_results[-15:]) <= 10:
            pause_until_ts = ts_ms + 7 * 24 * 3600 * 1000
            continue

        return_z = float(row.get("return_1h_zscore_50") or 0.0)
        rsi = float(row.get("rsi") or 0.0)
        atr_z = float(row.get("atr_pct_zscore_50") or 0.0)
        atr = float(row.get("atr") or 0.0)
        close = float(row.get("close") or 0.0)
        if atr <= 0 or close <= 0:
            continue
        side = "NEUTRAL"
        if return_z <= -sigma and rsi <= rsi_low and atr_z >= atr_z_min:
            side = "LONG"
        elif return_z >= sigma and rsi >= rsi_high and atr_z >= atr_z_min:
            side = "SHORT"
        if side == "NEUTRAL":
            continue
        stop_distance = atr * sl_atr
        if stop_distance <= 0:
            continue
        if side == "LONG":
            stop = round(close - stop_distance, 2)
            if stop == round(close, 2):
                stop = round(close - (atr * 1.2), 2)
            tp = round(close + (atr * tp_atr), 2)
        else:
            stop = round(close + stop_distance, 2)
            if stop == round(close, 2):
                stop = round(close + (atr * 1.2), 2)
            tp = round(close - (atr * tp_atr), 2)
        risk_usdt = balance * risk_pct
        qty = risk_usdt / abs(close - stop)
        if qty <= 0:
            continue
        open_trade = OpenTrade(
            side=side,
            entry_idx=idx,
            entry_ts=ts_ms,
            entry_price=close,
            stop_price=stop,
            tp_price=tp,
            qty=qty,
            risk_usdt=risk_usdt,
            opened_day=day,
            reason=f"return_z={return_z:.2f} rsi={rsi:.1f} atr_z={atr_z:.2f}",
            peak_price=close,
            trough_price=close,
        )
        daily_trades += 1

    r_values = [float(t["r_multiple"]) for t in trades]
    holding_bars = [int(t["holding_bars"]) for t in trades]
    wins = sum(1 for t in trades if float(t["pnl"]) > 0)
    losses = sum(1 for t in trades if float(t["pnl"]) < 0)
    gross_profit = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = abs(sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = float((wins / max(1, len(trades))) * 100.0)
    sharpe = _annualized_sharpe(r_values, holding_bars, timeframe_minutes=60)
    max_dd = _max_drawdown_pct(equity_curve)
    expectancy_r = float(np.mean(r_values)) if r_values else 0.0
    metrics = {
        "trades": int(len(trades)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor if math.isfinite(profit_factor) else 999.0),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "expectancy_R": float(expectancy_r),
        "final_balance": float(balance),
        "start_balance": 10000.0,
        "net_pnl_pct": float(((balance - 10000.0) / 10000.0) * 100.0),
    }
    metrics["passes"] = bool(
        metrics["trades"] >= 30
        and metrics["win_rate"] >= 40.0
        and metrics["profit_factor"] >= 1.1
        and metrics["max_drawdown_pct"] < 50.0
        and metrics["sharpe"] >= 0.2
    )
    return {"trades": trades, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest BTC mean-reversion overshoot")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-05-09")
    parser.add_argument("--risk-per-trade", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    df = _load_history(str(args.symbol or "BTCUSDT"), str(args.start), str(args.end))
    cfg = {
        "risk_pct_per_trade": float(args.risk_per_trade),
        "meanrev_sigma": 2.5,
        "meanrev_rsi_low": 25.0,
        "meanrev_rsi_high": 75.0,
        "meanrev_atr_z": 1.5,
        "meanrev_sl_atr": 1.0,
        "meanrev_tp_atr": 0.6,
        "meanrev_time_stop_bars": 12,
        "meanrev_breakeven_r": 0.3,
        "meanrev_lock_r": 0.5,
        "meanrev_lock_stop_r": 0.1,
    }
    result = run_backtest(df, cfg, verbose=bool(args.verbose))
    metrics = result["metrics"]
    report = {
        "symbol": str(args.symbol or "BTCUSDT"),
        "start": str(args.start),
        "end": str(args.end),
        "risk_per_trade": float(args.risk_per_trade),
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "Total 1h candles: {candles}\n"
        "Trades: {trades}\n"
        "Win rate: {win_rate:.1f}%\n"
        "Profit factor: {pf:.2f}\n"
        "Sharpe: {sharpe:.2f}\n"
        "Max DD: {dd:.2f}%\n"
        "Expectancy R: {exp:.2f}\n"
        "Final balance: {bal:.2f} USDT\n"
        "Passes gate: {passes}".format(
            candles=len(df),
            trades=metrics["trades"],
            win_rate=metrics["win_rate"],
            pf=metrics["profit_factor"],
            sharpe=metrics["sharpe"],
            dd=metrics["max_drawdown_pct"],
            exp=metrics["expectancy_R"],
            bal=metrics["final_balance"],
            passes=metrics["passes"],
        )
    )
    return 0 if metrics["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
