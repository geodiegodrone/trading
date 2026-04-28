from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

import bot_config
import multi_bot
import regime as regime_detector
from strategies import signal_breakout, signal_meanrev, signal_trend


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "backtest_walkforward_report.json"


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _interval_label(timeframe: int) -> str:
    minutes = int(timeframe)
    if minutes < 60:
        return f"{minutes}m"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int | None = None, limit: int = 1500) -> List[list]:
    url = "https://fapi.binance.com/fapi/v1/klines?" + urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
            **({"endTime": end_ms} if end_ms is not None else {}),
        }
    )
    with urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_history(symbol: str, timeframe: int, days: int) -> pd.DataFrame:
    interval = _interval_label(int(timeframe))
    end_ms = _utc_now_ms()
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _fetch_klines(symbol, interval, cursor, end_ms, limit=1500)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1500:
            break
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"])
    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"],
    )
    df = df[["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


@dataclass
class SimTrade:
    side: str
    entry_ts: int
    entry_price: float
    qty: float
    sl: float
    tp: float
    risk_usdt: float
    strategy: str
    regime: str
    peak_price: float
    trough_price: float
    entry_idx: int
    exit_ts: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    reason: str = ""


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _annualized_sharpe(r_values: List[float], holding_bars: List[int], timeframe_minutes: int) -> float:
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
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak * 100.0)
    return max_dd


def _trade_usdt(balance: float, price: float, stop_distance: float, leverage: float) -> float:
    if price <= 0 or stop_distance <= 0 or leverage <= 0:
        return 50.0
    risk_target = balance * 0.005
    return max(15.0, risk_target * price / (leverage * stop_distance))


def _confirmation_row(confirmation_df: pd.DataFrame, ts_value: int) -> pd.Series | None:
    subset = confirmation_df[confirmation_df["ts"] <= ts_value]
    if subset.empty:
        return None
    return subset.iloc[-1]


def _htf_confirms(signal: str, confirm_row: pd.Series | None, cfg: Dict[str, float | int]) -> bool:
    if signal == "NEUTRAL" or confirm_row is None:
        return False
    ema50 = _safe_float(confirm_row.get(f"ema{int(cfg.get('ema_confirm', 50))}"))
    ema200 = _safe_float(confirm_row.get(f"ema{int(cfg.get('ema_trend', 200))}"))
    close = _safe_float(confirm_row.get("close"))
    if signal == "LONG":
        return close > ema200 and ema50 > ema200
    return close < ema200 and ema50 < ema200


def _mode_signal(mode: str, primary_slice: pd.DataFrame, confirm_row: pd.Series | None, cfg: Dict[str, float | int]) -> tuple[str, str]:
    last = primary_slice.iloc[-1].to_dict()
    regime_name = "TREND"
    if mode == "trend":
        signal = signal_trend(last, primary_slice, cfg)
    elif mode == "meanrev":
        regime_name = "RANGE"
        signal = signal_meanrev(last, primary_slice, cfg)
    elif mode == "breakout":
        regime_name = "VOLATILE"
        signal = signal_breakout(last, primary_slice, cfg)
    else:
        regime_info = regime_detector.classify_regime(primary_slice)
        regime_name = str(regime_info.get("regime", "MIXED")).upper()
        if regime_name == "TREND":
            signal = signal_trend(last, primary_slice, cfg)
        elif regime_name == "RANGE":
            signal = signal_meanrev(last, primary_slice, cfg)
        elif regime_name == "VOLATILE":
            signal = signal_breakout(last, primary_slice, cfg)
        else:
            signal = "NEUTRAL"
    if signal != "NEUTRAL" and not _htf_confirms(signal, confirm_row, cfg):
        return "NEUTRAL", regime_name
    return signal, regime_name


def _apply_circuit_breaker(state: Dict[str, object], balance: float, daily_start: float, weekly_start: float, row: pd.Series, cfg: Dict[str, float | int], ts_dt: datetime) -> tuple[bool, int]:
    pause_until = int(state.get("pause_until", 0) or 0)
    if pause_until and int(row["ts"]) < pause_until:
        return True, 0
    paused = False
    triggers = 0
    if daily_start > 0 and balance <= daily_start * (1 - float(cfg["cb_daily_loss_pct"]) / 100.0):
        state["pause_until"] = int((ts_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp() * 1000)
        paused = True
    elif weekly_start > 0 and balance <= weekly_start * (1 - float(cfg["cb_weekly_loss_pct"]) / 100.0):
        state["pause_until"] = int((ts_dt + timedelta(hours=24)).timestamp() * 1000)
        paused = True
    elif int(state.get("consecutive_losses", 0) or 0) >= int(cfg["cb_consecutive_losses"]):
        state["pause_until"] = int((ts_dt + timedelta(hours=float(cfg["cb_cooldown_hours"]))).timestamp() * 1000)
        paused = True
    else:
        recent = list(state.get("last_trade_results", []) or [])[-int(cfg["cb_rolling_window"]):]
        if len(recent) >= 10 and (sum(recent) / max(1, len(recent))) < float(cfg["cb_rolling_winrate_min"]):
            state["pause_until"] = int((ts_dt + timedelta(hours=float(cfg["cb_cooldown_hours"]) * 2.0)).timestamp() * 1000)
            paused = True
        else:
            open_ = _safe_float(row.get("open"))
            spike_pct = abs((_safe_float(row.get("high")) - _safe_float(row.get("low"))) / open_ * 100.0) if open_ > 0 else 0.0
            if spike_pct > float(cfg["cb_volatility_spike_pct"]):
                state["pause_until"] = int((ts_dt + timedelta(hours=1)).timestamp() * 1000)
                paused = True
            else:
                equity_curve = list(state.get("equity_curve", []) or [])
                if _max_drawdown_pct(equity_curve[-100:]) > 15.0:
                    state["pause_until"] = int((ts_dt + timedelta(hours=24)).timestamp() * 1000)
                    paused = True
    if paused:
        triggers = 1
    return paused, triggers


def simulate_mode(primary_df: pd.DataFrame, confirmation_df: pd.DataFrame, cfg: Dict[str, float | int], mode: str, start_balance: float = 5000.0) -> Dict[str, float | int]:
    primary_df = multi_bot._compute_indicators(primary_df.copy(), cfg)
    confirmation_df = multi_bot._compute_indicators(confirmation_df.copy(), cfg)
    warmup = max(220, int(cfg.get("ema_trend", 200)) + 5)
    balance = start_balance
    equity_curve = [balance]
    trades: List[SimTrade] = []
    open_trade: SimTrade | None = None
    cb_state: Dict[str, object] = {
        "pause_until": 0,
        "consecutive_losses": 0,
        "last_trade_results": [],
        "equity_curve": equity_curve,
    }
    current_day = None
    current_week = None
    daily_start = balance
    weekly_start = balance
    cb_triggers = 0

    for i in range(warmup, len(primary_df) - 1):
        row = primary_df.iloc[i]
        ts_dt = datetime.fromtimestamp(int(row["ts"]) / 1000, tz=timezone.utc)
        day_key = ts_dt.strftime("%Y-%m-%d")
        week_key = f"{ts_dt.isocalendar().year}-W{ts_dt.isocalendar().week:02d}"
        if current_day != day_key:
            current_day = day_key
            daily_start = balance
        if current_week != week_key:
            current_week = week_key
            weekly_start = balance

        signal, regime_name = _mode_signal(mode, primary_df.iloc[: i + 1], _confirmation_row(confirmation_df, int(row["ts"])), cfg)
        price = _safe_float(row["close"])
        atr = _safe_float(row.get("atr"))
        if open_trade:
            move_r = ((price - open_trade.entry_price) * open_trade.qty if open_trade.side == "LONG" else (open_trade.entry_price - price) * open_trade.qty) / max(open_trade.risk_usdt, 1e-9)
            if open_trade.side == "LONG":
                open_trade.peak_price = max(open_trade.peak_price, price)
                if move_r >= float(cfg.get("trail_start_r", 1.5)) and atr > 0:
                    open_trade.sl = max(open_trade.sl, open_trade.peak_price - 3.0 * atr)
                if i - open_trade.entry_idx >= 24 and move_r < 0.3:
                    exit_price = price
                    reason = "EXIT_TIMEOUT"
                elif _safe_float(row["low"]) <= open_trade.sl:
                    exit_price = open_trade.sl
                    reason = "SL"
                elif _safe_float(row["high"]) >= open_trade.tp:
                    exit_price = open_trade.tp
                    reason = "TP"
                elif signal == "SHORT":
                    exit_price = price
                    reason = "REV"
                else:
                    exit_price = None
                    reason = ""
            else:
                open_trade.trough_price = min(open_trade.trough_price, price)
                if move_r >= float(cfg.get("trail_start_r", 1.5)) and atr > 0:
                    open_trade.sl = min(open_trade.sl, open_trade.trough_price + 3.0 * atr)
                if i - open_trade.entry_idx >= 24 and move_r < 0.3:
                    exit_price = price
                    reason = "EXIT_TIMEOUT"
                elif _safe_float(row["high"]) >= open_trade.sl:
                    exit_price = open_trade.sl
                    reason = "SL"
                elif _safe_float(row["low"]) <= open_trade.tp:
                    exit_price = open_trade.tp
                    reason = "TP"
                elif signal == "LONG":
                    exit_price = price
                    reason = "REV"
                else:
                    exit_price = None
                    reason = ""
            if exit_price is not None:
                pnl = (exit_price - open_trade.entry_price) * open_trade.qty if open_trade.side == "LONG" else (open_trade.entry_price - exit_price) * open_trade.qty
                balance += pnl
                open_trade.exit_ts = int(row["ts"])
                open_trade.exit_price = float(exit_price)
                open_trade.pnl = float(pnl)
                open_trade.reason = reason
                trades.append(open_trade)
                cb_state["consecutive_losses"] = 0 if pnl > 0 else int(cb_state.get("consecutive_losses", 0) or 0) + 1
                last_results = list(cb_state.get("last_trade_results", []) or [])
                last_results.append(1 if pnl > 0 else 0)
                cb_state["last_trade_results"] = last_results[-int(cfg["cb_rolling_window"]):]
                equity_curve.append(balance)
                open_trade = None
                cb_state["equity_curve"] = equity_curve

        paused, triggers = _apply_circuit_breaker(cb_state, balance, daily_start, weekly_start, row, cfg, ts_dt)
        cb_triggers += triggers
        if open_trade is not None or paused:
            continue
        if signal == "NEUTRAL" or atr <= 0:
            continue
        next_row = primary_df.iloc[i + 1]
        entry = _safe_float(next_row["open"], price)
        if mode == "meanrev" or regime_name == "RANGE":
            stop_distance = atr * float(cfg.get("meanrev_atr_mult", 0.5))
        elif mode == "breakout" or regime_name == "VOLATILE":
            stop_distance = atr * float(cfg.get("volatile_atr_mult", 2.0) if regime_name == "VOLATILE" else cfg.get("breakout_atr_mult", 1.0))
        else:
            stop_distance = atr * float(cfg.get("atr_mult", 1.5))
        trade_usdt = _trade_usdt(balance, entry, stop_distance, float(cfg.get("leverage", 2)))
        qty = (trade_usdt * float(cfg.get("leverage", 2))) / entry if entry > 0 else 0.0
        if qty <= 0:
            continue
        if signal == "LONG":
            sl = entry - stop_distance
            if mode == "meanrev" or regime_name == "RANGE":
                sma20 = _safe_float(primary_df["close"].iloc[max(0, i - 19): i + 1].mean())
                tp = sma20 if sma20 > entry else entry + stop_distance
            elif mode == "breakout" or regime_name == "VOLATILE":
                tp = entry + atr * float(cfg.get("breakout_tp_atr_mult", 3.0))
            else:
                tp = entry + stop_distance * 2.0
        else:
            sl = entry + stop_distance
            if mode == "meanrev" or regime_name == "RANGE":
                sma20 = _safe_float(primary_df["close"].iloc[max(0, i - 19): i + 1].mean())
                tp = sma20 if 0 < sma20 < entry else entry - stop_distance
            elif mode == "breakout" or regime_name == "VOLATILE":
                tp = entry - atr * float(cfg.get("breakout_tp_atr_mult", 3.0))
            else:
                tp = entry - stop_distance * 2.0
        open_trade = SimTrade(
            side=signal,
            entry_ts=int(next_row["ts"]),
            entry_price=entry,
            qty=qty,
            sl=sl,
            tp=tp,
            risk_usdt=max(1e-9, qty * stop_distance),
            strategy=mode,
            regime=regime_name,
            peak_price=entry,
            trough_price=entry,
            entry_idx=i + 1,
        )

    realized_pnls = [trade.pnl for trade in trades]
    wins = len([p for p in realized_pnls if p > 0])
    losses = len([p for p in realized_pnls if p <= 0])
    r_values = [trade.pnl / trade.risk_usdt for trade in trades if trade.risk_usdt > 0]
    holding_bars = [max(1, int((trade.exit_ts - trade.entry_ts) / (int(cfg["primary_timeframe"]) * 60 * 1000))) for trade in trades if trade.exit_ts]
    return {
        "mode": mode,
        "trades": len(trades),
        "pnl_pct": ((balance - start_balance) / start_balance * 100.0) if start_balance else 0.0,
        "win_pct": (wins / max(1, len(trades))) * 100.0,
        "sharpe": _annualized_sharpe(r_values, holding_bars, int(cfg["primary_timeframe"])),
        "maxdd_pct": _max_drawdown_pct(equity_curve),
        "cb_triggers": cb_triggers,
        "wins": wins,
        "losses": losses,
    }


def compare_modes(days: int = 180, symbol: str = "BTCUSDT") -> List[Dict[str, float | int]]:
    cfg = multi_bot._get_symbol_config(symbol)
    primary_df = fetch_history(symbol, int(cfg["primary_timeframe"]), days)
    confirmation_df = fetch_history(symbol, int(cfg["confirmation_timeframe"]), days)
    results = []
    for mode in ("trend", "meanrev", "breakout", "regime"):
        mode_cfg = dict(cfg)
        mode_cfg["strategy_mode"] = mode
        results.append(simulate_mode(primary_df, confirmation_df, mode_cfg, mode))
    return results


def _print_table(results: List[Dict[str, float | int]]) -> None:
    print("mode      | trades | pnl%  | win%  | sharpe | maxdd% | cb_triggers")
    for row in results:
        print(
            f"{str(row['mode']):<9} | {int(row['trades']):>6} | {float(row['pnl_pct']):>5.1f} | "
            f"{float(row['win_pct']):>5.1f} | {float(row['sharpe']):>6.2f} | {float(row['maxdd_pct']):>6.2f} | {int(row['cb_triggers']):>11}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest BTC strategy modes on higher timeframes")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    results = compare_modes(days=args.days, symbol="BTCUSDT")
    _print_table(results)
    best = sorted(results, key=lambda item: (float(item["sharpe"]), float(item["pnl_pct"])), reverse=True)[0]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "results": results,
        "best_mode": best["mode"],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best["mode"] != "regime":
        print(f"warning: regime no fue el mejor modo; mejor sharpe={best['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
