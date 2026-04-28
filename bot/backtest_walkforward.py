from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import pandas_ta as ta

import bot_config
import multi_bot


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "backtest_walkforward_report.json"


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


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
    interval = f"{int(timeframe)}m"
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
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    for col in ["ts", "open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def _compute_indicators(df: pd.DataFrame, cfg: Dict[str, float | int]) -> pd.DataFrame:
    df = df.copy()
    fast = int(cfg.get("ema_fast", 9))
    slow = int(cfg.get("ema_slow", 21))
    trend = int(cfg.get("ema_trend", 200))
    rsi_period = int(cfg.get("rsi_period", 14))
    adx_period = int(cfg.get("adx_period", 14))
    st_period = int(cfg.get("supertrend_period", 14))
    st_mult = float(cfg.get("supertrend_mult", 3.5))
    df[f"ema{fast}"] = ta.ema(df["close"], length=fast)
    df[f"ema{slow}"] = ta.ema(df["close"], length=slow)
    df[f"ema{trend}"] = ta.ema(df["close"], length=trend)
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)
    adx = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
    adx_col = f"ADX_{adx_period}"
    if adx is not None and adx_col in adx:
        df["adx"] = adx[adx_col]
    st = ta.supertrend(df["high"], df["low"], df["close"], length=st_period, multiplier=st_mult)
    st_dir_col = f"SUPERTd_{st_period}_{st_mult}"
    if st is not None and st_dir_col in st:
        df["supertrend_direction"] = st[st_dir_col]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(cfg.get("atr_period", 14)))
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df.attrs["indicator_cols"] = {
        "ema_fast": f"ema{fast}",
        "ema_slow": f"ema{slow}",
        "ema_trend": f"ema{trend}",
        "supertrend_dir": "supertrend_direction",
    }
    return df


@dataclass
class Trade:
    side: str
    entry_ts: int
    entry_price: float
    signal_idx: int | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    reason: str = ""


def _trade_pnl(side: str, entry: float, exit_: float, qty: float) -> float:
    if side == "LONG":
        return (exit_ - entry) * qty
    return (entry - exit_) * qty


def _trade_move_r(side: str, entry: float, price: float, qty: float, risk_usdt: float) -> float:
    if entry <= 0 or qty <= 0 or risk_usdt <= 0:
        return 0.0
    pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
    return pnl / risk_usdt


def simulate(df: pd.DataFrame, cfg: Dict[str, float | int], trade_usdt: float) -> Dict[str, float | int | list]:
    if df.empty:
        return {"trades": [], "total": 0, "pnl": 0.0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    df = _compute_indicators(df, cfg)
    cols = df.attrs.get("indicator_cols", {})
    fast_col = cols.get("ema_fast", "ema9")
    slow_col = cols.get("ema_slow", "ema21")
    trend_col = cols.get("ema_trend", "ema200")
    st_col = cols.get("supertrend_dir", "supertrend_direction")

    warmup = max(220, int(cfg.get("ema_trend", 200)) + 5)
    start_idx = min(warmup, len(df) - 2)
    if start_idx < 0:
        start_idx = 0

    open_trade: Trade | None = None
    open_qty = 0.0
    open_sl = 0.0
    open_tp = 0.0
    open_risk_usdt = 0.0
    open_peak = 0.0
    open_trough = 0.0
    open_breakeven_moved = False
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    trades: List[Trade] = []

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        signal = multi_bot.evaluate_signal(row.to_dict(), df.iloc[: i + 1], cfg)
        next_row = df.iloc[i + 1]
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])
        atr = float(row.get("atr") or 0.0)

        if open_trade is not None:
            move_r = _trade_move_r(open_trade.side, open_trade.entry_price, close, open_qty, open_risk_usdt)
            if open_trade.side == "LONG":
                open_peak = max(open_peak, close)
                if not open_breakeven_moved and move_r >= float(cfg.get("breakeven_r", 1.0)):
                    buffer_pct = float(cfg.get("breakeven_buffer_pct", 0.0))
                    new_sl = open_trade.entry_price * (1 + buffer_pct / 100.0)
                    if new_sl > open_sl:
                        open_sl = new_sl
                        open_breakeven_moved = True
                if move_r >= float(cfg.get("trail_start_r", 1.5)) and atr > 0:
                    trail_candidate = close - atr * float(cfg.get("trail_atr_mult", 1.25))
                    if trail_candidate > open_sl:
                        open_sl = trail_candidate
            else:
                open_trough = min(open_trough, close)
                if not open_breakeven_moved and move_r >= float(cfg.get("breakeven_r", 1.0)):
                    buffer_pct = float(cfg.get("breakeven_buffer_pct", 0.0))
                    new_sl = open_trade.entry_price * (1 - buffer_pct / 100.0)
                    if open_sl <= 0 or new_sl < open_sl:
                        open_sl = new_sl
                        open_breakeven_moved = True
                if move_r >= float(cfg.get("trail_start_r", 1.5)) and atr > 0:
                    trail_candidate = close + atr * float(cfg.get("trail_atr_mult", 1.25))
                    if open_sl <= 0 or trail_candidate < open_sl:
                        open_sl = trail_candidate
            if open_trade.side == "LONG":
                hit_sl = low <= open_sl
                hit_tp = high >= open_tp
                if hit_sl and hit_tp:
                    exit_price = open_sl
                    reason = "SL"
                elif hit_sl:
                    exit_price = open_sl
                    reason = "SL"
                elif hit_tp:
                    exit_price = open_tp
                    reason = "TP"
                elif signal == "SHORT":
                    exit_price = close
                    reason = "REV"
                else:
                    exit_price = None
                    reason = ""
            else:
                hit_sl = high >= open_sl
                hit_tp = low <= open_tp
                if hit_sl and hit_tp:
                    exit_price = open_sl
                    reason = "SL"
                elif hit_sl:
                    exit_price = open_sl
                    reason = "SL"
                elif hit_tp:
                    exit_price = open_tp
                    reason = "TP"
                elif signal == "LONG":
                    exit_price = close
                    reason = "REV"
                else:
                    exit_price = None
                    reason = ""

            if exit_price is not None:
                pnl = _trade_pnl(open_trade.side, open_trade.entry_price, float(exit_price), open_qty)
                open_trade.exit_ts = int(row["ts"])
                open_trade.exit_price = float(exit_price)
                open_trade.pnl = pnl
                open_trade.reason = reason
                trades.append(open_trade)
                equity += pnl
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity * 100.0)
                open_trade = None
                open_qty = 0.0
                open_sl = 0.0
                open_tp = 0.0
                open_risk_usdt = 0.0
                open_peak = 0.0
                open_trough = 0.0
                open_breakeven_moved = False

        if open_trade is None and signal in {"LONG", "SHORT"} and float(next_row["open"]) > 0 and atr > 0:
            entry = float(next_row["open"])
            qty = (trade_usdt * float(cfg.get("leverage", 1))) / entry
            sl_dist = atr * float(cfg.get("atr_mult", 1.5))
            if signal == "LONG":
                sl = entry - sl_dist
                tp = entry + sl_dist * 2.0
            else:
                sl = entry + sl_dist
                tp = entry - sl_dist * 2.0
            open_trade = Trade(side=signal, signal_idx=i, entry_ts=int(next_row["ts"]), entry_price=entry)
            open_qty = qty
            open_sl = sl
            open_tp = tp
            open_risk_usdt = qty * sl_dist
            open_peak = entry
            open_trough = entry
            open_breakeven_moved = False

    if open_trade is not None:
        last_close = float(df.iloc[-1]["close"])
        pnl = _trade_pnl(open_trade.side, open_trade.entry_price, last_close, open_qty)
        open_trade.exit_ts = int(df.iloc[-1]["ts"])
        open_trade.exit_price = last_close
        open_trade.pnl = pnl
        open_trade.reason = "EOD"
        trades.append(open_trade)
        equity += pnl
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity * 100.0)
        open_risk_usdt = 0.0
        open_peak = 0.0
        open_trough = 0.0
        open_breakeven_moved = False

    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    profit_factor = (sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0))) if any(p < 0 for p in pnls) else float("inf") if any(p > 0 for p in pnls) else 0.0
    total = len(trades)
    win_rate = (wins / total * 100.0) if total else 0.0
    total_pnl = sum(pnls)
    return {
        "trades": [t.__dict__ for t in trades],
        "total": total,
        "pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def _windows(df: pd.DataFrame, folds: int) -> List[Tuple[int, int]]:
    if folds <= 1 or len(df) < 500:
        return [(0, len(df))]
    start = max(220, int(len(df) * 0.2))
    usable = len(df) - start
    if usable <= 0:
        return [(0, len(df))]
    step = max(1, usable // folds)
    windows = []
    for i in range(folds):
        s = start + i * step
        e = len(df) if i == folds - 1 else min(len(df), s + step)
        if s < e:
            windows.append((s, e))
    return windows or [(0, len(df))]


def compare_symbol(symbol: str, days: int, folds: int, trade_usdt: float) -> Dict[str, object]:
    symbol = bot_config.normalize_symbol(symbol)
    base_cfg = dict(bot_config.DEFAULT_CONFIG)
    base_cfg.update(
        {
            "timeframe": int(bot_config.DEFAULT_CONFIG.get("timeframe", 5)),
            "leverage": int(bot_config.DEFAULT_CONFIG.get("leverage", 2)),
            "ema_fast": 9,
            "ema_slow": 21,
            "ema_trend": 200,
            "rsi_period": 14,
            "adx_period": 14,
            "atr_period": 14,
            "atr_mult": 1.5,
            "supertrend_period": 14,
            "supertrend_mult": 3.5,
            "adx_threshold": 25,
            "volume_mult": 1.2,
            "rsi_min": 30,
            "rsi_max": 70,
            "ml_threshold": 0.55,
            "breakeven_r": 1.0,
            "breakeven_buffer_pct": 0.0,
            "trail_start_r": 1.5,
            "trail_atr_mult": 1.25,
        }
    )
    spec_cfg = multi_bot._get_symbol_config(symbol)
    spec_cfg["timeframe"] = int(spec_cfg.get("timeframe", 5))

    timeframe = int(spec_cfg.get("timeframe", 5))
    df = fetch_history(symbol, timeframe, days=days)
    windows = _windows(df, folds)

    baseline_runs = []
    specialized_runs = []
    for start, end in windows:
        window_df = df.iloc[start:end].reset_index(drop=True)
        baseline_runs.append(simulate(window_df, base_cfg, trade_usdt))
        specialized_runs.append(simulate(window_df, spec_cfg, trade_usdt))

    def agg(runs):
        pnl = sum(float(r["pnl"]) for r in runs)
        total = sum(int(r["total"]) for r in runs)
        wins = sum(int(r["wins"]) for r in runs)
        losses = sum(int(r["losses"]) for r in runs)
        win_rate = (wins / total * 100.0) if total else 0.0
        profit_factors = [float(r["profit_factor"]) for r in runs if r["profit_factor"] != float("inf")]
        max_drawdown = max((float(r["max_drawdown"]) for r in runs), default=0.0)
        return {
            "pnl": pnl,
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": (sum(profit_factors) / len(profit_factors)) if profit_factors else float("inf"),
            "max_drawdown": max_drawdown,
        }

    baseline = agg(baseline_runs)
    specialized = agg(specialized_runs)
    better = (
        specialized["pnl"] > baseline["pnl"]
        and specialized["win_rate"] >= baseline["win_rate"]
        and specialized["max_drawdown"] <= baseline["max_drawdown"] * 1.10 + 0.01
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "folds": folds,
        "baseline": baseline,
        "specialized": specialized,
        "better": better,
        "windows": [
            {
                "baseline": baseline_runs[idx],
                "specialized": specialized_runs[idx],
            }
            for idx in range(len(windows))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward comparison: baseline vs specialized per asset")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trade-usdt", type=float, default=50.0)
    parser.add_argument("--symbols", type=str, default="BTCUSDT")
    args = parser.parse_args()

    symbols = bot_config.parse_symbols(args.symbols)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "folds": args.folds,
        "trade_usdt": args.trade_usdt,
        "results": [],
    }

    overall_baseline_pnl = 0.0
    overall_specialized_pnl = 0.0
    overall_baseline_win_rate = []
    overall_specialized_win_rate = []
    approved_symbols = []

    for symbol in symbols:
        result = compare_symbol(symbol, args.days, args.folds, args.trade_usdt)
        report["results"].append(result)
        overall_baseline_pnl += float(result["baseline"]["pnl"])
        overall_specialized_pnl += float(result["specialized"]["pnl"])
        overall_baseline_win_rate.append(float(result["baseline"]["win_rate"]))
        overall_specialized_win_rate.append(float(result["specialized"]["win_rate"]))
        if result["better"]:
            approved_symbols.append(symbol)

    report["summary"] = {
        "baseline_pnl": overall_baseline_pnl,
        "specialized_pnl": overall_specialized_pnl,
        "baseline_avg_win_rate": sum(overall_baseline_win_rate) / len(overall_baseline_win_rate) if overall_baseline_win_rate else 0.0,
        "specialized_avg_win_rate": sum(overall_specialized_win_rate) / len(overall_specialized_win_rate) if overall_specialized_win_rate else 0.0,
        "approved_symbols": approved_symbols,
        "specialization_wins": len(approved_symbols),
        "specialization_losses": len(symbols) - len(approved_symbols),
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    for result in report["results"]:
        print(
            f"{result['symbol']}: baseline pnl={result['baseline']['pnl']:.2f}, "
            f"specialized pnl={result['specialized']['pnl']:.2f}, "
            f"baseline win={result['baseline']['win_rate']:.1f}%, "
            f"specialized win={result['specialized']['win_rate']:.1f}%, "
            f"better={result['better']}"
        )


if __name__ == "__main__":
    main()
