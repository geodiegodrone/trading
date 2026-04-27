import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

import activity_log
import binance_broker
import market_stream
from bot_config import DEFAULT_CONFIG, get_symbol_config, parse_symbols
import ml_model
import trade_log

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_bot")

SYMBOLS = parse_symbols()
PORTFOLIO_FILE = ROOT / "portfolio_state.json"
DAILY_FILE = ROOT / "daily_balance.json"
OPEN_TRADE_FILE = ROOT / "open_trade_id.json"

lock = threading.Lock()
stop_event = threading.Event()


def _load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path, payload):
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save %s: %s", path.name, exc)


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _get_symbol_config(symbol):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(get_symbol_config(symbol))
    cfg["timeframe"] = int(cfg.get("timeframe", 5))
    cfg["leverage"] = int(cfg.get("leverage", 2))
    cfg["ema_fast"] = int(cfg.get("ema_fast", 9))
    cfg["ema_slow"] = int(cfg.get("ema_slow", 21))
    cfg["ema_trend"] = int(cfg.get("ema_trend", 200))
    cfg["rsi_period"] = int(cfg.get("rsi_period", 14))
    cfg["adx_period"] = int(cfg.get("adx_period", 14))
    cfg["atr_period"] = int(cfg.get("atr_period", 14))
    cfg["atr_mult"] = float(cfg.get("atr_mult", 1.5))
    cfg["supertrend_period"] = int(cfg.get("supertrend_period", 14))
    cfg["supertrend_mult"] = float(cfg.get("supertrend_mult", 3.5))
    cfg["adx_threshold"] = float(cfg.get("adx_threshold", 25))
    cfg["volume_mult"] = float(cfg.get("volume_mult", 1.2))
    cfg["rsi_min"] = float(cfg.get("rsi_min", 30))
    cfg["rsi_max"] = float(cfg.get("rsi_max", 70))
    cfg["ml_threshold"] = float(cfg.get("ml_threshold", 0.55))
    cfg["daily_risk_cap"] = float(cfg.get("daily_risk_cap", 5.0))
    cfg["breakeven_r"] = float(cfg.get("breakeven_r", 1.0))
    cfg["breakeven_buffer_pct"] = float(cfg.get("breakeven_buffer_pct", 0.0))
    cfg["trail_start_r"] = float(cfg.get("trail_start_r", 1.5))
    cfg["trail_atr_mult"] = float(cfg.get("trail_atr_mult", 1.25))
    return cfg


def _fetch_candles(symbol, timeframe, limit=250):
    klines = market_stream.get_candles(symbol, timeframe, limit)
    if not klines:
        klines = binance_broker.get_kline(symbol, timeframe, limit)
    # Binance kline: [open_time, open, high, low, close, volume, ...]
    df = pd.DataFrame(klines, columns=["ts", "open", "high", "low", "close", "volume",
                                        "close_time", "quote_vol", "trades",
                                        "taker_base", "taker_quote", "ignore"])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    if df.empty:
        return df
    for col in ["ts", "open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _compute_indicators(df, cfg):
    if df.empty or len(df) < 220:
        return df
    fast = int(cfg["ema_fast"])
    slow = int(cfg["ema_slow"])
    trend = int(cfg["ema_trend"])
    rsi_period = int(cfg["rsi_period"])
    adx_period = int(cfg["adx_period"])
    st_period = int(cfg["supertrend_period"])
    st_mult = float(cfg["supertrend_mult"])
    df[f"ema{fast}"] = ta.ema(df["close"], length=fast)
    df[f"ema{slow}"] = ta.ema(df["close"], length=slow)
    df[f"ema{trend}"] = ta.ema(df["close"], length=trend)
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)
    adx = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
    adx_col = f"ADX_{adx_period}"
    if adx is not None and adx_col in adx:
        df["adx"] = adx[adx_col]
    st = ta.supertrend(df["high"], df["low"], df["close"], length=st_period, multiplier=st_mult)
    st_col = f"SUPERT_{st_period}_{st_mult}"
    st_dir_col = f"SUPERTd_{st_period}_{st_mult}"
    if st is not None and st_col in st and st_dir_col in st:
        df[st_col] = st[st_col]
        df[st_dir_col] = st[st_dir_col]
    df.attrs["indicator_cols"] = {
        "ema_fast": f"ema{fast}",
        "ema_slow": f"ema{slow}",
        "ema_trend": f"ema{trend}",
        "supertrend_dir": st_dir_col,
    }
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=cfg["atr_period"])
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def evaluate_signal(last, df, cfg):
    if last is None or df.empty:
        return "NEUTRAL"
    cols = df.attrs.get("indicator_cols", {})
    ema_fast = _as_float(last.get(cols.get("ema_fast", "ema9")))
    ema_slow = _as_float(last.get(cols.get("ema_slow", "ema21")))
    ema_trend = _as_float(last.get(cols.get("ema_trend", "ema200")))
    close = _as_float(last.get("close"))
    rsi = _as_float(last.get("rsi"))
    adx = _as_float(last.get("adx"))
    volume = _as_float(last.get("volume"))
    vol_ma20 = _as_float(last.get("vol_ma20"))
    supertrend_dir = int(_as_float(last.get(cols.get("supertrend_dir", "SUPERTd_14_3.5"))))

    long_ok = (
        ema_fast > ema_slow
        and close > ema_trend
        and supertrend_dir == 1
        and adx > float(cfg.get("adx_threshold", 25))
        and volume > (vol_ma20 * float(cfg.get("volume_mult", 1.2)) if vol_ma20 else 0)
        and float(cfg.get("rsi_min", 30)) < rsi < float(cfg.get("rsi_max", 70))
    )
    short_ok = (
        ema_fast < ema_slow
        and close < ema_trend
        and supertrend_dir == -1
        and adx > float(cfg.get("adx_threshold", 25))
        and volume > (vol_ma20 * float(cfg.get("volume_mult", 1.2)) if vol_ma20 else 0)
        and float(cfg.get("rsi_min", 30)) < rsi < float(cfg.get("rsi_max", 70))
    )
    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return "NEUTRAL"


def _load_daily_balance():
    data = _load_json(DAILY_FILE, {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("date") != today:
        bal = binance_broker.get_balance()
        data = {"date": today, "daily_start_balance": bal, "paused": False}
        _save_json(DAILY_FILE, data)
    elif _as_float(data.get("daily_start_balance"), 0.0) <= 0:
        data["daily_start_balance"] = binance_broker.get_balance()
        data.setdefault("paused", False)
        _save_json(DAILY_FILE, data)
    return data


def _daily_loss_triggered(balance):
    state = _load_daily_balance()
    start = _as_float(state.get("daily_start_balance"), balance)
    if start <= 0:
        return False
    if balance < start * 0.97:
        state["paused"] = True
        _save_json(DAILY_FILE, state)
        activity_log.push("PORTFOLIO", "risk", "Límite diario alcanzado — bot pausado hasta mañana UTC")
        return True
    return bool(state.get("paused"))


def _open_trade_registry():
    return _load_json(OPEN_TRADE_FILE, {})


def _save_trade_registry(data):
    _save_json(OPEN_TRADE_FILE, data)


def _portfolio_state():
    return _load_json(PORTFOLIO_FILE, {"open_positions": {}, "total_risk_pct": 0.0})


def _save_portfolio_state(state):
    _save_json(PORTFOLIO_FILE, state)


def _latest_open_trade(symbol):
    try:
        trades = trade_log.get_all_trades(symbol)
    except Exception:
        return None
    for trade in reversed(trades):
        if trade.get("pnl") is None and str(trade.get("result") or "").upper() == "OPEN":
            return trade
    return None


def _open_trade_rows(symbol):
    try:
        trades = trade_log.get_all_trades(symbol)
    except Exception:
        return []
    return [trade for trade in trades if trade.get("pnl") is None and str(trade.get("result") or "").upper() == "OPEN"]


def _local_position(symbol):
    try:
        state = _portfolio_state()
        return (state.get("open_positions", {}) or {}).get(symbol)
    except Exception:
        return None


def _position_from_trade_row(symbol, trade):
    if not trade:
        return None
    entry = _as_float(trade.get("entry_price"))
    size = _as_float(trade.get("qty"))
    if size <= 0 and entry > 0:
        notional = _as_float(trade.get("notional_usdt"))
        if notional > 0:
            size = notional / entry
    if entry <= 0 or size <= 0:
        return None
    side = str(trade.get("side") or "Buy")
    risk_usdt = _as_float(trade.get("risk_usdt"))
    stop_distance = risk_usdt / size if risk_usdt > 0 and size > 0 else max(entry * 0.005, 0.0)
    if side.lower() == "buy":
        sl = entry - stop_distance
        tp = entry + stop_distance * 2.0
    else:
        sl = entry + stop_distance
        tp = entry - stop_distance * 2.0
    opened_ts = trade.get("ts") or datetime.now(timezone.utc).isoformat()
    return {
        "side": side,
        "size": size,
        "entry_price": entry,
        "sl": sl,
        "tp": tp,
        "trade_id": trade.get("id"),
        "opened_ts": opened_ts,
        "atr_stop_distance": stop_distance,
        "risk_usdt": risk_usdt if risk_usdt > 0 else size * stop_distance,
        "peak_price": entry,
        "trough_price": entry,
        "breakeven_moved": False,
        "trail_stop": sl,
    }


def _register_live_position(symbol, live_pos, cfg, balance):
    price = _as_float(live_pos.get("entry_price"))
    if price <= 0:
        return None
    try:
        df = _compute_indicators(_fetch_candles(symbol, cfg["timeframe"]), cfg)
        if df.empty:
            return None
        last = df.iloc[-1].to_dict()
        cols = df.attrs.get("indicator_cols", {})
        side = live_pos.get("side", "Buy")
        qty = _position_size(live_pos)
        atr_estimate = _as_float(last.get("atr"))
        sl_dist = atr_estimate * float(cfg.get("atr_mult", 1.5)) if atr_estimate > 0 else max(price * 0.005, 0.0)
        if side.lower() == "buy":
            sl = price - sl_dist
            tp = price + sl_dist * 2.0
        else:
            sl = price + sl_dist
            tp = price - sl_dist * 2.0
        try:
            trade_id = trade_log.log_trade(
                side,
                price,
                _as_float(last.get(cols.get("ema_fast", "ema9"))),
                _as_float(last.get(cols.get("ema_slow", "ema21"))),
                _as_float(last.get("rsi")),
                _as_float(last.get(cols.get("ema_trend", "ema200"))),
                symbol=symbol,
                qty=qty,
                notional_usdt=qty * price,
                risk_usdt=qty * sl_dist,
            )
        except TypeError:
            trade_id = trade_log.log_trade(
                side,
                price,
                _as_float(last.get(cols.get("ema_fast", "ema9"))),
                _as_float(last.get(cols.get("ema_slow", "ema21"))),
                _as_float(last.get("rsi")),
                _as_float(last.get(cols.get("ema_trend", "ema200"))),
                symbol=symbol,
            )
        return {
            "side": side,
            "size": qty,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "trade_id": trade_id,
            "opened_ts": datetime.now(timezone.utc).isoformat(),
            "atr_stop_distance": sl_dist,
            "risk_usdt": qty * sl_dist,
            "peak_price": price,
            "trough_price": price,
            "breakeven_moved": False,
            "trail_stop": sl,
        }
    except Exception as exc:
        logger.warning("Failed to register live position for %s: %s", symbol, exc)
        return None


def _rebuild_position_from_exchange(symbol, cfg, balance):
    pos = _get_current_position(symbol)
    if not pos:
        return None
    trade = _latest_open_trade(symbol)
    price = _as_float(pos.get("entry_price")) or _as_float(trade.get("entry_price") if trade else 0.0)
    if price <= 0:
        return None
    atr_estimate = 0.0
    try:
        df = _compute_indicators(_fetch_candles(symbol, cfg["timeframe"]), cfg)
        if not df.empty:
            last = df.iloc[-1].to_dict()
            atr_estimate = _as_float(last.get("atr"))
    except Exception:
        atr_estimate = 0.0
    sl_dist = atr_estimate * float(cfg.get("atr_mult", 1.5)) if atr_estimate > 0 else max(price * 0.005, 0.0)
    side = pos.get("side", "Buy")
    if side.lower() == "buy":
        sl = price - sl_dist
        tp = price + sl_dist * 2.0
    else:
        sl = price + sl_dist
        tp = price - sl_dist * 2.0
    trade_id = trade.get("id") if trade else None
    state_entry = {
        "side": side,
        "size": _position_size(pos),
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "trade_id": trade_id,
        "opened_ts": trade.get("ts") if trade else datetime.now(timezone.utc).isoformat(),
        "atr_stop_distance": sl_dist,
        "risk_usdt": _position_size(pos) * sl_dist,
        "peak_price": price,
        "trough_price": price,
        "breakeven_moved": False,
        "trail_stop": sl,
    }
    return state_entry


def reconcile_state(force_refresh: bool = False):
    report = {
        "force_refresh": bool(force_refresh),
        "repairs": [],
        "before": {},
        "after": {},
        "counts": {},
    }
    try:
        balance = _as_float(binance_broker.get_balance(force_refresh=force_refresh))
        current = _portfolio_state()
        current_positions = current.get("open_positions", {}) if isinstance(current, dict) else {}
        reconciled_positions = {}
        reconciled_registry = {}
        before_open_rows = 0
        for symbol in SYMBOLS:
            live_pos = _get_current_position(symbol)
            open_rows = _open_trade_rows(symbol)
            before_open_rows += len(open_rows)
            local_pos = current_positions.get(symbol)
            registry_trade_id = _open_trade_registry().get(symbol)
            open_rows_sorted = sorted(open_rows, key=lambda row: int(row.get("id") or 0))
            if live_pos:
                rebuilt = _rebuild_position_from_exchange(symbol, _get_symbol_config(symbol), balance)
                active_trade_id = None
                if rebuilt:
                    active_trade_id = rebuilt.get("trade_id")
                    if (
                        active_trade_id is None
                        and not open_rows_sorted
                        and (local_pos is None or local_pos.get("trade_id") is None)
                        and registry_trade_id is None
                    ):
                        synthetic = _register_live_position(symbol, live_pos, _get_symbol_config(symbol), balance)
                        if synthetic:
                            rebuilt = synthetic
                            active_trade_id = synthetic.get("trade_id")
                            report["repairs"].append({
                                "symbol": symbol,
                                "action": "register_live_position",
                                "detail": "Creado registro local y registry desde posicion viva de Binance",
                            })
                    reconciled_positions[symbol] = rebuilt
                    trade_id = rebuilt.get("trade_id")
                    if trade_id is not None:
                        reconciled_registry[symbol] = trade_id
                        if registry_trade_id is None or int(registry_trade_id) != int(trade_id):
                            report["repairs"].append({
                                "symbol": symbol,
                                "action": "repair_registry",
                                "detail": f"open_trade_id sincronizado con trade {trade_id}",
                            })
                else:
                    if (
                        not open_rows_sorted
                        and (local_pos is None or local_pos.get("trade_id") is None)
                        and registry_trade_id is None
                    ):
                        synthetic = _register_live_position(symbol, live_pos, _get_symbol_config(symbol), balance)
                        if synthetic:
                            reconciled_positions[symbol] = synthetic
                            reconciled_registry[symbol] = synthetic.get("trade_id")
                            active_trade_id = synthetic.get("trade_id")
                            report["repairs"].append({
                                "symbol": symbol,
                                "action": "register_live_position",
                                "detail": "Reconstruido registro desde posicion viva sin metadata local",
                            })
                    if active_trade_id is None and local_pos and local_pos.get("trade_id") is not None:
                        active_trade_id = int(local_pos.get("trade_id"))
                    if active_trade_id is None and registry_trade_id is not None:
                        active_trade_id = int(registry_trade_id)
                for row in open_rows:
                    row_id = row.get("id")
                    if row_id is None:
                        continue
                    if active_trade_id is not None and int(row_id) == int(active_trade_id):
                        continue
                    try:
                        trade_log.reconcile_trade(int(row_id), float(row.get("entry_price") or 0.0))
                        report["repairs"].append({
                            "symbol": symbol,
                            "action": "close_orphan_trade_row",
                            "trade_id": int(row_id),
                            "detail": "Fila OPEN huérfana cerrada como reconciliada",
                        })
                    except Exception as exc:
                        logger.warning("Failed to reconcile orphan trade %s for %s: %s", row_id, symbol, exc)
                if live_pos and not rebuilt:
                    reconciled_positions[symbol] = {
                        "side": live_pos.get("side"),
                        "size": _position_size(live_pos),
                        "entry_price": _as_float(live_pos.get("entry_price")),
                        "sl": _as_float(live_pos.get("entry_price")),
                        "tp": _as_float(live_pos.get("entry_price")),
                        "trade_id": None,
                        "opened_ts": datetime.now(timezone.utc).isoformat(),
                        "atr_stop_distance": 0.0,
                        "risk_usdt": 0.0,
                        "peak_price": _as_float(live_pos.get("entry_price")),
                        "trough_price": _as_float(live_pos.get("entry_price")),
                        "breakeven_moved": False,
                        "trail_stop": _as_float(live_pos.get("entry_price")),
                    }
            else:
                active_trade_id = None
                recovered = None
                if local_pos:
                    recovered = dict(local_pos)
                    active_trade_id = int(recovered.get("trade_id")) if recovered.get("trade_id") is not None else None
                if active_trade_id is None and registry_trade_id is not None:
                    active_trade_id = int(registry_trade_id)
                if recovered is None and open_rows_sorted:
                    recovered = _position_from_trade_row(symbol, open_rows_sorted[-1])
                    if active_trade_id is None and open_rows_sorted[-1].get("id") is not None:
                        active_trade_id = int(open_rows_sorted[-1]["id"])
                if recovered:
                    if active_trade_id is not None:
                        recovered["trade_id"] = int(active_trade_id)
                        reconciled_registry[symbol] = int(active_trade_id)
                    elif recovered.get("trade_id") is not None:
                        reconciled_registry[symbol] = int(recovered["trade_id"])
                    reconciled_positions[symbol] = recovered
                    report["repairs"].append({
                        "symbol": symbol,
                        "action": "recover_local_position",
                        "trade_id": int(active_trade_id) if active_trade_id is not None else None,
                        "detail": "Estado local reconstruido desde DB/registry",
                    })
                for row in open_rows_sorted:
                    row_id = row.get("id")
                    if row_id is None:
                        continue
                    if active_trade_id is not None and int(row_id) == int(active_trade_id):
                        continue
                    try:
                        trade_log.reconcile_trade(int(row_id), float(row.get("entry_price") or 0.0))
                        report["repairs"].append({
                            "symbol": symbol,
                            "action": "close_stale_trade_row",
                            "trade_id": int(row_id),
                            "detail": "Fila OPEN stale cerrada como reconciliada",
                        })
                    except Exception as exc:
                        logger.warning("Failed to reconcile stale trade %s for %s: %s", row_id, symbol, exc)
        if not reconciled_positions and not reconciled_registry and not current_positions:
            report["counts"] = {
                "live_positions": 0,
                "registry_entries": 0,
                "open_trade_rows_before": before_open_rows,
                "open_trade_rows_after": before_open_rows,
            }
            report["after"] = {"open_positions": {}, "total_risk_pct": 0.0}
            return report
        state = _calc_portfolio_risk(balance, {"open_positions": reconciled_positions, "total_risk_pct": 0.0})
        _save_portfolio_state(state)
        _save_trade_registry(reconciled_registry)
        report["after"] = state
        report["counts"] = {
            "live_positions": len(reconciled_positions),
            "registry_entries": len(reconciled_registry),
            "open_trade_rows_before": before_open_rows,
            "open_trade_rows_after": len([t for t in trade_log.get_all_trades() if t.get("pnl") is None]),
        }
        logger.info("State reconciled: %s live positions, %s registry entries", len(reconciled_positions), len(reconciled_registry))
        return report
    except Exception as exc:
        logger.warning("State reconciliation failed: %s", exc)
        report["error"] = str(exc)
        return report


def repair_state(force_refresh: bool = True):
    report = reconcile_state(force_refresh=force_refresh)
    try:
        daily = _load_daily_balance()
        report.setdefault("repairs", []).append({
            "symbol": "PORTFOLIO",
            "action": "refresh_daily_balance",
            "detail": f"daily_balance sincronizado para {daily.get('date')}",
        })
        report["daily"] = daily
    except Exception as exc:
        report["repairs"].append({
            "symbol": "PORTFOLIO",
            "action": "refresh_daily_balance_failed",
            "detail": str(exc),
        })
    return report


def _calc_portfolio_risk(balance, state):
    total = 0.0
    for pos in state.get("open_positions", {}).values():
        try:
            qty = abs(float(pos.get("size", 0.0)))
            entry = float(pos.get("entry_price", 0.0))
            atr_stop = float(pos.get("atr_stop_distance", 0.0))
            if balance > 0:
                total += (qty * entry * atr_stop) / balance * 100.0
        except Exception:
            continue
    state["total_risk_pct"] = round(total, 4)
    return state


def _kelly_trade_usdt(balance):
    try:
        stats = trade_log.get_stats()
        total = int(stats.get("total", 0))
        if total < 20:
            return float(os.getenv("TRADE_USDT", "50"))
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        if total <= 0 or wins <= 0 or losses <= 0:
            return float(os.getenv("TRADE_USDT", "50"))
        trades = trade_log.get_all_trades()
        pnl_values = [float(t.get("pnl", 0.0)) for t in trades]
        positive = [p for p in pnl_values if p > 0]
        negative = [p for p in pnl_values if p < 0]
        if not positive or not negative:
            return float(os.getenv("TRADE_USDT", "50"))
        win_rate = wins / total
        avg_win = sum(positive) / len(positive)
        avg_loss = abs(sum(negative) / len(negative))
        if avg_loss <= 0:
            return float(os.getenv("TRADE_USDT", "50"))
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        half_kelly = max(kelly / 2.0, 0.02)
        return float(balance) * min(half_kelly, 0.10)
    except Exception:
        return float(os.getenv("TRADE_USDT", "50"))


def _ml_confidence(features, symbol):
    try:
        if ml_model.is_ready(symbol):
            return float(ml_model.predict(features, symbol))
    except TypeError:
        try:
            if ml_model.is_ready():
                return float(ml_model.predict(features))
        except Exception:
            pass
    except Exception:
        pass
    return 0.5


def _retrain_if_ready(symbol):
    try:
        stats = trade_log.get_stats(symbol)
        trades = trade_log.get_all_trades(symbol)
        try:
            ml_model.train(trades, symbol)
        except TypeError:
            ml_model.train(trades)
        activity_log.push(symbol, "ml", f"ML reentrenado con {int(stats.get('total', 0))} trades cerrados. Win rate: {float(stats.get('win_rate', 0.0)):.1f}%")
    except Exception as exc:
        logger.warning("ML retrain failed for %s: %s", symbol, exc)


def _position_side(position):
    if not position:
        return None
    return position.get("side") or position.get("positionIdx")


def _position_size(position):
    if not position:
        return 0.0
    return abs(_as_float(position.get("size") or position.get("qty")))


def _position_pnl(position):
    if not position:
        return 0.0
    return _as_float(position.get("unrealisedPnl") or position.get("unrealised_pnl"))


def _position_risk_usdt(position):
    if not position:
        return 0.0
    risk_usdt = _as_float(position.get("risk_usdt"))
    if risk_usdt > 0:
        return risk_usdt
    size = _position_size(position)
    stop_distance = _as_float(position.get("atr_stop_distance"))
    if size > 0 and stop_distance > 0:
        return abs(size * stop_distance)
    return 0.0


def _position_move_r(position, price):
    if not position:
        return 0.0
    entry = _as_float(position.get("entry_price"))
    size = _position_size(position)
    risk_usdt = _position_risk_usdt(position)
    if entry <= 0 or size <= 0 or risk_usdt <= 0:
        return 0.0
    side = str(position.get("side", "Buy")).lower()
    pnl = (price - entry) * size if side == "buy" else (entry - price) * size
    return pnl / risk_usdt


def _get_current_position(symbol):
    try:
        return binance_broker.get_position(symbol)
    except Exception:
        return _local_position(symbol)


def _close_trade(symbol, position, exit_price, reason="EXIT"):
    side = position.get("side", "Buy")
    entry = _as_float(position.get("entry_price"))
    size = _position_size(position)
    pnl = _position_pnl(position)
    risk_usdt = _position_risk_usdt(position)
    if pnl == 0.0:
        if side.lower() == "buy":
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size
    trade_id = position.get("trade_id")
    if trade_id is not None:
        try:
            trade_log.close_trade(trade_id, exit_price, pnl, risk_usdt=risk_usdt, exit_reason=reason)
        except TypeError:
            trade_log.close_trade(trade_id, exit_price, pnl)
    try:
        binance_broker.close_position(symbol, position)
    except Exception as exc:
        logger.warning("close_position failed for %s: %s", symbol, exc)
    r_multiple = (pnl / risk_usdt) if risk_usdt > 0 else 0.0
    activity_log.push(symbol, "trade_close", f'CERRÓ {side} ({reason}) | PnL={pnl:+.2f} USDT | R={r_multiple:+.2f}')
    threading.Thread(target=_retrain_if_ready, args=(symbol,), daemon=True).start()


def _open_trade(symbol, signal, last, df, cfg, balance, trade_usdt):
    price = _as_float(last["close"])
    atr = _as_float(last.get("atr"))
    if price <= 0 or atr <= 0:
        return
    side = "Buy" if signal == "LONG" else "Sell"
    qty = binance_broker.normalize_quantity(symbol, (trade_usdt * cfg["leverage"]) / price)
    if qty <= 0:
        logger.warning("Skipping %s entry because normalized quantity is too small", symbol)
        return
    sl_dist = atr * cfg["atr_mult"]
    if side == "Buy":
        sl = price - sl_dist
        tp = price + sl_dist * 2.0
        try:
            binance_broker.open_long(symbol, qty)
        except Exception as exc:
            logger.warning("open_long failed for %s: %s", symbol, exc)
            return
    else:
        sl = price + sl_dist
        tp = price - sl_dist * 2.0
        try:
            binance_broker.open_short(symbol, qty)
        except Exception as exc:
            logger.warning("open_short failed for %s: %s", symbol, exc)
            return
    cols = df.attrs.get("indicator_cols", {})
    trade_id = None
    try:
        trade_id = trade_log.log_trade(
            side,
            price,
            _as_float(last.get(cols.get("ema_fast", "ema9"))),
            _as_float(last.get(cols.get("ema_slow", "ema21"))),
            _as_float(last.get("rsi")),
            _as_float(last.get(cols.get("ema_trend", "ema200"))),
            symbol=symbol,
            qty=qty,
            notional_usdt=qty * price,
            risk_usdt=qty * sl_dist,
        )
    except TypeError:
        trade_id = trade_log.log_trade(
            side,
            price,
            _as_float(last.get(cols.get("ema_fast", "ema9"))),
            _as_float(last.get(cols.get("ema_slow", "ema21"))),
            _as_float(last.get("rsi")),
            _as_float(last.get(cols.get("ema_trend", "ema200"))),
            symbol=symbol,
        )
    state = _portfolio_state()
    state.setdefault("open_positions", {})[symbol] = {
        "side": side,
        "size": qty,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "trade_id": trade_id,
        "opened_ts": datetime.now(timezone.utc).isoformat(),
        "atr_stop_distance": sl_dist,
        "risk_usdt": qty * sl_dist,
        "peak_price": price,
        "trough_price": price,
        "breakeven_moved": False,
        "trail_stop": sl,
    }
    _save_portfolio_state(_calc_portfolio_risk(balance, state))
    try:
        _save_trade_registry({**_open_trade_registry(), symbol: trade_id})
    except Exception:
        pass
    activity_log.push(symbol, "trade_open", f'ABRIÓ {side} @ ${price:,.2f} | SL=${sl:,.2f} TP=${tp:,.2f} | qty={qty:.6f}')


def _update_position(symbol, last, signal, cfg):
    state = _portfolio_state()
    position = state.get("open_positions", {}).get(symbol)
    if not position:
        return
    price = _as_float(last["close"])
    atr = _as_float(last.get("atr"))
    side = position.get("side", "Buy")
    sl = _as_float(position.get("sl"))
    tp = _as_float(position.get("tp"))
    entry = _as_float(position.get("entry_price"))
    risk_usdt = _position_risk_usdt(position)
    move_r = _position_move_r(position, price)
    hit_exit = False
    reason = "EXIT"
    if side.lower() == "buy":
        position["peak_price"] = max(_as_float(position.get("peak_price"), price), price)
        if risk_usdt > 0 and not position.get("breakeven_moved") and move_r >= float(cfg.get("breakeven_r", 1.0)):
            buffer_pct = float(cfg.get("breakeven_buffer_pct", 0.0))
            new_sl = entry * (1 + buffer_pct / 100.0)
            if new_sl > sl:
                sl = new_sl
                position["sl"] = sl
                position["breakeven_moved"] = True
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Stop moved to breakeven @ {sl:,.2f} (R={move_r:.2f})")
        if risk_usdt > 0 and atr > 0 and move_r >= float(cfg.get("trail_start_r", 1.5)):
            trail_candidate = price - atr * float(cfg.get("trail_atr_mult", 1.25))
            if trail_candidate > sl:
                sl = trail_candidate
                position["sl"] = sl
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Trailing stop updated @ {sl:,.2f} (R={move_r:.2f})")
        if price <= sl or price >= tp or signal == "SHORT":
            hit_exit = True
            reason = "SL" if price <= sl else "TP" if price >= tp else "REV"
    else:
        position["trough_price"] = min(_as_float(position.get("trough_price"), price), price)
        if risk_usdt > 0 and not position.get("breakeven_moved") and move_r >= float(cfg.get("breakeven_r", 1.0)):
            buffer_pct = float(cfg.get("breakeven_buffer_pct", 0.0))
            new_sl = entry * (1 - buffer_pct / 100.0)
            if sl <= 0 or new_sl < sl:
                sl = new_sl
                position["sl"] = sl
                position["breakeven_moved"] = True
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Stop moved to breakeven @ {sl:,.2f} (R={move_r:.2f})")
        if risk_usdt > 0 and atr > 0 and move_r >= float(cfg.get("trail_start_r", 1.5)):
            trail_candidate = price + atr * float(cfg.get("trail_atr_mult", 1.25))
            if sl <= 0 or trail_candidate < sl:
                sl = trail_candidate
                position["sl"] = sl
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Trailing stop updated @ {sl:,.2f} (R={move_r:.2f})")
        if price >= sl or price <= tp or signal == "LONG":
            hit_exit = True
            reason = "SL" if price >= sl else "TP" if price <= tp else "REV"
    if hit_exit:
        _close_trade(symbol, position, price, reason=reason)
        state["open_positions"].pop(symbol, None)
        _save_portfolio_state(state)
        registry = _open_trade_registry()
        registry.pop(symbol, None)
        _save_trade_registry(registry)
    else:
        state["open_positions"][symbol] = position
        _save_portfolio_state(_calc_portfolio_risk(_as_float(binance_broker.get_balance()), state))


def _market_paused(balance):
    state = _load_daily_balance()
    if state.get("paused"):
        if datetime.now(timezone.utc).strftime("%Y-%m-%d") != state.get("date"):
            _load_daily_balance()
            return False
        return True
    return _daily_loss_triggered(balance)


def run_symbol(symbol, cfg, shared_stop_event):
    if not binance_broker.is_symbol_trading(symbol):
        logger.warning("Skipping %s because Binance reports it as not trading", symbol)
        return
    try:
        binance_broker.set_leverage(symbol, cfg["leverage"])
    except Exception:
        pass
    while not shared_stop_event.is_set():
        try:
            balance = _as_float(binance_broker.get_balance())
            if _market_paused(balance):
                time.sleep(5)
                continue
            df = _fetch_candles(symbol, cfg["timeframe"])
            df = _compute_indicators(df, cfg)
            if df.empty or len(df) < 50:
                time.sleep(5)
                continue
            last = df.iloc[-1].to_dict()
            signal = evaluate_signal(last, df, cfg)
            cols = df.attrs.get("indicator_cols", {})
            ema_fast = _as_float(last.get(cols.get("ema_fast", "ema9")))
            ema_slow = _as_float(last.get(cols.get("ema_slow", "ema21")))
            ema_trend = _as_float(last.get(cols.get("ema_trend", "ema200")))
            rsi = _as_float(last.get("rsi"))
            adx = _as_float(last.get("adx"))
            activity_log.push(symbol, "signal", f"Se?al: {signal} | EMA{cfg['ema_fast']}={ema_fast:.1f} EMA{cfg['ema_slow']}={ema_slow:.1f} RSI={rsi:.1f} ADX={adx:.1f}")
            if adx < float(cfg.get("adx_threshold", 25)):
                activity_log.push(symbol, "regime", f"Mercado lateral (ADX={adx:.1f}<{float(cfg.get('adx_threshold', 25)):.1f}) - sin operar")
                _update_position(symbol, last, signal, cfg)
                time.sleep(5)
                continue
            features = {
                "ema9_minus_ema21_pct": (ema_fast - ema_slow) / ema_slow * 100 if ema_slow else 0.0,
                "rsi": rsi,
                "price_vs_ema200_pct": ((_as_float(last.get("close")) - ema_trend) / ema_trend * 100) if ema_trend else 0.0,
                "candle_body_pct": abs(_as_float(last.get("close")) - _as_float(last.get("open"))) / _as_float(last.get("open")) * 100 if _as_float(last.get("open")) else 0.0,
                "adx": adx,
                "supertrend_direction": int(_as_float(last.get(cols.get("supertrend_dir", "SUPERTd_14_3.5")))),
                "volume_vs_ma20_ratio": _as_float(last.get("volume")) / _as_float(last.get("vol_ma20")) if _as_float(last.get("vol_ma20")) else 0.0,
                "atr_pct": _as_float(last.get("atr")) / _as_float(last.get("close")) * 100 if _as_float(last.get("close")) else 0.0,
            }
            confidence = _ml_confidence(features, symbol)
            ml_threshold = float(cfg.get("ml_threshold", 0.55))
            activity_log.push(symbol, "ml", f"ML confianza: {confidence:.0%} {'se?al aceptada' if confidence >= ml_threshold else 'se?al rechazada'}")
            position = _get_current_position(symbol)
            state = _portfolio_state()
            state = _calc_portfolio_risk(balance, state)
            _save_portfolio_state(state)
            if position:
                _update_position(symbol, last, signal, cfg)
                time.sleep(5)
                continue
            if signal == "NEUTRAL":
                time.sleep(5)
                continue
            if confidence < ml_threshold and ml_model.is_ready(symbol):
                time.sleep(5)
                continue
            trade_usdt = _kelly_trade_usdt(balance)
            price = _as_float(last.get("close"))
            atr = _as_float(last.get("atr"))
            qty_preview = binance_broker.normalize_quantity(symbol, (trade_usdt * cfg["leverage"]) / price) if price > 0 else 0.0
            risk_pct = (qty_preview * price * (atr * cfg["atr_mult"])) / balance * 100 if balance > 0 else 0.0
            projected = state.get("total_risk_pct", 0.0) + risk_pct
            if projected > float(cfg.get("daily_risk_cap", 5.0)):
                activity_log.push("PORTFOLIO", "risk", "Límite de riesgo del portafolio (5%) alcanzado — nueva posición bloqueada")
                time.sleep(5)
                continue
            _open_trade(symbol, signal, last, df, cfg, balance, trade_usdt)
        except Exception as exc:
            logger.exception("Symbol loop error for %s", symbol)
            activity_log.push(symbol, "error", f"Error: {exc}")
        time.sleep(5)


def main():
    market_stream.start([(symbol, _get_symbol_config(symbol)["timeframe"]) for symbol in SYMBOLS])
    reconcile_state()
    threads = []
    for symbol in SYMBOLS:
        cfg = _get_symbol_config(symbol)
        thread = threading.Thread(target=run_symbol, args=(symbol, cfg, stop_event), daemon=True)
        thread.start()
        threads.append(thread)
        logger.info("Started thread for %s", symbol)
    try:
        last_reconcile = time.time()
        while True:
            if time.time() - last_reconcile >= 60:
                reconcile_state()
                last_reconcile = time.time()
            for thread in threads:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
        logger.info("Stopping all symbol threads...")
        for thread in threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
