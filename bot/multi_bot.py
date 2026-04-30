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
import circuit_breaker
import market_stream
from bot_config import DEFAULT_CONFIG, get_symbol_config, parse_symbols
from features import FEATURE_COLUMNS, build_features
import ml_model
import regime as regime_detector
from strategies import enrich_indicators, signal_breakout, signal_meanrev, signal_trend, volume_filter_passes
import trade_log

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_bot")

SYMBOLS = ["BTCUSDT"]
PORTFOLIO_FILE = ROOT / "portfolio_state.json"
DAILY_FILE = ROOT / "daily_balance.json"
WEEKLY_FILE = ROOT / "weekly_balance.json"
OPEN_TRADE_FILE = ROOT / "open_trade_id.json"

lock = threading.Lock()
stop_event = threading.Event()
ML_RETRAIN_EVERY = max(1, int(os.getenv("ML_RETRAIN_EVERY", "10")))
ML_THRESHOLD_OVERRIDE = os.getenv("ML_THRESHOLD_OVERRIDE")
_ML_NOT_READY_STATE = {}
_LAST_DECISION_LOG = {}
_ML_BOOTSTRAP_STATE = {"thread": None, "running": False, "last_started": 0.0, "last_completed": 0.0}
_ML_RETRAIN_FAILURES = {}


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
    cfg["strategy_mode"] = str(cfg.get("strategy_mode", "regime")).lower()
    cfg["primary_timeframe"] = int(cfg.get("primary_timeframe", cfg.get("timeframe", 60)))
    cfg["confirmation_timeframe"] = int(cfg.get("confirmation_timeframe", 240))
    cfg["timeframe"] = int(cfg.get("timeframe", cfg["primary_timeframe"]))
    cfg["leverage"] = int(cfg.get("leverage", 2))
    cfg["ema_fast"] = int(cfg.get("ema_fast", 9))
    cfg["ema_slow"] = int(cfg.get("ema_slow", 21))
    cfg["ema_confirm"] = int(cfg.get("ema_confirm", 50))
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
    cfg["ml_auto_bootstrap_days"] = int(cfg.get("ml_auto_bootstrap_days", 365))
    cfg["ml_watchdog_hours"] = float(cfg.get("ml_watchdog_hours", 24.0))
    cfg["ml_force_bootstrap_after_failures"] = int(cfg.get("ml_force_bootstrap_after_failures", 3))
    cfg["daily_risk_cap"] = float(cfg.get("daily_risk_cap", 5.0))
    cfg["breakeven_r"] = float(cfg.get("breakeven_r", 1.0))
    cfg["breakeven_buffer_pct"] = float(cfg.get("breakeven_buffer_pct", 0.0))
    cfg["trail_start_r"] = float(cfg.get("trail_start_r", 1.5))
    cfg["trail_atr_mult"] = float(cfg.get("trail_atr_mult", 1.25))
    cfg["circuit_breaker_enabled"] = bool(cfg.get("circuit_breaker_enabled", True))
    cfg["cb_daily_loss_pct"] = float(cfg.get("cb_daily_loss_pct", 3.0))
    cfg["cb_weekly_loss_pct"] = float(cfg.get("cb_weekly_loss_pct", 7.0))
    cfg["cb_consecutive_losses"] = int(cfg.get("cb_consecutive_losses", 4))
    cfg["cb_rolling_window"] = int(cfg.get("cb_rolling_window", 20))
    cfg["cb_rolling_winrate_min"] = float(cfg.get("cb_rolling_winrate_min", 0.30))
    cfg["cb_volatility_spike_pct"] = float(cfg.get("cb_volatility_spike_pct", 8.0))
    cfg["cb_cooldown_hours"] = float(cfg.get("cb_cooldown_hours", 12.0))
    cfg["meanrev_atr_mult"] = float(cfg.get("meanrev_atr_mult", 0.5))
    cfg["meanrev_tp_at_sma"] = bool(cfg.get("meanrev_tp_at_sma", True))
    cfg["breakout_atr_mult"] = float(cfg.get("breakout_atr_mult", 1.0))
    cfg["breakout_tp_atr_mult"] = float(cfg.get("breakout_tp_atr_mult", 3.0))
    cfg["volatile_atr_mult"] = float(cfg.get("volatile_atr_mult", 2.0))
    cfg["adx_threshold_1h"] = float(cfg.get("adx_threshold_1h", 18.0))
    cfg["volume_mult_1h"] = float(cfg.get("volume_mult_1h", 1.0))
    return cfg


def _fetch_candles(symbol, timeframe, limit=250):
    klines = market_stream.get_candles(symbol, timeframe, limit)
    if not klines:
        klines = binance_broker.get_kline(symbol, timeframe, limit)
    # Binance kline: [open_time, open, high, low, close, volume, ...]
    df = pd.DataFrame(
        klines,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_vol",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    df = df[["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]]
    if df.empty:
        return df
    for col in ["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _compute_indicators(df, cfg):
    if df.empty or len(df) < 220:
        return df
    fast = int(cfg["ema_fast"])
    slow = int(cfg["ema_slow"])
    confirm = int(cfg.get("ema_confirm", 50))
    trend = int(cfg["ema_trend"])
    rsi_period = int(cfg["rsi_period"])
    adx_period = int(cfg["adx_period"])
    st_period = int(cfg["supertrend_period"])
    st_mult = float(cfg["supertrend_mult"])
    df[f"ema{fast}"] = ta.ema(df["close"], length=fast)
    df[f"ema{slow}"] = ta.ema(df["close"], length=slow)
    df[f"ema{confirm}"] = ta.ema(df["close"], length=confirm)
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
    return enrich_indicators(df)


def evaluate_signal(last, df, cfg):
    return signal_trend(last, df, cfg)


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


def _load_weekly_balance():
    data = _load_json(WEEKLY_FILE, {})
    now = datetime.now(timezone.utc)
    year, week, _ = now.isocalendar()
    week_key = f"{year}-W{week:02d}"
    if data.get("week") != week_key:
        bal = binance_broker.get_balance()
        data = {"week": week_key, "weekly_start_balance": bal}
        _save_json(WEEKLY_FILE, data)
    elif _as_float(data.get("weekly_start_balance"), 0.0) <= 0:
        data["weekly_start_balance"] = binance_broker.get_balance()
        _save_json(WEEKLY_FILE, data)
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
        feature_context = build_features(df).iloc[-1].to_dict()
        trade_log.set_trade_context(feature_context.get("candle_body_pct", 0.0))
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
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_fast", "ema9"))),
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_slow", "ema21"))),
                _as_float(last.get("rsi")),
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_trend", "ema200"))),
                symbol=symbol,
                qty=qty,
                notional_usdt=qty * price,
                risk_usdt=qty * sl_dist,
                return_3_pct=feature_context.get("return_3_pct", 0.0),
                return_5_pct=feature_context.get("return_5_pct", 0.0),
                range_5_pct=feature_context.get("range_5_pct", 0.0),
                body_avg_3_pct=feature_context.get("body_avg_3_pct", 0.0),
                volume_trend_3_ratio=feature_context.get("volume_trend_3_ratio", 0.0),
                close_pos_5=feature_context.get("close_pos_5", 0.0),
                ema9_slope_3_pct=feature_context.get("ema9_slope_3_pct", 0.0),
            )
        except TypeError:
            trade_id = trade_log.log_trade(
                side,
                price,
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_fast", "ema9"))),
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_slow", "ema21"))),
                _as_float(last.get("rsi")),
                _as_float(last.get(df.attrs.get("indicator_cols", {}).get("ema_trend", "ema200"))),
                symbol=symbol,
                return_3_pct=feature_context.get("return_3_pct", 0.0),
                return_5_pct=feature_context.get("return_5_pct", 0.0),
                range_5_pct=feature_context.get("range_5_pct", 0.0),
                body_avg_3_pct=feature_context.get("body_avg_3_pct", 0.0),
                volume_trend_3_ratio=feature_context.get("volume_trend_3_ratio", 0.0),
                close_pos_5=feature_context.get("close_pos_5", 0.0),
                ema9_slope_3_pct=feature_context.get("ema9_slope_3_pct", 0.0),
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


def _kelly_trade_usdt(balance, price, stop_distance, leverage):
    base_trade = float(os.getenv("TRADE_USDT", "50"))
    try:
        stats = trade_log.get_stats()
        total = int(stats.get("total", 0))
        if total < 20:
            half_kelly = 0.02
        else:
            wins = int(stats.get("wins", 0))
            losses = int(stats.get("losses", 0))
            if total <= 0 or wins <= 0 or losses <= 0:
                half_kelly = 0.02
            else:
                trades = trade_log.get_all_trades()
                pnl_values = [float(t.get("pnl", 0.0)) for t in trades if t.get("pnl") is not None]
                positive = [p for p in pnl_values if p > 0]
                negative = [p for p in pnl_values if p < 0]
                if not positive or not negative:
                    half_kelly = 0.02
                else:
                    win_rate = wins / total
                    avg_win = sum(positive) / len(positive)
                    avg_loss = abs(sum(negative) / len(negative))
                    if avg_loss <= 0:
                        half_kelly = 0.02
                    else:
                        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                        half_kelly = max(kelly / 2.0, 0.01)
        half_kelly = min(half_kelly, 0.05)
        balance = float(balance)
        risk_target_usdt = balance * 0.005
        if price > 0 and stop_distance > 0 and leverage > 0:
            vol_target_trade = risk_target_usdt * price / (leverage * stop_distance)
        else:
            vol_target_trade = base_trade
        kelly_trade = balance * half_kelly
        trade_usdt = max(15.0, min(vol_target_trade, kelly_trade))
        status = circuit_breaker.get_status()
        last_resume = status.get("last_resume_at")
        if last_resume:
            try:
                resumed = datetime.fromisoformat(str(last_resume).replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - resumed).total_seconds() <= 48 * 3600:
                    trade_usdt *= 0.5
            except Exception:
                pass
        return max(10.0, trade_usdt)
    except Exception:
        return base_trade


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


def _volume_regime_name(strategy_kind, regime_name):
    strategy = str(strategy_kind or "").lower()
    regime = str(regime_name or "").upper()
    if strategy == "meanrev":
        return "MEANREV"
    if strategy == "breakout":
        return "VOLATILE" if regime == "VOLATILE" else "BREAKOUT"
    if regime in {"TREND", "TRENDING"}:
        return "TRENDING"
    if regime == "RANGE":
        return "MEANREV"
    return regime or "DEFAULT"


def _primary_only_active(ml_ready, ml_info):
    return not bool(ml_ready)


def _primary_only_trade_usdt(trade_usdt, ml_ready):
    return float(trade_usdt) * 0.5 if not bool(ml_ready) else float(trade_usdt)


def _loop_context_hash(ctx):
    return "|".join(
        [
            str(ctx.get("regime") or "N/A"),
            str(ctx.get("strategy_used") or "none"),
            str(ctx.get("signal") or "NEUTRAL"),
            str(ctx.get("ml_ready")),
            f"{_as_float(ctx.get('ml_confidence')):.4f}",
            f"{_as_float(ctx.get('ml_threshold_used')):.4f}",
            str(ctx.get("circuit_breaker") or "active"),
            str(ctx.get("decision") or "SALTAR"),
            str(ctx.get("decision_reason") or "-"),
        ]
    )


def _log_decision(symbol, ctx):
    now = time.time()
    digest = _loop_context_hash(ctx)
    prev = _LAST_DECISION_LOG.get(symbol) or {}
    if prev.get("hash") == digest and (now - float(prev.get("ts", 0.0))) < 300:
        return
    ml_ready = bool(ctx.get("ml_ready"))
    confidence = _as_float(ctx.get("ml_confidence"), 0.0)
    threshold = _as_float(ctx.get("ml_threshold_used"), 0.0)
    ml_text = f"{confidence:.0%}/{threshold:.2f}{'✓' if ml_ready else '✗ml_not_ready'}"
    message = (
        f"regime={ctx.get('regime', 'N/A')} strat={ctx.get('strategy_used', 'none')} "
        f"signal={ctx.get('signal', 'NEUTRAL')} ml={ml_text} cb={ctx.get('circuit_breaker', 'active')} "
        f"→ {ctx.get('decision', 'SALTAR')} ({ctx.get('decision_reason', '-')})"
    )
    activity_log.push(symbol, "loop", message)
    _LAST_DECISION_LOG[symbol] = {"ts": now, "hash": digest}


def _ml_bootstrap_reason(symbol, cfg):
    reasons = []
    try:
        model_path = ml_model._model_path(symbol)
    except Exception:
        model_path = ROOT / f"model_{symbol}.pkl"
    info = ml_model.model_info(symbol)
    expected_features = len(FEATURE_COLUMNS)
    if not model_path.exists():
        reasons.append("modelo ausente")
    if int(info.get("trained_on", 0) or 0) == 0:
        reasons.append("trained_on=0")
    if int(info.get("feature_count", 0) or 0) != expected_features:
        reasons.append(f"feature_count mismatch {int(info.get('feature_count', 0) or 0)}!={expected_features}")
    last_trained = str(info.get("last_trained_at") or "").strip()
    if not last_trained:
        reasons.append("sin last_trained_at")
    else:
        try:
            trained_at = datetime.fromisoformat(last_trained.replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - trained_at).total_seconds() > 7 * 24 * 3600:
                reasons.append("modelo vencido >7d")
        except Exception:
            reasons.append("last_trained_at inválido")
    return reasons, info, expected_features


def _ensure_ml_bootstrapped(symbol, cfg, force=False, reason_override=None):
    reasons, info, _ = _ml_bootstrap_reason(symbol, cfg)
    if reason_override:
        reasons = [reason_override]
    if not reasons and not force:
        return False
    thread = _ML_BOOTSTRAP_STATE.get("thread")
    if thread is not None and getattr(thread, "is_alive", lambda: False)():
        return False

    def _runner():
        _ML_BOOTSTRAP_STATE["running"] = True
        try:
            import bootstrap_ml

            activity_log.push(symbol, "ml", f"⚙️ Auto-bootstrap iniciado (motivo: {'; '.join(reasons)})")
            result = bootstrap_ml.bootstrap(
                symbol="BTCUSDT",
                days=int(cfg.get("ml_auto_bootstrap_days", 365)),
                timeframe=int(cfg.get("primary_timeframe", 60)),
                trade_usdt=50,
                min_samples=150,
                quiet=True,
            )
            trained_on = int(result.get("labels") or ml_model.model_info(symbol).get("trained_on", 0))
            ready = bool(ml_model.is_ready(symbol))
            sharpe = float(result.get("val_sharpe", ml_model.model_info(symbol).get("val_sharpe", 0.0)))
            threshold = float(result.get("suggested_threshold", ml_model.model_info(symbol).get("suggested_threshold", cfg.get("ml_threshold", 0.55))))
            activity_log.push(symbol, "ml", f"✅ Bootstrap finalizado: trained={trained_on} ready={ready} sharpe={sharpe:.2f} thr={threshold:.2f}")
        except Exception as exc:
            activity_log.push(symbol, "ml", f"❌ Bootstrap error: {exc}")
            logger.exception("Auto-bootstrap failed for %s", symbol)
        finally:
            _ML_BOOTSTRAP_STATE["running"] = False
            _ML_BOOTSTRAP_STATE["last_completed"] = time.time()

    worker = threading.Thread(target=_runner, daemon=True)
    _ML_BOOTSTRAP_STATE["thread"] = worker
    _ML_BOOTSTRAP_STATE["last_started"] = time.time()
    worker.start()
    return True


def _retrain_if_ready(symbol):
    try:
        cfg = _get_symbol_config(symbol)
        stats = trade_log.get_stats(symbol)
        closed_trades = int(stats.get("total", 0))
        if closed_trades <= 0 or closed_trades % ML_RETRAIN_EVERY != 0:
            return
        info = ml_model.model_info(symbol)
        if not bool(info.get("ready", False)) and int(info.get("trained_on", 0) or 0) == 0:
            threading.Thread(target=lambda: _ensure_ml_bootstrapped(symbol, cfg), daemon=True).start()
            return
        if not bool(info.get("ready", False)):
            failures = int(_ML_RETRAIN_FAILURES.get(symbol, 0)) + 1
            _ML_RETRAIN_FAILURES[symbol] = failures
            activity_log.push(symbol, "ml", f"Modelo no listo tras cierre #{closed_trades}; intento {failures}/{int(cfg.get('ml_force_bootstrap_after_failures', 3))}")
            if failures >= int(cfg.get("ml_force_bootstrap_after_failures", 3)):
                _ML_RETRAIN_FAILURES[symbol] = 0
                _ensure_ml_bootstrapped(symbol, cfg, force=True, reason_override=f"retrain attempts={failures}")
            return
        _ML_RETRAIN_FAILURES[symbol] = 0
        _ensure_ml_bootstrapped(symbol, cfg, force=True, reason_override=f"{closed_trades} trades cerrados")
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
    circuit_breaker.update_on_trade_close(pnl, pnl > 0)
    activity_log.push(symbol, "trade_close", f'CERRÓ {side} ({reason}) | PnL={pnl:+.2f} USDT | R={r_multiple:+.2f}')
    threading.Thread(target=_retrain_if_ready, args=(symbol,), daemon=True).start()


def _open_trade(symbol, signal, last, df, cfg, balance, trade_usdt, strategy_kind="trend", regime_name="TREND"):
    price = _as_float(last["close"])
    atr = _as_float(last.get("atr"))
    if price <= 0 or atr <= 0:
        return
    side = "Buy" if signal == "LONG" else "Sell"
    qty = binance_broker.normalize_quantity(symbol, (trade_usdt * cfg["leverage"]) / price)
    if qty <= 0:
        logger.warning("Skipping %s entry because normalized quantity is too small", symbol)
        return
    sma20 = _as_float(pd.Series(df["close"]).rolling(20).mean().iloc[-1]) if len(df) >= 20 else 0.0
    if strategy_kind == "meanrev":
        sl_dist = atr * float(cfg.get("meanrev_atr_mult", 0.5))
    elif strategy_kind == "breakout":
        sl_dist = atr * float(cfg.get("volatile_atr_mult", 2.0) if regime_name == "VOLATILE" else cfg.get("breakout_atr_mult", 1.0))
    else:
        sl_dist = atr * cfg["atr_mult"]
    if side == "Buy":
        sl = price - sl_dist
        if strategy_kind == "meanrev" and bool(cfg.get("meanrev_tp_at_sma", True)) and sma20 > price:
            tp = sma20
        elif strategy_kind == "breakout":
            tp = price + atr * float(cfg.get("breakout_tp_atr_mult", 3.0))
        else:
            tp = price + sl_dist * 2.0
        try:
            binance_broker.open_long(symbol, qty)
        except Exception as exc:
            logger.warning("open_long failed for %s: %s", symbol, exc)
            return
    else:
        sl = price + sl_dist
        if strategy_kind == "meanrev" and bool(cfg.get("meanrev_tp_at_sma", True)) and 0 < sma20 < price:
            tp = sma20
        elif strategy_kind == "breakout":
            tp = price - atr * float(cfg.get("breakout_tp_atr_mult", 3.0))
        else:
            tp = price - sl_dist * 2.0
        try:
            binance_broker.open_short(symbol, qty)
        except Exception as exc:
            logger.warning("open_short failed for %s: %s", symbol, exc)
            return
    cols = df.attrs.get("indicator_cols", {})
    feature_context = build_features(df).iloc[-1].to_dict()
    trade_log.set_trade_context(feature_context.get("candle_body_pct", 0.0))
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
            return_3_pct=feature_context.get("return_3_pct", 0.0),
            return_5_pct=feature_context.get("return_5_pct", 0.0),
            range_5_pct=feature_context.get("range_5_pct", 0.0),
            body_avg_3_pct=feature_context.get("body_avg_3_pct", 0.0),
            volume_trend_3_ratio=feature_context.get("volume_trend_3_ratio", 0.0),
            close_pos_5=feature_context.get("close_pos_5", 0.0),
            ema9_slope_3_pct=feature_context.get("ema9_slope_3_pct", 0.0),
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
            return_3_pct=feature_context.get("return_3_pct", 0.0),
            return_5_pct=feature_context.get("return_5_pct", 0.0),
            range_5_pct=feature_context.get("range_5_pct", 0.0),
            body_avg_3_pct=feature_context.get("body_avg_3_pct", 0.0),
            volume_trend_3_ratio=feature_context.get("volume_trend_3_ratio", 0.0),
            close_pos_5=feature_context.get("close_pos_5", 0.0),
            ema9_slope_3_pct=feature_context.get("ema9_slope_3_pct", 0.0),
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
        "strategy_kind": strategy_kind,
        "regime_name": regime_name,
        "primary_timeframe": int(cfg.get("primary_timeframe", cfg.get("timeframe", 60))),
        "timeout_bars": 24,
        "opened_ts_ms": int(_as_float(last.get("ts"))),
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
    timeframe_minutes = int(position.get("primary_timeframe", cfg.get("primary_timeframe", 60)))
    opened_ts_ms = int(_as_float(position.get("opened_ts_ms"), 0.0))
    current_ts_ms = int(_as_float(last.get("ts"), opened_ts_ms))
    elapsed_bars = 0
    if opened_ts_ms > 0 and current_ts_ms >= opened_ts_ms and timeframe_minutes > 0:
        elapsed_bars = int((current_ts_ms - opened_ts_ms) / (timeframe_minutes * 60 * 1000))
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
            trail_candidate = _as_float(position.get("peak_price"), price) - atr * 3.0
            if trail_candidate > sl:
                sl = trail_candidate
                position["sl"] = sl
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Chandelier stop updated @ {sl:,.2f} (R={move_r:.2f})")
        if elapsed_bars >= int(position.get("timeout_bars", 24)) and move_r < 0.3:
            hit_exit = True
            reason = "EXIT_TIMEOUT"
        elif price <= sl or price >= tp or signal == "SHORT":
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
            trail_candidate = _as_float(position.get("trough_price"), price) + atr * 3.0
            if sl <= 0 or trail_candidate < sl:
                sl = trail_candidate
                position["sl"] = sl
                position["trail_stop"] = sl
                activity_log.push(symbol, "risk", f"Chandelier stop updated @ {sl:,.2f} (R={move_r:.2f})")
        if elapsed_bars >= int(position.get("timeout_bars", 24)) and move_r < 0.3:
            hit_exit = True
            reason = "EXIT_TIMEOUT"
        elif price >= sl or price <= tp or signal == "LONG":
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


def _htf_confirms(signal, confirmation_df, cfg):
    if signal == "NEUTRAL" or confirmation_df.empty:
        return False
    row = confirmation_df.iloc[-1]
    ema50 = _as_float(row.get(f"ema{int(cfg.get('ema_confirm', 50))}"))
    ema200 = _as_float(row.get(f"ema{int(cfg.get('ema_trend', 200))}"))
    close = _as_float(row.get("close"))
    if signal == "LONG":
        return close > ema200 and ema50 > ema200
    return close < ema200 and ema50 < ema200


def _regime_threshold(regime_name, cfg):
    mapping = {
        "TREND": 0.55,
        "RANGE": 0.60,
        "VOLATILE": 0.65,
        "MIXED": 1.0,
    }
    return float(mapping.get(str(regime_name or "").upper(), cfg.get("ml_threshold", 0.55)))


def _select_signal(symbol, primary_df, confirmation_df, cfg):
    regime_name = "TREND"
    mode = str(cfg.get("strategy_mode", "regime")).lower()
    strategy_name = mode
    meta = {"mixed_fallback": False, "reason": "", "source": mode}
    last = primary_df.iloc[-1].to_dict()
    if mode == "regime":
        regime_info = regime_detector.classify_regime(primary_df)
        regime_name = str(regime_info.get("regime", "MIXED")).upper()
        activity_log.push(symbol, "regime", f"Regime {regime_name} | source={regime_info.get('source')} | hurst={float(regime_info.get('hurst', 0.0)):.2f} | vol={float(regime_info.get('realized_volatility_pct', 0.0)):.2f}%")
        if regime_name == "TREND":
            strategy_name = "trend"
            signal = signal_trend(last, primary_df, cfg)
        elif regime_name == "RANGE":
            strategy_name = "meanrev"
            signal = signal_meanrev(last, primary_df, cfg)
        elif regime_name == "VOLATILE":
            strategy_name = "breakout"
            signal = signal_breakout(last, primary_df, cfg)
        else:
            strategy_name = "trend"
            meta["mixed_fallback"] = True
            meta["reason"] = "regime MIXED -> fallback a trend con threshold elevado"
            signal = signal_trend(last, primary_df, cfg)
    elif mode == "meanrev":
        regime_name = "RANGE"
        signal = signal_meanrev(last, primary_df, cfg)
    elif mode == "breakout":
        regime_name = "VOLATILE"
        signal = signal_breakout(last, primary_df, cfg)
    else:
        regime_name = "TREND"
        strategy_name = "trend"
        signal = signal_trend(last, primary_df, cfg)
    if signal != "NEUTRAL" and not _htf_confirms(signal, confirmation_df, cfg):
        activity_log.push(symbol, "signal", f"HTF 4h no confirma {signal}")
        meta["reason"] = f"HTF 4h no confirma {signal}"
        return "NEUTRAL", strategy_name, regime_name, meta
    return signal, strategy_name, regime_name, meta


def _startup_diagnostics(symbol, cfg):
    info = ml_model.model_info(symbol)
    cb_status = circuit_breaker.get_status()
    daily = _load_daily_balance()
    trades = trade_log.get_all_trades(symbol)
    stats = trade_log.get_stats(symbol)
    expected = len(FEATURE_COLUMNS)
    model_count = int(info.get("feature_count", 0) or 0)
    mismatch = model_count not in (0, expected)
    lines = [
        "=" * 60,
        "BOT STARTUP DIAGNOSTICS",
        f"Strategy mode: {cfg['strategy_mode']}",
        f"Primary TF: {int(cfg['primary_timeframe'])}m | Confirmation TF: {int(cfg['confirmation_timeframe'])}m",
        f"ML model: trained_on={int(info.get('trained_on', 0) or 0)} ready={bool(info.get('ready', False))} sharpe={float(info.get('val_sharpe', 0.0)):.2f} thr={float(info.get('suggested_threshold', cfg.get('ml_threshold', 0.55))):.2f}",
        f"ML feature count expected={expected} / model={model_count} {'MISMATCH!' if mismatch else 'OK'}",
        f"Circuit breaker: enabled={bool(cfg.get('circuit_breaker_enabled', True))} paused={bool(cb_status.get('paused', False))}",
        f"Trades in DB: total={len(trades)} closed_real={int(stats.get('total', 0))}",
        f"Daily start balance: {_as_float(daily.get('daily_start_balance'), 0.0):.2f} USDT",
        "=" * 60,
    ]
    return lines


def _emit_startup_diagnostics(symbol, cfg):
    for line in _startup_diagnostics(symbol, cfg):
        logger.info(line)
        activity_log.push(symbol, "startup", line)


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
            daily_state = _load_daily_balance()
            weekly_state = _load_weekly_balance()
            primary_df = _compute_indicators(_fetch_candles(symbol, cfg["primary_timeframe"], limit=450), cfg)
            confirmation_df = _compute_indicators(_fetch_candles(symbol, cfg["confirmation_timeframe"], limit=320), cfg)
            if primary_df.empty or confirmation_df.empty or len(primary_df) < 220 or len(confirmation_df) < 220:
                _log_decision(symbol, {"regime": "N/A", "strategy_used": "none", "signal": "NEUTRAL", "ml_ready": False, "ml_confidence": 0.0, "ml_threshold_used": float(cfg.get("ml_threshold", 0.55)), "circuit_breaker": "active", "decision": "SALTAR", "decision_reason": "velas insuficientes"})
                time.sleep(60)
                continue
            feature_frame = build_features(primary_df)
            last = primary_df.iloc[-1].to_dict()
            feature_context = feature_frame.iloc[-1].to_dict()
            signal, strategy_kind, regime_name, signal_meta = _select_signal(symbol, primary_df, confirmation_df, cfg)
            position = _get_current_position(symbol)
            if position:
                _update_position(symbol, last, signal, cfg)
                _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": bool(ml_model.is_ready(symbol)), "ml_confidence": 0.0, "ml_threshold_used": float(cfg.get("ml_threshold", 0.55)), "circuit_breaker": "active", "decision": "POSICION_ABIERTA", "decision_reason": "gestion de posicion en curso"})
                time.sleep(60)
                continue
            cb_status_text = "active"
            if cfg.get("circuit_breaker_enabled", True):
                paused, reason = circuit_breaker.is_paused()
                if not paused:
                    paused, reason = circuit_breaker.check_circuit_breaker(balance, _as_float(daily_state.get("daily_start_balance"), balance), _as_float(weekly_state.get("weekly_start_balance"), balance), last, cfg)
                if paused:
                    cb_status_text = f"paused: {reason}"
                    activity_log.push(symbol, "circuit_breaker", f"CB paused: {reason}")
                    _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": False, "ml_confidence": 0.0, "ml_threshold_used": float(cfg.get("ml_threshold", 0.55)), "circuit_breaker": cb_status_text, "decision": "CB_PAUSED", "decision_reason": reason})
                    time.sleep(60)
                    continue
            features = {**feature_context}
            features["side_buy"] = 1.0 if signal == "LONG" else 0.0
            features["supertrend_aligned_side"] = 1.0 if ((features["side_buy"] >= 0.5 and _as_float(features.get("supertrend_direction")) > 0) or (features["side_buy"] < 0.5 and _as_float(features.get("supertrend_direction")) < 0)) else 0.0
            features["regime_code"] = {"TREND": 1.0, "RANGE": -1.0, "VOLATILE": 2.0, "MIXED": 0.0}.get(regime_name, 0.0)
            confidence = _ml_confidence(features, symbol)
            ml_threshold = float(cfg.get("ml_threshold", 0.55))
            ml_ready = False
            ml_info = {}
            try:
                ml_ready = bool(ml_model.is_ready(symbol))
                ml_info = ml_model.model_info(symbol)
            except Exception:
                ml_ready = False
                ml_info = {}
            if ML_THRESHOLD_OVERRIDE not in (None, ""):
                ml_threshold = float(ML_THRESHOLD_OVERRIDE)
            elif ml_ready:
                regime_floor = _regime_threshold("TREND", cfg) + 0.05 if signal_meta.get("mixed_fallback") else _regime_threshold(regime_name, cfg)
                ml_threshold = max(float(ml_info.get("suggested_threshold", cfg["ml_threshold"])), min(0.70, regime_floor))
            else:
                ml_threshold = float(cfg["ml_threshold"])
                not_ready_reason = str(ml_info.get("not_ready_reason") or "gates de validacion no cumplidas")
                if _ML_NOT_READY_STATE.get(symbol) != not_ready_reason:
                    activity_log.push(symbol, "ml", f"Modelo no listo: {not_ready_reason}; usando heuristica cfg")
                    _ML_NOT_READY_STATE[symbol] = not_ready_reason
            if ml_ready:
                _ML_NOT_READY_STATE.pop(symbol, None)
            primary_only_mode = _primary_only_active(ml_ready, ml_info)
            state = _portfolio_state()
            state = _calc_portfolio_risk(balance, state)
            _save_portfolio_state(state)
            if signal == "NEUTRAL":
                _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": ml_ready, "ml_confidence": confidence, "ml_threshold_used": ml_threshold, "circuit_breaker": cb_status_text, "decision": "SALTAR", "decision_reason": signal_meta.get("reason") or "sin se?al v?lida"})
                time.sleep(60)
                continue
            volume_regime = _volume_regime_name(strategy_kind, regime_name)
            volume_cfg = dict(cfg)
            volume_cfg["_signal_side"] = signal
            volume_ok, volume_reason = volume_filter_passes(primary_df, volume_regime, volume_cfg)
            if not volume_ok:
                _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": ml_ready, "ml_confidence": confidence, "ml_threshold_used": ml_threshold, "circuit_breaker": cb_status_text, "decision": "RECHAZAR", "decision_reason": volume_reason})
                time.sleep(60)
                continue
            if confidence < ml_threshold and ml_ready:
                _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": ml_ready, "ml_confidence": confidence, "ml_threshold_used": ml_threshold, "circuit_breaker": cb_status_text, "decision": "RECHAZAR", "decision_reason": f"meta-modelo {confidence:.0%} < {ml_threshold:.2f}"})
                time.sleep(60)
                continue
            price = _as_float(last.get("close"))
            atr = _as_float(last.get("atr"))
            if strategy_kind == "meanrev":
                atr_stop_distance = atr * float(cfg.get("meanrev_atr_mult", 0.5))
            elif strategy_kind == "breakout":
                atr_stop_distance = atr * float(cfg.get("volatile_atr_mult", 2.0) if regime_name == "VOLATILE" else cfg.get("breakout_atr_mult", 1.0))
            else:
                atr_stop_distance = atr * float(cfg.get("atr_mult", 1.5))
            trade_usdt = _kelly_trade_usdt(balance, price, atr_stop_distance, cfg["leverage"])
            trade_usdt = _primary_only_trade_usdt(trade_usdt, ml_ready)
            qty_preview = binance_broker.normalize_quantity(symbol, (trade_usdt * cfg["leverage"]) / price) if price > 0 else 0.0
            risk_pct = (qty_preview * atr_stop_distance) / balance * 100 if balance > 0 else 0.0
            projected = state.get("total_risk_pct", 0.0) + risk_pct
            if projected > float(cfg.get("daily_risk_cap", 5.0)):
                activity_log.push("PORTFOLIO", "risk", "Limite de riesgo del portafolio alcanzado - nueva posicion bloqueada")
                _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": ml_ready, "ml_confidence": confidence, "ml_threshold_used": ml_threshold, "circuit_breaker": cb_status_text, "decision": "RECHAZAR", "decision_reason": f"riesgo proyectado {projected:.2f}% > cap {float(cfg.get('daily_risk_cap', 5.0)):.2f}%"})
                time.sleep(60)
                continue
            if primary_only_mode:
                activity_log.push(symbol, "ml", f"[ML-OFF] operating primary signal at 50% size, reason={ml_info.get('not_ready_reason') or 'modelo no listo'}")
            _open_trade(symbol, signal, last, primary_df, cfg, balance, trade_usdt, strategy_kind=strategy_kind, regime_name=regime_name)
            open_reason = signal_meta.get("reason") or "se?al confirmada"
            if primary_only_mode:
                open_reason = f"{open_reason} | PRIMARY ONLY 50% size"
            else:
                open_reason = f"{open_reason} | {volume_reason}"
            _log_decision(symbol, {"regime": regime_name, "strategy_used": strategy_kind, "signal": signal, "ml_ready": ml_ready, "ml_confidence": confidence, "ml_threshold_used": ml_threshold, "circuit_breaker": cb_status_text, "decision": "ABRIR", "decision_reason": open_reason})
        except Exception as exc:
            logger.exception("Symbol loop error for %s", symbol)
            activity_log.push(symbol, "error", f"Error: {exc}")
            _log_decision(symbol, {"regime": "N/A", "strategy_used": "none", "signal": "NEUTRAL", "ml_ready": False, "ml_confidence": 0.0, "ml_threshold_used": float(cfg.get("ml_threshold", 0.55)), "circuit_breaker": "active", "decision": "SALTAR", "decision_reason": f"error: {exc}"})
        time.sleep(60)


def main():
    base_cfg = _get_symbol_config("BTCUSDT")
    stream_pairs = []
    for symbol in SYMBOLS:
        cfg = _get_symbol_config(symbol)
        stream_pairs.append((symbol, cfg["primary_timeframe"]))
        stream_pairs.append((symbol, cfg["confirmation_timeframe"]))
    market_stream.start(stream_pairs)
    reconcile_state()
    _emit_startup_diagnostics("BTCUSDT", base_cfg)
    _ensure_ml_bootstrapped("BTCUSDT", base_cfg)
    threads = []
    for symbol in SYMBOLS:
        cfg = _get_symbol_config(symbol)
        thread = threading.Thread(target=run_symbol, args=(symbol, cfg, stop_event), daemon=True)
        thread.start()
        threads.append(thread)
        logger.info("Started thread for %s", symbol)
    try:
        last_reconcile = time.time()
        last_watchdog = time.time()
        watchdog_sec = max(3600.0, float(base_cfg.get("ml_watchdog_hours", 24.0)) * 3600.0)
        while True:
            now = time.time()
            if now - last_reconcile >= 60:
                reconcile_state()
                last_reconcile = now
            if now - last_watchdog >= watchdog_sec:
                _ensure_ml_bootstrapped("BTCUSDT", _get_symbol_config("BTCUSDT"))
                last_watchdog = now
            for thread in threads:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
        logger.info("Stopping all symbol threads...")
        for thread in threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
