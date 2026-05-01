import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

import activity_log
import binance_broker
import circuit_breaker
import historical_data
import market_stream
from bot_config import get_symbol_config, parse_symbols
from features import build_features
import ml_model
import regime as regime_detector
import search_loop
from strategies import signal_breakout, signal_meanrev, signal_trend, volume_filter_passes
import trade_log

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = FastAPI()

HTML_PATH = ROOT / "dashboard_multi.html"
PORTFOLIO_FILE = ROOT / "portfolio_state.json"
DAILY_FILE = ROOT / "daily_balance.json"
OPEN_TRADE_FILE = ROOT / "open_trade_id.json"

SYMBOLS = ["BTCUSDT"]
TAB_SYMBOLS = ["BTCUSDT"]
HISTORY_TIMEFRAMES = [5, 15, 60, 240]
CHART_TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _recent_activity(symbol: str | None = None, limit: int = 20):
    try:
        events = [evt for evt in activity_log.get_recent(100) if not _is_healthcheck_event(evt)]
    except Exception:
        return []
    if symbol is None:
        return events[:limit]
    key = symbol.upper()
    filtered = []
    for event in events:
        evt_symbol = str(event.get("symbol") or "").upper()
        if evt_symbol == key or evt_symbol == "PORTFOLIO":
            filtered.append(event)
        if len(filtered) >= limit:
            break
    return filtered


def _is_healthcheck_event(event) -> bool:
    try:
        symbol = str(event.get("symbol") or "").upper()
        message = str(event.get("message") or "").lower()
        if symbol == "PORTFOLIO" and "chequeo de salud" in message:
            return True
    except Exception:
        return False
    return False


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _portfolio_state():
    return _load_json(PORTFOLIO_FILE, {"open_positions": {}, "total_risk_pct": 0.0})


def _local_position(symbol: str):
    state = _portfolio_state()
    return (state.get("open_positions", {}) or {}).get(symbol)


def _position_snapshot(symbol: str, price: float | None = None):
    try:
        pos = binance_broker.get_position(symbol)
        if pos:
            return {
                "position": {
                    "side": pos.get("side"),
                    "size": pos.get("size"),
                    "entry_price": pos.get("entry_price"),
                    "unrealisedPnl": pos.get("unrealised_pnl", 0.0),
                    "_source": "binance",
                },
                "source": "binance",
                "error": None,
            }
        return {"position": None, "source": "binance", "error": None}
    except Exception as exc:
        local = _local_position(symbol)
        if local:
            recovered = dict(local)
            if price is not None:
                recovered["unrealisedPnl"] = _unrealized_pnl_from_position(recovered, price)
            else:
                recovered["unrealisedPnl"] = _as_float(recovered.get("unrealisedPnl"), 0.0)
            recovered["_source"] = "local"
            recovered["_exchange_error"] = str(exc)
            return {"position": recovered, "source": "local", "error": str(exc)}
        return {"position": None, "source": "error", "error": str(exc)}


def _open_trade_registry():
    return _load_json(OPEN_TRADE_FILE, {})


def _daily_state():
    data = _load_json(DAILY_FILE, {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("date") != today:
        try:
            data = {"date": today, "daily_start_balance": _get_balance(), "paused": False}
        except Exception:
            data = {"date": today, "daily_start_balance": 0.0, "paused": False}
    return data


def _get_balance():
    try:
        return binance_broker.get_balance()
    except Exception:
        return 0.0


def _fetch_candles(symbol, timeframe=5, limit=250):
    klines = market_stream.get_candles(symbol, timeframe, limit)
    if not klines:
        klines = binance_broker.get_kline(symbol, timeframe, limit)
    df = pd.DataFrame(
        klines,
        columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"],
    )
    df = df[["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]]
    if df.empty:
        return df
    for col in ["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _parse_chart_timeframe(raw: str | None) -> tuple[str, int]:
    label = str(raw or "1d").strip().lower()
    if label in CHART_TIMEFRAMES:
        return label, CHART_TIMEFRAMES[label]
    return "1d", CHART_TIMEFRAMES["1d"]




def _compute_indicators(df, cfg):
    if df.empty or len(df) < 50:
        return df
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
def _position(symbol):
    return _position_snapshot(symbol).get("position")


def _kelly_fraction():
    try:
        stats = trade_log.get_stats()
        total = int(stats.get("total", 0))
        if total < 20:
            return 0.0
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        if total <= 0 or wins <= 0 or losses <= 0:
            return 0.0
        trades = trade_log.get_all_trades()
        pnl_values = [float(t.get("pnl", 0.0)) for t in trades]
        positive = [p for p in pnl_values if p > 0]
        negative = [p for p in pnl_values if p < 0]
        if not positive or not negative:
            return 0.0
        win_rate = wins / total
        avg_win = sum(positive) / len(positive)
        avg_loss = abs(sum(negative) / len(negative))
        if avg_loss <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        return max(kelly / 2.0, 0.02)
    except Exception:
        return 0.0


def _daily_loss_pct(balance):
    state = _daily_state()
    start = _as_float(state.get("daily_start_balance"), balance)
    if start <= 0:
        return 0.0
    if balance >= start:
        return 0.0
    return min(((start - balance) / start) / 0.03 * 100.0, 100.0)


def _unrealized_pnl_from_position(position, price):
    if not position or price <= 0:
        return 0.0
    entry = _as_float(position.get("entry_price"))
    size = _as_float(position.get("size") or position.get("qty"))
    if entry <= 0 or size <= 0:
        return _as_float(position.get("unrealisedPnl"), 0.0)
    side = str(position.get("side") or "Buy").lower()
    if side == "buy":
        return (price - entry) * size
    return (entry - price) * size


def _signal_from_last(last, cfg=None):
    cfg = cfg or get_symbol_config("BTCUSDT")
    cols = {
        "ema_fast": f"ema{int(cfg.get('ema_fast', 9))}",
        "ema_slow": f"ema{int(cfg.get('ema_slow', 21))}",
        "ema_trend": f"ema{int(cfg.get('ema_trend', 200))}",
    }
    ema_fast = _as_float(last.get(cols["ema_fast"]))
    ema_slow = _as_float(last.get(cols["ema_slow"]))
    ema_trend = _as_float(last.get(cols["ema_trend"]))
    rsi = _as_float(last.get("rsi"))
    adx = _as_float(last.get("adx"))
    st = int(_as_float(last.get("supertrend_direction")))
    close = _as_float(last.get("close"))
    primary_tf = int(cfg.get("primary_timeframe", cfg.get("timeframe", 60)))
    adx_threshold = float(cfg.get("adx_threshold_1h", 18)) if primary_tf >= 60 else float(cfg.get("adx_threshold", 25))
    rsi_min = float(cfg.get("rsi_min", 30))
    rsi_max = float(cfg.get("rsi_max", 70))
    if (
        ema_fast > ema_slow
        and close > ema_trend
        and st == 1
        and adx > adx_threshold
        and rsi_min < rsi < rsi_max
    ):
        return "LONG"
    if (
        ema_fast < ema_slow
        and close < ema_trend
        and st == -1
        and adx > adx_threshold
        and rsi_min < rsi < rsi_max
    ):
        return "SHORT"
    return "NEUTRAL"


def _dashboard_volume_regime(strategy_kind, regime_name):
    strategy = str(strategy_kind or "").lower()
    regime = str(regime_name or "").upper()
    if strategy == "meanrev":
        return "VOLATILE" if regime == "VOLATILE" else "MEANREV"
    if strategy == "breakout":
        return "BREAKOUT"
    if regime in {"TREND", "TRENDING"}:
        return "TRENDING"
    if regime == "RANGE":
        return "MEANREV"
    return regime or "DEFAULT"


def _dashboard_strategy_state(df, cfg):
    if df.empty:
        return {
            "regime_name": "N/A",
            "strategy_kind": "none",
            "signal": "NEUTRAL",
            "signal_reason": "sin datos",
            "volume_ok": False,
            "volume_reason": "sin datos",
            "volume_regime": "DEFAULT",
        }
    last = df.iloc[-1].to_dict()
    regime_info = regime_detector.classify_regime(df)
    mode = str(cfg.get("strategy_mode", "regime")).lower()
    strategy_kind = mode
    signal_reason = ""
    if mode == "regime":
        regime_name = str(regime_info.get("regime") or "MIXED").upper()
        if regime_name in {"TREND", "TRENDING"}:
            regime_name = "TRENDING"
            strategy_kind = "trend"
            signal = signal_trend(last, df, cfg)
        elif regime_name == "RANGE":
            strategy_kind = "meanrev"
            signal = signal_meanrev(last, df, cfg)
        elif regime_name == "VOLATILE":
            strategy_kind = "meanrev"
            signal_reason = "regime VOLATILE -> meanrev con riesgo reducido"
            signal = signal_meanrev(last, df, cfg)
        else:
            regime_name = "MIXED"
            strategy_kind = "breakout"
            signal = signal_breakout(last, df, cfg)
            signal_reason = "regime MIXED -> breakout"
    elif mode == "meanrev":
        regime_name = "RANGE"
        strategy_kind = "meanrev"
        signal = signal_meanrev(last, df, cfg)
    elif mode == "breakout":
        regime_name = "BREAKOUT"
        strategy_kind = "breakout"
        signal = signal_breakout(last, df, cfg)
    else:
        regime_name = "TRENDING"
        strategy_kind = "trend"
        signal = signal_trend(last, df, cfg)
    volume_regime = _dashboard_volume_regime(strategy_kind, regime_name)
    volume_cfg = dict(cfg)
    volume_cfg["_signal_side"] = signal
    volume_ok, volume_reason = volume_filter_passes(df, volume_regime, volume_cfg)
    return {
        "regime_name": regime_name,
        "strategy_kind": strategy_kind,
        "signal": signal,
        "signal_reason": signal_reason,
        "volume_ok": bool(volume_ok),
        "volume_reason": str(volume_reason),
        "volume_regime": volume_regime,
    }


def _trade_gate_state(last, df, cfg, position=None, ml_ready=False, ml_confidence=0.5, ml_info=None):
    if last is None:
        return {"status": "SIN DATOS", "reason": "Esperando velas suficientes", "ready": False}
    if position:
        return {"status": "POSICI?N ABIERTA", "reason": "Ya hay una operaci?n activa", "ready": False}
    signal_state = _dashboard_strategy_state(df, cfg)
    signal = signal_state["signal"]
    if signal == "NEUTRAL":
        return {"status": "ESPERANDO", "reason": signal_state.get("signal_reason") or "Sin confluencia de entrada", "ready": False}
    if not signal_state["volume_ok"]:
        return {"status": "ESPERANDO", "reason": signal_state["volume_reason"], "ready": False}
    if not ml_ready:
        reason = str((ml_info or {}).get("not_ready_reason") or "ML no listo")
        return {"status": "PRIMARY ONLY", "reason": f"MODO PRIMARY (sin ML, half-size) | {reason}", "ready": True}
    ml_threshold = float((ml_info or {}).get("suggested_threshold", cfg.get("ml_threshold", 0.55)))
    if ml_confidence < ml_threshold:
        return {"status": "FILTRADO", "reason": f"ML bajo umbral ({ml_confidence:.2f} < {ml_threshold:.2f})", "ready": False}
    return {"status": "LISTO PARA OPERAR", "reason": f"Se?al {signal} confirmada por filtros", "ready": True}


def _health_issues_for_symbol(exchange_pos, local_pos, registry_trade_id, open_rows, exchange_error=None):
    issues = []
    active_trade_id = registry_trade_id
    if active_trade_id is None and local_pos and local_pos.get("trade_id") is not None:
        try:
            active_trade_id = int(local_pos.get("trade_id"))
        except Exception:
            active_trade_id = registry_trade_id
    if exchange_pos and not local_pos:
        issues.append("Binance tiene posicion pero portfolio_state no")
    if local_pos and not exchange_pos and exchange_error is None:
        issues.append("portfolio_state tiene posicion pero Binance no")
    if exchange_pos and registry_trade_id is None:
        issues.append("Falta registro open_trade_id")
    if local_pos and registry_trade_id is None and active_trade_id is None:
        issues.append("portfolio_state no esta reflejado en el registry")
    orphan_rows = []
    for row in open_rows:
        row_id = row.get("id")
        if row_id is None:
            orphan_rows.append(row)
            continue
        if active_trade_id is None or int(row_id) != int(active_trade_id):
            orphan_rows.append(row)
    if orphan_rows:
        issues.append(f"Existen {len(orphan_rows)} trades abiertos huérfanos")
    if exchange_pos and registry_trade_id is not None:
        live_trade_ids = {int(row.get("id")) for row in open_rows if row.get("id") is not None}
        if live_trade_ids and int(registry_trade_id) not in live_trade_ids:
            issues.append("open_trade_id no coincide con el trade abierto")
    if exchange_error and not local_pos and active_trade_id is None:
        issues.append("Binance no responde y no hay respaldo local")
    return issues, active_trade_id, orphan_rows


def _run_healthcheck():
    import multi_bot

    before_portfolio = _portfolio_state()
    before_registry = _open_trade_registry()
    before_open_rows = [t for t in trade_log.get_all_trades() if t.get("pnl") is None]

    multi_bot.reconcile_state()

    portfolio = _portfolio_state()
    registry = _open_trade_registry()
    daily = _daily_state()
    daily_file = _load_json(DAILY_FILE, {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    per_symbol = []
    issues = []
    live_count = 0
    orphan_count = 0
    exchange_errors = 0
    for symbol in SYMBOLS:
        snapshot = _position_snapshot(symbol)
        live = snapshot.get("position")
        exchange_error = snapshot.get("error")
        source = snapshot.get("source")
        local = portfolio.get("open_positions", {}).get(symbol)
        registry_trade_id = registry.get(symbol)
        open_rows = [row for row in trade_log.get_all_trades(symbol) if row.get("pnl") is None]
        if live and source == "binance":
            live_count += 1
        if exchange_error:
            exchange_errors += 1
        symbol_issues, active_trade_id, orphan_rows = _health_issues_for_symbol(
            live,
            local,
            registry_trade_id,
            open_rows,
            exchange_error=exchange_error,
        )
        orphan_count += len(orphan_rows)
        if symbol_issues:
            for issue in symbol_issues:
                issues.append(f"{symbol}: {issue}")
        per_symbol.append(
            {
                "symbol": symbol,
                "live": bool(live and source == "binance"),
                "local": bool(local),
                "registry": registry_trade_id is not None,
                "open_rows": len(open_rows),
                "active_trade_id": active_trade_id,
                "orphan_rows": len(orphan_rows),
                "source": source,
                "exchange_error": exchange_error,
                "issues": symbol_issues,
            }
        )

    daily_issues = []
    if daily_file.get("date") != today:
        daily_issues.append("daily_balance.json no corresponde al dia actual")
    if _as_float(daily.get("daily_start_balance")) <= 0:
        daily_issues.append("daily_start_balance invalido")
    if daily.get("paused") not in (True, False):
        daily_issues.append("paused invalido")
    if daily_issues:
        for issue in daily_issues:
            issues.append(f"DAILY: {issue}")

    open_rows_after = [t for t in trade_log.get_all_trades() if t.get("pnl") is None]
    local_count = len(portfolio.get("open_positions", {}) or {})
    registry_count = len(registry)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": len(issues) == 0,
        "reconciled": True,
        "counts": {
            "live_positions": live_count,
            "local_positions": local_count,
            "registry_entries": registry_count,
            "open_trade_rows_before": len(before_open_rows),
            "open_trade_rows_after": len(open_rows_after),
            "orphan_trade_rows_after": orphan_count,
            "exchange_errors": exchange_errors,
        },
        "checks": [
            {
                "name": "Binance vs estado local",
                "status": "ok" if exchange_errors == 0 and live_count == local_count and not any(item["issues"] for item in per_symbol) else "warn",
                "details": f"{live_count} posiciones binance, {local_count} posiciones locales, {exchange_errors} errores de exchange",
            },
            {
                "name": "Registry de trades",
                "status": "ok" if registry_count == local_count else "warn",
                "details": f"{registry_count} entradas activas | {local_count} posiciones locales",
            },
            {
                "name": "Trades abiertos huerfanos",
                "status": "ok" if orphan_count == 0 else "warn",
                "details": f"{orphan_count} filas huérfanas | {len(open_rows_after)} abiertas en DB",
            },
            {
                "name": "Balance diario",
                "status": "ok" if not daily_issues else "warn",
                "details": f"{daily.get('date') or '--'} | start {float(daily.get('daily_start_balance') or 0.0):.2f}",
            },
        ],
        "symbols": per_symbol,
        "issues": issues,
        "before": {
            "open_trade_rows": len(before_open_rows),
            "local_positions": len(before_portfolio.get("open_positions", {}) or {}),
            "registry_entries": len(before_registry),
        },
    }

    return summary


def _run_recovery(force_refresh: bool = True, auto: bool = False):
    import multi_bot

    started = datetime.now(timezone.utc).isoformat()
    report = {
        "timestamp": started,
        "running": True,
        "status": "running",
        "activity": "repairing",
        "force_refresh": bool(force_refresh),
        "auto": bool(auto),
        "issues": [],
        "repairs": [],
        "counts": {},
    }
    _store_recovery_report(report)
    try:
        reconcile_report = multi_bot.reconcile_state(force_refresh=force_refresh) or {}
        report["reconcile"] = reconcile_report
        report["repairs"] = reconcile_report.get("repairs", []) if isinstance(reconcile_report, dict) else []
        report["counts"] = reconcile_report.get("counts", {}) if isinstance(reconcile_report, dict) else {}
        report["issues"] = list((_get_health_report() or {}).get("issues", []))
        _cache["portfolio"] = _portfolio_payload()
        _cache["agents"] = _agents_payload()
        for sym in TAB_SYMBOLS:
            try:
                _cache[sym] = _build_symbol_payload(sym)
            except Exception:
                logger.exception("Failed to refresh symbol cache during recovery for %s", sym)
        health = _run_healthcheck()
        _store_health_report(health)
        report["health"] = health
        report["recovery"] = {
            "before_issues": len(report.get("issues", [])),
            "after_issues": len(health.get("issues", [])) if isinstance(health, dict) else 0,
            "notes": [item.get("action") for item in report.get("repairs", []) if item.get("action")],
            "auto": bool(auto),
        }
        report["ok"] = bool(health.get("ok")) if isinstance(health, dict) else False
        report["status"] = "done" if report["ok"] else "warn"
        report["activity"] = "monitoring" if report["ok"] else "alert"
        report["running"] = False
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        activity_log.push(
            "PORTFOLIO",
            "recovery",
            "Recovery Agent ejecutó una reparación conservadora y actualizó estado, registry y cache.",
        )
        return _store_recovery_report(report)
    except Exception as exc:
        report["running"] = False
        report["status"] = "error"
        report["activity"] = "alert"
        report["error"] = str(exc)
        report["issues"] = report.get("issues") or [str(exc)]
        activity_log.push("PORTFOLIO", "error", f"Recovery Agent falló: {exc}")
        return _store_recovery_report(report)


def _build_symbol_payload(symbol):
    cfg = get_symbol_config(symbol)
    cfg_timeframe = int(cfg.get("primary_timeframe", cfg.get("timeframe", os.getenv("TIMEFRAME", "60"))))
    df = _fetch_candles(symbol, cfg_timeframe)
    df = _compute_indicators(df, cfg)
    events = _recent_activity(symbol, 20)
    updated_at = datetime.now(timezone.utc)
    cb_status = circuit_breaker.get_status()
    if df.empty or len(df) < 50:
        return {
            "price": 0.0,
            "ema9": 0.0,
            "ema21": 0.0,
            "rsi": 0.0,
            "signal": "NEUTRAL",
            "position": None,
            "candles": [],
            "kpis": {},
            "indicators": {},
            "trade_gate": {"status": "SIN DATOS", "reason": "Esperando velas suficientes", "ready": False},
            "circuit_breaker": cb_status,
            "events": events,
            "updated_at": updated_at.isoformat(),
            "updated_label": updated_at.strftime("%H:%M:%S UTC"),
        }
    last = df.iloc[-1].to_dict()
    cols = df.attrs.get("indicator_cols", {})
    price = _as_float(last.get("close"))
    position_snapshot = _position_snapshot(symbol, price)
    position = position_snapshot.get("position")
    open_trades = 1 if position else 0
    balance = _get_balance()
    stats = trade_log.get_stats(symbol)
    symbol_trades = trade_log.get_all_trades(symbol)
    try:
        ml_training_trades = len(ml_model.label_closed_trades(symbol_trades))
    except Exception:
        ml_training_trades = int(stats.get("total", 0))
    ml_min_train_trades = int(getattr(ml_model, "MIN_TRAIN_TRADES", 20))
    try:
        ml_info = ml_model.model_info(symbol)
    except Exception:
        ml_info = {}
    ml_validation_accuracy = _as_float(ml_info.get("validation_accuracy"), 0.0)
    ml_validation_trades = int(ml_info.get("validation_trades") or 0)
    no_profitable_threshold = bool(ml_info.get("no_profitable_threshold", False))
    features = build_features(df).iloc[-1].to_dict()
    ml_ready = False
    ml_confidence = 0.5
    try:
        ml_ready = bool(ml_model.is_ready(symbol))
    except TypeError:
        try:
            ml_ready = bool(ml_model.is_ready())
        except Exception:
            ml_ready = False
    except Exception:
        ml_ready = False
    if ml_ready:
        try:
            ml_confidence = _as_float(ml_model.predict(features, symbol), 0.5)
        except TypeError:
            try:
                ml_confidence = _as_float(ml_model.predict(features), 0.5)
            except Exception:
                ml_confidence = 0.5
        except Exception:
            ml_confidence = 0.5
    signal_state = _dashboard_strategy_state(df, cfg)
    primary_only_mode = not ml_ready
    kpis = {
        "balance": balance,
        "balance_start": 50000.0,
        "pnl_realized": sum(_as_float(t.get("pnl"), 0.0) for t in symbol_trades),
        "pnl_unrealized": _as_float(position.get("unrealisedPnl")) if position else 0.0,
        "pnl_total": 0.0,
        "trade_count": int(stats.get("total", 0)),
        "open_trades": open_trades,
        "closed_trades": int(stats.get("total", 0)),
        "wins": int(stats.get("wins", 0)),
        "losses": int(stats.get("losses", 0)),
        "win_rate": _as_float(stats.get("win_rate")),
        "best_trade": _as_float(stats.get("best")),
        "worst_trade": _as_float(stats.get("worst")),
        "ml_ready": ml_ready,
        "ml_confidence": ml_confidence,
        "ml_training_trades": ml_training_trades,
        "ml_min_train_trades": ml_min_train_trades,
        "ml_validation_accuracy": ml_validation_accuracy,
        "ml_validation_trades": ml_validation_trades,
        "no_profitable_threshold": no_profitable_threshold,
        "primary_only_mode": primary_only_mode,
        "trade_stats": {**stats, "reconciled_count": int(stats.get("reconciled_count", stats.get("reconciled", 0) or 0))},
    }
    kpis["pnl_total"] = kpis["pnl_realized"] + kpis["pnl_unrealized"]
    indicators = {
        "adx": _as_float(last.get("adx")),
        "supertrend_direction": int(_as_float(last.get(cols.get("supertrend_dir", "supertrend_direction")))) if last.get(cols.get("supertrend_dir", "supertrend_direction")) is not None else 0,
        "volume_ratio": _as_float(last.get("volume")) / _as_float(last.get("vol_ma20")) if _as_float(last.get("vol_ma20")) else 0.0,
        "volume_z": _as_float(features.get("volume_z")),
        "volume_quantile": _as_float(features.get("volume_quantile")),
        "volume_filter_ok": bool(signal_state.get("volume_ok", False)),
        "volume_filter_reason": str(signal_state.get("volume_reason") or ""),
        "volume_filter_regime": str(signal_state.get("volume_regime") or "DEFAULT"),
        "atr": _as_float(last.get("atr")),
        "atr_pct": _as_float(last.get("atr_pct")),
        "ema9": _as_float(last.get(cols.get("ema_fast", "ema9"))),
        "ema21": _as_float(last.get(cols.get("ema_slow", "ema21"))),
        "ema200": _as_float(last.get(cols.get("ema_trend", "ema200"))),
        "rsi": _as_float(last.get("rsi")),
        "market_regime": signal_state.get("regime_name", "MIXED"),
        "daily_loss_pct": _daily_loss_pct(balance),
        "daily_risk_used": _daily_loss_pct(balance),
        "kelly_fraction": _kelly_fraction(),
        "primary_only_mode": primary_only_mode,
    }
    trade_gate = _trade_gate_state(last, df, cfg, position=position, ml_ready=ml_ready, ml_confidence=ml_confidence, ml_info=ml_info)
    candles = [
        {
            "ts": int(row["ts"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in df.tail(50).iterrows()
    ]
    return {
        "price": price,
        "ema9": _as_float(last.get(cols.get("ema_fast", "ema9"))),
        "ema21": _as_float(last.get(cols.get("ema_slow", "ema21"))),
        "rsi": _as_float(last.get("rsi")),
        "signal": signal_state.get("signal", _signal_from_last(last, cfg)),
        "signal_meta": {"side": signal_state.get("signal", _signal_from_last(last, cfg)), "source": signal_state.get("strategy_kind", "none")},
        "position": position,
        "candles": candles,
        "kpis": kpis,
        "indicators": indicators,
        "trade_gate": {**trade_gate, "can_trade": bool(trade_gate.get("ready"))},
        "ml": {
            "ready": ml_ready,
            "suggested_threshold": _as_float(ml_info.get("suggested_threshold"), 0.55),
            "val_sharpe": _as_float(ml_info.get("val_sharpe")),
            "val_auc": _as_float(ml_info.get("val_auc")),
            "trained_on": int(ml_info.get("trained_on", 0) or 0),
            "not_ready_reason": str(ml_info.get("not_ready_reason") or ""),
            "no_profitable_threshold": bool(ml_info.get("no_profitable_threshold", False)),
        },
        "circuit_breaker": cb_status,
        "last_candle": {
            "open": _as_float(last.get("open")),
            "high": _as_float(last.get("high")),
            "low": _as_float(last.get("low")),
            "close": _as_float(last.get("close")),
            "volume": _as_float(last.get("volume")),
        },
        "events": events,
        "position_source": position_snapshot.get("source"),
        "position_error": position_snapshot.get("error"),
        "updated_at": updated_at.isoformat(),
        "updated_label": updated_at.strftime("%H:%M:%S UTC"),
        "balance": kpis["balance"],
        "balance_start": kpis["balance_start"],
        "pnl_total": kpis["pnl_total"],
        "pnl_realized": kpis["pnl_realized"],
        "win_rate": kpis["win_rate"],
        "win_rate_hint": "Rendimiento sólido" if kpis["win_rate"] >= 60 else "Rendimiento mixto" if kpis["win_rate"] >= 40 else "Rendimiento débil",
        "trades": kpis["trade_count"],
        "trades_sub": f"{kpis['closed_trades']} cerradas ({kpis['open_trades']} abiertas)",
        "best_worst": f"{kpis['best_trade']:.2f}/{kpis['worst_trade']:.2f}",
        "regime": indicators["market_regime"],
        "adx": indicators["adx"],
        "supertrend": "BULL" if indicators["supertrend_direction"] == 1 else "BEAR" if indicators["supertrend_direction"] == -1 else "NEUTRAL",
        "volume_ratio": indicators["volume_ratio"],
        "volume_quantile": indicators["volume_quantile"],
        "volume_z": indicators["volume_z"],
        "volume_filter_passes": indicators["volume_filter_ok"],
        "volume_filter_reason": indicators["volume_filter_reason"],
        "daily_risk_pct": indicators["daily_loss_pct"],
        "daily_risk_used": indicators["daily_risk_used"],
        "kelly_pct": indicators["kelly_fraction"] * 100.0,
        "indicators_snapshot": {
            "ema9": _as_float(last.get(cols.get("ema_fast", "ema9"))),
            "ema21": _as_float(last.get(cols.get("ema_slow", "ema21"))),
            "ema200": _as_float(last.get(cols.get("ema_trend", "ema200"))),
            "rsi": _as_float(last.get("rsi")),
            "atr_pct": _as_float(last.get("atr_pct")),
        },
    }


def _build_chart_payload(symbol: str, timeframe_raw: str = "1d"):
    timeframe_label, timeframe_minutes = _parse_chart_timeframe(timeframe_raw)
    cfg = get_symbol_config(symbol)
    df = _fetch_candles(symbol, timeframe_minutes, 260)
    df = _compute_indicators(df, cfg)
    updated_at = datetime.now(timezone.utc)
    if df.empty or len(df) < 50:
        return {
            "symbol": symbol,
            "timeframe": timeframe_label,
            "price": 0.0,
            "ema9": 0.0,
            "ema21": 0.0,
            "rsi": 0.0,
            "candles": [],
            "ohlc": None,
            "updated_at": updated_at.isoformat(),
            "updated_label": updated_at.strftime("%H:%M:%S UTC"),
        }
    last = df.iloc[-1].to_dict()
    prev_close = _as_float(df.iloc[-2]["close"]) if len(df) > 1 else _as_float(last.get("open"))
    cols = df.attrs.get("indicator_cols", {})
    latest_ts = int(_as_float(last.get("ts"), 0))
    open_price = _as_float(last.get("open"))
    high_price = _as_float(last.get("high"))
    low_price = _as_float(last.get("low"))
    close_price = _as_float(last.get("close"))
    change_pct = ((close_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
    range_pct = ((high_price - low_price) / open_price * 100.0) if open_price else 0.0
    candles = [
        {
            "ts": int(row["ts"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in df.tail(220).iterrows()
    ]
    return {
        "symbol": symbol,
        "timeframe": timeframe_label,
        "price": close_price,
        "ema9": _as_float(last.get(cols.get("ema_fast", "ema9"))),
        "ema21": _as_float(last.get(cols.get("ema_slow", "ema21"))),
        "rsi": _as_float(last.get("rsi")),
        "candles": candles,
        "ohlc": {
            "ts": latest_ts,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "change_pct": change_pct,
            "range_pct": range_pct,
        },
        "updated_at": updated_at.isoformat(),
        "updated_label": updated_at.strftime("%H:%M:%S UTC"),
    }


def _portfolio_payload():
    balance = _get_balance()
    trades = trade_log.get_all_trades()
    stats = trade_log.get_stats()
    realized = sum(_as_float(t.get("pnl"), 0.0) for t in trades)
    open_positions = {}
    unrealized = 0.0
    state = _portfolio_state()
    for symbol in SYMBOLS:
        pos = _position(symbol)
        if pos:
            open_positions[symbol] = {
                "side": pos.get("side"),
                "size": pos.get("size"),
                "entry": pos.get("entry_price"),
                "pnl": pos.get("unrealisedPnl"),
            }
            unrealized += _as_float(pos.get("unrealisedPnl"))
        else:
            open_positions[symbol] = None
    total_risk = _as_float(state.get("total_risk_pct"))
    start = 50000.0
    daily = _daily_state()
    start_balance = _as_float(daily.get("daily_start_balance"), balance)
    daily_loss_pct = _daily_loss_pct(balance)
    cb_status = circuit_breaker.get_status()
    return {
        "total_balance": balance,
        "total_pnl_realized": realized,
        "total_pnl_unrealized": unrealized,
        "total_pnl_pct": ((realized + unrealized) / start_balance * 100.0) if start_balance else 0.0,
        "portfolio_risk_pct": total_risk,
        "positions": open_positions,
        "daily_loss_pct": daily_loss_pct,
        "trade_count": int(stats.get("total", 0)),
        "wins": int(stats.get("wins", 0)),
        "losses": int(stats.get("losses", 0)),
        "win_rate": _as_float(stats.get("win_rate")),
        "best_trade": _as_float(stats.get("best")),
        "worst_trade": _as_float(stats.get("worst")),
        "trade_stats": stats,
        "ml_confidence": 0.5,
        "circuit_breaker": cb_status,
    }
def _ml_info_payload():
    info = dict(ml_model.model_info("BTCUSDT"))
    info["n_trades_real_closed"] = int(trade_log.get_stats("BTCUSDT").get("total", 0))
    info["retraining"] = bool(_ml_retrain_state.get("running"))
    info["retraining_started_at"] = _ml_retrain_state.get("started_at")
    info["last_error"] = _ml_retrain_state.get("last_error")
    info["last_result"] = _ml_retrain_state.get("last_result")
    return info


def _history_payload():
    rows = []
    for timeframe in HISTORY_TIMEFRAMES:
        rows.append(historical_data.history_metadata("BTCUSDT", timeframe))
    return {
        "symbol": "BTCUSDT",
        "timeframes": rows,
        "refresh_running": bool(_history_refresh_state.get("running")),
        "refresh_started_at": _history_refresh_state.get("started_at"),
        "last_error": _history_refresh_state.get("last_error"),
        "last_result": _history_refresh_state.get("last_result"),
    }


def _search_payload():
    status = dict(search_loop.load_search_status())
    gate = dict(search_loop.load_live_gate())
    status["live_gate"] = gate
    status["search_log_tail"] = []
    if search_loop.SEARCH_LOG_PATH.exists():
        try:
            lines = [json.loads(line) for line in search_loop.SEARCH_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
            status["search_log_tail"] = lines[-10:]
        except Exception:
            logger.exception("Failed to read search log tail")
    return status


def _live_gate_payload():
    return dict(search_loop.load_live_gate())


def _start_history_refresh(days: int = 730):
    if _history_refresh_state.get("running"):
        return False

    def _worker():
        _history_refresh_state["running"] = True
        _history_refresh_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _history_refresh_state["last_error"] = None
        try:
            results = []
            for timeframe in HISTORY_TIMEFRAMES:
                results.append(historical_data.download_history("BTCUSDT", timeframe, days, force=False))
            _history_refresh_state["last_result"] = {"timeframes": results, "completed_at": datetime.now(timezone.utc).isoformat()}
            summary = ", ".join([f"{int(row['timeframe'])}m={int(row['rows'])}" for row in results])
            activity_log.push("BTCUSDT", "history", f"Hist?rico actualizado: {summary}")
            _history_refresh_state["last_error"] = str(exc)
            logger.exception("History refresh failed")
            activity_log.push("BTCUSDT", "history", f"Error actualizando hist?rico: {exc}")
        finally:
            _history_refresh_state["running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _start_search_run(budget: int = 10, base_days: int = 730):
    status = search_loop.load_search_status()
    if bool(status.get("running")) or _search_run_state.get("running"):
        return False

    def _worker():
        _search_run_state["running"] = True
        _search_run_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _search_run_state["last_error"] = None
        try:
            activity_log.push("BTCUSDT", "search", f"Loop de b?squeda iniciado budget={budget} base_days={base_days}")
            result = search_loop.run_search(budget=budget, base_days=base_days, symbol="BTCUSDT")
            _search_run_state["last_result"] = result
            activity_log.push("BTCUSDT", "search", f"B?squeda finalizada: ready_for_live={bool(result.get('ready_for_live'))} exhausted={bool(result.get('exhausted'))}")
        except Exception as exc:
            _search_run_state["last_error"] = str(exc)
            logger.exception("Search loop failed")
            activity_log.push("BTCUSDT", "search", f"Error en b?squeda de edge: {exc}")
        finally:
            _search_run_state["running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _start_ml_retrain():
    if _ml_retrain_state.get("running"):
        return False

    def _worker():
        _ml_retrain_state["running"] = True
        _ml_retrain_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _ml_retrain_state["last_error"] = None
        try:
            import bootstrap_ml

            cfg = get_symbol_config("BTCUSDT")
            activity_log.push("BTCUSDT", "ml", "⚙️ Reentreno manual iniciado desde dashboard")
            result = bootstrap_ml.bootstrap(
                symbol="BTCUSDT",
                days=int(cfg.get("ml_auto_bootstrap_days", 120)),
                timeframe=int(cfg.get("primary_timeframe", cfg.get("timeframe", 60))),
                trade_usdt=50,
                quiet=True,
            )
            _ml_retrain_state["last_result"] = result
            info = ml_model.model_info("BTCUSDT")
            activity_log.push(
                "BTCUSDT",
                "ml",
                f"✅ Reentreno manual finalizado: trained={int(info.get('trained_on', 0))} ready={bool(info.get('ready', False))} sharpe={float(info.get('val_sharpe', 0.0)):.2f} thr={float(info.get('suggested_threshold', 0.55)):.2f}",
            )
        except Exception as exc:
            _ml_retrain_state["last_error"] = str(exc)
            activity_log.push("BTCUSDT", "ml", f"❌ Reentreno manual error: {exc}")
        finally:
            _ml_retrain_state["running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


_cache: dict = {}
_health_cache: dict = {"report": None}
_recovery_cache: dict = {"report": None}
_auto_recovery_state: dict = {"last_ts": 0.0}
_history_refresh_state: dict = {"running": False, "started_at": None, "last_error": None, "last_result": None}
_search_run_state: dict = {"running": False, "started_at": None, "last_error": None, "last_result": None}
_ml_retrain_state: dict = {"running": False, "started_at": None, "last_error": None, "last_result": None}


def _store_health_report(report: dict):
    _health_cache["report"] = report
    _cache["health"] = report
    return report


def _get_health_report():
    report = _health_cache.get("report") or _cache.get("health")
    return report if isinstance(report, dict) else None


def _store_recovery_report(report: dict):
    _recovery_cache["report"] = report
    _cache["recovery"] = report
    return report


def _get_recovery_report():
    report = _recovery_cache.get("report") or _cache.get("recovery")
    return report if isinstance(report, dict) else None


def _recoverable_health_issues(report: dict) -> list[str]:
    issues = []
    for issue in report.get("issues", []) if isinstance(report, dict) else []:
        text = str(issue).lower()
        if (
            "huérf" in text
            or "huerf" in text
            or "portfolio_state" in text
            or "open_trade_id" in text
            or "daily_balance" in text
        ):
            issues.append(str(issue))
    return issues


def _should_auto_recover(report: dict) -> bool:
    if not isinstance(report, dict) or report.get("ok"):
        return False
    if report.get("recovery", {}).get("running"):
        return False
    if not _recoverable_health_issues(report):
        return False
    cooldown = max(60, int(os.getenv("RECOVERY_COOLDOWN_SEC", "300")))
    last_ts = float(_auto_recovery_state.get("last_ts") or 0.0)
    return (time.time() - last_ts) >= cooldown


def _agents_payload():
    portfolio = _cache.get("portfolio") or _portfolio_payload()
    health = _get_health_report() or {}
    recovery = _get_recovery_report() or {}
    symbols = {sym: _cache.get(sym) or {} for sym in SYMBOLS}
    open_positions = portfolio.get("positions", {}) or {}
    trade_gates = {
        sym: (symbols.get(sym, {}) or {}).get("trade_gate", {})
        for sym in SYMBOLS
    }
    ready_symbols = [sym for sym, gate in trade_gates.items() if gate.get("ready")]
    active_symbols = [sym for sym, pos in open_positions.items() if pos]
    warnings = health.get("issues", []) if isinstance(health, dict) else []
    recovery_issues = recovery.get("issues", []) if isinstance(recovery, dict) else []
    recovery_repairs = recovery.get("repairs", []) if isinstance(recovery, dict) else []
    daily_loss = _as_float((portfolio or {}).get("daily_loss_pct"))
    risk_pct = _as_float((portfolio or {}).get("portfolio_risk_pct"))
    recovery_running = bool(recovery.get("running")) if isinstance(recovery, dict) else False
    recovery_activity = str(recovery.get("activity") or "monitoring").lower() if isinstance(recovery, dict) else "monitoring"
    recovery_status = str(recovery.get("status") or "idle").lower() if isinstance(recovery, dict) else "idle"
    return {
        "title": "Agentes auxiliares",
        "subtitle": "Supervisión pixel art de market, risk, health, recovery, execution y reporting.",
        "agents": [
            {
                "id": "market-scout",
                "name": "Market Scout",
                "role": "Explora señales y vigila BTCUSDT en Binance.",
                "status": "active" if ready_symbols else "idle",
                "activity": "reading" if ready_symbols else "idle",
                "task": "Detectando confluencia lista para entrada en BTCUSDT.",
                "energy": min(100, 30 + len(ready_symbols) * 15),
                "accent": "#7dd3fc",
                "palette": ["#111827", "#22c55e", "#fbbf24"],
                "metrics": [
                    {"label": "ready", "value": len(ready_symbols)},
                    {"label": "watching", "value": len(SYMBOLS)},
                ],
            },
            {
                "id": "risk-warden",
                "name": "Risk Warden",
                "role": "Controla riesgo de portafolio y freno diario.",
                "status": "warn" if daily_loss >= 66 or risk_pct >= 4 else "active",
                "activity": "monitoring" if daily_loss < 66 and risk_pct < 4 else "alert",
                "task": "Manteniendo el tamaño dentro del límite y vigilando drawdown.",
                "energy": max(10, 100 - int(max(daily_loss, risk_pct * 20))),
                "accent": "#fbbf24",
                "palette": ["#111827", "#f59e0b", "#f87171"],
                "metrics": [
                    {"label": "risk%", "value": f"{risk_pct:.2f}"},
                    {"label": "daily%", "value": f"{daily_loss:.1f}"},
                ],
            },
            {
                "id": "recovery-agent",
                "name": "Recovery Agent",
                "role": "Repara desajustes y reconstruye el estado seguro desde Binance.",
                "status": "active" if recovery_running or warnings else "idle",
                "activity": recovery_activity if recovery_running else ("alert" if warnings else "monitoring"),
                "task": (
                    "Reparando registry, DB y snapshots desde Binance."
                    if recovery_running
                    else (
                        "Auto-repara desajustes del healthcheck con acciones conservadoras."
                        if warnings
                        else "Vigilando y listo para auto-reparar cuando aparezcan desajustes."
                    )
                ),
                "energy": 95 if recovery_running else 80 if warnings else 60,
                "accent": "#f59e0b",
                "palette": ["#111827", "#f59e0b", "#fb7185"],
                "metrics": [
                    {"label": "issues", "value": len(warnings)},
                    {"label": "repairs", "value": len(recovery_repairs)},
                    {"label": "mode", "value": "auto"},
                ],
                "status_hint": recovery_status,
            },
            {
                "id": "health-sentinel",
                "name": "Health Sentinel",
                "role": "Reconciliación y consistencia entre Binance, DB y JSON.",
                "status": "warn" if warnings else "active",
                "activity": "waiting" if warnings else "reading",
                "task": "Comparando estado local contra Binance en busca de desajustes.",
                "energy": 100 if not warnings else 55,
                "accent": "#36d399",
                "palette": ["#111827", "#36d399", "#60a5fa"],
                "metrics": [
                    {"label": "issues", "value": len(warnings)},
                    {"label": "synced", "value": "yes" if not warnings else "check"},
                ],
            },
            {
                "id": "execution-runner",
                "name": "Execution Runner",
                "role": "Abre, gestiona y cierra posiciones con control de riesgo.",
                "status": "active" if active_symbols else "idle",
                "activity": "typing" if active_symbols else "walking",
                "task": "Manteniendo posiciones vivas y aplicando break-even / trailing.",
                "energy": min(100, 35 + len(active_symbols) * 20),
                "accent": "#c084fc",
                "palette": ["#111827", "#c084fc", "#22c55e"],
                "metrics": [
                    {"label": "open", "value": len(active_symbols)},
                    {"label": "watch", "value": len([s for s in SYMBOLS if s not in active_symbols])},
                ],
            },
            {
                "id": "report-scribe",
                "name": "Report Scribe",
                "role": "Resume historial, actividad y estado del tablero.",
                "status": "active",
                "activity": "typing",
                "task": "Escribiendo notas breves para revisión humana.",
                "energy": 88,
                "accent": "#60a5fa",
                "palette": ["#111827", "#60a5fa", "#f59e0b"],
                "metrics": [
                    {"label": "logs", "value": len(_recent_activity(None, 20))},
                    {"label": "assets", "value": len(SYMBOLS)},
                ],
            },
        ],
        "summary": {
            "ready_symbols": ready_symbols,
            "active_symbols": active_symbols,
            "warnings": warnings[:3],
            "recovery_issues": recovery_issues[:3],
            "recovery_repairs": recovery_repairs[:3],
        },
    }


async def _refresh_cache():
    while True:
        try:
            _cache["portfolio"] = await asyncio.to_thread(_portfolio_payload)
        except Exception:
            logger.exception("Failed to refresh portfolio cache")
        try:
            _cache["agents"] = await asyncio.to_thread(_agents_payload)
        except Exception:
            logger.exception("Failed to refresh agents cache")
        try:
            _cache["activity"] = [evt for evt in activity_log.get_recent(50) if not _is_healthcheck_event(evt)]
        except Exception:
            logger.exception("Failed to refresh activity cache")
        for sym in TAB_SYMBOLS:
            try:
                _cache[sym] = await asyncio.to_thread(_build_symbol_payload, sym)
            except Exception:
                logger.exception("Failed to refresh symbol cache for %s", sym)
        await asyncio.sleep(1)


async def _healthcheck_loop():
    interval = max(60, int(os.getenv("HEALTHCHECK_INTERVAL_SEC", "900")))
    await asyncio.sleep(10)
    while True:
        try:
            report = await asyncio.to_thread(_run_healthcheck)
            _store_health_report(report)
            if _should_auto_recover(report):
                _auto_recovery_state["last_ts"] = time.time()
                recovery_report = await asyncio.to_thread(_run_recovery, True, True)
                _store_recovery_report(recovery_report)
        except Exception as exc:
            _store_health_report({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "reconciled": False,
                "error": str(exc),
                "issues": [str(exc)],
            })
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup():
    stream_specs = {
        (symbol, int(get_symbol_config(symbol).get("primary_timeframe", get_symbol_config(symbol).get("timeframe", os.getenv("TIMEFRAME", "60")))))
        for symbol in SYMBOLS
    }
    for symbol in SYMBOLS:
        stream_specs.add((symbol, int(get_symbol_config(symbol).get("confirmation_timeframe", 240))))
        for minutes in CHART_TIMEFRAMES.values():
            stream_specs.add((symbol, minutes))
    market_stream.start(sorted(stream_specs))
    asyncio.create_task(_refresh_cache())
    asyncio.create_task(_healthcheck_loop())


@app.get("/")
async def index():
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/healthcheck")
async def api_healthcheck():
    report = await asyncio.to_thread(_run_healthcheck)
    return _store_health_report(report)


@app.post("/api/recovery")
async def api_recovery():
    report = await asyncio.to_thread(_run_recovery)
    return _store_recovery_report(report)


@app.get("/api/recovery/latest")
async def api_recovery_latest():
    report = _get_recovery_report()
    if report is None:
        report = await asyncio.to_thread(_run_recovery)
        _store_recovery_report(report)
    return report


@app.get("/api/healthcheck/latest")
async def api_healthcheck_latest():
    report = _get_health_report()
    if report is None:
        report = await asyncio.to_thread(_run_healthcheck)
        _store_health_report(report)
    return report


@app.get("/api/circuit-breaker")
async def api_circuit_breaker():
    return circuit_breaker.get_status()


@app.post("/api/circuit-breaker/resume")
async def api_circuit_breaker_resume():
    circuit_breaker.set_manual_override(True)
    ok = circuit_breaker.force_resume()
    return {"ok": ok, "status": circuit_breaker.get_status()}


@app.get("/api/ml/info")
async def api_ml_info():
    return _ml_info_payload()


@app.post("/api/ml/retrain")
async def api_ml_retrain():
    return {"started": _start_ml_retrain()}


@app.get("/api/loop/status")
async def api_loop_status():
    return activity_log.get_recent(50, event_type="loop")


@app.get("/api/history")
async def api_history():
    return await asyncio.to_thread(_history_payload)


@app.post("/api/history/refresh")
async def api_history_refresh():
    return {"started": _start_history_refresh()}


@app.get("/api/search")
async def api_search():
    return await asyncio.to_thread(_search_payload)


@app.post("/api/search/run")
async def api_search_run():
    return {"started": _start_search_run()}


@app.get("/api/live-gate")
async def api_live_gate():
    return await asyncio.to_thread(_live_gate_payload)


@app.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await asyncio.to_thread(_portfolio_payload)
            _cache["portfolio"] = payload
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except Exception:
        logger.exception("Portfolio websocket failed")


@app.websocket("/ws/activity")
async def ws_activity(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = [evt for evt in activity_log.get_recent(50) if not _is_healthcheck_event(evt)]
            _cache["activity"] = payload
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except Exception:
        logger.exception("Activity websocket failed")


@app.websocket("/ws/agents")
async def ws_agents(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await asyncio.to_thread(_agents_payload)
            _cache["agents"] = payload
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except Exception:
        logger.exception("Agents websocket failed")


@app.websocket("/ws/chart/{symbol}/{timeframe}")
async def ws_chart(websocket: WebSocket, symbol: str, timeframe: str):
    await websocket.accept()
    try:
        while True:
            payload = await asyncio.to_thread(_build_chart_payload, symbol, timeframe)
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except Exception:
        logger.exception("Chart websocket failed for %s %s", symbol, timeframe)


@app.websocket("/ws/{symbol}")
async def ws_symbol(websocket: WebSocket, symbol: str):
    await websocket.accept()
    try:
        while True:
            payload = await asyncio.to_thread(_build_symbol_payload, symbol)
            _cache[symbol] = payload
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except Exception:
        logger.exception("Symbol websocket failed for %s", symbol)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard_multi:app", host="0.0.0.0", port=8000, reload=False)
