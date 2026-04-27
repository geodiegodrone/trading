import asyncio
import json
import os
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
import market_stream
from bot_config import get_symbol_config, parse_symbols
import ml_model
import trade_log

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

app = FastAPI()

HTML_PATH = ROOT / "dashboard_multi.html"
PORTFOLIO_FILE = ROOT / "portfolio_state.json"
DAILY_FILE = ROOT / "daily_balance.json"
OPEN_TRADE_FILE = ROOT / "open_trade_id.json"

SYMBOLS = parse_symbols()
TAB_SYMBOLS = list(SYMBOLS)


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
    vol = _as_float(last.get("volume"))
    vol_ma = _as_float(last.get("vol_ma20"))
    close = _as_float(last.get("close"))
    adx_threshold = float(cfg.get("adx_threshold", 25))
    volume_mult = float(cfg.get("volume_mult", 1.2))
    rsi_min = float(cfg.get("rsi_min", 30))
    rsi_max = float(cfg.get("rsi_max", 70))
    if (
        ema_fast > ema_slow
        and close > ema_trend
        and st == 1
        and adx > adx_threshold
        and vol > vol_ma * volume_mult
        and rsi_min < rsi < rsi_max
    ):
        return "LONG"
    if (
        ema_fast < ema_slow
        and close < ema_trend
        and st == -1
        and adx > adx_threshold
        and vol > vol_ma * volume_mult
        and rsi_min < rsi < rsi_max
    ):
        return "SHORT"
    return "NEUTRAL"


def _trade_gate_state(last, cfg, position=None, ml_ready=False, ml_confidence=0.5):
    if last is None:
        return {"status": "SIN DATOS", "reason": "Esperando velas suficientes", "ready": False}

    cols = {
        "ema_fast": f"ema{int(cfg.get('ema_fast', 9))}",
        "ema_slow": f"ema{int(cfg.get('ema_slow', 21))}",
        "ema_trend": f"ema{int(cfg.get('ema_trend', 200))}",
    }
    ema_fast = _as_float(last.get(cols["ema_fast"]))
    ema_slow = _as_float(last.get(cols["ema_slow"]))
    ema_trend = _as_float(last.get(cols["ema_trend"]))
    close = _as_float(last.get("close"))
    rsi = _as_float(last.get("rsi"))
    adx = _as_float(last.get("adx"))
    volume = _as_float(last.get("volume"))
    vol_ma20 = _as_float(last.get("vol_ma20"))
    supertrend_dir = int(_as_float(last.get("supertrend_direction")))
    adx_threshold = float(cfg.get("adx_threshold", 25))
    volume_mult = float(cfg.get("volume_mult", 1.2))
    rsi_min = float(cfg.get("rsi_min", 30))
    rsi_max = float(cfg.get("rsi_max", 70))
    ml_threshold = float(cfg.get("ml_threshold", 0.55))

    if position:
        return {"status": "POSICIÓN ABIERTA", "reason": "Ya hay una operación activa", "ready": False}

    long_ok = (
        ema_fast > ema_slow
        and close > ema_trend
        and supertrend_dir == 1
        and adx > adx_threshold
        and volume > (vol_ma20 * volume_mult if vol_ma20 else 0)
        and rsi_min < rsi < rsi_max
    )
    short_ok = (
        ema_fast < ema_slow
        and close < ema_trend
        and supertrend_dir == -1
        and adx > adx_threshold
        and volume > (vol_ma20 * volume_mult if vol_ma20 else 0)
        and rsi_min < rsi < rsi_max
    )
    signal = "LONG" if long_ok else "SHORT" if short_ok else "NEUTRAL"
    if signal == "NEUTRAL":
        if adx <= adx_threshold:
            return {"status": "ESPERANDO", "reason": f"ADX bajo ({adx:.2f} <= {adx_threshold:.2f})", "ready": False}
        if vol_ma20 and volume <= vol_ma20 * volume_mult:
            ratio = (volume / vol_ma20) if vol_ma20 else 0.0
            return {"status": "ESPERANDO", "reason": f"Volumen bajo (x{ratio:.2f} < x{volume_mult:.2f})", "ready": False}
        if not (rsi_min < rsi < rsi_max):
            return {"status": "ESPERANDO", "reason": f"RSI fuera de rango ({rsi:.2f} vs {rsi_min:.0f}-{rsi_max:.0f})", "ready": False}
        return {"status": "ESPERANDO", "reason": "Sin confluencia de entrada", "ready": False}
    if not ml_ready:
        return {"status": "FILTRADO", "reason": "ML entrenando", "ready": False}
    if ml_confidence < ml_threshold:
        return {"status": "FILTRADO", "reason": f"ML bajo umbral ({ml_confidence:.2f} < {ml_threshold:.2f})", "ready": False}
    return {"status": "LISTO PARA OPERAR", "reason": f"Señal {signal} confirmada por filtros", "ready": True}


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
                pass
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
    cfg_timeframe = int(cfg.get("timeframe", os.getenv("TIMEFRAME", "5")))
    df = _fetch_candles(symbol, cfg_timeframe)
    df = _compute_indicators(df, cfg)
    events = _recent_activity(symbol, 20)
    updated_at = datetime.now(timezone.utc)
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
    features = ml_model.snapshot_features(last, df.tail(10).to_dict("records"), cols)
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
        "trade_stats": stats,
    }
    kpis["pnl_total"] = kpis["pnl_realized"] + kpis["pnl_unrealized"]
    indicators = {
        "adx": _as_float(last.get("adx")),
        "supertrend_direction": int(_as_float(last.get(cols.get("supertrend_dir", "supertrend_direction")))) if last.get(cols.get("supertrend_dir", "supertrend_direction")) is not None else 0,
        "volume_ratio": _as_float(last.get("volume")) / _as_float(last.get("vol_ma20")) if _as_float(last.get("vol_ma20")) else 0.0,
        "atr": _as_float(last.get("atr")),
        "ema200": _as_float(last.get(cols.get("ema_trend", "ema200"))),
        "market_regime": "TRENDING" if _as_float(last.get("adx")) > float(cfg.get("adx_threshold", 25)) else "RANGING",
        "daily_loss_pct": _daily_loss_pct(balance),
        "kelly_fraction": _kelly_fraction(),
    }
    trade_gate = _trade_gate_state(last, cfg, position=position, ml_ready=ml_ready, ml_confidence=ml_confidence)
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
        "signal": _signal_from_last(last, cfg),
        "position": position,
        "candles": candles,
        "kpis": kpis,
        "indicators": indicators,
        "trade_gate": trade_gate,
        "events": events,
        "position_source": position_snapshot.get("source"),
        "position_error": position_snapshot.get("error"),
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
    }




_cache: dict = {}
_health_cache: dict = {"report": None}
_recovery_cache: dict = {"report": None}
_auto_recovery_state: dict = {"last_ts": 0.0}


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
                "role": "Explora señales y vigila BTC, ETH, XRP, SOL, XAU y XAG.",
                "status": "active" if ready_symbols else "idle",
                "activity": "reading" if ready_symbols else "idle",
                "task": "Detectando activos con confluencia lista para entrada.",
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
            pass
        try:
            _cache["agents"] = await asyncio.to_thread(_agents_payload)
        except Exception:
            pass
        try:
            _cache["activity"] = [evt for evt in activity_log.get_recent(50) if not _is_healthcheck_event(evt)]
        except Exception:
            pass
        for sym in TAB_SYMBOLS:
            try:
                _cache[sym] = await asyncio.to_thread(_build_symbol_payload, sym)
            except Exception:
                pass
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
    market_stream.start([(symbol, int(get_symbol_config(symbol).get("timeframe", os.getenv("TIMEFRAME", "5")))) for symbol in SYMBOLS])
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
        pass


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
        pass


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
        pass


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
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard_multi:app", host="0.0.0.0", port=8000, reload=False)
