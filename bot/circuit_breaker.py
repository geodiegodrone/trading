from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import trade_log

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "circuit_breaker_state.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _default_state() -> Dict[str, Any]:
    return {
        "paused": False,
        "pause_reason": "",
        "paused_at": None,
        "resume_after": None,
        "consecutive_losses": 0,
        "last_trade_results": [],
        "last_pause_event": {},
        "manual_override": False,
        "recent_trade_pnls": [],
        "last_resume_at": None,
    }


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            state = _default_state()
            state.update(data)
            return state
    except Exception:
        pass
    return _default_state()


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _next_utc_day() -> datetime:
    now = _utc_now()
    next_day = (now + timedelta(days=1)).date()
    return datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)


def _pause(reason: str, resume_after: datetime) -> Tuple[bool, str]:
    state = _load_state()
    state["paused"] = True
    state["pause_reason"] = reason
    state["paused_at"] = _utc_now().isoformat()
    state["resume_after"] = resume_after.isoformat()
    state["last_pause_event"] = {
        "reason": reason,
        "paused_at": state["paused_at"],
        "resume_after": state["resume_after"],
    }
    _save_state(state)
    return True, reason


def update_on_trade_close(pnl: float, win: bool) -> None:
    state = _load_state()
    state["consecutive_losses"] = 0 if win else int(state.get("consecutive_losses", 0)) + 1
    results = list(state.get("last_trade_results") or [])
    results.append(1 if win else 0)
    state["last_trade_results"] = results[-max(20, len(results)) :]
    pnls = list(state.get("recent_trade_pnls") or [])
    pnls.append(_safe_float(pnl))
    state["recent_trade_pnls"] = pnls[-100:]
    _save_state(state)


def is_paused() -> Tuple[bool, str]:
    state = _load_state()
    if not state.get("paused"):
        return False, ""
    resume_after = _parse_dt(state.get("resume_after"))
    if resume_after and _utc_now() >= resume_after and not bool(state.get("manual_override")):
        state["paused"] = False
        state["pause_reason"] = ""
        state["paused_at"] = None
        state["resume_after"] = None
        state["last_resume_at"] = _utc_now().isoformat()
        _save_state(state)
        return False, ""
    return True, str(state.get("pause_reason") or "Pausado")


def force_pause(reason: str, hours: float) -> None:
    resume_after = _utc_now() + timedelta(hours=max(0.0, _safe_float(hours, 0.0)))
    _pause(reason, resume_after)


def set_manual_override(enabled: bool) -> None:
    state = _load_state()
    state["manual_override"] = bool(enabled)
    _save_state(state)


def force_resume() -> bool:
    state = _load_state()
    if not bool(state.get("manual_override")):
        return False
    state["paused"] = False
    state["pause_reason"] = ""
    state["paused_at"] = None
    state["resume_after"] = None
    state["last_resume_at"] = _utc_now().isoformat()
    _save_state(state)
    return True


def _rolling_drawdown() -> float:
    state = _load_state()
    pnls = list(state.get("recent_trade_pnls") or [])
    if len(pnls) < 5:
        try:
            trades = trade_log.get_all_trades("BTCUSDT")
        except Exception:
            trades = []
        pnls = [_safe_float(row.get("pnl")) for row in trades if row.get("pnl") is not None and str(row.get("result") or "").upper() != "RECONCILED"][-100:]
    if not pnls:
        return 0.0
    equity = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        equity.append(running)
    peak = float("-inf")
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    base = max(abs(peak), 1.0)
    return (max_dd / base) * 100.0


def check_circuit_breaker(balance: float, daily_start: float, weekly_start: float, last_candle: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    balance = _safe_float(balance)
    daily_start = _safe_float(daily_start, balance)
    weekly_start = _safe_float(weekly_start, balance)
    state = _load_state()

    if daily_start > 0 and balance <= daily_start * (1 - float(cfg["cb_daily_loss_pct"]) / 100.0):
        pct = ((daily_start - balance) / daily_start) * 100.0
        return _pause(f"Pérdida diaria {pct:.1f}% >= {cfg['cb_daily_loss_pct']}%", _next_utc_day())

    if weekly_start > 0 and balance <= weekly_start * (1 - float(cfg["cb_weekly_loss_pct"]) / 100.0):
        pct = ((weekly_start - balance) / weekly_start) * 100.0
        return _pause(f"Pérdida semanal {pct:.1f}% >= {cfg['cb_weekly_loss_pct']}%", _utc_now() + timedelta(hours=24))

    consecutive_losses = int(state.get("consecutive_losses", 0))
    if consecutive_losses >= int(cfg["cb_consecutive_losses"]):
        return _pause(f"{consecutive_losses} pérdidas consecutivas", _utc_now() + timedelta(hours=float(cfg["cb_cooldown_hours"])))

    recent = list(state.get("last_trade_results") or [])[-int(cfg["cb_rolling_window"]):]
    if len(recent) >= 10:
        winrate = sum(recent) / max(1, len(recent))
        if winrate < float(cfg["cb_rolling_winrate_min"]):
            return _pause(
                f"Winrate {winrate:.0%} en últimos {len(recent)} trades < {float(cfg['cb_rolling_winrate_min']):.0%}",
                _utc_now() + timedelta(hours=float(cfg["cb_cooldown_hours"]) * 2.0),
            )

    open_ = _safe_float(last_candle.get("open"))
    high = _safe_float(last_candle.get("high"))
    low = _safe_float(last_candle.get("low"))
    if open_ > 0:
        spike_pct = abs((high - low) / open_) * 100.0
        if spike_pct > float(cfg["cb_volatility_spike_pct"]):
            return _pause(f"Vela extrema {spike_pct:.1f}% > {cfg['cb_volatility_spike_pct']}%", _utc_now() + timedelta(hours=1))

    drawdown_pct = _rolling_drawdown()
    if drawdown_pct > 15.0:
        return _pause(f"Drawdown equity {drawdown_pct:.1f}% > 15%", _utc_now() + timedelta(hours=24))

    return False, ""


def get_status() -> Dict[str, Any]:
    paused, reason = is_paused()
    state = _load_state()
    return {
        "paused": paused,
        "reason": reason,
        "paused_at": state.get("paused_at"),
        "resume_after": state.get("resume_after"),
        "consecutive_losses": int(state.get("consecutive_losses", 0) or 0),
        "last_trade_results": list(state.get("last_trade_results") or [])[-20:],
        "last_pause_event": dict(state.get("last_pause_event") or {}),
        "manual_override": bool(state.get("manual_override")),
        "last_resume_at": state.get("last_resume_at"),
    }
