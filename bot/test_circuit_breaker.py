from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import circuit_breaker


CFG = {
    "cb_daily_loss_pct": 3.0,
    "cb_weekly_loss_pct": 7.0,
    "cb_consecutive_losses": 4,
    "cb_rolling_window": 20,
    "cb_rolling_winrate_min": 0.30,
    "cb_volatility_spike_pct": 8.0,
    "cb_cooldown_hours": 12.0,
}


def _reset() -> None:
    if circuit_breaker.STATE_PATH.exists():
        circuit_breaker.STATE_PATH.unlink()


def test_consecutive_losses_pause() -> None:
    _reset()
    for _ in range(5):
        circuit_breaker.update_on_trade_close(-10.0, False)
    paused, reason = circuit_breaker.check_circuit_breaker(5000.0, 5000.0, 5000.0, {"open": 100, "high": 101, "low": 99}, CFG)
    assert paused and "pérdidas consecutivas" in reason.lower(), reason


def test_daily_loss_pause() -> None:
    _reset()
    paused, reason = circuit_breaker.check_circuit_breaker(4800.0, 5000.0, 5000.0, {"open": 100, "high": 101, "low": 99}, CFG)
    assert paused and "pérdida diaria" in reason.lower(), reason


def test_rolling_winrate_pause() -> None:
    _reset()
    outcomes = [True, False, False, False, True, False, False, False, True, False, False, False, True, False, False]
    for win in outcomes:
        circuit_breaker.update_on_trade_close(5.0 if win else -5.0, win)
    paused, reason = circuit_breaker.check_circuit_breaker(5000.0, 5000.0, 5000.0, {"open": 100, "high": 101, "low": 99}, CFG)
    assert paused and "winrate" in reason.lower(), reason


def test_resume_after_cooldown() -> None:
    _reset()
    circuit_breaker.force_pause("test", 1)
    state = json.loads(circuit_breaker.STATE_PATH.read_text(encoding="utf-8"))
    state["resume_after"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    circuit_breaker.STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    paused, _ = circuit_breaker.is_paused()
    assert not paused
    assert circuit_breaker.STATE_PATH.exists()


if __name__ == "__main__":
    test_consecutive_losses_pause()
    test_daily_loss_pause()
    test_rolling_winrate_pause()
    test_resume_after_cooldown()
    print("test_circuit_breaker_ok")
