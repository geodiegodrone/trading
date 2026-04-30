from __future__ import annotations

import multi_bot


def test_primary_only_mode_halves_size_and_logs(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(multi_bot.activity_log, "push", lambda symbol, event_type, message, data=None: events.append({"symbol": symbol, "type": event_type, "message": message, "data": data or {}}))
    multi_bot._LAST_DECISION_LOG.clear()

    full_trade = 100.0
    reduced_trade = multi_bot._primary_only_trade_usdt(full_trade, False)
    assert reduced_trade == 50.0

    multi_bot._log_decision(
        "BTCUSDT",
        {
            "regime": "TRENDING",
            "strategy_used": "trend",
            "signal": "LONG",
            "ml_ready": False,
            "ml_confidence": 0.5,
            "ml_threshold_used": 0.55,
            "circuit_breaker": "ok",
            "decision": "ABRIR",
            "decision_reason_code": "open",
            "decision_reason": "PRIMARY ONLY 50% size",
            "mode": "primary_only",
            "size_pct": "50%",
        },
    )

    assert events, "expected loop log"
    assert "mode=primary_only size=50%" in events[-1]["message"], events[-1]["message"]
    assert events[-1]["data"]["decision_reason_code"] == "open"
