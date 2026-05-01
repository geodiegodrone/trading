from __future__ import annotations

import pandas as pd

import multi_bot


def test_live_gate_false_blocks_open_trade(monkeypatch) -> None:
    opened = []
    cfg = multi_bot._get_symbol_config("BTCUSDT")
    stop_event = multi_bot.threading.Event()

    frame = pd.DataFrame(
        {
            "ts": list(range(250)),
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.5] * 250,
            "volume": [100.0] * 250,
            "atr": [1.0] * 250,
            "supertrend_direction": [1.0] * 250,
        }
    )
    frame.attrs["indicator_cols"] = {
        "ema_fast": "ema9",
        "ema_slow": "ema21",
        "ema_trend": "ema200",
        "supertrend_dir": "supertrend_direction",
    }

    monkeypatch.setattr(multi_bot.binance_broker, "is_symbol_trading", lambda symbol: True)
    monkeypatch.setattr(multi_bot.binance_broker, "set_leverage", lambda symbol, leverage: None)
    monkeypatch.setattr(multi_bot.binance_broker, "get_balance", lambda: 5000.0)
    monkeypatch.setattr(multi_bot, "_load_live_gate", lambda: {"ready_for_live": False, "reason": "test gate", "config": {}})
    monkeypatch.setattr(multi_bot, "_load_daily_balance", lambda: {"daily_start_balance": 5000.0})
    monkeypatch.setattr(multi_bot, "_load_weekly_balance", lambda: {"weekly_start_balance": 5000.0})
    monkeypatch.setattr(multi_bot, "_fetch_candles", lambda symbol, timeframe, limit=450: frame.copy())
    monkeypatch.setattr(multi_bot, "_compute_indicators", lambda df, runtime_cfg: df)
    monkeypatch.setattr(multi_bot, "build_features", lambda df: pd.DataFrame([{"side_buy": 1.0, "supertrend_direction": 1.0}], index=[df.index[-1]]))
    monkeypatch.setattr(multi_bot, "_select_signal", lambda symbol, primary_df, confirmation_df, runtime_cfg: ("LONG", "trend", "TRENDING", {"reason": "test signal"}))
    monkeypatch.setattr(multi_bot, "_get_current_position", lambda symbol: None)
    monkeypatch.setattr(multi_bot, "_log_decision", lambda symbol, ctx: None)
    monkeypatch.setattr(multi_bot.activity_log, "push", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi_bot, "_open_trade", lambda *args, **kwargs: opened.append(args))
    monkeypatch.setattr(multi_bot.time, "sleep", lambda seconds: stop_event.set())

    multi_bot.run_symbol("BTCUSDT", cfg, stop_event)
    assert opened == []


if __name__ == "__main__":
    from pytest import MonkeyPatch

    test_live_gate_false_blocks_open_trade(MonkeyPatch())
    print("test_multi_bot_gate_ok")
