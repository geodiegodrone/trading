from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ml_model


def _make_trade(index: int, win: bool) -> dict:
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    if win:
        trade = {
            "entry_price": 100.0,
            "ema9": 110.0,
            "ema21": 100.0,
            "ema200": 90.0,
            "rsi": 80.0,
            "adx": 30.0,
            "supertrend_dir": 1,
            "volume_ratio": 2.0,
            "atr_pct": 0.5,
            "return_1_pct": 2.0,
            "return_2_pct": 2.0,
            "return_3_pct": 2.0,
            "return_4_pct": 2.0,
            "return_5_pct": 2.0,
            "range_1_pct": 1.0,
            "range_2_pct": 1.0,
            "range_3_pct": 1.0,
            "range_5_pct": 1.0,
            "body_1_pct": 0.8,
            "body_2_pct": 0.8,
            "body_3_pct": 0.8,
            "body_avg_3_pct": 0.8,
            "volume_trend_1": 1.5,
            "volume_trend_2": 1.5,
            "volume_trend_3": 1.5,
            "volume_trend_3_ratio": 1.5,
            "close_pos_5": 0.8,
            "ema9_slope_3_pct": 0.5,
            "side": "Buy",
            "r_multiple": 1.5,
            "pnl": 1.5,
            "result": "WIN",
        }
    else:
        trade = {
            "entry_price": 100.0,
            "ema9": 90.0,
            "ema21": 100.0,
            "ema200": 110.0,
            "rsi": 20.0,
            "adx": 30.0,
            "supertrend_dir": -1,
            "volume_ratio": 0.5,
            "atr_pct": 0.5,
            "return_1_pct": -2.0,
            "return_2_pct": -2.0,
            "return_3_pct": -2.0,
            "return_4_pct": -2.0,
            "return_5_pct": -2.0,
            "range_1_pct": 1.0,
            "range_2_pct": 1.0,
            "range_3_pct": 1.0,
            "range_5_pct": 1.0,
            "body_1_pct": 0.8,
            "body_2_pct": 0.8,
            "body_3_pct": 0.8,
            "body_avg_3_pct": 0.8,
            "volume_trend_1": 0.5,
            "volume_trend_2": 0.5,
            "volume_trend_3": 0.5,
            "volume_trend_3_ratio": 0.5,
            "close_pos_5": 0.2,
            "ema9_slope_3_pct": -0.5,
            "side": "Sell",
            "r_multiple": -1.0,
            "pnl": -1.0,
            "result": "LOSS",
        }
    trade.update(
        {
            "ts": base_time.isoformat(),
            "id": index + 1,
            "symbol": "BTCUSDT",
            "qty": 0.01,
            "notional_usdt": 1.0,
            "risk_usdt": 1.0,
            "candle_body_pct": 0.8,
        }
    )
    return trade


def main() -> None:
    trades = []
    for index in range(120):
        trades.append(_make_trade(index, win=(index % 2 == 0)))

    ml_model.train(trades, symbol="BTCUSDT")

    assert ml_model.is_ready("BTCUSDT"), "model should be ready after training"

    info = ml_model.model_info("BTCUSDT")
    assert info["trained_on"] >= 50, info

    winning_trade = _make_trade(120, win=True)
    winning_features = ml_model.features_from_trade_row(winning_trade)

    score = ml_model.predict(winning_features, symbol="BTCUSDT")
    assert score > 0.5, score

    print("smoke_ok", info["trained_on"], f"{score:.3f}")


if __name__ == "__main__":
    main()
