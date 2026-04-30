from __future__ import annotations

import pandas as pd

import dashboard_multi


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(80):
        price = 75000.0 + i * 10.0
        rows.append(
            {
                "ts": 1777000000000 + i * 3600000,
                "open": price - 5.0,
                "high": price + 15.0,
                "low": price - 20.0,
                "close": price,
                "volume": 100.0 + i,
                "ema9": price - 4.0,
                "ema21": price - 8.0,
                "ema200": price - 60.0,
                "rsi": 54.0,
                "adx": 28.0,
                "atr": 320.0,
                "atr_pct": 0.42,
                "vol_ma20": 95.0,
                "supertrend_direction": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["indicator_cols"] = {
        "ema_fast": "ema9",
        "ema_slow": "ema21",
        "ema_trend": "ema200",
        "supertrend_dir": "supertrend_direction",
    }
    return frame


def test_dashboard_payload_contains_frontend_schema(monkeypatch) -> None:
    frame = _frame()
    monkeypatch.setattr(dashboard_multi, "_fetch_candles", lambda symbol, timeframe, limit=500: frame.copy())
    monkeypatch.setattr(dashboard_multi, "_compute_indicators", lambda df, cfg: df)
    monkeypatch.setattr(
        dashboard_multi,
        "build_features",
        lambda df: pd.DataFrame(
            [{"volume_z": -0.25, "volume_quantile": 0.35}],
            index=[df.index[-1]],
        ),
    )
    monkeypatch.setattr(dashboard_multi, "_get_balance", lambda: 4922.52)
    monkeypatch.setattr(dashboard_multi, "_recent_activity", lambda symbol, limit=20: [])
    monkeypatch.setattr(dashboard_multi, "_position_snapshot", lambda symbol, price=0.0: {"position": None, "source": "binance", "error": None})
    monkeypatch.setattr(dashboard_multi.trade_log, "get_stats", lambda symbol=None: {"total": 1, "wins": 0, "losses": 1, "win_rate": 0.0, "best": -0.12, "worst": -0.12, "reconciled": 0})
    monkeypatch.setattr(dashboard_multi.trade_log, "get_all_trades", lambda symbol=None: [{"pnl": -0.12, "symbol": "BTCUSDT"}])
    monkeypatch.setattr(
        dashboard_multi.ml_model,
        "model_info",
        lambda symbol="BTCUSDT": {
            "ready": False,
            "trained_on": 315,
            "val_sharpe": 1.06,
            "val_auc": 0.493,
            "suggested_threshold": 0.45,
            "not_ready_reason": "auc < 0.55",
            "no_profitable_threshold": False,
            "validation_trades": 33,
        },
    )
    monkeypatch.setattr(dashboard_multi.ml_model, "is_ready", lambda symbol="BTCUSDT": False)
    monkeypatch.setattr(dashboard_multi.circuit_breaker, "get_status", lambda: {"paused": False, "reason": "", "paused_at": None, "resume_after": None, "manual_override": False})

    payload = dashboard_multi._build_symbol_payload("BTCUSDT")
    expected = {
        "balance",
        "balance_start",
        "pnl_total",
        "pnl_realized",
        "win_rate",
        "trades",
        "trades_sub",
        "best_worst",
        "regime",
        "adx",
        "supertrend",
        "volume_ratio",
        "volume_quantile",
        "volume_z",
        "volume_filter_passes",
        "volume_filter_reason",
        "daily_risk_pct",
        "daily_risk_used",
        "kelly_pct",
        "ml",
        "trade_gate",
        "circuit_breaker",
        "signal",
        "position",
        "last_candle",
        "indicators_snapshot",
        "kpis",
        "indicators",
        "updated_label",
    }
    assert expected.issubset(payload.keys()), payload.keys()
    assert {"balance", "pnl_total", "ml_ready", "trade_stats", "primary_only_mode"}.issubset(payload["kpis"].keys())
    assert {"adx", "volume_filter_ok", "volume_quantile", "volume_z", "market_regime"}.issubset(payload["indicators"].keys())

    portfolio = dashboard_multi._portfolio_payload()
    portfolio_expected = {
        "total_balance",
        "total_pnl_realized",
        "total_pnl_unrealized",
        "total_pnl_pct",
        "portfolio_risk_pct",
        "positions",
        "daily_loss_pct",
        "circuit_breaker",
    }
    assert portfolio_expected.issubset(portfolio.keys()), portfolio.keys()
