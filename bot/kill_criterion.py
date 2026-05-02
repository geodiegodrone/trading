from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

import trade_log


ROOT = Path(__file__).resolve().parent
KILL_FLAG_PATH = ROOT / "KILL.flag"
LOG_PATH = ROOT / "kill_criterion_log.json"
SYMBOL = "BTCUSDT"
MIN_DAYS = 90
RECHECK_DAYS = 30


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return dict(default)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _real_closed_trades(symbol: str = SYMBOL) -> list[dict[str, Any]]:
    rows = trade_log.get_all_trades(symbol)
    return [row for row in rows if row.get("pnl") is not None and str(row.get("result") or "").upper() != "RECONCILED"]


def _metrics(symbol: str = SYMBOL) -> Dict[str, float]:
    trades = _real_closed_trades(symbol)
    pnl_values = np.array([float(row.get("pnl") or 0.0) for row in trades], dtype=float)
    r_values = np.array([float(row.get("r_multiple") or 0.0) for row in trades], dtype=float)
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    sharpe = 0.0
    if len(r_values) >= 2 and float(r_values.std(ddof=0)) > 1e-9:
        sharpe = float((r_values.mean() / r_values.std(ddof=0)) * np.sqrt(len(r_values)))
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0)
    win_rate = float((pnl_values > 0).mean() * 100.0) if len(pnl_values) else 0.0
    expectancy_r = float(r_values.mean()) if len(r_values) else 0.0
    return {
        "sharpe": sharpe,
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999.0),
        "total_trades": int(len(trades)),
        "win_rate": win_rate,
        "expectancy_R": expectancy_r,
    }


def _default_report() -> Dict[str, Any]:
    started_at = _utc_now().isoformat()
    return {
        "symbol": SYMBOL,
        "started_at": started_at,
        "last_evaluated_at": started_at,
        "days_running": 0,
        "days_required": MIN_DAYS,
        "eta_days": MIN_DAYS,
        "next_eval_at": (_utc_now() + timedelta(days=1)).isoformat(),
        "thresholds": {
            "sharpe_min": 0.3,
            "profit_factor_min": 1.1,
            "total_trades_min": 30,
            "expectancy_R_min": 0.0,
        },
        "metrics": _metrics(SYMBOL),
        "ready_to_continue": True,
        "killed": False,
        "kill_reason": "",
    }


def status(symbol: str = SYMBOL) -> Dict[str, Any]:
    report = _load_json(LOG_PATH, _default_report())
    report.setdefault("metrics", _metrics(symbol))
    report["kill_flag"] = KILL_FLAG_PATH.exists()
    if report["kill_flag"] and not report.get("killed"):
        report["killed"] = True
        try:
            report["kill_reason"] = KILL_FLAG_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            report["kill_reason"] = "KILL.flag presente"
    return report


def evaluate(symbol: str = SYMBOL, min_days: int = MIN_DAYS) -> Dict[str, Any]:
    report = _load_json(LOG_PATH, _default_report())
    started_at_raw = report.get("started_at") or _utc_now().isoformat()
    try:
        started_at = datetime.fromisoformat(str(started_at_raw).replace("Z", "+00:00"))
    except Exception:
        started_at = _utc_now()
    now = _utc_now()
    days_running = max(0, int((now - started_at).total_seconds() // 86400))
    metrics = _metrics(symbol)
    report.update(
        {
            "symbol": symbol,
            "started_at": started_at.isoformat(),
            "last_evaluated_at": now.isoformat(),
            "days_running": days_running,
            "days_required": int(min_days),
            "eta_days": max(0, int(min_days) - days_running),
            "metrics": metrics,
            "kill_flag": KILL_FLAG_PATH.exists(),
        }
    )
    if days_running < int(min_days):
        report["next_eval_at"] = (started_at + timedelta(days=int(min_days))).isoformat()
        report["ready_to_continue"] = True
        report["killed"] = False
        report["kill_reason"] = ""
        _save_json(LOG_PATH, report)
        return report
    kill_reason = ""
    if metrics["sharpe"] < 0.3:
        kill_reason = f"sharpe={metrics['sharpe']:.2f} < 0.30"
    elif metrics["profit_factor"] < 1.1:
        kill_reason = f"profit_factor={metrics['profit_factor']:.2f} < 1.10"
    elif metrics["total_trades"] < 30:
        kill_reason = f"total_trades={metrics['total_trades']} < 30"
    elif metrics["expectancy_R"] < 0.0:
        kill_reason = f"expectancy_R={metrics['expectancy_R']:.2f} < 0"
    if kill_reason:
        report["ready_to_continue"] = False
        report["killed"] = True
        report["kill_reason"] = kill_reason
        report["next_eval_at"] = None
        KILL_FLAG_PATH.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "killed_at": now.isoformat(),
                    "reason": kill_reason,
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        report["ready_to_continue"] = True
        report["killed"] = False
        report["kill_reason"] = ""
        report["next_eval_at"] = (now + timedelta(days=RECHECK_DAYS)).isoformat()
        if KILL_FLAG_PATH.exists():
            KILL_FLAG_PATH.unlink()
    _save_json(LOG_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 90-day kill criterion for BTC observation bot")
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args()
    print(json.dumps(evaluate(symbol=str(args.symbol or SYMBOL)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
