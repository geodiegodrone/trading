from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from bot_config import DEFAULT_CONFIG
from historical_data import download_history, history_metadata, load_history
from walkforward_validator import run_walkforward


ROOT = Path(__file__).resolve().parent
SEARCH_LOG_PATH = ROOT / "search_log.jsonl"
SEARCH_STATUS_PATH = ROOT / "search_status.json"
LIVE_GATE_PATH = ROOT / "live_gate.json"
_SEARCH_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return dict(default)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_live_gate() -> Dict[str, Any]:
    return _load_json(
        LIVE_GATE_PATH,
        {
            "ready_for_live": False,
            "exhausted": False,
            "reason": "live_gate ausente",
            "config": {},
            "sealed_at": None,
        },
    )


def load_search_status() -> Dict[str, Any]:
    return _load_json(
        SEARCH_STATUS_PATH,
        {
            "running": False,
            "status": "idle",
            "iter_current": 0,
            "budget": 0,
            "best_config": {},
            "best_valid_sharpe": 0.0,
            "best_pf": 0.0,
            "updated_at": None,
        },
    )


def _config_signature(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _base_runtime_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg["strategy_mode"] = str(cfg.get("strategy_mode", "regime")).lower()
    cfg["primary_timeframe"] = int(cfg.get("primary_timeframe", cfg.get("timeframe", 60)))
    cfg["confirmation_timeframe"] = int(cfg.get("confirmation_timeframe", 240))
    cfg["t_bars"] = int(cfg.get("t_bars", 24) or 24)
    cfg["tp_ratio"] = float(cfg.get("tp_ratio", 2.0) or 2.0)
    cfg["atr_mult"] = float(cfg.get("atr_mult", 1.5) or 1.5)
    cfg["cusum_mode"] = str(cfg.get("cusum_mode", "2xatr_pct"))
    cfg["feature_fraction"] = float(cfg.get("feature_fraction", 0.8) or 0.8)
    return cfg


def _candidate_configs(base_days: int) -> List[Dict[str, Any]]:
    return [
        {"name": "baseline", "data_window_days": int(base_days), "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
        {"name": "long_horizon", "data_window_days": int(base_days), "t_bars": 48, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
        {"name": "scalp_horizon", "data_window_days": int(base_days), "t_bars": 12, "atr_mult": 1.5, "tp_ratio": 1.5, "cusum_mode": "2xatr_pct"},
        {"name": "wide_stop", "data_window_days": int(base_days), "t_bars": 24, "atr_mult": 2.0, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
        {"name": "cusum_dense", "data_window_days": int(base_days), "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "atr_pct"},
        {"name": "cusum_sparse", "data_window_days": int(base_days), "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "3xatr_pct"},
        {"name": "flow_heavy", "data_window_days": int(base_days), "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct", "feature_fraction": 1.0},
        {"name": "hybrid_primary", "data_window_days": int(base_days), "strategy_mode": "hybrid", "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
        {"name": "three_years", "data_window_days": 1095, "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
        {"name": "five_years", "data_window_days": 1825, "t_bars": 24, "atr_mult": 1.5, "tp_ratio": 2.0, "cusum_mode": "2xatr_pct"},
    ]


def _slice_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cutoff = int(df["ts"].max()) - int(days) * 24 * 60 * 60 * 1000
    return df[df["ts"] >= cutoff].copy().reset_index(drop=True)


def _split_history(primary_df: pd.DataFrame, confirmation_df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    if primary_df.empty:
        return {"search": (primary_df.copy(), confirmation_df.copy()), "valid": (primary_df.copy(), confirmation_df.copy()), "holdout": (primary_df.copy(), confirmation_df.copy())}
    n_rows = len(primary_df)
    search_stop = max(1, int(n_rows * 0.70))
    valid_stop = max(search_stop + 1, int(n_rows * 0.85))
    boundaries = {
        "search": (int(primary_df.iloc[0]["ts"]), int(primary_df.iloc[search_stop - 1]["ts"])),
        "valid": (int(primary_df.iloc[search_stop]["ts"]), int(primary_df.iloc[valid_stop - 1]["ts"])),
        "holdout": (int(primary_df.iloc[valid_stop]["ts"]), int(primary_df.iloc[-1]["ts"])),
    }
    blocks: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (start_ts, end_ts) in boundaries.items():
        p_slice = primary_df[(primary_df["ts"] >= start_ts) & (primary_df["ts"] <= end_ts)].copy().reset_index(drop=True)
        c_slice = confirmation_df[(confirmation_df["ts"] >= start_ts) & (confirmation_df["ts"] <= end_ts)].copy().reset_index(drop=True)
        blocks[name] = (p_slice, c_slice)
    return blocks


def _passes_valid(metrics: Dict[str, Any]) -> bool:
    return bool(
        metrics.get("median_fold_sharpe", 0.0) >= 0.7
        and metrics.get("profit_factor", 0.0) >= 1.3
        and metrics.get("folds_positive", 0) >= 7
        and metrics.get("max_drawdown_pct", 999.0) <= 20.0
        and metrics.get("expectancy_R", 0.0) >= 0.15
        and metrics.get("total_trades", 0) >= 50
        and metrics.get("val_auc", 0.0) >= 0.55
    )


def _passes_holdout(valid_metrics: Dict[str, Any], holdout_metrics: Dict[str, Any]) -> tuple[bool, str]:
    valid_sharpe = float(valid_metrics.get("median_fold_sharpe", 0.0))
    valid_pf = float(valid_metrics.get("profit_factor", 0.0))
    valid_dd = float(valid_metrics.get("max_drawdown_pct", 0.0))
    holdout_sharpe = float(holdout_metrics.get("median_fold_sharpe", 0.0))
    holdout_pf = float(holdout_metrics.get("profit_factor", 0.0))
    holdout_dd = float(holdout_metrics.get("max_drawdown_pct", 0.0))
    if holdout_sharpe < valid_sharpe * 0.7:
        return False, "holdout_sharpe demasiado bajo"
    if holdout_pf < 1.2:
        return False, "holdout_profit_factor < 1.2"
    if valid_dd > 0 and holdout_dd > valid_dd * 1.5:
        return False, "holdout_drawdown demasiado alto"
    return True, ""


def _recommendation(best_entry: Dict[str, Any] | None) -> str:
    if not best_entry:
        return "no hubo configuraciones evaluables; revisar integridad de datos, gaps y generacion de eventos"
    valid = dict(best_entry.get("valid_metrics") or {})
    trades = int(valid.get("total_trades", 0) or 0)
    auc = float(valid.get("val_auc", 0.0) or 0.0)
    sharpe = float(valid.get("median_fold_sharpe", 0.0) or 0.0)
    pf = float(valid.get("profit_factor", 0.0) or 0.0)
    if trades < 50:
        return "muy pocos eventos utiles; revisar trigger CUSUM, horizontes y considerar otro timeframe o instrumento"
    if auc < 0.55:
        return "primary signal no tiene edge meta-modelable; considerar funding rate, open interest, basis, liquidaciones y order flow real"
    if sharpe < 0.7 or pf < 1.3:
        return "primary signal no tiene edge robusto; redisenar reglas de entrada/salida antes de habilitar live"
    return "config prometedora en validacion pero no confirmada en holdout; revisar sobreajuste y simplificar modelo"


def _status_payload(status: str, budget: int, iter_current: int, best_config: Dict[str, Any], best_metrics: Dict[str, Any], running: bool, message: str = "") -> Dict[str, Any]:
    return {
        "running": bool(running),
        "status": str(status),
        "iter_current": int(iter_current),
        "budget": int(budget),
        "best_config": dict(best_config or {}),
        "best_valid_sharpe": float(best_metrics.get("median_fold_sharpe", 0.0) if best_metrics else 0.0),
        "best_pf": float(best_metrics.get("profit_factor", 0.0) if best_metrics else 0.0),
        "message": str(message or ""),
        "updated_at": _utc_now(),
    }


def run_search(budget: int = 10, base_days: int = 730, symbol: str = "BTCUSDT", force: bool = False) -> Dict[str, Any]:
    with _SEARCH_LOCK:
        configs = _candidate_configs(base_days)[: max(1, int(budget))]
        best_entry: Dict[str, Any] | None = None
        status = _status_payload("running", len(configs), 0, {}, {}, True, "descargando historico")
        _save_json(SEARCH_STATUS_PATH, status)
        gate = load_live_gate()
        for idx, patch in enumerate(configs, start=1):
            cfg = _base_runtime_config()
            cfg.update(patch)
            days = int(cfg.get("data_window_days", base_days))
            primary_tf = int(cfg.get("primary_timeframe", 60))
            confirm_tf = int(cfg.get("confirmation_timeframe", 240))
            download_history(symbol, primary_tf, days, force=False)
            download_history(symbol, confirm_tf, days, force=False)
            primary_df = _slice_recent(load_history(symbol, primary_tf), days)
            confirm_df = _slice_recent(load_history(symbol, confirm_tf), days)
            blocks = _split_history(primary_df, confirm_df)
            search_metrics = run_walkforward(blocks["search"][0], cfg, n_folds=12, confirmation_df=blocks["search"][1])
            valid_metrics = run_walkforward(blocks["valid"][0], cfg, n_folds=12, confirmation_df=blocks["valid"][1])
            entry: Dict[str, Any] = {
                "iter": idx,
                "config": cfg,
                "search_metrics": search_metrics,
                "valid_metrics": valid_metrics,
                "decision": "continue",
                "evaluated_at": _utc_now(),
            }
            if best_entry is None or float(valid_metrics.get("median_fold_sharpe", 0.0)) > float(best_entry.get("valid_metrics", {}).get("median_fold_sharpe", 0.0)):
                best_entry = entry
            status = _status_payload("running", len(configs), idx, best_entry.get("config", {}) if best_entry else {}, best_entry.get("valid_metrics", {}) if best_entry else {}, True, f"iter {idx}/{len(configs)}")
            _save_json(SEARCH_STATUS_PATH, status)
            if _passes_valid(valid_metrics):
                signature = _config_signature(cfg)
                if gate.get("sealed_at") and gate.get("config_signature") == signature and gate.get("holdout_metrics"):
                    holdout_metrics = dict(gate.get("holdout_metrics") or {})
                    decision = "holdout_reused"
                else:
                    holdout_metrics = run_walkforward(blocks["holdout"][0], cfg, n_folds=12, confirmation_df=blocks["holdout"][1])
                    decision = "holdout_checked"
                entry["holdout_metrics"] = holdout_metrics
                passes_holdout, holdout_reason = _passes_holdout(valid_metrics, holdout_metrics)
                if passes_holdout:
                    gate = {
                        "ready_for_live": True,
                        "exhausted": False,
                        "config": cfg,
                        "config_signature": signature,
                        "search_metrics": search_metrics,
                        "valid_metrics": valid_metrics,
                        "holdout_metrics": holdout_metrics,
                        "sealed_at": _utc_now(),
                        "reason": "",
                    }
                    _save_json(LIVE_GATE_PATH, gate)
                    entry["decision"] = "ready_for_live"
                    _append_jsonl(SEARCH_LOG_PATH, entry)
                    _save_json(SEARCH_STATUS_PATH, _status_payload("ready_for_live", len(configs), idx, cfg, valid_metrics, False, "holdout aprobado"))
                    return gate
                entry["decision"] = f"overfit:{holdout_reason}"
            else:
                entry["decision"] = "reject_valid"
            _append_jsonl(SEARCH_LOG_PATH, entry)

        recommendation = _recommendation(best_entry)
        final_gate = {
            "ready_for_live": False,
            "exhausted": True,
            "best_iter": int(best_entry.get("iter", 0)) if best_entry else 0,
            "best_config": dict(best_entry.get("config", {}) if best_entry else {}),
            "best_metrics": dict(best_entry.get("valid_metrics", {}) if best_entry else {}),
            "recommendation": recommendation,
            "config_signature": _config_signature(best_entry.get("config", {})) if best_entry else "",
            "sealed_at": _utc_now(),
            "reason": "search_loop agotado sin edge demostrado",
        }
        _save_json(LIVE_GATE_PATH, final_gate)
        _save_json(SEARCH_STATUS_PATH, _status_payload("exhausted", len(configs), len(configs), final_gate.get("best_config", {}), final_gate.get("best_metrics", {}), False, recommendation))
        return final_gate


def _print_iteration(entry: Dict[str, Any]) -> None:
    valid = dict(entry.get("valid_metrics") or {})
    print(
        f"iter={int(entry.get('iter', 0)):>2} name={entry.get('config', {}).get('name', '-'):<14} "
        f"trades={int(valid.get('total_trades', 0)):>4} sharpe={float(valid.get('median_fold_sharpe', 0.0)):>6.2f} "
        f"pf={float(valid.get('profit_factor', 0.0)):>5.2f} auc={float(valid.get('val_auc', 0.0)):>5.3f} "
        f"dd={float(valid.get('max_drawdown_pct', 0.0)):>5.2f}% decision={entry.get('decision')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative BTC edge search with holdout gate")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--base-days", type=int, default=730)
    args = parser.parse_args()
    result = run_search(budget=args.budget, base_days=args.base_days)
    if SEARCH_LOG_PATH.exists():
        lines = [json.loads(line) for line in SEARCH_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        for entry in lines[-args.budget:]:
            _print_iteration(entry)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
