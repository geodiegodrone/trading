from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import circuit_breaker
import trade_log


ROOT = Path(__file__).resolve().parent
OPEN_TRADE_PATH = ROOT / "open_trade_id.json"


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return dict(default)


def detect_synthetic_circuit_breaker(state: Dict[str, Any]) -> bool:
    pnls = list(state.get("recent_trade_pnls") or [])
    results = list(state.get("last_trade_results") or [])
    if len(pnls) < 8 or len(results) != len(pnls):
        return False
    if any(abs(float(value)) != 5.0 for value in pnls):
        return False
    if any(int(results[idx]) != (1 if float(pnls[idx]) > 0 else 0) for idx in range(len(pnls))):
        return False
    pattern = [5.0, -5.0, -5.0, -5.0]
    for idx, value in enumerate(pnls):
        if float(value) != pattern[idx % len(pattern)]:
            return False
    return True


def _reset_circuit_breaker_state() -> Dict[str, Any]:
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


def _cleanup_open_trade_registry(apply: bool) -> Dict[str, Any]:
    registry = _load_json(OPEN_TRADE_PATH, {})
    if not registry:
        return {"changed": False, "reason": "registry vacío", "entries": {}}
    valid_ids = {int(row.get("id")) for row in trade_log.get_all_trades("BTCUSDT") if row.get("id") is not None}
    cleaned = {}
    removed = {}
    for symbol, trade_id in registry.items():
        try:
            trade_key = int(trade_id)
        except Exception:
            removed[symbol] = trade_id
            continue
        if trade_key in valid_ids:
            cleaned[symbol] = trade_key
        else:
            removed[symbol] = trade_id
    changed = cleaned != registry
    if apply and changed:
        OPEN_TRADE_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return {"changed": changed, "removed": removed, "entries": cleaned}


def run_cleanup(apply: bool = False) -> Dict[str, Any]:
    state = _load_json(circuit_breaker.STATE_PATH, {})
    synthetic = detect_synthetic_circuit_breaker(state)
    registry = _cleanup_open_trade_registry(apply)
    report = {
        "apply": bool(apply),
        "synthetic_circuit_breaker": synthetic,
        "registry": registry,
        "changed": False,
    }
    if synthetic:
        report["changed"] = True
        report["circuit_breaker_before"] = state
        report["circuit_breaker_after"] = _reset_circuit_breaker_state()
        if apply:
            circuit_breaker.STATE_PATH.write_text(json.dumps(report["circuit_breaker_after"], indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup synthetic bot runtime state")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = run_cleanup(apply=bool(args.apply))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
