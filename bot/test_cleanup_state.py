from __future__ import annotations

import json

import cleanup_state


def test_cleanup_state_detects_synthetic_and_resets(tmp_path, monkeypatch) -> None:
    cb_path = tmp_path / "circuit_breaker_state.json"
    open_path = tmp_path / "open_trade_id.json"
    synthetic = {
        "paused": True,
        "recent_trade_pnls": [5, -5, -5, -5, 5, -5, -5, -5],
        "last_trade_results": [1, 0, 0, 0, 1, 0, 0, 0],
    }
    cb_path.write_text(json.dumps(synthetic), encoding="utf-8")
    open_path.write_text(json.dumps({"BTCUSDT": 999}), encoding="utf-8")

    monkeypatch.setattr(cleanup_state.circuit_breaker, "STATE_PATH", cb_path)
    monkeypatch.setattr(cleanup_state, "OPEN_TRADE_PATH", open_path)
    monkeypatch.setattr(cleanup_state.trade_log, "get_all_trades", lambda symbol=None: [{"id": 1}])

    report = cleanup_state.run_cleanup(apply=True)
    assert report["synthetic_circuit_breaker"] is True
    after = json.loads(cb_path.read_text(encoding="utf-8"))
    assert after["paused"] is False
    registry = json.loads(open_path.read_text(encoding="utf-8"))
    assert registry == {}


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from pytest import MonkeyPatch

    with TemporaryDirectory() as tmp:
        test_cleanup_state_detects_synthetic_and_resets(Path(tmp), MonkeyPatch())
    print("test_cleanup_state_ok")
