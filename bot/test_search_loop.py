from __future__ import annotations

import json

import pandas as pd

import search_loop


def _frame(block: str) -> pd.DataFrame:
    return pd.DataFrame({"ts": [1, 2, 3], "block": [block, block, block]})


def test_search_loop_converges_to_profitable_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(search_loop, "SEARCH_LOG_PATH", tmp_path / "search_log.jsonl")
    monkeypatch.setattr(search_loop, "SEARCH_STATUS_PATH", tmp_path / "search_status.json")
    monkeypatch.setattr(search_loop, "LIVE_GATE_PATH", tmp_path / "live_gate.json")
    monkeypatch.setattr(search_loop, "download_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(search_loop, "load_history", lambda symbol, tf: pd.DataFrame({"ts": list(range(100))}))
    monkeypatch.setattr(search_loop, "_slice_recent", lambda df, days: df)
    monkeypatch.setattr(search_loop, "_split_history", lambda primary, confirmation: {"search": (_frame("search"), _frame("search")), "valid": (_frame("valid"), _frame("valid")), "holdout": (_frame("holdout"), _frame("holdout"))})

    def fake_walkforward(df, cfg, n_folds=12, confirmation_df=None):
        block = str(df["block"].iloc[0])
        name = str(cfg.get("name"))
        if block == "holdout":
            return {"median_fold_sharpe": 1.0, "profit_factor": 1.25, "max_drawdown_pct": 8.0}
        if name == "wide_stop" and block == "valid":
            return {"median_fold_sharpe": 0.9, "profit_factor": 1.4, "folds_positive": 8, "max_drawdown_pct": 10.0, "expectancy_R": 0.2, "total_trades": 60, "val_auc": 0.57}
        return {"median_fold_sharpe": 0.1, "profit_factor": 0.9, "folds_positive": 2, "max_drawdown_pct": 30.0, "expectancy_R": 0.01, "total_trades": 10, "val_auc": 0.50}

    monkeypatch.setattr(search_loop, "run_walkforward", fake_walkforward)

    result = search_loop.run_search(budget=4, base_days=730)
    assert result["ready_for_live"] is True
    assert result["config"]["name"] == "wide_stop"
    assert (tmp_path / "search_log.jsonl").exists()


def test_search_loop_exhausts_when_all_configs_fail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(search_loop, "SEARCH_LOG_PATH", tmp_path / "search_log.jsonl")
    monkeypatch.setattr(search_loop, "SEARCH_STATUS_PATH", tmp_path / "search_status.json")
    monkeypatch.setattr(search_loop, "LIVE_GATE_PATH", tmp_path / "live_gate.json")
    monkeypatch.setattr(search_loop, "download_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(search_loop, "load_history", lambda symbol, tf: pd.DataFrame({"ts": list(range(100))}))
    monkeypatch.setattr(search_loop, "_slice_recent", lambda df, days: df)
    monkeypatch.setattr(search_loop, "_split_history", lambda primary, confirmation: {"search": (_frame("search"), _frame("search")), "valid": (_frame("valid"), _frame("valid")), "holdout": (_frame("holdout"), _frame("holdout"))})
    monkeypatch.setattr(search_loop, "run_walkforward", lambda *args, **kwargs: {"median_fold_sharpe": 0.0, "profit_factor": 0.8, "folds_positive": 1, "max_drawdown_pct": 30.0, "expectancy_R": 0.0, "total_trades": 12, "val_auc": 0.50})

    result = search_loop.run_search(budget=3, base_days=730)
    assert result["ready_for_live"] is False
    assert result["exhausted"] is True
    lines = [json.loads(line) for line in (tmp_path / "search_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from pytest import MonkeyPatch

    with TemporaryDirectory() as tmp:
        test_search_loop_converges_to_profitable_config(__import__("pathlib").Path(tmp), MonkeyPatch())
    with TemporaryDirectory() as tmp:
        test_search_loop_exhausts_when_all_configs_fail(__import__("pathlib").Path(tmp), MonkeyPatch())
    print("test_search_loop_ok")
