from __future__ import annotations

import search_loop


def test_holdout_overfit_guard_blocks_degradation() -> None:
    valid = {"median_fold_sharpe": 2.0, "profit_factor": 1.5, "max_drawdown_pct": 10.0}
    holdout = {"median_fold_sharpe": 0.3, "profit_factor": 1.25, "max_drawdown_pct": 11.0}
    ok, reason = search_loop._passes_holdout(valid, holdout)
    assert ok is False
    assert "sharpe" in reason


if __name__ == "__main__":
    test_holdout_overfit_guard_blocks_degradation()
    print("test_holdout_overfit_guard_ok")
