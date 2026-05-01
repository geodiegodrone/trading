from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
import ml_model
import walkforward_validator


def _market_frame(rows: int = 320) -> pd.DataFrame:
    ts = np.arange(rows) * 3600000
    base = np.linspace(100.0, 140.0, rows)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": base - 0.4,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.linspace(100, 200, rows),
            "quote_vol": np.linspace(1000, 2000, rows),
            "taker_base": np.linspace(40, 120, rows),
            "taker_quote": np.linspace(400, 1200, rows),
        }
    )


def _dataset(rows: int = 240, profitable: bool = True) -> pd.DataFrame:
    labels = np.array(([1, 0] * (rows // 2))[:rows], dtype=int)
    data = {column: np.zeros(rows, dtype=float) for column in FEATURE_COLUMNS}
    if profitable:
        data[FEATURE_COLUMNS[0]] = np.where(labels == 1, 0.9, 0.1)
        r_values = np.where(labels == 1, 0.45, -0.10)
    else:
        data[FEATURE_COLUMNS[0]] = 0.5
        r_values = np.where(labels == 1, 0.05, -0.08)
    frame = pd.DataFrame(data)
    frame["label"] = labels
    frame["signal_idx"] = np.arange(rows)
    frame["exit_idx"] = np.arange(rows) + 6
    frame["r_multiple"] = r_values
    frame["holding_bars"] = 6
    return frame


class _DummyModel:
    pass


class _PassthroughIsotonic:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.asarray(x, dtype=float)


class _SingleFoldSplitter:
    def __init__(self, n_splits=12, embargo=0):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, dataset):
        size = len(dataset)
        test_start = int(size * 0.75)
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, size)
        yield train_idx, test_idx


def test_walkforward_positive_metrics(monkeypatch) -> None:
    monkeypatch.setattr(walkforward_validator, "_compute_indicators", lambda df, cfg: df)
    monkeypatch.setattr(walkforward_validator, "_event_dataset", lambda primary, confirmation, cfg: (_dataset(profitable=True), pd.DataFrame({"label": [1, 0] * 120})))
    monkeypatch.setattr(walkforward_validator, "_fit_model", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(walkforward_validator, "IsotonicRegression", _PassthroughIsotonic)
    monkeypatch.setattr(ml_model, "PurgedKFold", _SingleFoldSplitter)
    monkeypatch.setattr(ml_model, "_positive_proba", lambda model, x: x[:, 0])
    monkeypatch.setattr(ml_model, "_optimize_threshold", lambda probs, labels, r_values, holding_bars, signal_idx: {"tau": 0.55})

    metrics = walkforward_validator.run_walkforward(_market_frame(), {"strategy_mode": "trend"}, n_folds=12, confirmation_df=_market_frame())
    assert metrics["median_fold_sharpe"] > 0
    assert metrics["profit_factor"] > 1.0
    assert metrics["expectancy_R"] > 0
    assert metrics["total_trades"] > 0


def test_walkforward_random_fails_gates(monkeypatch) -> None:
    monkeypatch.setattr(walkforward_validator, "_compute_indicators", lambda df, cfg: df)
    monkeypatch.setattr(walkforward_validator, "_event_dataset", lambda primary, confirmation, cfg: (_dataset(profitable=False), pd.DataFrame({"label": [1, 0] * 120})))
    monkeypatch.setattr(walkforward_validator, "_fit_model", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(walkforward_validator, "IsotonicRegression", _PassthroughIsotonic)
    monkeypatch.setattr(ml_model, "PurgedKFold", _SingleFoldSplitter)
    monkeypatch.setattr(ml_model, "_positive_proba", lambda model, x: x[:, 0])
    monkeypatch.setattr(ml_model, "_optimize_threshold", lambda probs, labels, r_values, holding_bars, signal_idx: {"tau": 0.55})

    metrics = walkforward_validator.run_walkforward(_market_frame(), {"strategy_mode": "trend"}, n_folds=12, confirmation_df=_market_frame())
    assert metrics["passes"] is False
    assert metrics["profit_factor"] < 1.3 or metrics["val_auc"] < 0.55


if __name__ == "__main__":
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    test_walkforward_positive_metrics(monkeypatch)
    monkeypatch.undo()
    monkeypatch = MonkeyPatch()
    test_walkforward_random_fails_gates(monkeypatch)
    print("test_walkforward_validator_ok")
