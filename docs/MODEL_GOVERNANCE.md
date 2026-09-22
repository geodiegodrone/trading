# Model Governance

## Model Lifecycle

1. Research candidate.
2. Backtest.
3. Walk-forward validation.
4. Paper trading.
5. Limited live pilot.
6. Production monitoring.
7. Retirement.

## Artifact Control

Persisted model files must record:

- training data period,
- feature schema,
- library versions,
- calibration method,
- validation metrics,
- owner,
- promotion date.

Every completed training call records an MLflow run with the validation protocol,
feature count, sample counts, readiness decision, threshold, AUC, F1, Sharpe,
drawdown, coverage, fitted estimator, calibrator, and deployment-state artifact.
Runs are stored under `bot/mlruns/` by default. Set `MLFLOW_TRACKING_URI` to a
remote tracking server or `TRADING_MLFLOW_ENABLED=false` to disable tracking.
Tracking failures are reported and do not prevent saving the local model.

## Current Warning

Local tests report scikit-learn model persistence warnings when loading artifacts trained under a different scikit-learn version. Treat this as a governance issue: retrain or pin exact versions before production use.
