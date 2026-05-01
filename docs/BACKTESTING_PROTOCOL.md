# Backtesting Protocol

## Minimum Standard

1. Use chronological train/test splits.
2. Use walk-forward validation.
3. Include fees and slippage.
4. Track max drawdown.
5. Track trade count and exposure.
6. Compare against a passive or no-trade baseline.
7. Report failed experiments.

## Anti-Overfit Rules

- Do not tune on the final holdout window.
- Do not select only the best symbol/timeframe.
- Require stability across regimes.
- Prefer simple rules until complex models prove value.
- Penalize high turnover.

## Promotion Gate

A strategy may move from research to paper trading only if:

- tests pass,
- walk-forward report is generated,
- drawdown is within defined risk,
- trade count is sufficient,
- circuit breaker is enabled.
