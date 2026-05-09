# Decision Log

## What We Proved
- Meta-labeling and SWINGVOLUME did not produce durable edge.
- BTC 1h/H4 was too efficient for the previous setups.
- The live path is now a single BTC 1h mean-reversion overshoot setup.

## Why We Pivoted
- The previous experiments were honest failures.
- Further tuning of those setups would have been p-hacking.
- The last remaining hypothesis is volatility overshoot mean reversion with fixed gates.

## Current Active Setup
- `meanrev`
- BTC-only
- 1h primary timeframe
- Risk per trade: `0.15%`
- One trade per day maximum
- One open trade maximum

## Mean-Reversion Spec
- LONG when:
  - 1h return z-score <= -2.5
  - RSI <= 25
  - ATR z-score >= 1.5
- SHORT mirrors the same conditions.
- Stop loss: `1.0 ATR`
- Take profit: `0.6 ATR`
- Time stop: `12` bars

## Kill Criterion
- Evaluation window: `90 days`
- Continue only if:
  - sharpe >= `0.4`
  - profit_factor >= `1.2`
  - win_rate >= `55%`
  - max_drawdown < `35%`
  - total_trades >= `15`
- If any gate fails, write `KILL.flag` and stop the bot.

## What Not To Do
- Do not reintroduce ML into the critical path.
- Do not add another setup before this one survives its 90-day gate.
- Do not relax the gates to force a pass.

## If This Fails
- Close the bot.
- Revisit the market hypothesis with derivatives data:
  - funding rate
  - open interest
  - basis
  - liquidations
  - order flow

## Final Backtest Result
- Backtest 2024-01-01 to 2026-05-09 on BTCUSDT 1h:
  - trades: 15
  - win_rate: 66.7%
  - profit_factor: 1.20
  - sharpe: 7.99
  - max_drawdown: 0.45%
  - expectancy_R: 0.07
- The setup did not pass because `trades < 30`.
- Decision: close the bot rather than relax gates or invent a fourth iteration.
