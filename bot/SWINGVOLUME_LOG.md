# SWINGVOLUME

## Decision

The previous bot did not show robust edge in BTC 1h:

- meta-labeling with 57 features
- 18 search iterations
- up to 5 years of data
- out-of-sample holdout did not replicate

Decision: remove ML from the critical path. Trade a single quantified discretionary setup. If it does not pass 90 demo days, shut it down.

## Setup

Timeframes:

- macro context: `D1`
- trigger: `H4`

Daily bias:

- `EMA20_D1 > EMA200_D1` -> only `LONG`
- `EMA20_D1 < EMA200_D1` -> only `SHORT`
- daily close cannot sit at an extreme of the candle:
  - `close_pos = (close-low)/(high-low)`
  - requires `0.02 <= close_pos <= 0.98`

MACD divergence on H4:

- use histogram `MACD(12,26,9)`
- normalize as `% of price`: `macd_hist_pct = macd_hist / close`
- for `LONG`:
  - first deep low `<= -0.0005`
  - second low 2 to 8 H4 bars later
  - price makes a lower low
  - histogram makes a higher low
- for `SHORT`: mirror
- divergence expires after `4` H4 bars

Volume candle on H4:

- `volume > vol_ma20 * 1.3`
- `vol_zscore_20 >= 1.5`
- two previous candles with `volume < vol_ma20 * 0.9`
- body `>= 60%` of candle range
- directional candle:
  - `LONG`: `close > open`
  - `SHORT`: `close < open`

MACD recovery:

- `LONG`: `-0.0001 <= macd_hist_pct <= 0.0005`
- or cross from negative to positive
- `SHORT`: mirror around zero

## Real Diagnosis

Backtest `2024-01-01 .. 2026-05-01`:

- `Total H4 candles`: `4892`
- `Daily bias OK`: `4844`
- `MACD divergence detected`: `171`
- `Divergence not expired`: `171`
- `Volume MA OK`: `1258`
- `Volume z-score OK`: `536`
- `Previous volume low`: `1961`
- `Candle body >= 60%`: `1335`
- `MACD range OK`: `479`
- `All gates pass`: `0`

Relaxation tested:

- `MACD cross` instead of fixed range
- result: `0 trades`

Conclusion:

- the setup did not produce enough entries in BTC H4 2024-2026
- no demo
- no parameter tuning to force activity
- honest close of the SWINGVOLUME experiment on BTC H4

## Execution

Entry:

- at the close of the H4 candle that confirms everything

Stop:

- `open` of the trigger candle
- if that is not useful, fallback `1.2 * ATR(14)`

Target:

- `TP = 3R`

Time stop:

- `5` H4 candles

Trailing:

- `+0.3R` -> stop to breakeven
- `+0.7R` -> stop to `+0.2R`

## Risk

- `0.15%` of balance per trade
- maximum `1` open trade
- maximum `1` new trade per UTC day
- daily cap `0.5%`
- streak of `3` losses -> pause `48h`
- rolling `7` losses in `20` trades (`win rate < 35%`) -> pause `7d`

## Kill Criterion

At day `90`:

- `sharpe >= 0.4`
- `profit_factor >= 1.15`
- `win_rate >= 50%`
- `total_trades >= 15`
- `max_drawdown < 30%`

If any fail:

- write `KILL.flag`
- stop opening trades

## Governance

Do not reintroduce ML into the critical path without updating this note and justifying it with real out-of-sample edge.
