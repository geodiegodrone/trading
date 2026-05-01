# Risk Disclosure

This repository is for research, education, and engineering demonstration only.

It is not financial advice, investment advice, or a recommendation to trade any asset.

## Key Risks

- Strategies can be overfit to historical data.
- Backtests can omit fees, slippage, latency, and funding.
- Futures and leverage can cause liquidation.
- Exchange APIs can fail, throttle, or return stale data.
- Network outages can prevent exits.
- Market regimes can shift abruptly.
- Model files can become incompatible across library versions.
- Paper-trading performance may not transfer to live markets.

## Required Safety Controls

- Run paper trading before live execution.
- Use small position sizes.
- Define max daily loss.
- Define max drawdown shutdown.
- Use circuit breakers.
- Log every decision.
- Review execution manually.
- Keep API keys restricted and withdraw-disabled.

## Performance Claims

Any performance claim must report:

- evaluation period,
- train/test split,
- walk-forward protocol,
- fees,
- slippage,
- funding,
- max drawdown,
- number of trades,
- Sharpe or Sortino,
- benchmark comparison.
