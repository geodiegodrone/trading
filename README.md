# Trading Monorepo

This repository contains:

- `bot/`: Binance demo futures trading bot and live dashboard
- `flowsurface/`: Rust market interface

## VPS deployment

For always-on Linux deployment, see:

- [deploy/vps/README.md](deploy/vps/README.md)

## Local run

### Bot

```bash
cd bot
python -m pip install -r requirements.txt
python dashboard_multi.py
python multi_bot.py
```

### Flowsurface

```bash
cd flowsurface
cargo run --release
```

## Notes

- Keep `bot/.env` out of version control.
- `bot/dashboard_multi.py` exposes the live dashboard.
- `bot/multi_bot.py` drives execution and reconciliation.

