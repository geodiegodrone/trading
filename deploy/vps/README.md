# VPS Deployment

This folder contains the Linux/systemd deployment for the Binance bot.

## Layout on the VPS

Suggested path:

```text
/opt/trading/
  bot/
  flowsurface/
  venv/
```

## Services

- `trading-bot.service` runs `bot/multi_bot.py`
- `trading-dashboard.service` runs `bot/dashboard_multi.py`

## Quick install

1. Copy the `trading` folder to the VPS.
2. Edit `bot/.env` on the VPS with your Binance credentials.
3. Run:

```bash
sudo bash deploy/vps/install.sh
sudo systemctl enable --now trading-bot trading-dashboard
```

## Health checks

```bash
systemctl status trading-bot
systemctl status trading-dashboard
journalctl -u trading-bot -f
journalctl -u trading-dashboard -f
```

