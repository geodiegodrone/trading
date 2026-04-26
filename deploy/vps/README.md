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
- `nginx.conf` is a ready-made reverse proxy for the dashboard

## One-command bootstrap

If the repo is already on the VPS:

```bash
sudo bash deploy/vps/bootstrap.sh
```

The script will:

- install Linux packages
- create `/opt/trading`
- clone or update this repo
- ask for your Binance API key and secret
- create `bot/.env`
- install Python dependencies
- register `systemd` services
- optionally configure `nginx`

After that, the services start automatically.

## Manual install

1. Copy the `trading` repo to the VPS or clone it there.
2. Run `sudo bash deploy/vps/install.sh`.
3. Start services:

```bash
sudo systemctl start trading-bot trading-dashboard
```

## Health checks

```bash
systemctl status trading-bot
systemctl status trading-dashboard
journalctl -u trading-bot -f
journalctl -u trading-dashboard -f
```
