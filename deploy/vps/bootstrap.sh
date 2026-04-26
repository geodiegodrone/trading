#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/geodiegodrone/trading.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/opt/trading}"
APP_USER="${APP_USER:-trading}"
ENABLE_NGINX="${ENABLE_NGINX:-1}"

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root: sudo bash deploy/vps/bootstrap.sh"
  exit 1
fi

read -r -p "Binance API key: " BINANCE_API_KEY
read -r -s -p "Binance API secret: " BINANCE_API_SECRET
echo
read -r -p "Use Binance demo futures? [Y/n] " USE_DEMO
USE_DEMO="${USE_DEMO:-Y}"
read -r -p "Optional domain for dashboard (leave blank for IP only): " DOMAIN

echo "[1/9] Installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git python3 python3-venv python3-pip curl ca-certificates nginx

echo "[2/9] Creating app user if needed"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi

echo "[3/9] Preparing app directory"
mkdir -p "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR" || true

echo "[4/9] Cloning or updating repo"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch --all --prune
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
  rm -rf "$APP_DIR"/*
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

echo "[5/9] Writing bot environment"
cat > "$APP_DIR/bot/.env" <<EOF
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET
EOF
if [[ "${USE_DEMO^^}" != "N" ]]; then
  cat >> "$APP_DIR/bot/.env" <<EOF
BINANCE_DEMO=1
EOF
fi
chown "$APP_USER:$APP_USER" "$APP_DIR/bot/.env"
chmod 600 "$APP_DIR/bot/.env"

echo "[6/9] Setting up Python virtualenv"
if [[ ! -d "$APP_DIR/venv" ]]; then
  python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel setuptools
if [[ -f "$APP_DIR/bot/requirements.txt" ]]; then
  "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/bot/requirements.txt"
fi

echo "[7/9] Installing systemd services"
install -m 0644 "$APP_DIR/deploy/vps/trading-bot.service" /etc/systemd/system/trading-bot.service
install -m 0644 "$APP_DIR/deploy/vps/trading-dashboard.service" /etc/systemd/system/trading-dashboard.service
systemctl daemon-reload
systemctl enable trading-bot trading-dashboard

echo "[8/9] Configuring nginx"
if [[ "$ENABLE_NGINX" == "1" ]]; then
  cat > /etc/nginx/sites-available/trading <<EOF
server {
  listen 80;
  server_name ${DOMAIN:-_};

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host \$host;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
  }
}
EOF
  ln -sf /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/trading
  rm -f /etc/nginx/sites-enabled/default || true
  nginx -t
  systemctl enable nginx
fi

echo "[9/9] Starting services"
systemctl restart trading-bot trading-dashboard
if [[ "$ENABLE_NGINX" == "1" ]]; then
  systemctl restart nginx
fi

cat <<EOF
Done.

What you need to do:
1. Confirm the VPS IP or domain.
2. Open port 80 in your VPS firewall / cloud panel.
3. If you want HTTPS, tell me the domain and I will add certbot steps.

Useful commands:
  systemctl status trading-bot trading-dashboard nginx
  journalctl -u trading-bot -f
  journalctl -u trading-dashboard -f
EOF

