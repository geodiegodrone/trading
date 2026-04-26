#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/trading}"
APP_USER="${APP_USER:-trading}"
REPO_DIR="${REPO_DIR:-$APP_DIR}"
BOT_DIR="$REPO_DIR/bot"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash deploy/vps/install.sh"
  exit 1
fi

echo "[1/6] Checking directories"
mkdir -p "$APP_DIR"

echo "[2/6] Creating app user if needed"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR" || true

echo "[3/6] Ensuring Python tooling"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi
if ! command -v pip3 >/dev/null 2>&1; then
  echo "pip3 is required"
  exit 1
fi

echo "[4/6] Creating virtualenv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools
if [[ -f "$BOT_DIR/requirements.txt" ]]; then
  "$VENV_DIR/bin/pip" install -r "$BOT_DIR/requirements.txt"
fi

echo "[5/6] Installing systemd units"
install -m 0644 "$(dirname "$0")/trading-bot.service" "$SYSTEMD_DIR/trading-bot.service"
install -m 0644 "$(dirname "$0")/trading-dashboard.service" "$SYSTEMD_DIR/trading-dashboard.service"

echo "[6/6] Reloading systemd"
systemctl daemon-reload
systemctl enable trading-bot trading-dashboard

cat <<EOF
Done.

Next:
  sudo systemctl start trading-bot trading-dashboard
  sudo systemctl status trading-bot trading-dashboard
EOF
