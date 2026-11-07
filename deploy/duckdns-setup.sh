#!/usr/bin/env bash
# Idempotent installer for the DuckDNS self-updater. Writes the token to
# /etc/duckdns/config (root-only), copies the script + systemd units into
# place, and enables the timer. Safe to re-run (re-running just refreshes
# the files).
#
# Usage: sudo ./duckdns-setup.sh <subdomain> <token>
set -euo pipefail

DOMAIN="${1:?usage: duckdns-setup.sh <subdomain> <token>}"
TOKEN="${2:?usage: duckdns-setup.sh <subdomain> <token>}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$DOMAIN" =~ ^[a-z0-9-]+$ ]] || { echo "bad subdomain: $DOMAIN" >&2; exit 1; }
[[ "$TOKEN" =~ ^[0-9a-fA-F-]{36}$ ]] || { echo "bad token format" >&2; exit 1; }

install -d -m 700 /etc/duckdns
cat >/etc/duckdns/config <<EOF
DOMAIN=${DOMAIN}
TOKEN=${TOKEN}
EOF
chmod 600 /etc/duckdns/config

install -d -m 755 /opt/duckdns /usr/local/share/duckdns
install -m 755 "$SRC_DIR/duckdns.sh" /opt/duckdns/duckdns.sh
install -m 644 "$SRC_DIR/duckdns.service" /usr/local/share/duckdns/duckdns.service
install -m 644 "$SRC_DIR/duckdns.timer" /usr/local/share/duckdns/duckdns.timer
cp /usr/local/share/duckdns/duckdns.service /etc/systemd/system/duckdns.service
cp /usr/local/share/duckdns/duckdns.timer /etc/systemd/system/duckdns.timer

systemctl daemon-reload
systemctl enable --now duckdns.timer
systemctl start duckdns.service
echo "duckdns: timer enabled; status:"
systemctl is-enabled duckdns.timer
systemctl is-active duckdns.timer
