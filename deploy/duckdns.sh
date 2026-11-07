#!/usr/bin/env bash
# DuckDNS updater: keeps <domain>.duckdns.org pointing at this instance's
# CURRENT public IP. Run by systemd (duckdns.service + duckdns.timer) at every
# boot and every 5 minutes while up, so the DNS name survives stop/start even
# though the public IP changes every time.
#
# Config lives in /etc/duckdns/config (mode 600, root-only) as:
#   DOMAIN=peds
#   TOKEN=<uuid>
# The token NEVER leaves this box - it is not in the repo.
#
# The IP is always sent EXPLICITLY (never auto-detect): DuckDNS would otherwise
# record the IP of whoever called it, which is wrong if the script is ever run
# by hand from a laptop. IMDSv2 is used (token PUT first, 3s TTL).
set -euo pipefail

CONFIG=/etc/duckdns/config
LOG_TAG=duckdns

[[ -r "$CONFIG" ]] || { echo "$LOG_TAG: missing $CONFIG" >&2; exit 0; }
# shellcheck disable=SC1090
source "$CONFIG"
DOMAIN="${DOMAIN:-}"
TOKEN="${TOKEN:-}"
[[ -n "$DOMAIN" && -n "$TOKEN" ]] || { echo "$LOG_TAG: DOMAIN/TOKEN empty in $CONFIG" >&2; exit 0; }

IP=$(curl -fsS -m 5 -H "X-aws-ec2-metadata-token: $(curl -fsS -m 3 -X PUT \
  'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 3')" \
  'http://169.254.169.254/latest/meta-data/public-ipv4') \
  || { echo "$LOG_TAG: could not read public IP from IMDS (probably stopped/starting)" >&2; exit 0; }

RES=$(curl -fsS -m 10 "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=${IP}") \
  || { echo "$LOG_TAG: update call failed for ${DOMAIN}.duckdns.org -> ${IP}" >&2; exit 1; }

if [[ "$RES" == "OK" ]]; then
  echo "$LOG_TAG: ${DOMAIN}.duckdns.org -> ${IP}"
else
  echo "$LOG_TAG: unexpected response '$RES' for ${DOMAIN}.duckdns.org -> ${IP}" >&2
  exit 1
fi
