#!/bin/bash
# =============================================================================
#  EC2 user-data — bootstrap + auto-deploy loop for the pedestrian API.
#
#  Why a poller instead of SSH/SSM: the IAM user that owns this account may
#  not create roles (iam:CreateRole denied), so the instance gets NO instance
#  profile -> it cannot authenticate to ECR or be reached by SSM, and there
#  is no key pair to SSH with (ec2:CreateKeyPair denied too). Publishing the
#  image to GHCR from CI and letting the box poll the tag turns "deploy" into
#  "wait <= 5 minutes", with zero credentials anywhere.
#
#  Trust model: the GHCR package is public (repo is public), so the pull is
#  anonymous — nothing here is a secret, which is why this file can live in
#  the repository.
#
#  Lifecycle:  run-instances --user-data file://deploy/user-data.sh
#  Updates:    push to main -> CI publishes ghcr...:latest -> this loop
#              notices a new image id and replaces the container.
#  Debug:      journalctl -u pedestrian-deployer -f
# =============================================================================
set -u

IMAGE="ghcr.io/pravinkotta/pedestrian-detection:latest"
CONTAINER="pedestrian-api"

exec > >(tee /var/log/pedestrian-bootstrap.log) 2>&1
echo "[bootstrap] $(date -Is) starting"

# --- docker ----------------------------------------------------------------
dnf -y install docker || { echo "[bootstrap] docker install FAILED"; exit 1; }
systemctl enable --now docker
# ec2-user is AL2023's default login; docker group lets a human in the console
# run docker without sudo when debugging.
usermod -aG docker ec2-user || true

# --- the deploy loop (as a unit, so it survives reboots and cloud-init) ----
cat > /usr/local/bin/pedestrian-deployer <<'LOOP'
#!/bin/bash
# Pull :latest; replace the container only when the image id actually changed.
set -u
IMAGE="ghcr.io/pravinkotta/pedestrian-detection:latest"
CONTAINER="pedestrian-api"

deploy_once() {
  # Anonymous pull of a public package. A failure here is almost always the
  # package still being private -> say so loudly instead of looping silently.
  if ! docker pull "$IMAGE"; then
    echo "[deployer] pull failed - is the GHCR package public?" >&2
    return 1
  fi
  local want have
  want=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo none)
  have=$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || echo none)
  if [ "$want" != "$have" ]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    # Step 9: keep the drift capture log across container replacement.
    # 10001 = the unprivileged `app` user inside the image.
    mkdir -p /var/lib/pedestrian && chown 10001:10001 /var/lib/pedestrian
    # `--restart always`: this box exists to serve the API, so the container
    # must come back on EVERY boot/daemon restart - even if someone manually
    # stopped it while debugging (`unless-stopped` would stay down). Cost is
    # gated by the INSTANCE state, not this flag: stopped instance = nothing
    # runs at all; started instance = API live within ~1-2 min, unattended.
    docker run -d --name "$CONTAINER" --restart always \
      -p 8000:8000 \
      -v /var/lib/pedestrian:/var/lib/pedestrian \
      -e FEATURES_LOG=/var/lib/pedestrian/features.jsonl \
      "$IMAGE" || return 1
    echo "[deployer] $(date -Is) deployed ${want#sha256:} (was ${have#sha256:})"
  fi
  # Every deploy leaves the previous image behind; on the 8 GB EBS volume
  # these pile up and eventually fill the 97%-full disk, wedging the NEXT
  # pull with "no space left on device" (which itself looks to users like
  # "the DNS/API is down"). After every successful cycle, drop everything
  # not referenced by a running container (-a includes the just-replaced
  # old image; the running container's own image is never touched).
  docker image prune -a -f >/dev/null 2>&1 || true
}

while true; do
  deploy_once || echo "[deployer] attempt failed, retrying in 5 min" >&2
  sleep 300
done
LOOP
chmod +x /usr/local/bin/pedestrian-deployer

cat > /etc/systemd/system/pedestrian-deployer.service <<'UNIT'
[Unit]
Description=Pedestrian API auto-deploy loop (pulls ghcr :latest)
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
ExecStart=/usr/local/bin/pedestrian-deployer
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now pedestrian-deployer
echo "[bootstrap] $(date -Is) deployer started"

# ---------------------------------------------------------------------------
# DuckDNS (optional): a PERMANENT domain instead of the ephemeral public IP.
# The current instance was set up by SSH (deploy/duckdns-setup.sh). For a NEW
# instance, fill both values in BEFORE launching and the self-updater installs
# itself at first boot. Leave them EMPTY to skip entirely — the repo always
# ships with placeholders, so no token is ever committed. duckdns.sh sends the
# instance's real public IP explicitly (never auto-detect) and the timer
# re-runs at every boot + every 5 min, so the name follows stop/start cycles.
DUCKDNS_DOMAIN=""
DUCKDNS_TOKEN=""
if [ -n "$DUCKDNS_DOMAIN" ] && [ -n "$DUCKDNS_TOKEN" ]; then
  mkdir -p /opt/duckdns /etc/duckdns
  install -m 755 /dev/null /opt/duckdns/duckdns.sh
  cat >/etc/duckdns/config <<'CFG'
DOMAIN=
TOKEN=
CFG
  printf 'DOMAIN=%s\nTOKEN=%s\n' "$DUCKDNS_DOMAIN" "$DUCKDNS_TOKEN" >/etc/duckdns/config
  chmod 600 /etc/duckdns/config
  cat >/opt/duckdns/duckdns.sh <<'DUCK'
#!/usr/bin/env bash
set -euo pipefail
. /etc/duckdns/config
T=$(curl -fsS -m 3 -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 3')
IP=$(curl -fsS -m 5 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/public-ipv4)
R=$(curl -fsS -m 10 "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=${IP}")
[ "$R" = "OK" ] && echo "duckdns: ${DOMAIN}.duckdns.org -> ${IP}" || { echo "duckdns: $R" >&2; exit 1; }
DUCK
  chmod 755 /opt/duckdns/duckdns.sh
  cat >/etc/systemd/system/duckdns.service <<'UNIT'
[Unit]
Description=DuckDNS self-updater (keeps the DNS name on this instance's current public IP)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/duckdns/duckdns.sh
UNIT
  cat >/etc/systemd/system/duckdns.timer <<'UNIT'
[Unit]
Description=Run the DuckDNS updater at every boot and every 5 minutes

[Timer]
OnBootSec=45s
OnUnitActiveSec=5min
AccuracySec=10s

[Install]
WantedBy=timers.target
UNIT
  systemctl daemon-reload
  systemctl enable --now duckdns.timer
fi
