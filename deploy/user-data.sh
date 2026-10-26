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
    docker run -d --name "$CONTAINER" --restart unless-stopped \
      -p 8000:8000 "$IMAGE" || return 1
    echo "[deployer] $(date -Is) deployed ${want#sha256:} (was ${have#sha256:})"
  fi
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
