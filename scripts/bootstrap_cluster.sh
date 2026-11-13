#!/usr/bin/env bash
# One command from a fresh Oracle VM to the whole cluster (section 14 of the plan: a free VM can be
# reclaimed, so it must be rebuildable from the repo alone). Run on the VM, from a clone of the repo,
# after scripts/harden_vm.sh:
#
#   sudo bash scripts/bootstrap_cluster.sh                          # a brand-new cluster
#   sudo bash scripts/bootstrap_cluster.sh --sealed-secrets-key F   # a rebuild: restore the key
#   sudo bash scripts/bootstrap_cluster.sh --restore-archive        # a rebuild: refill the archive
#   bash scripts/bootstrap_cluster.sh --skip-k3s --context NAME     # rehearse the rest on kind
#
# What it does, in order: k3s, Helm, the namespaces with their Pod Security levels, the restored
# Sealed Secrets key if given, Argo CD, then the root Application. From there Argo CD pulls the rest
# from GitHub: Sealed Secrets, the secrets, monitoring, and WikiGuard (k8s/argocd/apps). Nothing
# reaches the cluster from outside, so nothing needs to be opened in Oracle's security list.
set -euo pipefail

K3S_VERSION="v1.37.2+k3s1"
HELM_VERSION="v4.3.1"
ARGOCD_CHART_VERSION="10.18.0"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_K3S=false
CONTEXT=""
SEALED_KEY=""
RESTORE_ARCHIVE=false
started=$(date +%s)

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-k3s) SKIP_K3S=true ;;
    --context) CONTEXT="$2"; shift ;;
    --sealed-secrets-key) SEALED_KEY="$2"; shift ;;
    --restore-archive) RESTORE_ARCHIVE=true ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n== %s (%ss)\n' "$1" "$(( $(date +%s) - started ))"; }
kc() { if [ -n "$CONTEXT" ]; then kubectl --context "$CONTEXT" "$@"; else kubectl "$@"; fi; }
hc() { if [ -n "$CONTEXT" ]; then helm --kube-context "$CONTEXT" "$@"; else helm "$@"; fi; }

arch() {
  case "$(uname -m)" in
    aarch64|arm64) echo arm64 ;;
    x86_64|amd64) echo amd64 ;;
    *) echo "unsupported CPU: $(uname -m)" >&2; exit 1 ;;
  esac
}

if [ "$SKIP_K3S" = false ]; then
  [ "$(id -u)" -eq 0 ] || { echo "run with sudo: it installs k3s" >&2; exit 1; }
  say "k3s ${K3S_VERSION}"
  if ! command -v k3s >/dev/null; then
    # The install script checks the binary against the release's own SHA-256 list.
    # - traefik and servicelb off: nothing in the cluster takes public traffic.
    # - secrets-encryption: Secrets are encrypted at rest in k3s's datastore.
    # - The API answers on the tailnet address too, for kubectl from the laptop over Tailscale.
    tailnet_ip="$(tailscale ip -4 2>/dev/null || true)"
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - server \
      --disable traefik --disable servicelb \
      --secrets-encryption \
      --write-kubeconfig-mode 600 \
      ${tailnet_ip:+--tls-san "$tailnet_ip"}
  fi
  export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
  until kubectl get nodes 2>/dev/null | grep -q " Ready"; do sleep 3; done
  kubectl get nodes -o wide

  say "Helm ${HELM_VERSION}"
  if ! helm version --short 2>/dev/null | grep -q "${HELM_VERSION}"; then
    a="$(arch)"
    tmp="$(mktemp -d)"
    curl -sfL -o "$tmp/helm.tgz" "https://get.helm.sh/helm-${HELM_VERSION}-linux-${a}.tar.gz"
    curl -sfL -o "$tmp/helm.tgz.sha256" "https://get.helm.sh/helm-${HELM_VERSION}-linux-${a}.tar.gz.sha256sum"
    (cd "$tmp" && echo "$(cut -d' ' -f1 helm.tgz.sha256)  helm.tgz" | sha256sum --check --quiet)
    tar -xzf "$tmp/helm.tgz" -C "$tmp"
    install -m 0755 "$tmp/linux-${a}/helm" /usr/local/bin/helm
    rm -rf "$tmp"
  fi
fi

say "Namespaces and their Pod Security levels"
kc apply -f "$REPO_DIR/k8s/platform/namespaces.yaml"

if [ -n "$SEALED_KEY" ]; then
  # A rebuild: the old controller's key, from the owner's password manager, goes in before the
  # controller starts, so every SealedSecret already in the repo still opens.
  say "Restoring the Sealed Secrets key"
  kc apply -n sealed-secrets -f "$SEALED_KEY"
fi

say "Argo CD ${ARGOCD_CHART_VERSION}"
hc repo add argo https://argoproj.github.io/argo-helm >/dev/null 2>&1 || true
hc repo update argo >/dev/null
hc upgrade --install argocd argo/argo-cd --version "$ARGOCD_CHART_VERSION" \
  -n argocd -f "$REPO_DIR/k8s/platform/argocd-values.yaml" --wait --timeout 10m

say "The root Application: everything else comes from the repo"
kc apply -f "$REPO_DIR/k8s/argocd/root.yaml"

say "Waiting for Argo CD to sync the apps"
# Ten-second tries per app. A rehearsal before the first push can't sync, so it sets fewer.
ARGO_WAIT_TRIES="${ARGO_WAIT_TRIES:-120}"
for app in sealed-secrets secrets monitoring wikiguard; do
  for _ in $(seq 1 "$ARGO_WAIT_TRIES"); do
    status="$(kc -n argocd get application "$app" -o jsonpath='{.status.sync.status}/{.status.health.status}' 2>/dev/null || true)"
    [ "$status" = "Synced/Healthy" ] && break
    sleep 10
  done
  echo "$app: ${status:-not created yet}"
done

if [ "$RESTORE_ARCHIVE" = true ]; then
  # A rebuild starts with an empty archive volume. The nightly mirror holds the last 30 days.
  say "Refilling the archive from the bucket's mirror"
  kc -n wikiguard get cronjob wikiguard-archive-maintenance -o json \
    | python3 -c '
import json, sys
cron = json.load(sys.stdin)
job = {"apiVersion": "batch/v1", "kind": "Job",
       "metadata": {"name": "restore-archive", "namespace": "wikiguard"},
       "spec": cron["spec"]["jobTemplate"]["spec"]}
job["spec"]["template"]["spec"]["containers"][0]["args"] = ["--restore-mirror"]
print(json.dumps(job))' | kc apply -f -
  kc -n wikiguard wait --for=condition=complete job/restore-archive --timeout=60m
fi

say "Done"
cat <<'EOF'
Next, by hand (k8s/README.md has each step):
  - New cluster: back up the Sealed Secrets key to the password manager, then seal the secrets
    with scripts/seal_secrets.sh and commit them under k8s/secrets/prod.
  - Rebuild: run the restore steps in docs/runbooks/cluster-rebuild.md.
  - Grafana: kubectl -n monitoring port-forward svc/kps-grafana 3000:80, over Tailscale.
EOF
