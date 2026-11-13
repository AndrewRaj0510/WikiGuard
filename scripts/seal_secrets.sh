#!/usr/bin/env bash
# Seal one Secret for GitOps: encrypt it to the cluster's Sealed Secrets key, so the result is safe
# to commit to the public repo. Only the controller in that cluster can open it (section 8).
#
#   bash scripts/seal_secrets.sh NAMESPACE NAME ENV_FILE [--context CONTEXT]
#
# ENV_FILE holds KEY=value lines and lives outside the repo, or in data/ (git-ignored). The plain
# Secret goes from kubectl to kubeseal through a pipe and never touches the disk. The sealed file
# lands in k8s/secrets/prod/NAMESPACE-NAME.yaml, for the owner to review and commit.
#
# Which Secrets exist, and which keys each holds, is listed in k8s/README.md.
set -euo pipefail

[ $# -ge 3 ] || { sed -n '2,12p' "$0"; exit 2; }
namespace="$1"; name="$2"; env_file="$3"; shift 3
context_args=()
if [ "${1:-}" = "--context" ]; then context_args=(--context "$2"); fi

[ -f "$env_file" ] || { echo "no such file: $env_file" >&2; exit 1; }
command -v kubeseal >/dev/null || { echo "kubeseal isn't installed (k8s/README.md)" >&2; exit 1; }

# SEALED_DIR lets a rehearsal on kind write somewhere other than the repo.
out_dir="${SEALED_DIR:-$(cd "$(dirname "$0")/.." && pwd)/k8s/secrets/prod}"
mkdir -p "$out_dir"
out="$out_dir/${namespace}-${name}.yaml"

kubectl "${context_args[@]}" create secret generic "$name" -n "$namespace" \
  --from-env-file="$env_file" --dry-run=client -o yaml \
  | kubeseal "${context_args[@]}" --controller-namespace sealed-secrets \
      --controller-name sealed-secrets-controller --format yaml > "$out"

keys="$(awk '/encryptedData:/ {inside=1; next} inside && /^    [^ ]/ {n++} inside && /^  [^ ]/ {inside=0} END {print n+0}' "$out")"
echo "sealed: ${out#"$(pwd)/"} (${keys} keys). Review it, then commit it."
