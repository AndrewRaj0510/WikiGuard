#!/usr/bin/env bash
# Try real connections from inside the running pods, and compare each with what the NetworkPolicies
# should allow (k8s/charts/wikiguard/templates/networkpolicies.yaml). A policy that's only rendered
# proves nothing: this checks the cluster actually enforces it.
#
#   bash scripts/check_network_policies.sh                          # the current kubectl context
#   bash scripts/check_network_policies.sh --context kind-wikiguard
#
# Every probe opens a TCP connection with Python inside the pod, so it needs the WikiGuard images,
# and the alert sink on kind or any Python pod in `monitoring` (MONITORING_PROBE) for the last two.
# Exits 1 if any result differs from what's expected.
set -uo pipefail

kc=(kubectl)
if [ "${1:-}" = "--context" ]; then kc=(kubectl --context "$2"); fi
MONITORING_PROBE="${MONITORING_PROBE:-deploy/alert-sink}"
DATABASE_HOST="${DATABASE_HOST:-postgres}"
api="$("${kc[@]}" get endpointslices -n default -l kubernetes.io/service-name=kubernetes \
  -o jsonpath='{.items[0].endpoints[0].addresses[0]}')"
failed=0

probe() {  # namespace workload host port expected(open|closed)
  local result
  result="$("${kc[@]}" -n "$1" exec "$2" -- python -c "
import socket
try:
    socket.create_connection(('$3', $4), timeout=4).close()
    print('open')
except OSError:
    print('closed')" 2>/dev/null | tail -1)"
  if [ "$result" = "$5" ]; then mark=PASS; else mark=FAIL; failed=1; fi
  printf '%s  %-28s -> %-32s %-7s (expected %s)\n' "$mark" "$2" "$3:$4" "${result:-error}" "$5"
}

probe wikiguard deploy/wikiguard-scorer "$DATABASE_HOST" 5432 open
probe wikiguard deploy/wikiguard-scorer redpanda 9092 open
probe wikiguard deploy/wikiguard-scorer "$api" 6443 closed
probe wikiguard deploy/wikiguard-scorer 169.254.169.254 80 closed
probe wikiguard deploy/wikiguard-ingestor redpanda 9092 open
probe wikiguard deploy/wikiguard-ingestor "$DATABASE_HOST" 5432 closed
probe wikiguard deploy/wikiguard-ingestor en.wikipedia.org 443 open
probe wikiguard deploy/wikiguard-ingestor en.wikipedia.org 80 closed
probe wikiguard deploy/wikiguard-archiver "$DATABASE_HOST" 5432 closed
probe wikiguard deploy/wikiguard-archiver en.wikipedia.org 443 closed
probe wikiguard deploy/wikiguard-archiver wikiguard-scorer 8000 closed
probe monitoring "$MONITORING_PROBE" wikiguard-scorer.wikiguard 8000 open
probe monitoring "$MONITORING_PROBE" redpanda.wikiguard 9092 closed
exit "$failed"
