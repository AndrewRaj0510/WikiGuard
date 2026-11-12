#!/usr/bin/env bash
# Fuzz a running API from the schema it publishes, signed in as a guest. Used by
# .github/workflows/security-scan.yml against staging, and runnable against a local stack:
#
#   API_URL=http://127.0.0.1:8765 bash scripts/scan_api.sh
#
# A guest can read and can't change anything, so the data stays as it was. It does take the single
# seat for up to ten minutes, so this is for staging and local stacks, never production.
set -euo pipefail

: "${API_URL:?Set API_URL to where the API answers}"
TAB_ID="${TAB_ID:-tab-security-scan}"
MAX_SECONDS="${MAX_SECONDS:-600}"
REPORT_DIR="${REPORT_DIR:-schemathesis-report}"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
heartbeat_pid=""
session=()
signed_in=no

cleanup() {
  if [ -n "$heartbeat_pid" ]; then kill "$heartbeat_pid" 2>/dev/null || true; fi
  # If anything failed after signing in, still give the seat back, or the next visitor waits.
  if [ "$signed_in" = yes ]; then
    curl --silent --output /dev/null "${session[@]}" --request POST "${API_URL}/v1/auth/logout" \
      || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

# No Origin header: the API refuses one that isn't the site's own address, and a script, unlike a
# browser, doesn't send one.
curl --fail --silent --show-error --cookie-jar "$work/cookies.txt" \
  --header 'content-type: application/json' \
  --data "{\"tab_id\": \"${TAB_ID}\"}" "${API_URL}/v1/auth/guest" > "$work/session.json"
signed_in=yes
cookie=$(awk '$6 == "wikiguard_session" { print $7 }' "$work/cookies.txt")
csrf=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['csrf_token'])" "$work/session.json")
# Both are secrets for the next few minutes. In GitHub Actions this keeps them out of the log.
echo "::add-mask::${cookie}"
echo "::add-mask::${csrf}"
session=(--cookie "wikiguard_session=${cookie}" --header "x-tab-id: ${TAB_ID}"
  --header "x-csrf-token: ${csrf}")

# The seat is a lease. A browser renews it every 15 seconds, and a guest who does nothing for five
# minutes is signed out. Without this, the scan loses its session within a minute and tests little
# but the "not signed in" answer (F-138).
(
  while true; do
    curl --silent --output /dev/null "${session[@]}" --header 'content-type: application/json' \
      --data '{"active": true}' "${API_URL}/v1/gate/heartbeat" || true
    sleep 15
  done
) &
heartbeat_pid=$!

# Sign-in and seat routes are left out: fuzzing them ends the session the rest needs.
# never_crashes comes from scripts/fuzz_checks.py: no 500, and a 503 only when the API wrote it on
# purpose. The other two checks compare each answer with the schema. The rate stays under the API's
# own limit, so the scan tests the routes and not the limiter. A guest session lasts 15 minutes at
# most, so the scan stops at ten.
status=0
PYTHONPATH="${here}${PYTHONPATH:+:${PYTHONPATH}}" SCHEMATHESIS_HOOKS=fuzz_checks \
  schemathesis run "${API_URL}/v1/openapi.json" \
  --header "Cookie: wikiguard_session=${cookie}" \
  --header "x-tab-id: ${TAB_ID}" \
  --header "x-csrf-token: ${csrf}" \
  --exclude-path-regex '^/v1/(auth/(guest|login|logout)|gate/(claim|release))$' \
  --checks never_crashes,content_type_conformance,response_schema_conformance \
  --max-examples 40 --max-time "$MAX_SECONDS" --rate-limit 90/m --request-timeout 30 \
  --report junit --report-dir "$REPORT_DIR" || status=$?

kill "$heartbeat_pid" 2>/dev/null || true
heartbeat_pid=""

# Signing out answers 204 only while the session is alive. Anything else means the session ended
# during the scan, so most of it ran signed out and a clean result would prove little. This also
# frees the seat.
signed_out=$(curl --silent --output /dev/null --write-out '%{http_code}' "${session[@]}" \
  --request POST "${API_URL}/v1/auth/logout")
signed_in=no
echo "Sign-out answered ${signed_out}"
if [ "$signed_out" != "204" ]; then
  echo "The session didn't last the whole scan, so this run doesn't count." >&2
  exit 1
fi
exit "$status"
