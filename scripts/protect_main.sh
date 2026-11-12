#!/usr/bin/env bash
# Branch protection for main. Run once, by the owner, AFTER the Phase 6 workflows have run on
# GitHub at least once, so the checks named here exist:
#
#   bash scripts/protect_main.sh            # show what would change
#   bash scripts/protect_main.sh --apply    # change it
#
# It requires two checks before anything reaches main:
#   ci        the last job in .github/workflows/ci.yml, which fails if any other job failed
#   analyze   CodeQL, one result per language
# One summary check is used so that path filters work: a job the filter skipped still counts as
# passed, and a job that failed fails `ci`.
#
# Admins aren't forced through it (enforce_admins is false), because the owner works alone and
# pushes to main directly. The checks still run on every push, and a red X is visible at once.
# Force pushes and deleting the branch stay blocked for everyone.
set -euo pipefail

repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
body='{
  "required_status_checks": {
    "strict": false,
    "contexts": ["ci", "analyze (python)", "analyze (javascript-typescript)", "analyze (actions)"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}'

echo "Repository: ${repo}"
echo "Required checks now:"
gh api "repos/${repo}/branches/main/protection/required_status_checks" --jq '.contexts[]' \
  | sed 's/^/  /' || echo "  (none)"
echo "Required checks after:"
echo "${body}" | python -c "import json,sys; [print('  ' + c) for c in json.load(sys.stdin)['required_status_checks']['contexts']]"

if [ "${1:-}" != "--apply" ]; then
  echo "Nothing changed. Run again with --apply."
  exit 0
fi
echo "${body}" | gh api --method PUT "repos/${repo}/branches/main/protection" --input - > /dev/null
echo "Branch protection updated."
