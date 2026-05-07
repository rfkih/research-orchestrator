#!/usr/bin/env bash
# Read-only queue inspection for the quant-researcher agent.
# Replaces the inline curl|python one-liner that kept tripping permission prompts.
#
# Usage:
#   scripts/queue-status.sh                # default limit=20
#   scripts/queue-status.sh 50             # custom limit
set -euo pipefail

LIMIT="${1:-20}"
ORCH_HOST="${ORCH_HOST:-127.0.0.1}"
ORCH_PORT="${ORCH_PORT:-8082}"
AGENT="${ORCH_AGENT_NAME:-quant-researcher}"

# Resolve token: env wins; else read from .env next to the repo root.
if [[ -z "${ORCH_AUTH_TOKEN:-}" ]]; then
    ENV_FILE="$(dirname "$0")/../.env"
    if [[ -f "$ENV_FILE" ]]; then
        ORCH_AUTH_TOKEN="$(grep -E '^ORCH_AUTH_TOKEN=' "$ENV_FILE" | cut -d= -f2-)"
    fi
fi
: "${ORCH_AUTH_TOKEN:?ORCH_AUTH_TOKEN must be set (env or .env)}"

echo "=== FULL QUEUE (all statuses, limit=${LIMIT}) ==="
curl -fsS \
    -H "X-Orch-Token: ${ORCH_AUTH_TOKEN}" \
    -H "X-Agent-Name: ${AGENT}" \
    "http://${ORCH_HOST}:${ORCH_PORT}/queue?limit=${LIMIT}" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin)
for item in data.get("items", []):
    code = item["strategy_code"]
    interval = item["interval_name"]
    instrument = (item.get("instrument") or "")[:10]
    status = item["status"]
    it = item.get("iteration_number", 0)
    budget = item.get("iter_budget", "?")
    verdict = item.get("final_verdict", "N/A")
    qid = item["queue_id"][:12]
    print(f"{code:6} {interval:4} {instrument:10} {status:12} iter={it}/{budget} verdict={verdict} qid={qid}")
'
