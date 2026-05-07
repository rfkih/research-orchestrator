#!/usr/bin/env bash
# Read-only research_journal inspection for the quant-researcher agent.
#
# Usage:
#   scripts/journal-read.sh STRATEGY_OUTCOME           # default limit=10
#   scripts/journal-read.sh RUN_SUMMARY 25
#   scripts/journal-read.sh ANY 10                     # ANY = no entry_type filter
set -euo pipefail

ENTRY_TYPE="${1:?entry_type required (e.g. STRATEGY_OUTCOME, RUN_SUMMARY, ANY)}"
LIMIT="${2:-10}"
ORCH_HOST="${ORCH_HOST:-127.0.0.1}"
ORCH_PORT="${ORCH_PORT:-8082}"
AGENT="${ORCH_AGENT_NAME:-quant-researcher}"

if [[ -z "${ORCH_AUTH_TOKEN:-}" ]]; then
    ENV_FILE="$(dirname "$0")/../.env"
    if [[ -f "$ENV_FILE" ]]; then
        ORCH_AUTH_TOKEN="$(grep -E '^ORCH_AUTH_TOKEN=' "$ENV_FILE" | cut -d= -f2-)"
    fi
fi
: "${ORCH_AUTH_TOKEN:?ORCH_AUTH_TOKEN must be set (env or .env)}"

if [[ "$ENTRY_TYPE" == "ANY" ]]; then
    QUERY="limit=${LIMIT}"
    HEADER="=== JOURNAL (any entry_type, last ${LIMIT}) ==="
else
    QUERY="entry_type=${ENTRY_TYPE}&limit=${LIMIT}"
    HEADER="=== ${ENTRY_TYPE} entries (last ${LIMIT}) ==="
fi

echo "$HEADER"
curl -fsS \
    -H "X-Orch-Token: ${ORCH_AUTH_TOKEN}" \
    -H "X-Agent-Name: ${AGENT}" \
    "http://${ORCH_HOST}:${ORCH_PORT}/journal?${QUERY}" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin)
for e in data.get("items", []):
    strategy = e.get("strategy_code", "?")
    title = (e.get("title") or "")[:80]
    status = e.get("status", "?")
    created = e.get("created_time", "?")
    etype = e.get("entry_type", "?")
    print(f"[{etype}] strategy={strategy} title={title}")
    print(f"  status={status} created={created}")
    print()
'
