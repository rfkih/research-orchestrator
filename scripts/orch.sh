#!/usr/bin/env bash
# Generic HTTP wrapper for the research-orchestrator.
# Centralises token resolution, headers, base URL, idempotency key handling,
# and JSON body POSTing so the agent can issue any request as a single
# parser-friendly bash command.
#
# Usage:
#   scripts/orch.sh GET /readyz
#   scripts/orch.sh GET /queue?limit=20
#   scripts/orch.sh GET '/journal?entry_type=STRATEGY_OUTCOME&limit=10' --pretty
#   scripts/orch.sh HEAD /healthz
#   scripts/orch.sh POST /journal --body /tmp/body.json
#   scripts/orch.sh POST /tick --body /tmp/tick.json --ik my-key-1
#   scripts/orch.sh POST /walk-forward --body /tmp/wf.json --ik my-key-2 --pretty
#
# For POST: write the JSON body via the agent's Write tool first (NOT a bash
# heredoc — that trips CC's brace+quote guardrail). The bash invocation then
# contains no JSON literal, no multi-line curl with quoted headers, no
# parser-trip surface.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,/^set -euo pipefail$/p' "$0" | sed 's/^# \?//;/^set /d'
    exit 0
fi

METHOD="${1:?method required (GET|HEAD|POST). Run with --help for usage.}"
PATH_AND_QUERY="${2:?path required (e.g. /readyz, /queue?limit=20). Run with --help for usage.}"
shift 2 || true

case "$METHOD" in
    GET|HEAD|POST) ;;
    *) echo "orch.sh: only GET/HEAD/POST allowed (got: $METHOD)" >&2; exit 2 ;;
esac

PRETTY=0
BODY_FILE=""
IK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretty) PRETTY=1; shift ;;
        --body) BODY_FILE="${2:?--body requires a path}"; shift 2 ;;
        --ik|--idempotency-key) IK="${2:?--ik requires a value}"; shift 2 ;;
        *) echo "orch.sh: unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ "$METHOD" == "POST" ]]; then
    [[ -n "$BODY_FILE" ]] || { echo "orch.sh: POST requires --body <path>" >&2; exit 2; }
    [[ -f "$BODY_FILE" ]] || { echo "orch.sh: --body file not found: $BODY_FILE" >&2; exit 2; }
elif [[ -n "$BODY_FILE" || -n "$IK" ]]; then
    echo "orch.sh: --body / --ik only valid with POST" >&2
    exit 2
fi

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

[[ "$PATH_AND_QUERY" == /* ]] || PATH_AND_QUERY="/$PATH_AND_QUERY"
URL="http://${ORCH_HOST}:${ORCH_PORT}${PATH_AND_QUERY}"

CURL_ARGS=(-fsS -X "$METHOD")
CURL_ARGS+=(-H "X-Orch-Token: ${ORCH_AUTH_TOKEN}")
CURL_ARGS+=(-H "X-Agent-Name: ${AGENT}")

# Optional session grouping. If ORCH_SESSION_ID is exported the wrapper
# passes it through, so all calls in this shell land in the same activity
# session. When unset, the orchestrator synthesizes a deterministic
# per-(agent, UTC date) session UUID — no header needed for sane grouping.
if [[ -n "${ORCH_SESSION_ID:-}" ]]; then
    CURL_ARGS+=(-H "X-Session-Id: ${ORCH_SESSION_ID}")
fi

if [[ "$METHOD" == "POST" ]]; then
    CURL_ARGS+=(-H "Content-Type: application/json")
    CURL_ARGS+=(--data-binary "@${BODY_FILE}")
    if [[ -n "$IK" ]]; then
        CURL_ARGS+=(-H "Idempotency-Key: ${IK}")
    fi
fi

if [[ "$PRETTY" == "1" ]]; then
    curl "${CURL_ARGS[@]}" "$URL" | python3 -m json.tool
else
    curl "${CURL_ARGS[@]}" "$URL"
fi
