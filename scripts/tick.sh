#!/usr/bin/env bash
# Cron-callable. Replaces the old research-tick.sh — fires one iteration
# of the orchestrator. Cron drift is safe: the service-side claim uses
# SELECT FOR UPDATE SKIP LOCKED, so two ticks landing simultaneously will
# either pick different rows or one returns outcome=empty_queue.
set -euo pipefail

ORCH_HOST="${ORCH_HOST:-127.0.0.1}"
ORCH_PORT="${ORCH_PORT:-8082}"
ORCH_AUTH_TOKEN="${ORCH_AUTH_TOKEN:?ORCH_AUTH_TOKEN must be set}"
AGENT="${ORCH_AGENT_NAME:-cron}"

# Idempotency-Key only protects retries of an already-completed call —
# the orchestrator caches successful responses by key. Concurrent ticks
# are NOT serialised by this key; that's handled at the DB layer
# (SELECT FOR UPDATE SKIP LOCKED on the queue claim).
KEY="$(date -u +%Y%m%d%H%M)-${AGENT}"

curl -fsS -X POST \
    -H "X-Orch-Token: ${ORCH_AUTH_TOKEN}" \
    -H "X-Agent-Name: ${AGENT}" \
    -H "Idempotency-Key: ${KEY}" \
    -H "Content-Type: application/json" \
    "http://${ORCH_HOST}:${ORCH_PORT}/tick"
