"""Blackheart research orchestrator — agent-first FastAPI service.

Replaces the bash orchestrator (research-tick.sh and friends). Primary caller
is the quant-researcher Claude agent; secondary callers are the trading-JVM
dashboard and ad-hoc operator curl. Auth is a shared-secret header
(X-Orch-Token); identity comes from X-Agent-Name and lands in created_by on
every persisted row.

The service is long-running (uvicorn under systemd / Windows service). Cron
jobs collapse to thin curl callers — no orchestration logic in bash.
"""

__version__ = "0.1.0"
