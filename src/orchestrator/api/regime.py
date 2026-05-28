"""``POST /regime-analysis/{queue_id}`` — post-verdict regime breakdown.

Called by the researcher after a sweep yields PIVOT (INSUFFICIENT_EVIDENCE /
NO_EDGE) to label the queue row with its best market regime before deciding
whether to re-queue with a regime filter or pivot to a new archetype.

The endpoint:
  1. Reads the queue row (must be PARKED or COMPLETED).
  2. Resolves the best iteration's backtest_run_id.
  3. Calls services.regime_analysis.analyze().
  4. Writes regime_label + regime_analysis onto research_queue.
  5. Creates a REGIME_ANALYSIS journal entry.
  6. Returns the analysis so the agent can branch immediately.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends

from .deps import get_agent_name, get_db_conn
from ..errors import OrchestratorError
from ..repo import queue as queue_repo
from ..repo import queue_write
from ..repo import journal as journal_repo
from ..services import regime_analysis as ra_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regime-analysis", tags=["regime-analysis"])


@router.post("/{queue_id}")
async def run_regime_analysis(
    queue_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent_name: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Compute per-regime metrics for a completed/parked sweep and label it."""
    queue = await queue_repo.get_queue(conn, queue_id)
    if queue is None:
        raise OrchestratorError(
            status_code=404,
            error_code="queue_not_found",
            message=f"No research_queue row with queue_id={queue_id}.",
            retryable=False,
        )

    if queue["status"] not in ("PARKED", "COMPLETED", "FAILED"):
        raise OrchestratorError(
            status_code=409,
            error_code="queue_not_terminal",
            message=(
                f"Queue {queue_id} is {queue['status']}. Regime analysis runs only "
                "on PARKED / COMPLETED / FAILED queues."
            ),
            retryable=False,
            hint="Call after the sweep has finished (PIVOT or NO_EDGE terminal from /tick/drain).",
        )

    backtest_run_id = queue.get("last_run_id")
    if backtest_run_id is None:
        raise OrchestratorError(
            status_code=422,
            error_code="no_backtest_run",
            message=f"Queue {queue_id} has no last_run_id — no completed iteration to analyze.",
            retryable=False,
        )

    logger.info(
        "regime_analysis: queue_id=%s instrument=%s interval=%s backtest_run_id=%s",
        queue_id, queue["instrument"], queue["interval_name"], backtest_run_id,
    )

    result = await ra_service.analyze(
        conn,
        backtest_run_id=backtest_run_id,
        instrument=queue["instrument"],
        interval_name=queue["interval_name"],
    )

    await queue_write.update_regime_label(
        conn,
        queue_id=queue_id,
        regime_label=result["best_regime"],
        regime_analysis=result,
    )

    journal_content = _format_journal_content(queue, result)
    await journal_repo.insert_journal(
        conn,
        strategy_code=queue["strategy_code"],
        instrument=queue["instrument"],
        interval_name=queue["interval_name"],
        entry_type="REGIME_ANALYSIS",
        title=f"Regime analysis — {queue['strategy_code']} {queue['instrument']} {queue['interval_name']}",
        content=journal_content,
        status="ACTIVE",
        structured_data={
            "queue_id": str(queue_id),
            "best_regime": result["best_regime"],
            "best_pf": result["best_pf"],
            "is_promising": result["is_promising"],
            "regime_source": result["regime_source"],
            "n_total_trades": result["n_total_trades"],
            "breakdown": result["breakdown"],
        },
        iteration_id_refs=None,
        related_journal_ids=None,
        created_by=agent_name,
    )

    regime_signal = await _resolve_regime_signal(conn, queue["instrument"])

    return {
        **result,
        "queue_id": str(queue_id),
        "strategy_code": queue["strategy_code"],
        "instrument": queue["instrument"],
        "interval_name": queue["interval_name"],
        "next_actions": _next_actions(queue, result, regime_signal),
    }


def _format_journal_content(queue: dict, result: dict) -> str:
    lines = [
        f"Strategy: {queue['strategy_code']} × {queue['instrument']} {queue['interval_name']}",
        f"Regime source: {result['regime_source']}",
        f"Total classified trades: {result['n_classified']} / {result['n_total_trades']}",
        "",
        "Per-regime breakdown:",
    ]
    for regime, data in sorted(result["breakdown"].items()):
        lines.append(
            f"  {regime:20s}  n={data['n_trades']:4d}  PF={data['profit_factor']:.3f}"
            f"  WR={data['win_rate']:.2%}  mean_ret={data['mean_return_pct']:.3f}%"
        )
    lines.append("")
    if result["best_regime"]:
        lines.append(
            f"Best regime: {result['best_regime']}  PF={result['best_pf']:.3f}"
            f"  {'PROMISING — consider regime-gated re-queue' if result['is_promising'] else 'below threshold — pivot recommended'}"
        )
    else:
        lines.append("No regime with sufficient trades (n >= 10). Pivot recommended.")
    return "\n".join(lines)


async def _resolve_regime_signal(conn: asyncpg.Connection, instrument: str) -> str | None:
    """Return the name of the latest trained regime signal for this instrument.

    Queries ``signal_definition`` for rows whose name starts with ``regime_``
    and whose ``symbol`` matches the instrument. Returns the most recently
    created trained signal, or None if none exists.
    """
    row = await conn.fetchrow(
        """
        SELECT sd.name
        FROM signal_definition sd
        JOIN model_registry mr ON mr.id = sd.model_id
        WHERE sd.name LIKE 'regime_%'
          AND mr.symbol = $1
          AND mr.status IN ('trained', 'deployment_ready')
        ORDER BY sd.created_time DESC
        LIMIT 1
        """,
        instrument,
    )
    return row["name"] if row else None


def _next_actions(queue: dict, result: dict, regime_signal: str | None) -> list[dict]:
    actions = []
    if result["is_promising"]:
        if regime_signal:
            actions.append({
                "kind": "regime_retry",
                "description": (
                    f"Best regime '{result['best_regime']}' has PF={result['best_pf']:.3f} — "
                    f"re-queue with ML regime gate enabled using signal '{regime_signal}'."
                ),
                "recommended_sweep_override": {
                    "strategy_ml_gate_overrides": {
                        "signal_name": regime_signal,
                        "shadow_mode": False,
                    },
                },
                "regime_signal": regime_signal,
            })
        else:
            actions.append({
                "kind": "regime_retry_no_model",
                "description": (
                    f"Best regime '{result['best_regime']}' has PF={result['best_pf']:.3f} "
                    f"but no trained regime signal exists for {queue['instrument']}. "
                    "Train a regime model first (POST /ml/training-runs), then re-queue."
                ),
                "regime_signal": None,
            })
    else:
        actions.append({
            "kind": "pivot",
            "description": "No regime shows sufficient edge (PF < 1.15 or n < 10). Proceed to STRATEGY_OUTCOME and pivot.",
        })
    return actions
