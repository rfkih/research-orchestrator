"""Research paper generation.

``generate_paper`` is the single entry point. Given a ``queue_id`` it:
  1. Reads the research_queue row + all completed iterations.
  2. Selects the best iteration by geometric_return_pct_at_alloc_90.
  3. Fetches the backtest trade list and builds a normalised equity curve.
  4. Assembles title, abstract, and structured section data.
  5. Upserts ``research_paper`` + ``research_paper_trade_data`` atomically.

Called from ``POST /papers/{queue_id}/generate``. The /tick service may
also call this at Step 6 when a queue moves to COMPLETED or PARKED.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg

from ..repo import papers as papers_repo
from ..repo import trades as trades_repo
from ..errors import OrchestratorError

# Gate constants (mirror V11 / V60 — must not drift from analyze.py)
_MIN_TRADES = 100
_MIN_DSR = 0.95
_MIN_CAGR = 10.0

_CITATIONS: list[dict[str, str]] = [
    {
        "key": "bailey2014deflated",
        "text": (
            "Bailey, D. H., & López de Prado, M. (2014). "
            "The Deflated Sharpe Ratio: Correcting for Selection Bias, "
            "Backtest Overfitting and Non-Normality. "
            "Journal of Portfolio Management, 40(5), 94–107."
        ),
    },
    {
        "key": "harvey2016cross",
        "text": (
            "Harvey, C. R., Liu, Y., & Zhu, H. (2016). "
            "…and the Cross-Section of Expected Returns. "
            "Review of Financial Studies, 29(1), 5–68."
        ),
    },
    {
        "key": "ta4j",
        "text": (
            "Ta4j Contributors. (2024). "
            "Technical Analysis for Java (ta4j). "
            "GitHub: https://github.com/ta4j/ta4j"
        ),
    },
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_paper(
    conn: asyncpg.Connection,
    queue_id: UUID,
    agent_name: str,
) -> dict[str, Any]:
    """Generate or refresh a research paper for ``queue_id``.

    Returns the upserted ``research_paper`` row dict.
    Raises ``OrchestratorError`` (422) when the queue has no completed
    iterations yet — the caller should retry after the first /tick finishes.
    """
    queue = await papers_repo.get_queue_for_paper(conn, queue_id)
    if queue is None:
        raise OrchestratorError(
            status_code=404,
            error_code="queue_not_found",
            message=f"No research_queue row with queue_id={queue_id}.",
            retryable=False,
            hint="Check GET /queue to confirm the queue_id exists.",
        )

    iterations = await papers_repo.get_iterations_for_queue(conn, queue_id)
    if not iterations:
        raise OrchestratorError(
            status_code=422,
            error_code="no_iterations",
            message=(
                f"Queue {queue_id} ({queue['strategy_code']}) has no completed "
                "iterations yet. Run POST /tick first."
            ),
            retryable=True,
            hint="Call POST /tick to complete at least one iteration, then retry.",
        )

    best = _find_best_iteration(iterations)
    run_meta = await papers_repo.get_backtest_run_meta(conn, best["backtest_run_id"])
    raw_trades = await trades_repo.fetch_trades(conn, best["backtest_run_id"])

    initial_capital = float(run_meta["initial_capital"]) if run_meta else 1000.0
    equity_curve = _build_equity_curve(raw_trades, initial_capital)
    chart_trades = _format_trades_for_chart(raw_trades)

    journal_entries = await papers_repo.get_journal_for_strategy(
        conn, queue["strategy_code"], queue["instrument"]
    )

    paper_id = _build_paper_id(queue)
    title = _generate_title(queue)
    abstract = _generate_abstract(queue, best, iterations, run_meta)

    # Determine status: FINALIZED if queue is COMPLETED/PARKED and walk-forward
    # has run; WORKING_PAPER otherwise.
    paper_status = (
        "FINALIZED"
        if queue["status"] in ("COMPLETED", "PARKED") and queue.get("walk_forward_id")
        else "WORKING_PAPER"
    )

    existing = await papers_repo.get_paper(conn, paper_id)
    version = (existing["version"] + 1) if existing else 1

    async with conn.transaction():
        paper = await papers_repo.upsert_paper(
            conn,
            paper_id=paper_id,
            queue_id=queue_id,
            title=title,
            abstract=abstract,
            paper_status=paper_status,
            version=version,
            created_by=agent_name,
        )
        await papers_repo.upsert_trade_data(
            conn,
            paper_id=paper_id,
            iteration_id=best["iteration_id"],
            equity_curve=equity_curve,
            trades=chart_trades,
            wf_equity_curve=None,
            created_by=agent_name,
        )

    return {
        **paper,
        "metadata": _build_metadata(queue, best, iterations, run_meta),
        "best_iteration": _build_best_iter_section(best),
        "top_iterations": _build_top_iterations(iterations),
        "robustness": _build_robustness(best),
        "verdict_gate": _build_verdict_gate(best),
        "journal_entries": journal_entries,
        "citations": _CITATIONS,
    }


# ---------------------------------------------------------------------------
# Best iteration selection
# ---------------------------------------------------------------------------

def _find_best_iteration(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick highest geometric_return_pct_at_alloc_90; fall back to last row."""
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for it in iterations:
        metrics = it.get("metrics_snapshot") or {}
        geom = metrics.get("geometric_return_pct_at_alloc_90")
        if geom is None:
            continue
        try:
            score = float(geom)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score) and score > best_score:
            best_score = score
            best = it
    return best if best is not None else iterations[-1]


# ---------------------------------------------------------------------------
# Equity curve + trade formatter
# ---------------------------------------------------------------------------

def _build_equity_curve(
    trades: list[dict[str, Any]],
    initial_capital: float,
) -> list[dict[str, Any]]:
    """Normalised equity curve (base = 100) sampled at each trade close.

    Shape: [{t: epoch_ms, value: float}]
    """
    closed = sorted(
        (
            t for t in trades
            if t.get("exit_time") is not None
            and t.get("realized_pnl_amount") is not None
        ),
        key=lambda t: t["exit_time"],
    )
    if not closed:
        return []

    running = initial_capital
    first_entry: datetime = closed[0]["entry_time"]
    curve: list[dict[str, Any]] = [
        {"t": _to_epoch_ms(first_entry), "value": 100.0}
    ]
    for trade in closed:
        running += float(trade["realized_pnl_amount"])
        value = round((running / initial_capital) * 100.0, 4)
        curve.append({"t": _to_epoch_ms(trade["exit_time"]), "value": value})
    return curve


def _format_trades_for_chart(
    trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Minimal trade shape needed by the frontend P&L histogram.

    Shape: [{entry_time, exit_time, side, pnl, pnl_pct}]
    pnl_pct derived from realized_pnl_amount / notional_size when available.
    """
    result = []
    for t in trades:
        if t.get("exit_time") is None:
            continue
        pnl = _safe_float(t.get("realized_pnl_amount"))
        notional = _safe_float(t.get("notional_size"))
        pnl_pct: float | None = None
        if pnl is not None and notional and notional > 0:
            pnl_pct = round((pnl / notional) * 100.0, 6)
        result.append(
            {
                "entry_time": _iso(t.get("entry_time")),
                "exit_time": _iso(t.get("exit_time")),
                "side": t.get("side"),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Paper ID + text generation
# ---------------------------------------------------------------------------

def _build_paper_id(queue: dict[str, Any]) -> str:
    year = queue["created_time"].year if queue.get("created_time") else datetime.now().year
    code = queue["strategy_code"].upper()
    symbol = queue["instrument"].upper()
    interval = queue["interval_name"].upper()
    return f"BH-{year}-{code}-{symbol}-{interval}"


def _generate_title(queue: dict[str, Any]) -> str:
    strategy = queue["strategy_code"].upper()
    symbol = queue["instrument"]
    interval = queue["interval_name"].upper()
    return f"{strategy} Strategy on {symbol} ({interval}): An Empirical Backtest Study"


def _generate_abstract(
    queue: dict[str, Any],
    best: dict[str, Any],
    iterations: list[dict[str, Any]],
    run_meta: dict[str, Any] | None,
) -> str:
    metrics = best.get("metrics_snapshot") or {}
    analysis = metrics.get("analysis") or {}
    ci = best.get("confidence_intervals") or {}
    pf_ci = ci.get("pf_95") or {}

    n_trades = int(metrics.get("trade_count") or 0)
    pf = _safe_float(metrics.get("profit_factor"))
    cagr = _safe_float(metrics.get("geometric_return_pct_at_alloc_90"))
    dsr = _safe_float(analysis.get("dsr"))
    pf_lo = _safe_float(pf_ci.get("low"))
    pf_hi = _safe_float(pf_ci.get("high"))
    max_dd = _safe_float(metrics.get("max_drawdown_pct"))
    stat_verdict = best.get("statistical_verdict") or "NOT_TESTED"
    final_verdict = queue.get("final_verdict") or "PENDING"
    n_iters = len(iterations)

    sweep_cfg = queue.get("sweep_config") or {}
    sweep_type = "Bayesian (TPE)" if sweep_cfg.get("type") == "TPE" else "grid"

    period_str = ""
    if run_meta and run_meta.get("start_time") and run_meta.get("end_time"):
        s = run_meta["start_time"].strftime("%Y-%m")
        e = run_meta["end_time"].strftime("%Y-%m")
        period_str = f" over the period {s} to {e}"

    pf_str = f"{pf:.2f}" if pf is not None else "N/A"
    ci_str = (
        f" (95% CI: [{pf_lo:.2f}, {pf_hi:.2f}])"
        if pf_lo is not None and pf_hi is not None
        else ""
    )
    cagr_str = f"{cagr:.1f}%" if cagr is not None else "N/A"
    dsr_str = f"{dsr:.3f}" if dsr is not None else "N/A"
    dd_str = f"{abs(max_dd):.1f}%" if max_dd is not None else "N/A"

    hypothesis_sentence = (
        f" Hypothesis: {queue['hypothesis'].strip()}"
        if queue.get("hypothesis")
        else ""
    )

    verdict_clause = {
        "PASS": "meeting all statistical and economic gates",
        "DISCARD": "failing to meet the minimum edge threshold",
        "PARKED": "showing significant edge pending walk-forward validation",
        "PENDING": "with evaluation ongoing",
    }.get(final_verdict, f"with final verdict {final_verdict}")

    return (
        f"This paper presents an empirical evaluation of the {queue['strategy_code'].upper()} "
        f"trading strategy applied to {queue['instrument']} on the {queue['interval_name'].upper()} "
        f"timeframe{period_str}.{hypothesis_sentence} "
        f"Using a {sweep_type} parameter sweep, we evaluated {n_iters} parameter "
        f"configurations against statistical and economic performance gates. "
        f"The best-performing configuration achieved a compounded annual growth rate (CAGR) "
        f"of {cagr_str} at 90% Kelly allocation, a Profit Factor of {pf_str}{ci_str}, "
        f"a Deflated Sharpe Ratio (DSR) of {dsr_str}, a maximum drawdown of {dd_str}, "
        f"and {n_trades} closed trades. "
        f"Statistical assessment: {stat_verdict.replace('_', ' ').title()}. "
        f"The study concludes with a {final_verdict} verdict, {verdict_clause}."
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_metadata(
    queue: dict[str, Any],
    best: dict[str, Any],
    iterations: list[dict[str, Any]],
    run_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    sweep_cfg = queue.get("sweep_config") or {}
    return {
        "strategy_code": queue["strategy_code"],
        "instrument": queue["instrument"],
        "interval_name": queue["interval_name"],
        "hypothesis": queue.get("hypothesis"),
        "sweep_type": sweep_cfg.get("type", "GRID"),
        "sweep_config": sweep_cfg,
        "final_verdict": queue.get("final_verdict"),
        "queue_status": queue.get("status"),
        "n_iterations": len(iterations),
        "backtest_period": {
            "start": _iso(run_meta.get("start_time")) if run_meta else None,
            "end": _iso(run_meta.get("end_time")) if run_meta else None,
        },
        "initial_capital": _safe_float(run_meta.get("initial_capital")) if run_meta else None,
    }


def _build_best_iter_section(best: dict[str, Any]) -> dict[str, Any]:
    if not best or "iteration_id" not in best:
        return {}
    metrics = best.get("metrics_snapshot") or {}
    analysis = metrics.get("analysis") or {}
    ci = best.get("confidence_intervals") or {}
    pf_ci = ci.get("pf_95") or {}
    slippage = analysis.get("slippage_sensitivity") or {}

    return {
        "iteration_id": str(best["iteration_id"]),
        "iteration_number": best.get("iteration_number"),
        "backtest_run_id": str(best["backtest_run_id"]) if best.get("backtest_run_id") else None,
        "params": best.get("params_snapshot") or {},
        "metrics": {
            "trade_count": metrics.get("trade_count"),
            "profit_factor": _safe_float(metrics.get("profit_factor")),
            "cagr": _safe_float(metrics.get("geometric_return_pct_at_alloc_90")),
            "return_pct": _safe_float(metrics.get("return_pct")),
            "max_drawdown_pct": _safe_float(metrics.get("max_drawdown_pct")),
            "sharpe_ratio": _safe_float(metrics.get("sharpe_ratio")),
            "sortino_ratio": _safe_float(metrics.get("sortino_ratio")),
            "dsr": _safe_float(analysis.get("dsr")),
            "pf_ci_low": _safe_float(pf_ci.get("low")),
            "pf_ci_high": _safe_float(pf_ci.get("high")),
        },
        "slippage_sensitivity": slippage,
        "regime_breakdown": analysis.get("regime_breakdown") or {},
        "statistical_verdict": best.get("statistical_verdict"),
        "verdict": best.get("verdict"),
        "quant_audit_notes": best.get("quant_audit_notes"),
    }


def _build_top_iterations(
    iterations: list[dict[str, Any]],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Top N iterations ranked by CAGR descending."""
    def _score(it: dict[str, Any]) -> float:
        m = it.get("metrics_snapshot") or {}
        v = m.get("geometric_return_pct_at_alloc_90")
        try:
            return float(v) if v is not None else float("-inf")
        except (TypeError, ValueError):
            return float("-inf")

    ranked = sorted(iterations, key=_score, reverse=True)[:n]
    result = []
    for it in ranked:
        metrics = it.get("metrics_snapshot") or {}
        ci = it.get("confidence_intervals") or {}
        pf_ci = ci.get("pf_95") or {}
        analysis = metrics.get("analysis") or {}
        result.append(
            {
                "iteration_number": it.get("iteration_number"),
                "params": it.get("params_snapshot") or {},
                "trade_count": metrics.get("trade_count"),
                "profit_factor": _safe_float(metrics.get("profit_factor")),
                "pf_ci_low": _safe_float(pf_ci.get("low")),
                "pf_ci_high": _safe_float(pf_ci.get("high")),
                "cagr": _safe_float(metrics.get("geometric_return_pct_at_alloc_90")),
                "max_drawdown_pct": _safe_float(metrics.get("max_drawdown_pct")),
                "dsr": _safe_float(analysis.get("dsr")),
                "statistical_verdict": it.get("statistical_verdict"),
                "verdict": it.get("verdict"),
            }
        )
    return result


def _build_robustness(best: dict[str, Any]) -> dict[str, Any]:
    metrics = best.get("metrics_snapshot") or {}
    analysis = metrics.get("analysis") or {}
    return {
        "slippage_sensitivity": analysis.get("slippage_sensitivity") or {},
        "regime_breakdown": analysis.get("regime_breakdown") or {},
    }


def _build_verdict_gate(best: dict[str, Any]) -> dict[str, Any]:
    metrics = best.get("metrics_snapshot") or {}
    analysis = metrics.get("analysis") or {}
    ci = best.get("confidence_intervals") or {}
    pf_ci = ci.get("pf_95") or {}

    n_trades = int(metrics.get("trade_count") or 0)
    pf_lo = _safe_float(pf_ci.get("low"))
    dsr = _safe_float(analysis.get("dsr"))
    stat_verdict = best.get("statistical_verdict") or ""
    cagr = _safe_float(metrics.get("geometric_return_pct_at_alloc_90"))

    n_trades_ok = n_trades >= _MIN_TRADES
    pf_ci_ok = pf_lo is not None and pf_lo > 1.0
    dsr_ok = dsr is not None and dsr >= _MIN_DSR
    stat_ok = stat_verdict == "SIGNIFICANT_EDGE"
    cagr_ok = cagr is not None and cagr >= _MIN_CAGR

    return {
        "n_trades_ok": n_trades_ok,
        "n_trades_value": n_trades,
        "n_trades_threshold": _MIN_TRADES,
        "pf_ci_ok": pf_ci_ok,
        "pf_ci_low": pf_lo,
        "dsr_ok": dsr_ok,
        "dsr_value": dsr,
        "dsr_threshold": _MIN_DSR,
        "stat_verdict_ok": stat_ok,
        "stat_verdict": stat_verdict,
        "cagr_ok": cagr_ok,
        "cagr_value": cagr,
        "cagr_threshold": _MIN_CAGR,
        "all_gates_passed": all([n_trades_ok, pf_ci_ok, dsr_ok, stat_ok, cagr_ok]),
    }


# ---------------------------------------------------------------------------
# Public assembly helper — used by api/papers.py to avoid importing privates
# ---------------------------------------------------------------------------

async def build_paper_sections(
    conn: asyncpg.Connection,
    paper_row: dict[str, Any],
) -> dict[str, Any]:
    """Re-assemble all paper sections from DB for GET /papers/{id} and LaTeX export.

    Accepts an already-fetched ``research_paper`` row and enriches it with
    live queue + iteration data.  Callers avoid importing private helpers.
    """
    queue = await papers_repo.get_queue_for_paper(conn, paper_row["queue_id"])
    iterations = await papers_repo.get_iterations_for_queue(conn, paper_row["queue_id"])
    journal = await papers_repo.get_journal_for_strategy(
        conn,
        queue["strategy_code"] if queue else "",
        queue["instrument"] if queue else "",
    )

    best = _find_best_iteration(iterations) if iterations else {}
    run_meta = None
    if best and best.get("backtest_run_id"):
        run_meta = await papers_repo.get_backtest_run_meta(conn, best["backtest_run_id"])

    return {
        **paper_row,
        "metadata": _build_metadata(queue or {}, best, iterations, run_meta),
        "best_iteration": _build_best_iter_section(best),
        "top_iterations": _build_top_iterations(iterations),
        "robustness": _build_robustness(best),
        "verdict_gate": _build_verdict_gate(best),
        "journal_entries": journal,
        "citations": _CITATIONS,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_epoch_ms(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
