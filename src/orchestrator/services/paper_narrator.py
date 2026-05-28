"""AI-powered narrative section generator for research papers.

Calls the `claude` CLI (Max plan OAuth) to write IEEE-style prose sections
(Introduction, Data & Setup, Methodology, Results, Robustness, Discussion,
Conclusion) from structured iteration metrics.  Falls back to a deterministic
template if the CLI is unavailable or the call fails — the rest of
paper_generator.py is unaffected.

Entry point:  ``await generate_narrative(queue, best, iterations, run_meta, gate)``
Returns:       list[{"chapter": int, "title": str, "body": str}]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_CLAUDE_BIN = os.environ.get(
    "CLAUDE_BIN",
    r"C:\Users\rifki\AppData\Local\Microsoft\WinGet\Links\claude.exe",
)
_TIMEOUT_S = int(os.environ.get("ORCH_AI_NARRATIVE_TIMEOUT_S", "480"))


def _is_enabled() -> bool:
    # Checked at call time (not module-load) so load_dotenv() in __main__
    # has already run before the first paper request arrives.
    return os.environ.get("ORCH_AI_NARRATIVE_ENABLED", "").strip() in ("1", "true", "True", "yes")

_SYSTEM = (
    "You are an expert quantitative finance researcher writing a formal empirical "
    "study in IEEE conference paper style.  You write with rigour, precision, and "
    "intellectual honesty.  You explain what worked, what failed, and WHY — "
    "referencing specific numbers from the data.  You never hedge with empty phrases "
    "like 'further research is needed'; you state conclusions directly."
)

_SECTION_PROMPT = """\
{system}

Write the full body text for the following sections of an IEEE-style empirical \
backtest research paper.  Use the structured data below as your sole factual \
source — do not invent numbers.  Every quantitative claim must cite the \
exact figure from the data.

Tone: formal academic, active voice where possible, no bullet lists (prose \
paragraphs only).  Each section should be 2–5 paragraphs appropriate to its \
depth.  Regime analysis, failure mode analysis, and statistical interpretation \
must appear in the Results and Discussion sections.

Return ONLY valid JSON — an object whose keys are the section titles below and \
whose values are the plain-text prose body (no markdown, no LaTeX).

Sections to write:
1. Introduction
2. Data and Backtest Setup
3. Methodology
4. Results
5. Robustness Analysis
6. Discussion
7. Conclusion

--- STRATEGY DATA ---
{data}
--- END DATA ---

Return JSON only, no surrounding text.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_narrative(
    queue: dict[str, Any],
    best: dict[str, Any],
    iterations: list[dict[str, Any]],
    run_meta: dict[str, Any] | None,
    gate: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Return AI-generated sections or None if the CLI is unavailable/call fails."""
    if not _is_enabled():
        return None
    data_block = _build_data_block(queue, best, iterations, run_meta, gate)
    prompt = _SECTION_PROMPT.format(
        system=_SYSTEM,
        data=json.dumps(data_block, indent=2, default=str),
    )

    logger.info("paper_narrator: prompt_chars=%d, timeout_s=%d", len(prompt), _TIMEOUT_S)

    def _run_cli() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [_CLAUDE_BIN, "--print", "--allowed-tools", ""],
            input=prompt.encode(),
            capture_output=True,
            timeout=_TIMEOUT_S,
        )

    loop = asyncio.get_event_loop()
    try:
        completed = await asyncio.wait_for(
            loop.run_in_executor(None, _run_cli),
            timeout=_TIMEOUT_S + 10,
        )
    except FileNotFoundError:
        logger.warning("paper_narrator: claude CLI not found at %s; skipping AI narrative", _CLAUDE_BIN)
        return None
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        logger.warning("paper_narrator: claude CLI timed out after %ds; skipping AI narrative", _TIMEOUT_S)
        return None
    except Exception as exc:
        logger.warning("paper_narrator: subprocess error (%s); skipping AI narrative", exc)
        return None

    if completed.returncode != 0:
        logger.warning(
            "paper_narrator: claude CLI exited %d; stderr=%s",
            completed.returncode,
            completed.stderr.decode(errors="replace")[:400],
        )
        return None

    raw = completed.stdout.decode(errors="replace").strip()

    try:
        # Extract JSON from response — handle text before/after code fence
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        elif not raw.startswith("{"):
            # Try to find a bare JSON object anywhere in the response
            m2 = re.search(r"(\{[^{}].*\})", raw, re.DOTALL)
            if m2:
                raw = m2.group(1).strip()

        sections_map: dict[str, str] = json.loads(raw)
    except Exception as exc:
        logger.warning("paper_narrator: JSON parse failed (%s); skipping AI narrative", exc)
        return None

    ordered_titles = [
        "Introduction",
        "Data and Backtest Setup",
        "Methodology",
        "Results",
        "Robustness Analysis",
        "Discussion",
        "Conclusion",
    ]
    sections: list[dict[str, Any]] = []
    for chapter, title in enumerate(ordered_titles, start=1):
        body = sections_map.get(title, "").strip()
        if body:
            sections.append({"chapter": chapter, "title": title, "body": body})

    return sections if sections else None


# ---------------------------------------------------------------------------
# Data block builder
# ---------------------------------------------------------------------------

def _build_data_block(
    queue: dict[str, Any],
    best: dict[str, Any],
    iterations: list[dict[str, Any]],
    run_meta: dict[str, Any] | None,
    gate: dict[str, Any],
) -> dict[str, Any]:
    ms = best.get("metrics_snapshot") or {}
    analysis = ms.get("analysis") or {}
    ci = best.get("confidence_intervals") or {}
    pf_ci = ci.get("pf_95") or {}
    regimes = analysis.get("regimes") or {}
    by_quarter = regimes.get("by_quarter") or {}
    by_trend = regimes.get("by_trend_regime") or {}
    slip = analysis.get("slippage_sensitivity") or {}
    sweep_cfg = queue.get("sweep_config") or {}

    # Verdict distribution across all iterations
    dist: dict[str, int] = {}
    for it in iterations:
        sv = it.get("statistical_verdict") or "UNKNOWN"
        dist[sv] = dist.get(sv, 0) + 1

    # Top 3 and bottom 3 quarters by PnL for narrative
    quarters_sorted = sorted(
        by_quarter.items(),
        key=lambda kv: float(kv[1].get("pnl", 0)),
        reverse=True,
    )

    # Walk-forward results if present
    wf: dict[str, Any] = {}
    wf_id = queue.get("walk_forward_id")
    if wf_id:
        wf = {"walk_forward_id": str(wf_id), "note": "walk-forward was run for this queue"}

    return {
        "strategy": {
            "code": queue.get("strategy_code"),
            "instrument": queue.get("instrument"),
            "interval": queue.get("interval_name"),
            "hypothesis": queue.get("hypothesis"),
            "sweep_type": sweep_cfg.get("type", "GRID"),
            "n_params_tested": len(iterations),
            "final_verdict": queue.get("final_verdict"),
            "queue_status": queue.get("status"),
        },
        "backtest_period": {
            "start": str(run_meta["start_time"])[:10] if run_meta and run_meta.get("start_time") else None,
            "end": str(run_meta["end_time"])[:10] if run_meta and run_meta.get("end_time") else None,
            "window_days": analysis.get("window_days"),
            "initial_capital_usdt": float(run_meta["initial_capital"]) if run_meta and run_meta.get("initial_capital") else None,
        },
        "best_params": best.get("params_snapshot") or {},
        "best_iteration_metrics": {
            "n_trades": int(ms.get("total_trades") or 0),
            "profit_factor": _f(ms.get("profit_factor")),
            "pf_95_ci_low": _f(pf_ci.get("low")),
            "pf_95_ci_high": _f(pf_ci.get("high")),
            "pf_95_ci_note": "CI must have lower bound > 1.0 to pass V11 gate",
            "cagr_annualized_pct": _f(analysis.get("annualized_geometric_return_pct_at_alloc_90")),
            "cagr_gate_threshold_pct": 10.0,
            "cagr_gate_passed": gate.get("cagr_ok"),
            "dsr": _f(analysis.get("dsr")),
            "dsr_gate_threshold": 0.95,
            "dsr_gate_passed": gate.get("dsr_ok"),
            "sharpe_annualized": _f(analysis.get("sharpe_annualized")),
            "sortino_annualized": _f(analysis.get("sortino_annualized")),
            "calmar_ratio": _f(analysis.get("calmar")),
            "max_drawdown_pct": _f(analysis.get("max_drawdown_pct")),
            "win_rate_pct": _f(ms.get("win_rate")),
            "avg_win": _f(ms.get("avg_win")),
            "avg_loss": _f(ms.get("avg_loss")),
            "skewness": _f(analysis.get("skewness")),
            "excess_kurtosis": _f(analysis.get("excess_kurtosis")),
            "statistical_verdict": best.get("statistical_verdict"),
            "all_gates_passed": gate.get("all_gates_passed"),
        },
        "gate_results": {
            "n_trades_ok": gate.get("n_trades_ok"),
            "n_trades_value": gate.get("n_trades_value"),
            "pf_ci_ok": gate.get("pf_ci_ok"),
            "pf_ci_low": gate.get("pf_ci_low"),
            "dsr_ok": gate.get("dsr_ok"),
            "dsr_value": gate.get("dsr_value"),
            "stat_verdict_ok": gate.get("stat_verdict_ok"),
            "cagr_ok": gate.get("cagr_ok"),
            "cagr_value": gate.get("cagr_value"),
        },
        "regime_analysis": {
            "by_trend_regime": by_trend,
            "regime_note": (
                "BEAR/BULL/NEUTRAL are classified by a rolling 200-period EMA slope. "
                "pnl is raw USDT P&L; win_rate is fraction of trades that closed positive."
            ),
            "quarterly_breakdown": by_quarter,
            "top_3_quarters_by_pnl": [
                {"quarter": q, **v} for q, v in quarters_sorted[:3]
            ],
            "bottom_3_quarters_by_pnl": [
                {"quarter": q, **v} for q, v in quarters_sorted[-3:]
            ],
        },
        "slippage_sensitivity": slip,
        "sweep_verdict_distribution": dist,
        "walk_forward": wf,
        "fee_model": "0.075% per side (Binance SPOT maker/taker)",
    }


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return round(f, 6) if abs(f) < 1e6 else f
    except (TypeError, ValueError):
        return None
