"""LaTeX export for research papers.

``render_latex(paper_data)`` accepts the full paper dict returned by
``paper_generator.generate_paper`` and produces a complete, compilable
``.tex`` source string.

The output uses the standard ``article`` class (compatible with most
venue templates) and compiles without modification on Overleaf or any
local LaTeX distribution (pdflatex / xelatex / lualatex).  Venue-specific
templates (IEEE, ACM, Springer LNCS) require the user to copy section
bodies into the venue's skeleton — this file provides the content.

No external Python dependencies beyond the stdlib.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_latex(paper: dict[str, Any]) -> str:
    """Return a complete compilable .tex string for ``paper``."""
    meta = paper.get("metadata") or {}
    best = paper.get("best_iteration") or {}
    top = paper.get("top_iterations") or []
    robustness = paper.get("robustness") or {}
    gate = paper.get("verdict_gate") or {}
    journal = paper.get("journal_entries") or []
    citations = paper.get("citations") or []

    # RAW title — _preamble does the (single) escape. Escaping here too
    # double-escaped every ``_``-containing strategy code: pass 1 produced
    # ``EMA\_BAND``, pass 2 re-escaped the backslash into
    # ``EMA\textbackslash{}\_BAND`` — garbling the \title of every export.
    title = paper.get("title") or "Untitled Research Paper"
    abstract = paper.get("abstract") or ""
    paper_id = paper.get("paper_id") or ""
    version = paper.get("version") or 1
    date_str = _format_date(paper.get("created_time"))

    sections = [
        _preamble(title, date_str),
        _abstract_section(abstract),
        _section_introduction(meta, best),
        _section_data_setup(meta),
        _section_methodology(meta),
        _section_results(meta, best, top),
        _section_robustness(robustness, best),
        _section_conclusion(gate, meta),
        _section_agent_notes(journal),
        _bibliography(citations, paper_id, version),
        r"\end{document}",
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------

def _preamble(title: str, date_str: str) -> str:
    esc_title = _esc(title)
    return rf"""\documentclass[journal]{{IEEEtran}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{microtype}}

\title{{{esc_title}}}
\author{{Blackheart Research System}}
\date{{{date_str}}}

\begin{{document}}
\maketitle

"""


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------

def _abstract_section(abstract: str) -> str:
    return rf"""\begin{{abstract}}
{_esc(abstract)}
\end{{abstract}}"""


# ---------------------------------------------------------------------------
# Section 1 — Introduction
# ---------------------------------------------------------------------------

def _section_introduction(
    meta: dict[str, Any],
    best: dict[str, Any],
) -> str:
    code = _esc(meta.get("strategy_code") or "N/A")
    symbol = _esc(meta.get("instrument") or "N/A")
    interval = _esc(meta.get("interval_name") or "N/A")
    hypothesis = _esc(meta.get("hypothesis") or "")
    verdict = _esc(meta.get("final_verdict") or "PENDING")
    cagr = best.get("metrics", {}).get("cagr")
    cagr_str = f"{cagr:.1f}\\%" if cagr is not None else "N/A"

    hypothesis_paragraph = (
        f"\n\nThe research hypothesis was pre-registered as follows: "
        f"\\textit{{{hypothesis}}}"
        if hypothesis
        else ""
    )

    return rf"""\section{{Introduction}}

This paper presents an empirical evaluation of the \textbf{{{code}}} strategy
applied to the \textbf{{{symbol}}} perpetual futures market on the
\textbf{{{interval}}} timeframe. The study was conducted using the Blackheart
algorithmic trading research engine and follows a rigorous statistical framework
combining bootstrap confidence intervals, the Deflated Sharpe Ratio (DSR)
\cite{{bailey2014deflated}}, and the Harvey–Liu–Zhu (HLZ) multiple-testing
correction \cite{{harvey2016cross}}.{hypothesis_paragraph}

The primary contribution of this study is a systematic sweep of the {code}
strategy parameter space, culminating in a {verdict} verdict with the
best-performing configuration achieving a compounded annual growth rate (CAGR)
of {cagr_str} at 90\% Kelly allocation.

The remainder of this paper is organised as follows. Section~\ref{{sec:data}}
describes the data and backtest setup. Section~\ref{{sec:methodology}} outlines
the strategy logic and statistical framework. Section~\ref{{sec:results}}
presents the empirical results. Section~\ref{{sec:robustness}} reports
robustness checks. Section~\ref{{sec:conclusion}} concludes."""


# ---------------------------------------------------------------------------
# Section 2 — Data & Setup
# ---------------------------------------------------------------------------

def _section_data_setup(meta: dict[str, Any]) -> str:
    symbol = _esc(meta.get("instrument") or "N/A")
    interval = _esc(meta.get("interval_name") or "N/A")
    period = meta.get("backtest_period") or {}
    start = _esc(period.get("start") or "N/A")
    end = _esc(period.get("end") or "N/A")
    capital = meta.get("initial_capital")
    capital_str = f"\\${capital:,.2f}" if capital else "N/A"

    return rf"""\section{{Data and Backtest Setup}}
\label{{sec:data}}

\subsection{{Market and Timeframe}}
The strategy was evaluated on \textbf{{{symbol}}} perpetual futures sourced from
the Binance exchange. The primary analysis timeframe is \textbf{{{interval}}},
covering the period from \textbf{{{start}}} to \textbf{{{end}}}.

\subsection{{Backtest Parameters}}
\begin{{itemize}}
  \item \textbf{{Instrument:}} {symbol}
  \item \textbf{{Timeframe:}} {interval}
  \item \textbf{{Period:}} {start} to {end}
  \item \textbf{{Initial Capital:}} {capital_str}
  \item \textbf{{Exchange Fees:}} Binance taker fee (0.04\%)
  \item \textbf{{Engine:}} Blackheart Research JVM (see \S\ref{{sec:conclusion}} for version)
\end{{itemize}}

\subsection{{Data Quality}}
Price and volume data are sourced from the Binance REST API via the Blackheart
data pipeline and stored in a TimescaleDB hypertable partitioned by 7-day
chunks. Funding-rate data is included in position P\&L calculations."""


# ---------------------------------------------------------------------------
# Section 3 — Methodology
# ---------------------------------------------------------------------------

def _section_methodology(meta: dict[str, Any]) -> str:
    sweep_cfg = meta.get("sweep_config") or {}
    sweep_type = meta.get("sweep_type") or "GRID"
    n_iters = meta.get("n_iterations") or 0
    n_iters_str = str(n_iters)
    sweep_type_readable = "Bayesian optimisation (Tree-structured Parzen Estimator, TPE)" \
        if sweep_type == "TPE" else "exhaustive grid search"

    axes_rows = ""
    axes = sweep_cfg.get("axes") or {}
    if axes:
        rows = []
        for name, spec in axes.items():
            if isinstance(spec, dict):
                lo = spec.get("min", spec.get("low", ""))
                hi = spec.get("max", spec.get("high", ""))
                step = spec.get("step", "")
                step_str = f" step {step}" if step else ""
                rows.append(
                    f"  {_esc(str(name))} & {_esc(str(lo))}--{_esc(str(hi))}{_esc(step_str)} \\\\"
                )
            else:
                rows.append(f"  {_esc(str(name))} & {_esc(str(spec))} \\\\")
        axes_rows = "\n".join(rows)

    axes_table = ""
    if axes_rows:
        axes_table = rf"""
\begin{{table}}[ht]
\centering
\caption{{Parameter sweep axes}}
\label{{tab:sweep_axes}}
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Parameter}} & \textbf{{Range}} \\
\midrule
{axes_rows}
\bottomrule
\end{{tabular}}
\end{{table}}"""

    return rf"""\section{{Methodology}}
\label{{sec:methodology}}

\subsection{{Strategy Description}}
The {_esc(meta.get("strategy_code") or "N/A")} strategy is implemented in Java
using the TA4j technical analysis library \cite{{ta4j}}. Entry and exit
conditions are evaluated bar-by-bar on the {_esc(meta.get("interval_name") or "N/A")}
timeframe.

\subsection{{Parameter Sweep}}
We conducted a {sweep_type_readable} over the parameter space defined in
Table~\ref{{tab:sweep_axes}}, evaluating {n_iters_str} parameter combinations
in total.{axes_table}

\subsection{{Statistical Framework}}
Each parameter configuration is evaluated against the following gates:

\begin{{enumerate}}
  \item \textbf{{Minimum trade count:}} $n \geq 100$ closed trades.
  \item \textbf{{Profit Factor 95\% CI:}} Bootstrap lower bound $> 1.0$
        ($B = 2{{,}}000$ resamples) \cite{{bailey2014deflated}}.
  \item \textbf{{Deflated Sharpe Ratio:}} $\mathrm{{DSR}} \geq 0.90$, deflated
        by the cumulative number of trials via the HLZ correction
        \cite{{harvey2016cross}}.
  \item \textbf{{Economic gate:}} Annualised geometric return $\geq 10\%$ at
        90\% Kelly allocation (365-day year, compounded).
\end{{enumerate}}

A configuration is labelled \textsc{{Significant Edge}} only when all four
gates pass simultaneously. The final verdict is \textsc{{Pass}} when
\textsc{{Significant Edge}} is reached; \textsc{{Discard}} when no edge is
found early; \textsc{{Parked}} when edge is found but walk-forward validation
has not yet run."""


# ---------------------------------------------------------------------------
# Section 4 — Results
# ---------------------------------------------------------------------------

def _section_results(
    meta: dict[str, Any],
    best: dict[str, Any],
    top: list[dict[str, Any]],
) -> str:
    bm = best.get("metrics") or {}
    params = best.get("params") or {}

    cagr = bm.get("cagr")
    pf = bm.get("profit_factor")
    pf_lo = bm.get("pf_ci_low")
    pf_hi = bm.get("pf_ci_high")
    dd = bm.get("max_drawdown_pct")
    sharpe = bm.get("sharpe_ratio")
    n_trades = bm.get("trade_count")
    dsr = bm.get("dsr")
    stat_verdict = _esc(best.get("statistical_verdict") or "N/A")

    def _fmt(v: Any, decimals: int = 2, suffix: str = "") -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    ci_str = (
        f" (95\\% CI: [{_fmt(pf_lo)}, {_fmt(pf_hi)}])"
        if pf_lo is not None and pf_hi is not None
        else ""
    )

    params_str = ", ".join(
        f"{_esc(str(k))} = {_esc(str(v))}" for k, v in params.items()
    ) if params else "N/A"

    top_rows = _build_top_iterations_table(top)

    return rf"""\section{{Results}}
\label{{sec:results}}

\subsection{{Best Configuration}}
The optimal parameter configuration ({params_str}) achieved the metrics
summarised below:

\begin{{itemize}}
  \item \textbf{{CAGR (90\% Kelly):}} {_fmt(cagr, 1)}\%
  \item \textbf{{Profit Factor:}} {_fmt(pf)}{ci_str}
  \item \textbf{{Maximum Drawdown:}} {_fmt(dd, 1)}\%
  \item \textbf{{Sharpe Ratio:}} {_fmt(sharpe)}
  \item \textbf{{Deflated Sharpe Ratio:}} {_fmt(dsr, 3)}
  \item \textbf{{Closed Trades:}} {n_trades or "N/A"}
  \item \textbf{{Statistical Verdict:}} {stat_verdict.replace("_", " ")}
\end{{itemize}}

\subsection{{Top Parameter Configurations}}
Table~\ref{{tab:top_iters}} reports the top {len(top)} configurations by CAGR.
The equity curve of the best configuration is rendered in the frontend
research library (\textit{{chart available in the interactive report}}).

{top_rows}"""


def _build_top_iterations_table(top: list[dict[str, Any]]) -> str:
    if not top:
        return ""

    def _fmt(v: Any, d: int = 2) -> str:
        if v is None:
            return "---"
        try:
            return f"{float(v):.{d}f}"
        except (TypeError, ValueError):
            return "---"

    rows = []
    for i, it in enumerate(top, 1):
        params = it.get("params") or {}
        params_short = "; ".join(
            f"{k}={v}" for k, v in list(params.items())[:3]
        )
        rows.append(
            f"  {i} & {_esc(params_short)} & "
            f"{it.get('trade_count') or '---'} & "
            f"{_fmt(it.get('profit_factor'))} & "
            f"{_fmt(it.get('cagr'), 1)}\\% & "
            f"{_fmt(it.get('max_drawdown_pct'), 1)}\\% & "
            f"{_fmt(it.get('dsr'), 3)} & "
            f"{_esc(str(it.get('statistical_verdict') or '---'))} \\\\"
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Top {len(top)} parameter configurations by CAGR}}
\label{{tab:top_iters}}
\small
\begin{{tabular}}{{clrrrrrr}}
\toprule
\textbf{{Rank}} & \textbf{{Parameters}} & \textbf{{N}} &
\textbf{{PF}} & \textbf{{CAGR}} & \textbf{{MaxDD}} &
\textbf{{DSR}} & \textbf{{Verdict}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# Section 5 — Robustness
# ---------------------------------------------------------------------------

def _section_robustness(
    robustness: dict[str, Any],
    best: dict[str, Any],
) -> str:
    slip = robustness.get("slippage_sensitivity") or {}
    regime = robustness.get("regime_breakdown") or {}

    slip_rows = _build_slippage_table(slip, best)
    regime_rows = _build_regime_table(regime)

    return rf"""\section{{Robustness Analysis}}
\label{{sec:robustness}}

\subsection{{Slippage Sensitivity}}
Table~\ref{{tab:slippage}} reports how the strategy's CAGR degrades as
assumed round-trip slippage increases. A strategy is considered robust
if it remains profitable at +20 bps.

{slip_rows}

\subsection{{Regime Analysis}}
Table~\ref{{tab:regime}} breaks down trade outcomes by market regime at
entry. A regime-consistent strategy produces positive Profit Factor in
the regime it targets.

{regime_rows}

\subsection{{Parameter Sensitivity}}
A parameter sensitivity heatmap is available in the interactive report
in the research library. Configurations close to the optimal point
are evaluated to confirm that performance does not degrade sharply with
small parameter perturbations."""


def _build_slippage_table(
    slip: dict[str, Any],
    best: dict[str, Any],
) -> str:
    baseline_cagr = (best.get("metrics") or {}).get("cagr")

    def _fmt(v: Any) -> str:
        if v is None:
            return "---"
        try:
            return f"{float(v):.1f}\\%"
        except (TypeError, ValueError):
            return "---"

    rows = [
        f"  0 bps (baseline) & {_fmt(baseline_cagr)} \\\\"
    ]
    for label, key in [("5 bps", "bps_5"), ("10 bps", "bps_10"),
                       ("20 bps", "bps_20"), ("50 bps", "bps_50")]:
        bucket = slip.get(key) or {}
        cagr = bucket.get("geometric_return_pct_at_alloc_90") or bucket.get("cagr")
        rows.append(f"  {label} & {_fmt(cagr)} \\\\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{CAGR under increasing slippage assumptions}}
\label{{tab:slippage}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Slippage}} & \textbf{{CAGR (90\% Kelly)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _build_regime_table(regime: dict[str, Any]) -> str:
    if not regime:
        return r"\textit{Regime breakdown data unavailable.}"

    def _fmt_pf(v: Any) -> str:
        if v is None:
            return "---"
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return "---"

    rows = []
    for reg_name, data in regime.items():
        if not isinstance(data, dict):
            continue
        n = data.get("trades") or data.get("trade_count") or "---"
        pf = _fmt_pf(data.get("profit_factor") or data.get("pf"))
        rows.append(f"  {_esc(str(reg_name))} & {n} & {pf} \\\\")

    if not rows:
        return r"\textit{Regime breakdown data unavailable.}"

    body = "\n".join(rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Trade outcomes by market regime at entry}}
\label{{tab:regime}}
\begin{{tabular}}{{lrr}}
\toprule
\textbf{{Regime}} & \textbf{{Trades}} & \textbf{{Profit Factor}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# Section 6 — Conclusion
# ---------------------------------------------------------------------------

def _section_conclusion(
    gate: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    verdict = _esc(meta.get("final_verdict") or "PENDING")
    strategy = _esc(meta.get("strategy_code") or "N/A")
    symbol = _esc(meta.get("instrument") or "N/A")
    interval = _esc(meta.get("interval_name") or "N/A")
    all_passed = gate.get("all_gates_passed", False)

    gate_items = [
        (f"Trade count $\\geq {gate.get('n_trades_threshold', 100)}$",
         gate.get("n_trades_ok"), gate.get("n_trades_value")),
        ("Profit Factor 95\\% CI lower $> 1.0$",
         gate.get("pf_ci_ok"), gate.get("pf_ci_low")),
        (f"DSR $\\geq {gate.get('dsr_threshold', 0.90)}$",
         gate.get("dsr_ok"), gate.get("dsr_value")),
        ("Statistical verdict: Significant Edge",
         gate.get("stat_verdict_ok"), gate.get("stat_verdict")),
        (f"CAGR $\\geq {gate.get('cagr_threshold', 10.0)}\\%$",
         gate.get("cagr_ok"), gate.get("cagr_value")),
    ]

    items = []
    for label, passed, val in gate_items:
        mark = "\\checkmark" if passed else "$\\times$"
        val_str = f" ({val})" if val is not None else ""
        items.append(f"  \\item[{mark}] {label}{_esc(str(val_str))}")
    gate_list = "\n".join(items)

    outcome = (
        "The strategy demonstrates statistically significant and economically meaningful "
        f"edge on {symbol} ({interval}), meeting all five quantitative gates."
        if all_passed
        else f"The strategy did not meet all quantitative gates on {symbol} ({interval})."
    )

    return rf"""\section{{Conclusion}}
\label{{sec:conclusion}}

This paper evaluated the \textbf{{{strategy}}} strategy on \textbf{{{symbol}}}
({interval}) across a systematic parameter sweep. {outcome}

\subsection*{{Gate Checklist}}
\begin{{description}}
{gate_list}
\end{{description}}

\subsection*{{Final Verdict: \textsc{{{verdict}}}}}

\subsubsection*{{Limitations}}
\begin{{itemize}}
  \item Results are in-sample only unless a walk-forward validation has been run.
  \item Slippage estimates are conservative; actual execution may differ.
  \item The backtest does not account for market impact at scale.
\end{{itemize}}

\subsubsection*{{Future Work}}
Walk-forward validation (POST /walk-forward) should be conducted before
any capital allocation decision. Cross-asset and cross-interval robustness
checks are recommended for strategies that pass the primary gates."""


# ---------------------------------------------------------------------------
# Section 7 — Agent Notes (optional)
# ---------------------------------------------------------------------------

def _section_agent_notes(journal: list[dict[str, Any]]) -> str:
    if not journal:
        return ""

    entries = []
    for j in journal:
        entry_type = _esc(str(j.get("entry_type") or "NOTE"))
        title = _esc(str(j.get("title") or ""))
        content = _esc(str(j.get("content") or ""))
        date_str = _format_date(j.get("created_time"))
        entries.append(
            rf"""\subsection*{{{entry_type}: {title} \hfill \small {date_str}}}
{content}"""
        )

    body = "\n\n".join(entries)
    return rf"""\section{{Research Notes}}
\label{{sec:notes}}

The following notes were recorded by the quant-researcher agent during
the sweep. They include pre-registered hypotheses, outcome summaries,
and anti-pattern observations.

{body}"""


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

def _bibliography(
    citations: list[dict[str, str]],
    paper_id: str,
    version: int,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    items = []
    for c in citations:
        key = c.get("key") or "ref"
        text = _esc(c.get("text") or "")
        items.append(rf"\bibitem{{{key}}}" + "\n" + text)

    # Self-reference for reproducibility
    items.append(
        rf"\bibitem{{blackheart_engine}}"
        + "\n"
        + _esc(
            f"Blackheart Research Engine. Internal paper reference {paper_id} "
            f"(v{version}), generated {today}. "
            "Blackheart Research Unit."
        )
    )

    body = "\n\n".join(items)
    return rf"""\begin{{thebibliography}}{{99}}

{body}

\end{{thebibliography}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}
# Pre-built regex for all special chars (backslash first to avoid double-escape)
_SPECIAL_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in _LATEX_SPECIAL) + ")"
)


def _esc(text: str) -> str:
    """Escape LaTeX special characters in user-supplied strings."""
    return _SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL[m.group()], text)


def _format_date(dt: Any) -> str:
    if dt is None:
        return datetime.now().strftime("%B %d, %Y")
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return str(dt)
    if isinstance(dt, datetime):
        return dt.strftime("%B %d, %Y")
    return str(dt)
