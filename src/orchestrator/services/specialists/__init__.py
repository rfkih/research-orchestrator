"""Specialist services — Path A (Max-plan-only, 2026-05-19).

The dream-team architecture is here, but the LLM specialists run as
Claude Code Agent-tool sub-agents (their definitions live in
``.claude/agents/quant-*.md``) — NOT as orchestrator-side API clients.
Every token bills against the operator's Max plan; no separate billing.

This package's scope:

  * ``portfolio_math.py`` — Spearman correlations, HRP / mean-variance /
    equal-weight optimisers. Pure NumPy. No LLM.
  * ``prescreen.py`` — mechanical skeptic-style pattern checks over
    journal + iteration history. Returns metrics + threshold flags.
    The researcher reads these to decide whether to invoke the LLM
    skeptic agent.
  * ``agent_log.py`` — thin helper around ``repo/agent_decisions``;
    the researcher posts here after collecting a verdict from a
    spawned specialist sub-agent.

Out of scope (no API client, no Anthropic SDK dep):
  * Server-side LLM invocations. Specialists live in agent .md files.
"""

from .agent_log import log_agent_decision
from .portfolio_math import (
    equal_weight,
    hrp_weights,
    mean_variance_weights,
    spearman_corr_matrix,
)
from .ml_prescreen import ml_prescreen
from .prescreen import skeptic_prescreen

__all__ = [
    "equal_weight",
    "hrp_weights",
    "log_agent_decision",
    "mean_variance_weights",
    "ml_prescreen",
    "skeptic_prescreen",
    "spearman_corr_matrix",
]
