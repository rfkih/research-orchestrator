"""Drift detection between operator contract files.

Background: 2026-05-19 audit found the quant-researcher agent prompt still
listed "+20bps slippage still positive" as an active V11 gate in Hard rule
6, even though V60 retired that gate (`services/tick.py` lines 648-655).
A reviewer enforcing a retired gate produces false REJECTs — three sessions
worth of "exhaustion" verdicts were partly downstream of that drift
(memory: ``project_retired_cost_gate_methodology_bug``).

These tests read the prompt + playbook directly and fail loud the next time
a retired constant, stale terminal count, or mis-counted hard-rule list
sneaks back in. No orchestrator imports, no DB, no fixtures beyond file
reads — fast enough to run on every push.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_PROMPT = REPO_ROOT / ".claude" / "agents" / "quant-researcher.md"
WORKFLOW_PLAYBOOK = (
    REPO_ROOT
    / "blackheart-trading-engine"
    / "research"
    / "agent-playbooks"
    / "quant-researcher-workflow.md"
)

# Matches the retired-gate framings: "+20bps slippage still positive" /
# "+20bps slippage net positive" / "+20bps slippage > 0". The audit-only
# mentions ("slippage_haircut_pnl is still logged for audit", "+20bps
# slippage net check was retired") do NOT match this pattern, because their
# trailing tokens are "logged" / "check was retired" — not the gate-shape
# words above.
_RETIRED_GATE_RE = re.compile(
    r"\+20\s*bps\s+slippage\s+(still\s+positive|net\s+positive|>\s*0)",
    re.IGNORECASE,
)

# Phrases that, when within 200 chars of a +20bps match, make the match
# acceptable. They explicitly frame the gate as retired / audit-only /
# forbidden-to-reintroduce, so they're contract-affirming not drift.
_RETIRED_GATE_AFFIRMERS = (
    "retired",
    "audit-only",
    "audit only",
    "no longer gates",
    "re-introducing",
    "never enforce",
    "informational",
    "do not",
    "must not",
)


@pytest.fixture(scope="module")
def agent_prompt_text() -> str:
    return AGENT_PROMPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def playbook_text() -> str:
    return WORKFLOW_PLAYBOOK.read_text(encoding="utf-8")


def _unaffirmed_retired_gate_mentions(text: str) -> list[tuple[str, str]]:
    suspect: list[tuple[str, str]] = []
    for match in _RETIRED_GATE_RE.finditer(text):
        ctx_start = max(0, match.start() - 200)
        ctx_end = min(len(text), match.end() + 200)
        ctx = text[ctx_start:ctx_end].lower()
        if any(affirmer in ctx for affirmer in _RETIRED_GATE_AFFIRMERS):
            continue
        suspect.append((match.group(0), text[ctx_start:ctx_end]))
    return suspect


def test_agent_prompt_does_not_re_enforce_retired_20bps_gate(
    agent_prompt_text: str,
) -> None:
    suspect = _unaffirmed_retired_gate_mentions(agent_prompt_text)
    assert not suspect, (
        "Agent prompt re-introduces the retired +20bps slippage gate "
        f"as an active gate. Found {len(suspect)} unframed mention(s).\n"
        f"First context (200 chars each side):\n"
        f"{suspect[0][1] if suspect else ''}\n\n"
        "Fix: either remove the mention or surround it with explicit "
        "'retired' / 'audit-only' / 'do not re-introduce' framing. See "
        "memory project_retired_cost_gate_methodology_bug for the "
        "downstream impact when this drift was last live."
    )


def test_playbook_does_not_re_enforce_retired_20bps_gate(
    playbook_text: str,
) -> None:
    suspect = _unaffirmed_retired_gate_mentions(playbook_text)
    assert not suspect, (
        "Playbook re-introduces the retired +20bps slippage gate as an "
        f"active gate. Found {len(suspect)} unframed mention(s).\n"
        f"First context (200 chars each side):\n"
        f"{suspect[0][1] if suspect else ''}"
    )


def test_agent_prompt_cites_v60_economic_gate_field_name(
    agent_prompt_text: str,
) -> None:
    """The active economic gate is
    ``annualized_geometric_return_pct_at_alloc_90 >= 10``. The agent prompt
    must cite this constant verbatim at least once so a future drift cannot
    quietly substitute a different metric while keeping the "10%/yr" prose.
    """
    assert "annualized_geometric_return_pct_at_alloc_90" in agent_prompt_text, (
        "Agent prompt no longer cites the V60 economic gate field name "
        "verbatim. This is the iteration_log field the orchestrator reads "
        "to decide PASS/FAIL on the 10%/yr threshold; if it's not named in "
        "the prompt, a future edit could silently substitute a different "
        "metric while keeping the surface '10%/yr' framing."
    )


def test_agent_prompt_terminal_count_header_says_six(
    agent_prompt_text: str,
) -> None:
    """As of 2026-05-19 (operator-escalation addition) the terminals are
    GOAL_HIT, WALL_CLOCK_CAP, INFRA_HARD_FAIL, HARD_RULE_VIOLATION,
    ARCHETYPE_EXHAUSTION, OPERATOR_ESCALATION. The "Terminal conditions"
    header must say "ONLY six ways" — if a future edit adds a 7th
    terminal without updating the header, this fails loud.
    """
    match = re.search(
        r"Terminal conditions \(the ONLY (\w+) ways the loop ends\)",
        agent_prompt_text,
    )
    assert match is not None, (
        "Could not find the 'Terminal conditions (the ONLY N ways the loop "
        "ends)' header in the agent prompt. Either the header was renamed "
        "or the prompt structure drifted."
    )
    assert match.group(1) == "six", (
        f"Terminal conditions header says ONLY {match.group(1)} ways; "
        f"expected six. Update the header AND the terminals table AND "
        f"the 'no Nth exit' paragraph together, or update none of them."
    )


def test_agent_prompt_terminal_table_has_six_rows(
    agent_prompt_text: str,
) -> None:
    """The 'Terminal conditions' table must have exactly six data rows
    (GOAL_HIT, WALL_CLOCK_CAP, INFRA_HARD_FAIL, HARD_RULE_VIOLATION,
    ARCHETYPE_EXHAUSTION, OPERATOR_ESCALATION). Complements
    ``test_agent_prompt_terminal_count_header_says_six`` — that test
    pins the *header word*, this one pins the actual *table content*.
    Drift signal: a 7th row added silently without updating the header
    would pass the header-only test but fail this one.
    """
    section_match = re.search(
        r"\*\*Terminal conditions \(the ONLY \w+ ways the loop ends\):\*\*"
        r"(.+?)(?=\n\*\*|\n## )",
        agent_prompt_text,
        re.DOTALL,
    )
    assert section_match is not None, (
        "Could not locate the Terminal conditions table section."
    )
    section = section_match.group(1)
    row_headers = re.findall(
        r"^\|\s+\*\*[A-Z_]+\*\*\s+\|", section, re.MULTILINE
    )
    assert len(row_headers) == 6, (
        f"Terminal conditions table has {len(row_headers)} data rows; "
        f"expected 6. Rows found: {row_headers}. Update the header word "
        f"AND the table together — drift between them is the failure "
        f"mode this test is designed to catch."
    )


def test_lockout_title_prefix_does_not_collide_with_terminal_prefixes(
    agent_prompt_text: str,
) -> None:
    """Two terminal-fire rows write distinct title prefixes:
    ``ARCHETYPE_EXHAUSTION_<date>`` (24h lockout window) and
    ``OPERATOR_ESCALATION_<date>`` (12h lockout window). The resume-
    protocol lockout writes ``RESUME_LOCKOUT_<date>``. The lockout
    prefix MUST NOT start with EITHER terminal prefix — otherwise step
    1a's scan would match each lockout row and perpetuate the lockout
    indefinitely.

    Background: the original 2026-05-19 ARCHETYPE_EXHAUSTION
    implementation used ``ARCHETYPE_EXHAUSTION_PENDING_<date>`` for
    the lockout title, which shared the prefix and caused a perpetual-
    lockout bug — every session's lockout-exit became the new
    ``last_run_summary`` row that re-triggered the next session's
    lockout check. Fixed by renaming the lockout prefix to the
    disjoint ``RESUME_LOCKOUT_<...>`` template. The OPERATOR_ESCALATION
    terminal added later (same date) uses the same lockout mechanism
    and inherits the same invariant.
    """
    terminal_templates = (
        "ARCHETYPE_EXHAUSTION_<",
        "OPERATOR_ESCALATION_<",
    )
    lockout_template = "RESUME_LOCKOUT_<"

    for template in terminal_templates:
        assert template in agent_prompt_text, (
            f"Agent prompt no longer mentions the terminal-fire title "
            f"template {template!r}. Has the terminal been renamed or "
            f"removed?"
        )
    assert lockout_template in agent_prompt_text, (
        f"Agent prompt no longer mentions the lockout title template "
        f"{lockout_template!r}. Has the resume-protocol lockout been "
        f"renamed or removed? The lockout prefix must remain disjoint "
        f"from every terminal prefix or the perpetual-lockout bug "
        f"returns."
    )
    for template in terminal_templates:
        terminal_prefix = template.rstrip("<")
        assert not lockout_template.startswith(terminal_prefix), (
            f"Lockout title template {lockout_template!r} starts with "
            f"the terminal prefix {terminal_prefix!r}. Step 1a's "
            f"resume scan matches by prefix; collision causes the "
            f"perpetual-lockout bug. Use a fully-distinct prefix."
        )


def test_agent_prompt_hard_rule_count_matches_terminal_reference(
    agent_prompt_text: str,
) -> None:
    """The HARD_RULE_VIOLATION terminal references "violating one of the
    13 hard rules below". The Hard constraints section must contain exactly
    13 numbered rules. Drift signal: a 14th rule added without updating the
    cross-reference produces a contract inconsistency the auto-checklist
    won't catch.
    """
    section_match = re.search(
        r"## Hard constraints[^\n]*\n(.+?)(?=\n## )",
        agent_prompt_text,
        re.DOTALL,
    )
    assert section_match is not None, (
        "Could not find the '## Hard constraints' section in the agent "
        "prompt."
    )
    section = section_match.group(1)
    rule_headers = re.findall(r"^\d+\.\s+\*\*", section, re.MULTILINE)
    terminal_ref = re.search(
        r"violating one of the (\d+) hard rules below",
        agent_prompt_text,
    )
    assert terminal_ref is not None, (
        "HARD_RULE_VIOLATION terminal row does not cite a hard-rule count. "
        "Restore the 'violating one of the N hard rules below' phrasing."
    )
    declared_count = int(terminal_ref.group(1))
    assert len(rule_headers) == declared_count, (
        f"Hard constraints section has {len(rule_headers)} numbered rules, "
        f"but the HARD_RULE_VIOLATION terminal references {declared_count}. "
        f"Update both together — they must agree."
    )
