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


def test_agent_prompt_terminal_count_header_says_seven(
    agent_prompt_text: str,
) -> None:
    """As of 2026-05-20 (Path C async specialist review) the terminals are
    GOAL_HIT, SPECIALIST_REVIEW_PENDING, WALL_CLOCK_CAP, INFRA_HARD_FAIL,
    HARD_RULE_VIOLATION, ARCHETYPE_EXHAUSTION, OPERATOR_ESCALATION. The
    "Terminal conditions" header must say "ONLY seven ways" — if a future
    edit adds an 8th terminal (or removes one) without updating the header,
    this fails loud.
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
    assert match.group(1) == "seven", (
        f"Terminal conditions header says ONLY {match.group(1)} ways; "
        f"expected seven. Update the header AND the terminals table AND "
        f"the 'no Nth exit' paragraph together, or update none of them."
    )


def test_agent_prompt_terminal_table_has_seven_rows(
    agent_prompt_text: str,
) -> None:
    """The 'Terminal conditions' table must have exactly seven data rows
    (GOAL_HIT, SPECIALIST_REVIEW_PENDING, WALL_CLOCK_CAP, INFRA_HARD_FAIL,
    HARD_RULE_VIOLATION, ARCHETYPE_EXHAUSTION, OPERATOR_ESCALATION).
    Complements ``test_agent_prompt_terminal_count_header_says_seven`` —
    that test pins the *header word*, this one pins the actual *table
    content*. Drift signal: an 8th row added silently without updating
    the header would pass the header-only test but fail this one.
    """
    # Accept either the legacy bold-paragraph header (**Terminal conditions
    # (the ONLY N ways the loop ends):**) or the post-2026-05-20 slim-prompt
    # h2 header (## Terminal conditions (the ONLY N ways the loop ends)).
    # Both shapes were canonical at different points; the table content is
    # what this test pins, not the header style.
    section_match = re.search(
        r"(?:\*\*|##\s+)Terminal conditions \(the ONLY \w+ ways the loop ends\)"
        r"(?::\*\*|\s*)\n(.+?)(?=\n\*\*|\n## )",
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
    assert len(row_headers) == 7, (
        f"Terminal conditions table has {len(row_headers)} data rows; "
        f"expected 7. Rows found: {row_headers}. Update the header word "
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


# ── 2026-05-20 slim-prompt drift tests ─────────────────────────────────
#
# After the 2026-05-20 refactor, the researcher prompt is the "immutable
# lens" (mission, gates, terminals, hard rules) and the playbook is the
# "procedure" (HTTP recipes, step bodies). These three tests pin the
# partitioning so a future edit doesn't either (a) silently drop a
# load-bearing constant from the prompt or (b) silently put recipe-detail
# back into the prompt.

# Constants the prompt MUST cite verbatim. Drift on any of these is a
# contract failure — the gate or terminal that depends on it would fire
# wrong if the agent re-derived the value from prose.
_LOAD_BEARING_CONSTANTS = (
    "annualized_geometric_return_pct_at_alloc_90",  # V60 economic gate field
    "MIN_TRADES_FOR_SIG",                           # V11 stat-rigor field
    "BTCUSDT and ETHUSDT only",                     # universe constraint
    "5m / 15m / 1h / 4h",                           # interval constraint
    "LSR, VCB, VBO",                                # protected book
    "7.5",                                          # wall-clock cap (hours)
    "10%",                                          # profitability bar prose
    "PF lower 95% CI > 1.0",                        # V11 PF gate
    "DSR ≥ 0.95",                                   # V11 DSR threshold
    "research-mode",                                # never auto-promote
    "/specialist-review/request",                   # Path C trigger endpoint
    "PATH_C_RESUMING_",                             # crash-safety marker
)

# Terminal names that MUST appear in the prompt. The resume protocol
# branches on title prefixes; if a terminal name disappears from the
# prompt, the agent has no instruction to fire it even though the
# orchestrator may still expect it.
_TERMINAL_NAMES = (
    "GOAL_HIT",
    "SPECIALIST_REVIEW_PENDING",
    "WALL_CLOCK_CAP",
    "INFRA_HARD_FAIL",
    "HARD_RULE_VIOLATION",
    "ARCHETYPE_EXHAUSTION",
    "OPERATOR_ESCALATION",
)


def test_agent_prompt_cites_all_load_bearing_constants(
    agent_prompt_text: str,
) -> None:
    """Every load-bearing constant must appear verbatim in the prompt.

    Drift signal: if any of these silently disappears, the agent might
    re-derive the constant from prose ("10%/yr" → "10 percent per year")
    and the gate that consumes the verbatim field name (e.g. V60's
    `annualized_geometric_return_pct_at_alloc_90 >= 10`) won't trigger
    correctly.
    """
    missing = [c for c in _LOAD_BEARING_CONSTANTS if c not in agent_prompt_text]
    assert not missing, (
        "Slim prompt no longer cites load-bearing constants verbatim: "
        f"{missing}. Either restore them inline or update this pin list "
        "with a clear rationale (e.g. the constant moved into the "
        "playbook as part of a deliberate contract change)."
    )


def test_agent_prompt_names_all_seven_terminals(
    agent_prompt_text: str,
) -> None:
    """All 7 terminal names must appear in the prompt body — the
    resume protocol depends on each one being declared."""
    missing = [t for t in _TERMINAL_NAMES if t not in agent_prompt_text]
    assert not missing, (
        f"Slim prompt no longer names terminal(s) {missing}. The resume "
        "protocol matches by title prefix; missing names mean the agent "
        "cannot fire those terminals."
    )


# Heuristic regexes that indicate the prompt has absorbed procedural
# content that belongs in the playbook. Each pattern is conservative
# (single literal anchors) so the test doesn't false-positive on the
# loop-outline pseudocode block.
_RECIPE_LEAK_PATTERNS = (
    # Inline orch.sh invocations (recipe-shaped):
    re.compile(r"scripts/orch\.sh\s+(?:GET|POST)", re.IGNORECASE),
    # uuidgen-style idempotency keys (the deterministic-key formulae
    # belong in the playbook §Idempotency keys reference table):
    re.compile(r"\$\(uuidgen\)"),
    # python heredoc bodies (recipe-shaped):
    re.compile(r"python\s+-\s+<<'PY'", re.IGNORECASE),
    # jq pipelines:
    re.compile(r"\|\s*jq\s+"),
)


def test_agent_prompt_does_not_leak_recipe_content(
    agent_prompt_text: str,
) -> None:
    """The slim prompt is the invariant lens; HTTP recipes / shell
    commands / jq pipelines belong in the playbook. Any leak here means
    the partitioning is degrading — the prompt is silently re-absorbing
    procedural detail and we'll be back at 500+ lines in a few sessions.
    """
    leaks: list[tuple[str, str]] = []
    for pattern in _RECIPE_LEAK_PATTERNS:
        for match in pattern.finditer(agent_prompt_text):
            ctx_start = max(0, match.start() - 60)
            ctx_end = min(len(agent_prompt_text), match.end() + 60)
            leaks.append(
                (pattern.pattern, agent_prompt_text[ctx_start:ctx_end])
            )
    assert not leaks, (
        f"Slim prompt absorbed recipe content. {len(leaks)} leak(s) "
        f"found. First context:\n{leaks[0][1] if leaks else ''}\n\n"
        "Move the recipe to research/agent-playbooks/quant-researcher-"
        "workflow.md and keep the prompt outline-only."
    )


def test_playbook_has_section_for_every_loop_step_named_in_prompt(
    agent_prompt_text: str, playbook_text: str
) -> None:
    """Every loop step referenced in the prompt's outline must have a
    matching `### Step N` (or equivalent) heading in the playbook. The
    prompt is intentionally outline-only; if it names step 7.6 (Path C
    async checkpoint) without the playbook describing it, the agent
    has nothing to execute against.
    """
    # Step labels the prompt's outline references. Anchored on the
    # loop-pseudocode block — these are the steps the agent has to
    # execute. Format: tuple of (prompt_label, playbook_heading_hint).
    expected_steps = (
        ("step 1", "### 1."),
        ("step 2", "### 2."),
        ("step 3", "### 3."),
        ("step 4", "### 4."),
        ("step 7", "### 5"),     # playbook merges steps 5 + 6 + 7
        ("step 9", "### 7.45"),  # paired-delta gate at 9.0
        ("step 9d", "### 7.6"),  # Path C async checkpoint
        ("step 10", "### 7.5"),  # walk-forward request (step 7.5 in playbook covers this)
    )
    missing: list[tuple[str, str]] = []
    for prompt_label, playbook_hint in expected_steps:
        if prompt_label not in agent_prompt_text:
            missing.append((prompt_label, "absent from prompt"))
            continue
        if playbook_hint not in playbook_text:
            missing.append((prompt_label, f"playbook hint {playbook_hint!r} absent"))
    assert not missing, (
        f"Prompt-playbook step partitioning broken: {missing}. Either "
        "restore the missing playbook heading or update the prompt's "
        "loop outline to not reference it."
    )
