#!/usr/bin/env pwsh
# Headless quant-researcher tick. Invoked by Windows Task Scheduler every 6h.
# Runs `claude -p` from the trading project root so the quant-researcher
# subagent (.claude/agents/quant-researcher.md) is discoverable.

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\MyFiles\blackheart\blackheart'
$LogDir      = 'C:\MyFiles\blackheart\research-orchestrator\logs'
$ClaudeBin   = "$env:APPDATA\npm\claude.cmd"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stamp   = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$LogFile = Join-Path $LogDir "tick-$Stamp.log"

$Prompt = @'
Spawn the quant-researcher subagent and run ONE autonomous research-loop iteration against http://127.0.0.1:8082 (X-Orch-Token: dev-sentinel-not-for-prod, X-Agent-Name: cron-researcher).

Brief for the subagent:

1. GET /agent/state and check pending+running queue depth via /queue?status=PENDING&limit=1 + /queue?status=RUNNING&limit=1.
2. If pending+running >= 1: POST /tick (Idempotency-Key: tick-$(date +%s)-cron). Up to ~30 min synchronous - that is expected.
3. If pending+running == 0: pick ONE fresh hypothesis. Allowed instruments: BTCUSDT or ETHUSDT (Phase 3 shipped — ETH backfilled across all intervals 2023-11-29 → present). Interval in {5m,15m,1h,4h}, never LSR/VCB/VBO param changes, archetype+instrument combination not yet exhausted on /leaderboard. Prefer ETHUSDT for archetypes already exhausted on BTC (e.g. revisit a NO_EDGE BTC hypothesis on ETH before discarding the archetype entirely). Pre-register it as a HYPOTHESIS journal entry via POST /journal, then POST /queue with the matching sweep_config (>=3 dimensions, iter_budget<=5).
4. If a queue row is PARKED with final_verdict=SIGNIFICANT_EDGE and walk_forward_run has no row for it: POST /walk-forward {queue_id,...,n_folds:6}.
5. Append a short ACTIVE RUN_SUMMARY (<=200 words) to research_journal noting what you did and why.

Hard constraints from C:/MyFiles/blackheart/blackheart/.claude/agents/quant-researcher.md still apply. Do NOT promote, do NOT touch account_strategy, do NOT restart the JVM, do NOT deploy spec strategies. If you hit account_strategy_missing for TPR, skip TPR - operator hasn't seeded it yet.

Report back ONE LINE: action taken | queue depth before/after | any anomaly.
'@

Set-Location $ProjectRoot

"=== research-tick-cron $Stamp ===" | Out-File -FilePath $LogFile -Encoding utf8

try {
    $Prompt | & $ClaudeBin -p --dangerously-skip-permissions 2>&1 |
        Tee-Object -FilePath $LogFile -Append
    "=== exit code: $LASTEXITCODE ===" | Out-File -FilePath $LogFile -Append -Encoding utf8
} catch {
    "ERROR: $_" | Out-File -FilePath $LogFile -Append -Encoding utf8
    exit 1
}

# Prune logs older than 30 days
Get-ChildItem $LogDir -Filter 'tick-*.log' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force
