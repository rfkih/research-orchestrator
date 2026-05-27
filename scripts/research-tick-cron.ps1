#!/usr/bin/env pwsh
# Headless quant-researcher tick. Invoked by Windows Task Scheduler every 2h.
# Runs `claude -p` from the project root so the quant-researcher subagent
# (C:\Project\.claude\agents\quant-researcher.md) is discoverable.
#
# Persistence model (2026-05-24): user wants research to keep going until
# GOAL_HIT or hard-failure. Task Scheduler enforces no-overlap (mutex) — if
# a previous tick is still running when the next 2h trigger fires, the new
# instance is ignored. This means WALL_CLOCK_CAP exits (sessions that ran the
# full 8h) get their SESSION_CHECKPOINT picked up by the NEXT scheduler tick
# after they finish, not by an overlapping session. Idle gap between runs is
# ≤2h (down from the original 6h design).

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Project'
$OrchRoot    = 'C:\Project\blackheart-research-orchestrator'
$LogDir      = Join-Path $OrchRoot 'logs'
$EnvFile     = Join-Path $OrchRoot '.env'

# Resolve the Claude CLI binary. Get-Command works when PATH is populated;
# fall back to the WinGet shim path verified 2026-05-24.
$ClaudeBin = (Get-Command claude -ErrorAction SilentlyContinue).Source
if (-not $ClaudeBin) {
    $ClaudeBin = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\claude.exe"
}

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Load shared-secret + Telegram + ORCH base from .env so this script survives
# token rotations.
$OrchToken = $null
$TelegramBot = $null
$TelegramChat = $null
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^ORCH_AUTH_TOKEN=(.+)$')        { $OrchToken    = $Matches[1].Trim() }
        if ($line -match '^ORCH_TELEGRAM_BOT_TOKEN=(.+)$') { $TelegramBot  = $Matches[1].Trim() }
        if ($line -match '^ORCH_TELEGRAM_CHAT_ID=(.+)$')   { $TelegramChat = $Matches[1].Trim() }
    }
}

# Telegram is optional — if ORCH_TELEGRAM_BOT_TOKEN / ORCH_TELEGRAM_CHAT_ID
# are not in .env the Send-Telegram helper is a no-op (both guards check for
# non-empty values before sending). Never hardcode fallback credentials here.

function Send-Telegram {
    param([string]$Text)
    if (-not $TelegramBot -or -not $TelegramChat) { return }
    try {
        $body = @{ chat_id = $TelegramChat; text = $Text } | ConvertTo-Json -Compress
        Invoke-WebRequest -Method Post `
            -Uri "https://api.telegram.org/bot$TelegramBot/sendMessage" `
            -ContentType 'application/json' `
            -Body $body -TimeoutSec 5 -UseBasicParsing | Out-Null
    } catch {
        # Telegram failure is non-fatal — research must continue even if we
        # can't tell the operator about it. Log to file only.
        "telegram-send-failed: $_" | Out-File -FilePath $LogFile -Append -Encoding utf8
    }
}

if (-not $OrchToken) {
    "FATAL: ORCH_AUTH_TOKEN not found in $EnvFile" | Out-File -FilePath (Join-Path $LogDir 'tick-FATAL.log') -Append -Encoding utf8
    Send-Telegram "[research-tick] FATAL: ORCH_AUTH_TOKEN missing from $EnvFile"
    exit 2
}

$Stamp   = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$LogFile = Join-Path $LogDir "tick-$Stamp.log"
$StartTime = Get-Date

$Prompt = @'
Spawn the quant-researcher subagent and run ONE autonomous research-loop iteration against http://127.0.0.1:8082 (X-Orch-Token: __ORCH_TOKEN__, X-Agent-Name: cron-researcher).

Brief for the subagent:

1. GET /agent/state and check pending+running queue depth via /queue?status=PENDING&limit=1 + /queue?status=RUNNING&limit=1.
2. If pending+running >= 1: POST /tick (Idempotency-Key: tick-$(date +%s)-cron). Up to ~30 min synchronous - that is expected. Repeat /tick until queue depth = 0 OR your wall-clock check says CHECKPOINT_NOW.
3. If pending+running == 0: pick ONE fresh hypothesis. Allowed instruments: BTCUSDT or ETHUSDT (Phase 3 shipped — ETH backfilled across all intervals 2023-11-29 → present). Interval in {5m,15m,1h,4h}, never LSR/VCB/VBO param changes, archetype+instrument combination not yet exhausted on /leaderboard. Prefer ETHUSDT for archetypes already exhausted on BTC (e.g. revisit a NO_EDGE BTC hypothesis on ETH before discarding the archetype entirely). Pre-register it as a HYPOTHESIS journal entry via POST /journal, then POST /queue with the matching sweep_config (>=3 dimensions, iter_budget<=5).
4. If a queue row is PARKED with final_verdict=SIGNIFICANT_EDGE and walk_forward_run has no row for it: POST /walk-forward {queue_id,...,n_folds:6}.
5. Append a short ACTIVE RUN_SUMMARY (<=200 words) to research_journal noting what you did and why.

PERSISTENCE: the operator has flagged that you should run as long as possible inside the 8-hour wall-clock cap. Do NOT exit early just because step 2 returned ITERATE or step 1 found the queue empty. Loop: handle current queue state, then check again, then handle again, until either GOAL_HIT or wall-clock CHECKPOINT_NOW.

Hard constraints from C:\Project\.claude\agents\quant-researcher.md still apply. Do NOT promote, do NOT touch account_strategy, do NOT restart the JVM, do NOT deploy spec strategies. If you hit account_strategy_missing for TPR, skip TPR - operator hasn't seeded it yet.

Report back ONE LINE: action taken | queue depth before/after | terminal reason | any anomaly.
'@

$Prompt = $Prompt.Replace('__ORCH_TOKEN__', $OrchToken)

Set-Location $ProjectRoot

"=== research-tick-cron $Stamp ==="              | Out-File -FilePath $LogFile -Encoding utf8
"PROJECT_ROOT=$ProjectRoot"                       | Out-File -FilePath $LogFile -Append -Encoding utf8
"ORCH_TOKEN=...$($OrchToken.Substring([Math]::Max(0, $OrchToken.Length - 8)))" | Out-File -FilePath $LogFile -Append -Encoding utf8
"CLAUDE_BIN=$ClaudeBin"                           | Out-File -FilePath $LogFile -Append -Encoding utf8

Send-Telegram "[research-tick] Starting session $Stamp"

$exitCode = 0
try {
    $Prompt | & $ClaudeBin -p --dangerously-skip-permissions 2>&1 |
        Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE
    "=== exit code: $exitCode ===" | Out-File -FilePath $LogFile -Append -Encoding utf8
} catch {
    $exitCode = 1
    "ERROR: $_" | Out-File -FilePath $LogFile -Append -Encoding utf8
}

$Duration = (New-TimeSpan -Start $StartTime -End (Get-Date))
$DurationStr = "{0:hh\:mm\:ss}" -f $Duration

# Try to capture the agent's one-line summary from the tail of the log.
$Summary = ""
if (Test-Path $LogFile) {
    $tail = Get-Content $LogFile -Tail 50 -ErrorAction SilentlyContinue
    if ($tail) {
        $summaryLine = $tail | Where-Object {
            $_ -match '(queue depth|terminal reason|action taken|TICK|VERDICT|exit code)'
        } | Select-Object -Last 1
        if ($summaryLine) { $Summary = ($summaryLine -as [string]).Substring(0, [Math]::Min(300, $summaryLine.Length)) }
    }
}

$tag = if ($exitCode -eq 0) { 'OK' } else { 'FAIL' }
Send-Telegram "[research-tick] $tag in $DurationStr (exit=$exitCode). $Summary"

# Prune logs older than 30 days.
Get-ChildItem $LogDir -Filter 'tick-*.log' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force

exit $exitCode
