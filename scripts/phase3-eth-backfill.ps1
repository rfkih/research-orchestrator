#!/usr/bin/env pwsh
# Phase 3 ETH/USDT data plumbing.
#
# Submits a COVERAGE_REPAIR (mode=warmup) job against the research JVM's
# /api/v1/historical/jobs endpoint, then polls the job until terminal.
# Warmup fans out to 1h/15m/5m companion intervals automatically when the
# requested interval is 4h. Total runtime is typically 20-60 min.
#
# (Replaces the earlier script that hit the legacy synchronous /backfill
# endpoint, which has been retired in favor of the unified job system.)
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File phase3-eth-backfill.ps1
#
# You will be prompted for the admin email + password interactively.
# The credential is held only in memory for the duration of this script.

$ErrorActionPreference = 'Stop'

$TradingBase  = 'http://127.0.0.1:8080'
$ResearchBase = 'http://127.0.0.1:8081'
$Symbol       = 'ETHUSDT'
$Interval     = '4h'

Write-Host "=== Phase 3: ETHUSDT data plumbing ===" -ForegroundColor Cyan

# ── 1. Login ────────────────────────────────────────────────────────────────
$Email = Read-Host 'Admin email'
$SecurePass = Read-Host 'Admin password' -AsSecureString
$Pass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass))

$LoginBody = @{ email = $Email; password = $Pass } | ConvertTo-Json
$Pass = $null  # release ASAP

Write-Host "[1/4] Logging in as $Email ..."
try {
    $LoginResp = Invoke-RestMethod -Method Post `
        -Uri "$TradingBase/api/v1/users/login" `
        -Body $LoginBody -ContentType 'application/json' `
        -SessionVariable Sess
} catch {
    Write-Host "Login failed: $_" -ForegroundColor Red
    exit 1
}
$LoginBody = $null

# ── 2. Pre-flight: confirm ETH currently absent ─────────────────────────────
Write-Host "[2/4] Pre-flight: checking current ETH coverage ..."
$Pre = Invoke-Expression "psql -h 127.0.0.1 -U postgres -d trading_db -t -A -c `"SELECT COUNT(*) FROM market_data WHERE symbol='$Symbol';`""
Write-Host "   ETHUSDT market_data rows BEFORE: $Pre"

# ── 3. Submit COVERAGE_REPAIR job ───────────────────────────────────────────
$Started = Get-Date
$JobBody = @{
    jobType  = 'COVERAGE_REPAIR'
    symbol   = $Symbol
    interval = $Interval
    params   = @{ mode = 'warmup' }
} | ConvertTo-Json -Depth 5

Write-Host "[3/4] POST $ResearchBase/api/v1/historical/jobs (COVERAGE_REPAIR warmup)"
Write-Host "   Started: $Started"

try {
    $Submitted = Invoke-RestMethod -Method Post `
        -Uri "$ResearchBase/api/v1/historical/jobs" `
        -Body $JobBody -ContentType 'application/json' `
        -WebSession $Sess
} catch {
    Write-Host "Job submission failed: $_" -ForegroundColor Red
    exit 1
}

$JobId = $Submitted.jobId
Write-Host "   jobId: $JobId"

# ── 4. Poll until terminal (PENDING → RUNNING → SUCCESS/FAILED/CANCELLED) ──
Write-Host "[4/4] Polling /jobs/$JobId every 30s until terminal ..."
$Resp = $null
while ($true) {
    Start-Sleep -Seconds 30
    try {
        $Resp = Invoke-RestMethod -Method Get `
            -Uri "$ResearchBase/api/v1/historical/jobs/$JobId" `
            -WebSession $Sess
    } catch {
        Write-Host "Polling failed: $_" -ForegroundColor Red
        exit 1
    }
    $elapsed = ((Get-Date) - $Started).TotalMinutes.ToString('F1')
    Write-Host "   [$elapsed min] status=$($Resp.status) phase=$($Resp.phase) progress=$($Resp.progressDone)/$($Resp.progressTotal)"
    if ($Resp.status -in @('SUCCESS', 'FAILED', 'CANCELLED')) { break }
}

$Done = Get-Date
if ($Resp.status -eq 'SUCCESS') {
    Write-Host "   Done: $Done  (elapsed $(($Done - $Started).TotalMinutes.ToString('F1')) min)" -ForegroundColor Green
    $Resp.result | ConvertTo-Json -Depth 5 | Write-Host
} else {
    Write-Host "   Job ended with status=$($Resp.status): $($Resp.errorMessage)" -ForegroundColor Red
    exit 1
}

# ── 4. Post-flight verification ─────────────────────────────────────────────
Write-Host ""
Write-Host "=== Post-flight verification ===" -ForegroundColor Cyan
$VerifyMd = "SELECT \`"interval\`", COUNT(*) AS rows, MIN(start_time)::date AS oldest, MAX(start_time)::date AS newest FROM market_data WHERE symbol='$Symbol' GROUP BY \`"interval\`" ORDER BY \`"interval\`";"
$VerifyFs = "SELECT \`"interval\`", COUNT(*) AS rows FROM feature_store WHERE symbol='$Symbol' GROUP BY \`"interval\`" ORDER BY \`"interval\`";"
& psql -h 127.0.0.1 -U postgres -d trading_db -c $VerifyMd
& psql -h 127.0.0.1 -U postgres -d trading_db -c $VerifyFs

Write-Host ""
Write-Host "Phase 3 complete. ETH data is now available for backtests + research orchestrator." -ForegroundColor Green
