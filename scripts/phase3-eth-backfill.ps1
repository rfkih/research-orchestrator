#!/usr/bin/env pwsh
# Phase 3 ETH/USDT data plumbing.
#
# Hits the research JVM's /api/v1/historical/backfill?symbol=ETHUSDT&interval=4h
# endpoint, which is admin-gated. The endpoint fans out to 1h/15m/5m companion
# warmup for the same 4h date range (~2.4y) and computes FeatureStore indicators
# for all four intervals. Long-running (20-60 min synchronous).
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

Write-Host "[1/3] Logging in as $Email ..."
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
Write-Host "[2/3] Pre-flight: checking current ETH coverage ..."
$Pre = Invoke-Expression "psql -h 127.0.0.1 -U postgres -d trading_db -t -A -c `"SELECT COUNT(*) FROM market_data WHERE symbol='$Symbol';`""
Write-Host "   ETHUSDT market_data rows BEFORE: $Pre"

# ── 3. Fire backfill (long-running, synchronous) ────────────────────────────
$Started = Get-Date
Write-Host "[3/3] POST $ResearchBase/api/v1/historical/backfill?symbol=$Symbol&interval=$Interval"
Write-Host "   This blocks for 20-60 min. Progress visible in research JVM logs."
Write-Host "   Started: $Started"

try {
    $Resp = Invoke-RestMethod -Method Post `
        -Uri "$ResearchBase/api/v1/historical/backfill?symbol=$Symbol&interval=$Interval" `
        -WebSession $Sess -TimeoutSec 7200
    $Done = Get-Date
    Write-Host "   Done:    $Done  (elapsed $(($Done - $Started).TotalMinutes.ToString('F1')) min)" -ForegroundColor Green
    $Resp | ConvertTo-Json -Depth 5 | Write-Host
} catch {
    Write-Host "Backfill failed: $_" -ForegroundColor Red
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
