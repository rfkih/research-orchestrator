# Research Orchestrator — Startup Runbook

Everything you need to get the autonomous research loop running from scratch.

---

## Architecture recap

```
Windows Task Scheduler (every 6h)
        │
        ▼
research-tick-cron.ps1   ← fires claude -p with the quant-researcher prompt
        │
        ▼
quant-researcher agent   ← Claude subagent; reads .claude/agents/quant-researcher.md
        │  HTTP (X-Orch-Token)
        ▼
research-orchestrator    ← FastAPI, port 8082  (YOU must keep this running)
        │  HTTP
        ▼
Research JVM             ← Spring Boot, port 8081, profile=dev,research  (YOU must keep this running)
        │
        ▼
PostgreSQL
```

The **orchestrator** and **Research JVM** are long-lived processes you start manually (or on boot).  
The **researcher agent** fires automatically via Task Scheduler — it is NOT a persistent process.

---

## Prerequisites (one-time)

### 1. Python 3.12
The orchestrator requires Python ≥ 3.12. Check:
```powershell
py -3.12 --version
```
If missing, install via winget:
```powershell
winget install Python.Python.3.12
```

### 2. Install orchestrator dependencies (one-time, or after pulling changes)
```bash
cd C:/MyFiles/blackheart/research-orchestrator
py -3.12 -m pip install -e ".[dev]"
```

---

## Every time you restart your machine

### Step 1 — Start the Research JVM (port 8081)

Open **PowerShell** and run as a single line:
```powershell
Start-Process -FilePath "C:\graalvm\graalvm-jdk-21.0.4+8.1\bin\java.exe" -ArgumentList "-Xms512m -Xmx1500m -jar C:\MyFiles\blackheart\blackheart\build\libs\blackheart-research-0.0.1-SNAPSHOT.jar --spring.profiles.active=dev,research --server.port=8081" -WindowStyle Normal
```

Wait ~30 seconds, then verify it's up:
```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8081/api/v1/dev/login-as' -Method POST -ContentType 'application/json' -Body '{"email":"rfkih23@gmail.com"}' -UseBasicParsing | Select-Object StatusCode
```
Expected: `StatusCode: 200`

> **If you get 404**: the JAR wasn't built with the research profile, or the JVM hasn't finished booting yet. Wait longer and retry.

### Step 2 — Start the research orchestrator (port 8082)

Open **Git Bash** (or any terminal) and run:
```bash
cd C:/MyFiles/blackheart/research-orchestrator
py -3.12 -m orchestrator
```

Leave this terminal open — the orchestrator must stay running.

Verify it's up (new terminal):
```bash
curl http://127.0.0.1:8082/healthz
curl http://127.0.0.1:8082/readyz
```
Both should return `{"status":"ok"}` / `{"status":"ready"}`.

---

## One-time: register the Task Scheduler task (makes agent autonomous)

Open **PowerShell as Administrator** (Start → search PowerShell → right-click → Run as Administrator → approve UAC).

Paste this single line:
```powershell
$A = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NonInteractive -File "C:\MyFiles\blackheart\research-orchestrator\scripts\research-tick-cron.ps1"'; $T = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 6) -Once -At (Get-Date); Register-ScheduledTask -TaskName 'BlackheartResearchTick' -Action $A -Trigger $T -RunLevel Highest -Force
```

Verify:
```powershell
Get-ScheduledTask -TaskName 'BlackheartResearchTick' | Select-Object TaskName, State
```
Expected: `State: Ready`

> After this, the researcher agent fires every 6 hours automatically — no further action needed as long as the orchestrator and Research JVM are running.

---

## Run a tick manually (any time)

```powershell
powershell -File "C:\MyFiles\blackheart\research-orchestrator\scripts\research-tick-cron.ps1"
```

The tick takes up to ~30 minutes (synchronous backtest). Output goes to:
```
C:\MyFiles\blackheart\research-orchestrator\logs\tick-<timestamp>.log
```

---

## Check orchestrator state / leaderboard

```bash
# What should the agent do next?
curl -H "X-Orch-Token: dev-sentinel-not-for-prod" -H "X-Agent-Name: ops" http://127.0.0.1:8082/agent/state

# Top strategies by score
curl -H "X-Orch-Token: dev-sentinel-not-for-prod" -H "X-Agent-Name: ops" "http://127.0.0.1:8082/leaderboard?limit=15"

# Queue depth
curl -H "X-Orch-Token: dev-sentinel-not-for-prod" -H "X-Agent-Name: ops" http://127.0.0.1:8082/queue
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named orchestrator` | Dependencies not installed | `py -3.12 -m pip install -e ".[dev]"` |
| `requires Python >=3.12` error from pip | Using system Python 3.11 | Use `py -3.12 -m pip` not `pip` |
| `/api/v1/dev/login-as` returns 404 | Research JVM not running or wrong profile | Start JVM with `--spring.profiles.active=dev,research` |
| `jvm_auth_failed` in tick log | Research JVM down or not yet booted | Wait 30s and retry; check JVM window for errors |
| `Access is denied` on Register-ScheduledTask | Not running as Administrator | Open PowerShell via right-click → Run as Administrator |
| Task Scheduler fires but tick does nothing | Orchestrator (8082) not running | Start orchestrator before the 6h window fires |
