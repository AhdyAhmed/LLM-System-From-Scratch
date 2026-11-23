param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RepoRoot ".local_native_pids.json"

Write-Host "=== ScratchLLM Native Services Stopper (Day097) ===" -ForegroundColor Cyan

if (-not (Test-Path $PidFile)) {
    Write-Host "No active PID file found at $PidFile. Nothing to stop." -ForegroundColor Green
    exit 0
}

$Entries = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
foreach ($entry in $Entries) {
    if ($DryRun) {
        Write-Host "[DryRun] Would stop $($entry.name) (PID $($entry.pid))" -ForegroundColor Yellow
        continue
    }
    try {
        $proc = Get-Process -Id $entry.pid -ErrorAction SilentlyContinue
        if ($null -ne $proc) {
            Stop-Process -Id $entry.pid -Force
            Write-Host "Stopped $($entry.name) (PID $($entry.pid))" -ForegroundColor Green
        }
    } catch {
        Write-Host "Process $($entry.name) (PID $($entry.pid)) already exited." -ForegroundColor DarkGray
    }
}

if (-not $DryRun) {
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
}
