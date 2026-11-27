param(
    [string]$BackupLabel = "day103-snapshot",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "=== ScratchLLM Platform Backup Runner (Day103) ===" -ForegroundColor Cyan
Write-Host "BackupLabel: $BackupLabel"

if ($DryRun) {
    Write-Host "[DryRun] Would create backup manifest for llm_app, mlflow_db, airflow_db, DAGs, and pinned model packages" -ForegroundColor Green
    exit 0
}

python -m scratchlm.ops.backup_restore --mode backup --label $BackupLabel
if ($LASTEXITCODE -ne 0) {
    throw "Platform backup failed with exit code $LASTEXITCODE"
}
Write-Host "=== Platform Backup Completed Cleanly ===" -ForegroundColor Green
