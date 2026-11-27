param(
    [string]$BackupLabel = "day104-restore-drill",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "=== ScratchLLM Isolated Restore Drill Runner (Day104) ===" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DryRun] Would execute isolated restore drill, verify SHA-256 checksums, invalidate sessions, and verify inbox deduplication" -ForegroundColor Green
    exit 0
}

python -m scratchlm.ops.backup_restore --mode restore-drill --label $BackupLabel
if ($LASTEXITCODE -ne 0) {
    throw "Isolated restore drill failed with exit code $LASTEXITCODE"
}
Write-Host "=== Isolated Restore Drill Completed Cleanly ===" -ForegroundColor Green
