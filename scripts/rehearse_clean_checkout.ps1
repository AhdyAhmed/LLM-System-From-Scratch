param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "=== ScratchLLM Clean-Checkout Rehearsal Verifier (Day117) ===" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DryRun] Would verify all migrations (V1-V7), Compose profiles, model packages, cards, ADRs, and Postman collection" -ForegroundColor Green
    exit 0
}

$env:PYTHONPATH = Join-Path $RepoRoot "ml\src"
python -m scratchlm.release.final_release --mode all
if ($LASTEXITCODE -ne 0) {
    throw "Clean-checkout rehearsal failed with exit code $LASTEXITCODE"
}
Write-Host "=== Clean-Checkout Rehearsal Passed ===" -ForegroundColor Green
