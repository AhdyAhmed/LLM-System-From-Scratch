param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $RepoRoot "ml\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "=== ScratchLLM CPU-Only Demo & Benchmark Verification (Day097) ===" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DryRun] Would execute CPU-only ScratchGPT KV-cache and deterministic serving verification via $PythonExe" -ForegroundColor Green
    exit 0
}

& $PythonExe -m scratchlm.benchmarks.profiler --mode cpu-demo
if ($LASTEXITCODE -ne 0) {
    throw "CPU demo verification failed with exit code $LASTEXITCODE"
}
Write-Host "=== CPU-Only Demo Completed Cleanly ===" -ForegroundColor Green
