<#
.SYNOPSIS
    Unified local verification script for Phase 1 (Gate G1) and subsequent phases.
    Runs secret scanning, Docker Compose validation, environment probe, Python pytest,
    and Java Maven tests against PostgreSQL 16 + pgvector.
#>
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Write-Host "=== [1/5] Secret Pattern Scan ===" -ForegroundColor Cyan
$SecretPatterns = @(
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "AKIA[0-9A-Z]{16}",
    "ghp_[0-9a-zA-Z]{36}"
)
foreach ($pattern in $SecretPatterns) {
    $matchesFound = Select-String -Path "$RepoRoot\docs\*.md", "$RepoRoot\contracts\*.*", "$RepoRoot\infra\*.yml" -Pattern $pattern -ErrorAction SilentlyContinue
    if ($matchesFound) {
        Write-Host "Secret pattern detected: $pattern" -ForegroundColor Red
        exit 1
    }
}
Write-Host "Secret scan: PASS" -ForegroundColor Green

Write-Host "=== [2/5] Docker Compose Profile Validation ===" -ForegroundColor Cyan
docker compose --env-file "$RepoRoot\infra\.env.example" -f "$RepoRoot\infra\compose.yml" --profile full config -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Compose validation failed" -ForegroundColor Red
    exit 1
}
Write-Host "Docker Compose profile validation: PASS" -ForegroundColor Green

Write-Host "=== [3/5] Hardware & Runtime Environment Probe ===" -ForegroundColor Cyan
python "$RepoRoot\scripts\probe_environment.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment probe failed" -ForegroundColor Red
    exit 1
}

Write-Host "=== [4/5] Python ML & Shared Contracts Test Suite ===" -ForegroundColor Cyan
Push-Location "$RepoRoot\ml"
python -m pytest
$PytestExit = $LASTEXITCODE
Pop-Location
if ($PytestExit -ne 0) {
    Write-Host "Python pytest failed with exit code $PytestExit" -ForegroundColor Red
    exit $PytestExit
}

Write-Host "=== [5/5] Java 21 Spring Boot & PostgreSQL Integration Test Suite ===" -ForegroundColor Cyan
Push-Location "$RepoRoot\backend"
.\mvnw.cmd test
$MvnExit = $LASTEXITCODE
Pop-Location
if ($MvnExit -ne 0) {
    Write-Host "Maven test suite failed with exit code $MvnExit" -ForegroundColor Red
    exit $MvnExit
}

Write-Host "=== ALL CHECKS PASSED (0 errors, 0 failures) ===" -ForegroundColor Green
