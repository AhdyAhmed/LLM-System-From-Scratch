param(
    [ValidateSet("single", "two-replica")]
    [string]$Mode = "single",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $RepoRoot ".local_native_pids.json"

Write-Host "=== ScratchLLM Native Services Launcher (Day097) ===" -ForegroundColor Cyan
Write-Host "Mode: $Mode | RepoRoot: $RepoRoot"

$Plan = @(
    @{
        Name = "inference-gateway"
        Command = "python -m uvicorn scratchlm.serving.app:app --host 127.0.0.1 --port 8091"
        WorkingDir = (Join-Path $RepoRoot "ml")
        Port = 8091
    },
    @{
        Name = "spring-api-replica-1"
        Command = "mvn spring-boot:run -Dspring-boot.run.arguments=`"--server.port=8080 --scratchllm.replica.id=replica-1`""
        WorkingDir = (Join-Path $RepoRoot "backend")
        Port = 8080
    }
)

if ($Mode -eq "two-replica") {
    $Plan += @{
        Name = "spring-api-replica-2"
        Command = "mvn spring-boot:run -Dspring-boot.run.arguments=`"--server.port=8082 --scratchllm.replica.id=replica-2`""
        WorkingDir = (Join-Path $RepoRoot "backend")
        Port = 8082
    }
}

if ($DryRun) {
    Write-Host "[DryRun] Planned native processes:" -ForegroundColor Green
    foreach ($item in $Plan) {
        Write-Host "  - $($item.Name) (Port $($item.Port)): $($item.Command)"
    }
    exit 0
}

$Started = @()
foreach ($item in $Plan) {
    Write-Host "Starting $($item.Name) on port $($item.Port)..." -ForegroundColor Yellow
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-Command", $item.Command -WorkingDirectory $item.WorkingDir -PassThru
    $Started += @{
        name = $item.Name
        pid = $proc.Id
        port = $item.Port
        started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
}

$Started | ConvertTo-Json -Depth 4 | Set-Content -Path $PidFile -Encoding UTF8
Write-Host "Native services recorded in $PidFile" -ForegroundColor Green
