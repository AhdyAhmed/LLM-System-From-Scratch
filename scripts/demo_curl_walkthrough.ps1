<#
.SYNOPSIS
    Day015 & Day018 curl.exe cookie-jar walkthrough for the ScratchLLM headless API.
    Demonstrates CSRF bootstrap, registration, Redis session login, creator-private conversation CRUD,
    idempotent run submission, SSE streaming, message pagination, and logout.
#>
param(
    [string]$BaseUrl = "http://127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"
$CookieJar = Join-Path $env:TEMP "scratchllm-cookies-$([guid]::NewGuid().ToString('N')).txt"

try {
    Write-Host "1. Bootstrapping CSRF token via curl.exe cookie-jar..." -ForegroundColor Cyan
    $csrfJson = curl.exe -s -c $CookieJar -b $CookieJar "$BaseUrl/auth/csrf" | ConvertFrom-Json
    $CsrfToken = $csrfJson.token
    Write-Host "   CSRF Header: $($csrfJson.header_name) = $CsrfToken" -ForegroundColor Green

    $Email = "demo-$([guid]::NewGuid().ToString('N').Substring(0, 8))@example.com"
    $RegBody = (@{
        email = $Email
        password = "StrongDemoPassword#2026"
        workspace_name = "CLI Demo Workspace"
    } | ConvertTo-Json -Compress)

    Write-Host "2. Registering account $Email..." -ForegroundColor Cyan
    $profile = curl.exe -s -c $CookieJar -b $CookieJar `
        -H "Content-Type: application/json" `
        -H "X-CSRF-TOKEN: $CsrfToken" `
        -d $RegBody "$BaseUrl/auth/register" | ConvertFrom-Json
    $WorkspaceId = $profile.workspaces[0].workspace_id
    Write-Host "   Created user_id=$($profile.user_id), workspace_id=$WorkspaceId" -ForegroundColor Green

    Write-Host "3. Logging in and rotating Redis session cookie..." -ForegroundColor Cyan
    $LoginBody = (@{ email = $Email; password = "StrongDemoPassword#2026" } | ConvertTo-Json -Compress)
    $loginProfile = curl.exe -s -c $CookieJar -b $CookieJar `
        -H "Content-Type: application/json" `
        -H "X-CSRF-TOKEN: $CsrfToken" `
        -d $LoginBody "$BaseUrl/auth/login" | ConvertFrom-Json

    # Refresh CSRF token after login session rotation
    $csrfJson = curl.exe -s -c $CookieJar -b $CookieJar "$BaseUrl/auth/csrf" | ConvertFrom-Json
    $CsrfToken = $csrfJson.token

    Write-Host "4. Creating creator-private conversation..." -ForegroundColor Cyan
    $ConvBody = (@{ title = "Curl Cookie-Jar Walkthrough" } | ConvertTo-Json -Compress)
    $conv = curl.exe -s -c $CookieJar -b $CookieJar `
        -H "Content-Type: application/json" `
        -H "X-CSRF-TOKEN: $CsrfToken" `
        -d $ConvBody "$BaseUrl/api/v1/workspaces/$WorkspaceId/conversations" | ConvertFrom-Json
    $ConvId = $conv.conversation_id
    Write-Host "   Created conversation_id=$ConvId" -ForegroundColor Green

    Write-Host "5. Submitting idempotent generation run..." -ForegroundColor Cyan
    $RunBody = (@{
        prompt = "Once upon a time, the little robot counted tokens."
        model_alias = "fake-deterministic"
        sampling = @{
            max_new_tokens = 16
            temperature = 0.7
            top_k = 40
            top_p = 0.95
            seed = 42
        }
    } | ConvertTo-Json -Compress)
    $runAccepted = curl.exe -s -c $CookieJar -b $CookieJar `
        -H "Content-Type: application/json" `
        -H "X-CSRF-TOKEN: $CsrfToken" `
        -H "Idempotency-Key: idem-curl-walkthrough-0001" `
        -d $RunBody "$BaseUrl/api/v1/conversations/$ConvId/runs" | ConvertFrom-Json
    $RunId = $runAccepted.run_id
    Write-Host "   Run accepted: run_id=$RunId, events_url=$($runAccepted.events_url)" -ForegroundColor Green

    Write-Host "6. Streaming SSE events from $BaseUrl/api/v1/runs/$RunId/events..." -ForegroundColor Cyan
    curl.exe -s -N -b $CookieJar -H "Accept: text/event-stream" "$BaseUrl/api/v1/runs/$RunId/events"

    Write-Host "7. Logging out and invalidating Redis session..." -ForegroundColor Cyan
    curl.exe -s -c $CookieJar -b $CookieJar -X POST -H "X-CSRF-TOKEN: $CsrfToken" "$BaseUrl/auth/logout" | Out-Null
    Write-Host "Walkthrough completed successfully." -ForegroundColor Green
} finally {
    if (Test-Path $CookieJar) {
        Remove-Item $CookieJar -Force
    }
}
