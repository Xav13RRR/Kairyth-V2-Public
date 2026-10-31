$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$StateFile = Join-Path $Root 'data\logs\v2-instance.json'
if (-not (Test-Path -LiteralPath $StateFile -PathType Leaf)) { Write-Host 'No recorded V2 instance; nothing stopped.'; exit 0 }
$state = Get-Content -Raw -LiteralPath $StateFile | ConvertFrom-Json
if ($state.root -ne $Root -or $state.port -eq 7860 -or -not $state.instance_id) { throw 'Invalid V2 instance record; nothing stopped.' }
$Url = "http://127.0.0.1:$($state.port)"
try { $health = Invoke-RestMethod "$Url/api/health" -TimeoutSec 2 } catch { Write-Host 'V2 instance is not reachable; nothing stopped.'; exit 0 }
if ($health.instance_root -ne $Root -or $health.instance_id -ne $state.instance_id) { throw 'Instance identity mismatch; nothing stopped.' }
Invoke-RestMethod "$Url/api/shutdown" -Method Post -TimeoutSec 3 | Out-Null
# Server shutdown cancels its own workers/process groups. Never terminate a WSL distribution.
Write-Host 'Shutdown requested for the recorded V2 instance only.'
