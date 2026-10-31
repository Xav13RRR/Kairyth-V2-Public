param([ValidateRange(1024,65535)][int]$Port = 7861, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
try { $OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}
$Root = $PSScriptRoot
if ($Port -eq 7860) { throw 'Port 7860 is reserved for the original Kairyth deployment.' }
$env:UNIRIG_ROOT = Join-Path $Root 'third_party\UniRig'
if ($env:KAIRYTH_V2_ALLOW_EXTERNAL_ENV -ne '1' -or -not $env:UNIRIG_PYTHON) { $env:UNIRIG_PYTHON = Join-Path $Root 'third_party\envs\unirig\python.exe' }
$env:SKINTOKENS_ROOT = Join-Path $Root 'third_party\SkinTokens'
if ($env:KAIRYTH_V2_ALLOW_EXTERNAL_ENV -ne '1' -or -not $env:SKINTOKENS_PYTHON) { $env:SKINTOKENS_PYTHON = Join-Path $Root 'third_party\envs\skintokens\python.exe' }
$Url = "http://127.0.0.1:$Port"
$Python = Join-Path $Root 'runtime\python\Scripts\python.exe'
$LogDir = Join-Path $Root 'data\logs'
$StateFile = Join-Path $LogDir 'v2-instance.json'
foreach ($entry in @('data','runtime','third_party\envs')) {
    $item = Get-Item -LiteralPath (Join-Path $Root $entry) -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "V2 isolation forbids linked state/runtime: $entry" }
}
$client = New-Object Net.Sockets.TcpClient
try { $busy = $client.ConnectAsync('127.0.0.1',$Port).Wait(1000) -and $client.Connected } catch { $busy = $false } finally { $client.Dispose() }
if ($busy) {
    $state = if (Test-Path -LiteralPath $StateFile) { Get-Content -Raw -LiteralPath $StateFile | ConvertFrom-Json } else { $null }
    try { $health = Invoke-RestMethod "$Url/api/health" -TimeoutSec 2 } catch { throw "Port $Port is occupied; no process was stopped." }
    if ($state -and $health.instance_root -eq $Root -and $health.instance_id -eq $state.instance_id -and $state.port -eq $Port) {
        if (-not $NoBrowser) { Start-Process $Url }
        exit 0
    }
    throw "Port $Port belongs to an unrecognized instance; no process was stopped."
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'V2 runtime missing. Run the V2 install.ps1 first.' }
# Use the same strict Python loader as the API; failures must never imply local mode.
$ReleaseProfile = 'unconfigured'
try {
    $ProfileLoader = Join-Path $Root 'motionbuilder_app\release_profile.py'
    if (Test-Path -LiteralPath $ProfileLoader -PathType Leaf) {
        $profileOutput = @(& $Python -I -B -c "import runpy,sys; print(runpy.run_path(sys.argv[1])['capabilities']()['release_profile'])" $ProfileLoader 2>&1)
        if ($LASTEXITCODE -eq 0 -and $profileOutput.Count -eq 1 -and $profileOutput[0].ToString().Trim() -in @('local-research','public-workbench','unconfigured')) { $ReleaseProfile = $profileOutput[0].ToString().Trim() }
    }
} catch { Write-Warning 'Release profile unavailable; research inference stays disabled.' }
if ($ReleaseProfile -eq 'local-research') {
    if (-not $env:MOTIONBUILDER_WSL_DISTRO) { $env:MOTIONBUILDER_WSL_DISTRO = 'Kairyth-V2-GVHMR' }
    if ($env:MOTIONBUILDER_WSL_DISTRO -eq 'Kairyth-GVHMR') { throw 'Use a separate V2 WSL distribution; the original is protected.' }
    if (-not $env:MOTIONBUILDER_WSL_PYTHON) { $env:MOTIONBUILDER_WSL_PYTHON = '/home/gvhmr/miniforge3/envs/gvhmr/bin/python' }
} elseif ($ReleaseProfile -eq 'public-workbench') {
    Write-Host 'Experimental: new HybrIK/GVHMR inference is unavailable in this public release; compatibility packages are planned for a future update.'
} else {
    Write-Host 'Release configuration is not ready; new image/video inference is disabled. Core startup continues.'
}
# Profile-specific advisory arguments: preserve every unrelated manifest feature.
$ResourceArgs = @('--feature','all','--presence-only')
if ($ReleaseProfile -ne 'local-research') {
    $ResourceArgs = @()
    try {
        $resourceManifest = Get-Content -Raw -LiteralPath (Join-Path $Root 'docs\resource-license-evidence.json') | ConvertFrom-Json
        if (-not $resourceManifest.resources) { throw 'Missing resource inventory' }
        $advisoryFeatures = @($resourceManifest.resources | ForEach-Object { $_.feature } | Where-Object { $_ -and $_ -notin @('hybrik','gvhmr') } | Sort-Object -Unique)
        if (-not $advisoryFeatures.Count) { throw 'No non-research advisory features' }
        $ResourceArgs = @('--presence-only')
        foreach ($feature in $advisoryFeatures) { $ResourceArgs += @('--feature',[string]$feature) }
    } catch { $ResourceArgs = @(); Write-Warning 'Non-research resource advisory unavailable; no research resources were probed.' }
}
# End profile-specific advisory arguments.
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
try {
    if (-not $ResourceArgs.Count) { throw 'Resource advisory skipped safely' }
    $resourceNotice = @(& $Python (Join-Path $Root 'scripts\resource_readiness.py') @ResourceArgs 2>&1)
    $resourceNotice | Out-File -Encoding UTF8 -LiteralPath (Join-Path $LogDir 'resource-readiness.log')
    $resourceNotice | Select-Object -First 4 | ForEach-Object { Write-Host $_ }
    Write-Host 'Resource details: docs/RESOURCE_READINESS.md. Resource checks do not enable public experimental entries.'
    # Exit 2 means optional resources are missing, not that the base UI cannot start.
} catch { Write-Warning 'Resource advisory unavailable; see docs/RESOURCE_READINESS.md. Startup continues.' }
$env:KAIRYTH_V2_INSTANCE_ID = [guid]::NewGuid().ToString('N')
$server = Start-Process -FilePath $Python -ArgumentList @('-m','motionbuilder_app.server','--port',"$Port",'--no-browser') -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $LogDir 'server.out.log') -RedirectStandardError (Join-Path $LogDir 'server.err.log')
@{instance_id=$env:KAIRYTH_V2_INSTANCE_ID;root=$Root;port=$Port;pid=$server.Id} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $StateFile
$ready = $false
for ($i=0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 500
    $server.Refresh()
    if ($server.HasExited) { break }
    try {
        $health = Invoke-RestMethod "$Url/api/health" -TimeoutSec 1
        if ($health.instance_root -eq $Root -and $health.instance_id -eq $env:KAIRYTH_V2_INSTANCE_ID) { $ready=$true; break }
    } catch {}
}
if (-not $ready) {
    $server.Refresh()
    if (-not $server.HasExited) { $server.Kill() }
    throw "V2 startup failed. See $LogDir\server.err.log"
}
if (-not $NoBrowser) { Start-Process $Url }
$server.WaitForExit()
