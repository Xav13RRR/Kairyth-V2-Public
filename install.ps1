param(
    [string]$PythonExe = '',
    [switch]$SkipBlender
)

$ErrorActionPreference = 'Stop'
try { $OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}
$Root = $PSScriptRoot
if ((Split-Path -Leaf $Root) -eq 'Kairyth') { throw 'This V2 installer must not run in the original Kairyth deployment.' }
foreach ($entry in @('runtime','data','third_party\envs')) {
    $item = Get-Item -LiteralPath (Join-Path $Root $entry) -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "V2 isolation forbids linked runtime/state: $entry" }
}
$Runtime = Join-Path $Root 'runtime'
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null

$Python = Join-Path $Runtime 'python\Scripts\python.exe'
Write-Host 'Preparing the Kairyth base service (AI modules are optional)...' -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $VenvDir = Join-Path $Runtime 'python'
    if (Test-Path -LiteralPath $VenvDir) {
        throw "Incomplete runtime at $VenvDir. Back it up and repair it manually; this installer will not overwrite it."
    }
    if ($PythonExe) {
        & $PythonExe -c "import sys; assert sys.version_info[:2] == (3, 10), 'Use Python 3.10'"
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 validation failed' }
        & $PythonExe -m venv $VenvDir
    } else {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Install Python 3.10 x64, or pass -PythonExe <path-to-python.exe>. See docs/DEPLOYMENT_EN.md.'
        }
        & py -3.10 -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 environment creation failed' }
}
& $Python -c "import fastapi,uvicorn,numpy,scipy,trimesh; print('Windows runtime OK')"
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install -r (Join-Path $Root 'requirements-windows.txt') -c (Join-Path $Root 'requirements-service-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Windows WebUI dependency installation failed' }
}

if (-not $SkipBlender) {
    $BlenderRoot = Join-Path $Runtime 'blender'
    $BlenderExe = Get-ChildItem -LiteralPath $BlenderRoot -Recurse -File -Filter blender.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $BlenderExe) {
        $Version = '4.5.11'
        $DownloadDir = Join-Path $env:TEMP ('kairyth-v2-blender-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $DownloadDir | Out-Null
        $Zip = Join-Path $DownloadDir "blender-$Version-windows-x64.zip"
        $Url = "https://download.blender.org/release/Blender4.5/blender-$Version-windows-x64.zip"
        Write-Host "Downloading official Blender $Version LTS..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
        New-Item -ItemType Directory -Force -Path $BlenderRoot | Out-Null
        Expand-Archive -LiteralPath $Zip -DestinationPath $BlenderRoot -Force
        Remove-Item -LiteralPath $Zip -Force
        $BlenderExe = Get-ChildItem -LiteralPath $BlenderRoot -Recurse -File -Filter blender.exe | Select-Object -First 1
    }
    if (-not $BlenderExe) { throw 'Blender deployment failed' }
    $Portable = Join-Path $BlenderExe.Directory.FullName 'portable'
    New-Item -ItemType Directory -Force -Path $Portable | Out-Null
    & $BlenderExe.FullName --background --version | Select-Object -First 2
    if ($LASTEXITCODE -ne 0) { throw 'Blender self-test failed' }
}

& $Python -m compileall -q (Join-Path $Root 'motionbuilder_app')
if ($LASTEXITCODE -ne 0) { throw 'Application compile check failed' }
Write-Host 'Base service ready. Optional AI environments/weights: docs/DEPLOYMENT_EN.md. Start with start_motionbuilder.ps1.' -ForegroundColor Green
# Use the same strict Python loader as the API; failures must never imply local mode.
$ReleaseProfile = 'unconfigured'
try {
    $ProfileLoader = Join-Path $Root 'motionbuilder_app\release_profile.py'
    if (Test-Path -LiteralPath $ProfileLoader -PathType Leaf) {
        $profileOutput = @(& $Python -I -B -c "import runpy,sys; print(runpy.run_path(sys.argv[1])['capabilities']()['release_profile'])" $ProfileLoader 2>&1)
        if ($LASTEXITCODE -eq 0 -and $profileOutput.Count -eq 1 -and $profileOutput[0].ToString().Trim() -in @('local-research','public-workbench','unconfigured')) { $ReleaseProfile = $profileOutput[0].ToString().Trim() }
    }
} catch { Write-Warning 'Release profile unavailable; research inference stays disabled.' }
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
try {
    if (-not $ResourceArgs.Count) { throw 'Resource advisory skipped safely' }
    $notice = @(& $Python (Join-Path $Root 'scripts\resource_readiness.py') @ResourceArgs 2>&1)
    $notice | Select-Object -First 4 | ForEach-Object { Write-Host $_ }
    Write-Host 'Resource details: docs/RESOURCE_READINESS.md. Resource checks do not enable public experimental entries.'
    # Resource exit 2 is advisory; do not block the base UI on optional models.
} catch { Write-Warning 'Resource advisory unavailable; see docs/RESOURCE_READINESS.md.' }
