$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $ProjectRoot 'kimodo_runtime\src\kimodo\__init__.py'
$python = Join-Path $ProjectRoot 'runtime\python\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Project-local Kimodo source not found: $source"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python not found: $python"
}

Push-Location $ProjectRoot
try {
    & $python -c "from motionbuilder_app.kimodo_runtime import install_runtime_paths; install_runtime_paths(); import kimodo; print(kimodo.__file__)"
    if ($LASTEXITCODE -ne 0) { throw 'Project-local Kimodo import failed' }
} finally {
    Pop-Location
}
