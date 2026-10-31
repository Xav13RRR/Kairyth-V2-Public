param(
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $PythonExe) {
  $PythonExe = Join-Path $ProjectRoot "runtime\python\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
  throw "Kairyth Python not found: $PythonExe"
}

$requirements = Join-Path $ProjectRoot "requirements-kimodo.txt"
if (-not (Test-Path -LiteralPath $requirements)) {
  throw "Kimodo requirements file not found: $requirements"
}
$kimodoSource = Join-Path $ProjectRoot "kimodo_runtime\src\kimodo\__init__.py"
if (-not (Test-Path -LiteralPath $kimodoSource -PathType Leaf)) {
  throw "Project-local Kimodo source not found: $kimodoSource"
}

& $PythonExe -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Kimodo dependency installation failed" }
Push-Location $ProjectRoot
try {
  & $PythonExe -c "from motionbuilder_app.kimodo_runtime import install_runtime_paths; install_runtime_paths(); import kimodo; print(kimodo.__file__)"
  if ($LASTEXITCODE -ne 0) { throw "Project-local Kimodo import failed" }
} finally {
  Pop-Location
}

Write-Host "Kimodo dependencies installed into Kairyth runtime Python: $PythonExe"
