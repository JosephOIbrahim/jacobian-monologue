# Fully automated benchmark pass: env -> measure -> regression-check -> sync docs -> commit/push.
# Usage:  benchmarks\run.ps1 [-NoPush] [-Force]
#   -NoPush  do everything except git push
#   -Force   commit even when nothing material changed (refreshes the dated baseline)
param([switch]$NoPush, [switch]$Force)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

# ---- 1. environment (bootstrap once, reuse after) ----
if (-not (Test-Path $Py)) {
    Write-Host "[env] .venv missing -- bootstrapping" -ForegroundColor Yellow
    uv venv --python 3.12
    uv pip install -p $Py torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install -p $Py usd-core==26.8
    uv pip install -p $Py -e .
}

# ---- 2. measure ----
& $Py benchmarks\bench.py
$flip = $LASTEXITCODE   # 0 = decision flipped

# ---- 3. regression check + doc sync (docs regenerate from results.json) ----
$auto = & $Py benchmarks\autodoc.py 2>&1
$auto | ForEach-Object { Write-Host $_ }
$reg = $LASTEXITCODE    # 0 = none, 2 = tripped
$docChanged = ($auto | Select-String "DOC_CHANGED=true") -ne $null

# ---- 4. commit policy: only material change ships ----
$material = $docChanged -or ($reg -ne 0) -or ($flip -ne 0) -or $Force
if (-not $material) {
    git checkout -- benchmarks/results.json   # discard timing jitter; keep tree clean
    Write-Host "[done] no material change -- nothing committed" -ForegroundColor Green
    exit 0
}

$keyLine = $auto | Select-String '^KEY='
$key = if ($keyLine) { $keyLine.ToString().Substring(4) } else { "key line missing" }
$status = if ($reg -ne 0) { "REGRESSION" } elseif ($flip -ne 0) { "FLIP FAILED" } else { "refresh" }
git add benchmarks README.md
git commit -m "benchmarks: automated run ($status)" -m "$key"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fail] git commit failed -- nothing pushed" -ForegroundColor Red
    exit 1
}
if (-not $NoPush) {
    git push origin master
    if ($LASTEXITCODE -ne 0) { Write-Host "[fail] git push failed" -ForegroundColor Red; exit 1 }
}
Write-Host "[done] committed ($status)$(if (-not $NoPush) { ' and pushed' })" -ForegroundColor Green
exit $reg
