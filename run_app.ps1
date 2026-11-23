# Launch the D6 Unified Platform.
#
#   .\run_app.ps1              # create the venv if needed, install, run
#   .\run_app.ps1 -SkipInstall # straight to running
#
# Designed to work on a machine that is not the author's: it creates its own
# venv, installs pinned dependencies, and needs no configuration. The app opens
# usable within a couple of seconds -- the heavy engine load happens on a
# background thread while you read the brief.

param([switch]$SkipInstall, [int]$Port = 0)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    $launcher = (Get-Command py -ErrorAction SilentlyContinue)
    if ($null -eq $launcher) { $launcher = (Get-Command python -ErrorAction SilentlyContinue) }
    if ($null -eq $launcher) { throw "No Python found on PATH. Install Python 3.11+ and retry." }
    & $launcher.Source -m venv .venv
    if (-not (Test-Path $py)) { throw "venv creation failed." }
    $SkipInstall = $false
}

if (-not $SkipInstall) {
    Write-Host "Installing dependencies (pinned -- see requirements.txt)..." -ForegroundColor Cyan
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed." }
}

# Fail fast with a clear message rather than a stack trace three screens deep.
$required = @(
    "Urban Media Datasets\screens.csv",
    "Urban Media Datasets\cities.csv",
    "eda_output\audience_cube.csv",
    "eda_output\audience_profiles.csv",
    "eda_output\pricing_grain_d3.csv"
)
$missing = $required | Where-Object { -not (Test-Path (Join-Path $PSScriptRoot $_)) }
if ($missing.Count -gt 0) {
    Write-Host "Missing required data files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "The D1-D5 outputs must be present. See README."
}

if (-not (Test-Path (Join-Path $PSScriptRoot ".env"))) {
    Write-Host "No .env found -- running without LLM credentials." -ForegroundColor Yellow
    Write-Host "  The app is fully functional: feedback routing uses deterministic" -ForegroundColor Yellow
    Write-Host "  rules and the narrative uses D5's template. Copy .env.example to" -ForegroundColor Yellow
    Write-Host "  .env and fill it in to enable the model paths." -ForegroundColor Yellow
}

# Pick a free port. Streamlit exits with a bare "Port N is not available" if the
# port is taken, which is an unhelpful way to start a demo -- and 8501 is very
# often already serving something else on a dev machine.
function Test-PortFree([int]$p) {
    return -not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

if ($Port -eq 0) {
    $Port = 8501
    while ($Port -lt 8530 -and -not (Test-PortFree $Port)) { $Port++ }
    if (-not (Test-PortFree $Port)) { throw "No free port in 8501-8529." }
    if ($Port -ne 8501) {
        Write-Host "Port 8501 is in use; using $Port instead." -ForegroundColor Yellow
    }
} elseif (-not (Test-PortFree $Port)) {
    throw "Port $Port is already in use. Pass a different -Port."
}

Write-Host ""
Write-Host "Starting Streamlit on http://localhost:$Port" -ForegroundColor Green
& $py -m streamlit run app/main.py --server.port $Port --server.headless false
