# Competition toolkit setup — run this once per new project.
# Every step here exists because it was a REAL bug hit during testing today,
# not defensive boilerplate. Run from the project root, in PowerShell.
#
# v14 — switched from Anthropic Claude to Azure OpenAI, since that's the
# API the competition provides.

Write-Host "=== Competition toolkit setup (Azure OpenAI) ===" -ForegroundColor Cyan

# --- 1. venv ---
if (-not (Test-Path "venv")) {
    Write-Host "`n[1/6] Creating venv..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "`n[1/6] venv already exists, skipping." -ForegroundColor Yellow
}
& ".\venv\Scripts\Activate.ps1"

# --- 2. Packages ---
Write-Host "`n[2/6] Installing packages..." -ForegroundColor Yellow
pip install --quiet pandas numpy scipy scikit-learn joblib langgraph openai python-dotenv ortools

# --- 3. ds_toolkit, editable install ---
# CRITICAL: this script's OWN folder structure avoids the exact bug that
# cost 20+ minutes today -- never name a project folder the same as the
# Python package it contains. If you copy ds-toolkit-package/ elsewhere,
# keep that name distinct from the inner ds_toolkit/ folder.
Write-Host "`n[3/6] Installing ds_toolkit (editable)..." -ForegroundColor Yellow
$toolkitPath = Join-Path $PSScriptRoot "ds-toolkit-package"
pip install --quiet -e $toolkitPath
Write-Host "  Installed from: $toolkitPath"

# --- 4. .env ---
if (-not (Test-Path ".env")) {
    Write-Host "`n[4/6] Creating .env template -- ADD YOUR REAL AZURE VALUES BEFORE CONTINUING." -ForegroundColor Yellow
    @(
        "AZURE_OPENAI_API_KEY=REPLACE-ME",
        "AZURE_OPENAI_ENDPOINT=https://REPLACE-ME.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION=2024-10-21",
        "AZURE_OPENAI_DEPLOYMENT=REPLACE-ME"
    ) | Out-File -Encoding ascii .env
    Write-Host "  Created .env -- edit it now with the endpoint/key/deployment the" -ForegroundColor Red
    Write-Host "  competition organizers gave you, then re-run this script." -ForegroundColor Red
} else {
    Write-Host "`n[4/6] .env already exists, skipping." -ForegroundColor Yellow
}

# --- 5. Sanity check: package-name collision ---
# The exact bug from today: a bare folder in the CURRENT directory sharing
# a name with an installed package silently shadows the real install.
Write-Host "`n[5/6] Checking for package/folder name collisions..." -ForegroundColor Yellow
$collisionCheck = python -c "import ds_toolkit; print(ds_toolkit.__file__)" 2>&1
if ($collisionCheck -match "None" -or $collisionCheck -match "Error") {
    Write-Host "  WARNING: ds_toolkit did not import cleanly. Check for a bare" -ForegroundColor Red
    Write-Host "  'ds_toolkit' folder directly in this project's root -- that's" -ForegroundColor Red
    Write-Host "  the exact collision that cost 20+ minutes during testing." -ForegroundColor Red
    Write-Host "  Result: $collisionCheck"
} else {
    Write-Host "  OK: $collisionCheck"
}

# --- 6. Full environment verification ---
Write-Host "`n[6/6] Running full environment verification..." -ForegroundColor Yellow
Write-Host "  (this makes a REAL Azure OpenAI API call -- requires valid values in .env)`n"
python (Join-Path $PSScriptRoot "skills\env-verify\scripts\verify_environment.py")

Write-Host "`n=== Setup complete. Read README.md for the workflow order. ===" -ForegroundColor Cyan
