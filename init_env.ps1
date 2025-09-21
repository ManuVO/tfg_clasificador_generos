#!/usr/bin/env pwsh
# Script: init_env.ps1
# Description: Verifies requirements and activates the virtual environment (.venv) in the current terminal.

Write-Host "[INFO] Checking Python installation..." -ForegroundColor Cyan
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python is not installed or not in PATH. Install it before continuing." -ForegroundColor Red
    return
}

# Project root directory (where this script lives)
$projRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if the virtual environment exists
$venvPath = Join-Path $projRoot ".venv"
if (!(Test-Path $venvPath -PathType Container)) {
    Write-Host "[INFO] Virtual environment not found. Creating it at .venv..." -ForegroundColor Yellow
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create the virtual environment." -ForegroundColor Red
        return
    }
    Write-Host "[OK] Virtual environment created at .venv." -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already present at .venv. Reusing it." -ForegroundColor Green
}

# Path to the PowerShell activation script
$activateScript = Join-Path $venvPath "Scripts\\Activate.ps1"
if (!(Test-Path $activateScript -PathType Leaf)) {
    Write-Host "[ERROR] Activation script not found at $activateScript. Check the virtual environment installation." -ForegroundColor Red
    return
}

Write-Host "[INFO] Activating the virtual environment (.venv)..." -ForegroundColor Cyan
. $activateScript
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[ERROR] Virtual environment activation failed." -ForegroundColor Red
    return
}
Write-Host "[OK] Virtual environment activated. (VIRTUAL_ENV = $env:VIRTUAL_ENV)" -ForegroundColor Green

# Set PYTHONPATH so local modules under src can be imported
$srcPath = Join-Path $projRoot "src"
if (Test-Path $srcPath -PathType Container) {
    $env:PYTHONPATH = $srcPath
    Write-Host "[OK] PYTHONPATH set to $env:PYTHONPATH to expose project modules." -ForegroundColor Green
} else {
    Write-Host "[WARN] 'src' directory not found. Skipping PYTHONPATH configuration." -ForegroundColor Yellow
}

# Upgrade pip and install dependencies if requirements.txt is present
if (Test-Path "$projRoot\\requirements.txt" -PathType Leaf) {
    Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    & python -m pip install --upgrade pip
    & python -m pip install -r "$projRoot\\requirements.txt"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Issues occurred while installing dependencies. Review the log above." -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Dependencies installed or updated successfully." -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] requirements.txt not found at project root. Skipping dependency installation." -ForegroundColor Yellow
}

# Check for the configuration file (config.yaml or config.jang)
$configFound = $false
foreach ($cfgName in @("config.yaml", "config.jang")) {
    $cfgPath = Join-Path $projRoot $cfgName
    if (Test-Path $cfgPath -PathType Leaf) {
        Write-Host "[OK] Configuration file found: $cfgName." -ForegroundColor Green
        $configFound = $true
        break
    }
}
if (-not $configFound) {
    Write-Host "[WARN] Configuration file 'config.yaml' not found. Confirm the name or location." -ForegroundColor Yellow
}

# Test importing internal modules from src
Write-Host "[INFO] Verifying internal module imports..." -ForegroundColor Cyan
$modulesToTest = @(
    "features.melspectrogram",
    "features.augment",
    "data.dataset"
)
$importErrors = @()
foreach ($mod in $modulesToTest) {
    try {
        & python -c "import $mod"
    } catch {
        $importErrors += $mod
    }
}
if ($importErrors.Count -gt 0) {
    Write-Host "[WARN] Could not import the following modules: $($importErrors -join ', ')." -ForegroundColor Yellow
    Write-Host "       Check PYTHONPATH or install the project package in the environment." -ForegroundColor DarkYellow
} else {
    Write-Host "[OK] Internal modules imported successfully." -ForegroundColor Green
}

Write-Host ""
Write-Host "[DONE] Environment ready. You are now inside '.venv'." -ForegroundColor Green
Write-Host "       Close this terminal or run 'deactivate' to exit the virtual environment." -ForegroundColor Green
