# Setup test environment for LeRobot
# Run from repo root: .\scripts\setup_test_env.ps1

$ErrorActionPreference = "Continue"  # Don't stop on warnings

Write-Host "=== LeRobot Test Environment Setup ===" -ForegroundColor Cyan

# 1. Create venv
$venvPath = ".venv-lerobot-test"
if (Test-Path $venvPath) {
    Write-Host "Removing existing $venvPath..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv $venvPath

# 2. Activate and upgrade pip
& "$venvPath\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# 3. Install lerobot with test extras (ignore warnings)
Write-Host "Installing lerobot with test dependencies (this may take several minutes)..." -ForegroundColor Green
$env:PYTHONWARNINGS = "ignore"
pip install -e ".[test]" 2>&1 | Out-Host

if ($LASTEXITCODE -ne 0) {
    Write-Host "Install failed. Trying minimal install..." -ForegroundColor Yellow
    pip install -e . 2>&1 | Out-Host
    pip install pytest pytest-timeout pytest-cov 2>&1 | Out-Host
}

# 4. Verify
Write-Host "`nVerifying installation..." -ForegroundColor Green
python -c "import lerobot; print('lerobot:', lerobot.__version__)"
python -c "from lerobot.robots.brewie import BrewieConfig, BrewieBase; print('Brewie: OK')"

Write-Host "`n=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Activate: .\.venv-lerobot-test\Scripts\Activate.ps1"
Write-Host "Run tests: pytest tests -v --maxfail=5"
