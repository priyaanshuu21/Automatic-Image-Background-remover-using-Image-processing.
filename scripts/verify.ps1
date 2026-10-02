# Run the project checks in order and stop on the first failure.
# Usage:  .\scripts\verify.ps1
# Exits with the exit code of the first failing step.

$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Virtual environment not found at $python"
    exit 1
}

Write-Host "== ruff check =="
& $python -m ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ruff failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "== pytest =="
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "pytest failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "All checks passed."
exit 0
