# One-command setup for Windows teammates.
# Usage:  .\scripts\setup.ps1
#
# Equivalent for Mac/Linux (single line):
#   conda env create -f environment.yml && conda activate br41n-ssvep && pip install -e .

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating conda env 'br41n-ssvep' from environment.yml ..."
conda env create -f environment.yml

Write-Host "[2/3] Activating env ..."
conda activate br41n-ssvep

Write-Host "[3/3] Installing local package in editable mode ..."
pip install -e .

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  pytest tests/ -q"
Write-Host "  python scripts/run_baseline.py --synthetic"
