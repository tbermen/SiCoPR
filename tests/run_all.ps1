# COM Regression Harness
# Run all tests in order. Exits immediately on first failure.
# Usage: powershell -ExecutionPolicy Bypass -File tests/run_all.ps1

$ErrorActionPreference = "Stop"

function Run-Step {
    param([string]$Label, [string]$Command)
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label" -ForegroundColor Red
        exit 1
    }
}

Run-Step "Pre-flight audit" "python audit_stage0.py"
Run-Step "Assemble com.py" "python assemble_com.py"
Run-Step "Interface checks" "python check_interfaces.py"
Run-Step "Unit tests (153 functions)" "pytest com_functions/ -q --tb=short"
Run-Step "Smoke tests" "pytest tests/test_smoke.py -v"
Run-Step "Checkpoint tests (Stage 4)" "pytest tests/test_checkpoints.py -v"
Run-Step "End-to-end tests (Stage 5)" "pytest tests/test_end_to_end.py -v -s"

Write-Host ""
Write-Host "=== All tests passed ===" -ForegroundColor Green
