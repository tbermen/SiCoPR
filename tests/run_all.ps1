# COM Regression Harness
# Run all tests in order. Exits immediately on first failure.
# Usage: powershell -ExecutionPolicy Bypass -File tests/run_all.ps1
#
# Run from the repository root.

$ErrorActionPreference = "Stop"

# Resolve the repo root from this script's location so the harness works
# regardless of the caller's working directory. The audit/interface scripts live
# in dev/; this script used to invoke them at the repo root, which broke the
# harness at step 1 once they were moved.
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root

function Run-Step {
    param([string]$Label, [string]$Command)
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label" -ForegroundColor Red
        Pop-Location
        exit 1
    }
}

try {
    Run-Step "Pre-flight audit"        "python dev/audit_stage0.py"
    Run-Step "Assemble com.py"         "python assemble_com.py"
    Run-Step "Interface checks"        "python dev/check_interfaces.py"
    Run-Step "Unit tests (157 functions)" "python -m pytest com_functions/fn -q --tb=short"

    # tests/ holds two kinds of file. The three below are pytest modules; the
    # rest are standalone audit scripts that must be run directly -- pointing
    # pytest at the whole directory collects the audit scripts, hits their
    # module-level sys.exit, and reports "no tests ran" while exiting 0.
    Run-Step "Smoke tests"             "python -m pytest tests/test_smoke.py -v"
    Run-Step "Checkpoint tests (Stage 4)" "python -m pytest tests/test_checkpoints.py -v"
    Run-Step "End-to-end tests (Stage 5)" "python -m pytest tests/test_end_to_end.py -v -s"

    Write-Host ""
    Write-Host "=== Audit scripts ===" -ForegroundColor Cyan
    $failed = @()
    Get-ChildItem "$Root\tests\test_*.py" | Where-Object {
        $_.Name -notin @('test_smoke.py', 'test_checkpoints.py', 'test_end_to_end.py')
    } | ForEach-Object {
        Write-Host "--- $($_.Name)"
        python $_.FullName
        if ($LASTEXITCODE -ne 0) { $failed += $_.Name }
    }
    if ($failed.Count -gt 0) {
        Write-Host ""
        Write-Host "FAILED audit scripts: $($failed -join ', ')" -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Write-Host ""
    Write-Host "=== All tests passed ===" -ForegroundColor Green
}
finally {
    Pop-Location
}
