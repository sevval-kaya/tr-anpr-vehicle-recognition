#Requires -Version 5.1
# One-command local setup (Windows PowerShell). See setup.sh for the
# macOS/Linux bash equivalent, and README.md's "Hızlı Başlangıç" for
# both side by side (docs/decisions.md #45).
#
# If this script itself won't run ("running scripts is disabled on this
# system"), that's PowerShell's execution policy, not this script's
# doing -- run PowerShell as yourself (no admin needed) and allow
# locally-created scripts for your user:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# then re-run: .\setup.ps1

$ErrorActionPreference = "Stop"

# $ErrorActionPreference only governs PowerShell's own (cmdlet) errors --
# a native command like python/pip exiting nonzero does NOT trigger it,
# so every external call below is checked explicitly via this helper.
# Without it, e.g. a failed download_weights.py would be silently
# ignored and the script would carry on straight to `pytest`.
function Invoke-Checked {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Description basarisiz oldu (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}

$MinMajor = 3
$MinMinor = 11

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python bulunamadi. Python $MinMajor.$MinMinor+ kurulu olmali."
    exit 1
}

$versionOutput = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
$parts = $versionOutput -split '\.'
$major = [int]$parts[0]
$minor = [int]$parts[1]

if ($major -lt $MinMajor -or ($major -eq $MinMajor -and $minor -lt $MinMinor)) {
    Write-Error "Python $MinMajor.$MinMinor+ gerekli, bulunan: $versionOutput"
    exit 1
}
Write-Host "Python $versionOutput bulundu."

if (-not (Test-Path ".venv")) {
    Write-Host "Sanal ortam olusturuluyor (.venv)..."
    python -m venv .venv
    Invoke-Checked "Sanal ortam olusturma"
}

& .\.venv\Scripts\Activate.ps1

Write-Host "Bagimliliklar kuruluyor (dev, detection, ocr, serving)..."
python -m pip install --upgrade pip
Invoke-Checked "pip yukseltme"
pip install -e ".[dev,detection,ocr,serving]"
Invoke-Checked "Bagimlilik kurulumu"

Write-Host "Plaka dedektoru agirligi indiriliyor..."
python scripts/download_weights.py
Invoke-Checked "Model agirligi indirme"

Write-Host "Testler calistiriliyor..."
pytest
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Kurulum tamam. Calistirmak icin:"
    Write-Host "  .venv\Scripts\Activate.ps1"
    Write-Host "  python scripts/run_web.py"
} else {
    Write-Host ""
    Write-Warning "Kurulum tamamlandi ama bazi testler gecmedi -- yukaridaki ciktiyi kontrol edin."
    exit 1
}
