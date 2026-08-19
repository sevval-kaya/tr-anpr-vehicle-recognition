#!/usr/bin/env bash
# One-command local setup (macOS/Linux). See setup.ps1 for the Windows
# PowerShell equivalent, and README.md's "Hızlı Başlangıç" for both
# side by side (docs/decisions.md #45).
set -euo pipefail

MIN_MAJOR=3
MIN_MINOR=11
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Hata: '$PYTHON_BIN' bulunamadı. Python $MIN_MAJOR.$MIN_MINOR+ kurulu olmalı." >&2
    echo "(Farklı bir komutla kuruluysa: PYTHON_BIN=python3.11 ./setup.sh)" >&2
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt "$MIN_MAJOR" ] || { [ "$PYTHON_MAJOR" -eq "$MIN_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$MIN_MINOR" ]; }; then
    echo "Hata: Python $MIN_MAJOR.$MIN_MINOR+ gerekli, bulunan: $PYTHON_VERSION ($PYTHON_BIN)" >&2
    exit 1
fi
echo "Python $PYTHON_VERSION bulundu ($PYTHON_BIN)."

if [ ! -d .venv ]; then
    echo "Sanal ortam oluşturuluyor (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Bağımlılıklar kuruluyor (dev, detection, ocr, serving)..."
pip install --upgrade pip
pip install -e ".[dev,detection,ocr,serving]"

echo "Plaka dedektörü ağırlığı indiriliyor..."
python scripts/download_weights.py

echo "Testler çalıştırılıyor..."
if pytest; then
    echo ""
    echo "Kurulum tamam. Çalıştırmak için:"
    echo "  source .venv/bin/activate"
    echo "  python scripts/run_web.py"
else
    echo ""
    echo "UYARI: Kurulum tamamlandı ama bazı testler geçmedi — yukarıdaki çıktıyı kontrol edin." >&2
    exit 1
fi
