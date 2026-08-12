# Türk Plaka + Araç Marka/Model Tanıma Sistemi

Türk trafiğinde araç plakalarını okuyan (yalnızca Türk formatına özel) ve
aynı görüntüden aracın marka/modelini geniş bir kapsamda (dünya
genelindeki başlıca üreticiler dahil) tanıyan, üretime hazır bir
bilgisayarlı görü sistemi.

Kapsam, mimari kararları ve gerekçeleri için bkz. `docs/`. Kaynak proje
dokümanı bu repoyu başlatan brief'tir.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"      # temel + test araçları

# İhtiyaca göre ağır bağımlılık grupları:
pip install -e ".[detection]"       # ultralytics
pip install -e ".[ocr]"             # paddleocr
pip install -e ".[classification]"  # torch, timm
```

## Test

```bash
pytest
```

Şu anda tüm testler model ağırlığı veya veri seti gerektirmeden çalışır
(`tests/unit/`); model/veri gerektiren testler `tests/integration/`
altında ayrı tutulacak.

## Repo yapısı

```
src/plaka/
  detection/       araç + plaka tespiti (YOLO sarmalayıcıları)
  ocr/              plaka OCR + ön işleme
  classification/   marka/model sınıflandırma (timm sarmalayıcısı)
  validation/       Türk plaka format doğrulayıcı (model gerektirmez)
  pipeline/         şemalar, protokoller, tek-kare orkestrasyon
  data/             veri seti düzeni yardımcıları
  evaluation/       metrikler: CER, tam eşleşme, top-k doğruluk
  utils/            logging
configs/            aşama başına YAML config (ağırlık yolları, eşikler, hiperparametreler)
scripts/            CLI giriş noktaları (veri indirme, eğitim, çıkarım)
tests/unit/         her src modülü için, harici bağımlılık gerektirmeyen testler
tests/integration/  gerçek ağırlık/veri gerektiren testler için (şimdilik boş)
data/               raw/ external/ processed/ — git'e dahil değil, bkz. data/README.md
models/             eğitilmiş checkpoint'ler — git'e dahil değil
```

Detaylı mimari haritası: `docs/architecture.md`
Karar günlüğü (gerekçelerle): `docs/decisions.md`
Yol haritası ve ilerleme durumu: `docs/roadmap.md`

## Durum

Aşama 1 (Hazırlık) tamamlandı. Aşama 2'nin (Baseline) kod tarafı hazır;
veri indirme ve ilk eğitim, büyük kararlarda kullanıcı onayı bekliyor.
