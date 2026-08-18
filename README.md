# Türk Plaka Tanıma Sistemi (ANPR) + Araç Tipi Tespiti

## Hızlı Başlangıç

**Docker ile:**
```bash
docker compose up --build
```
Ardından tarayıcıda `http://localhost:8000` adresini açın.

**Yerel kurulum ile (macOS/Linux):**
```bash
./setup.sh
```

**Yerel kurulum ile (Windows PowerShell):**
```powershell
# Script engellenirse önce (yönetici gerekmez):
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\setup.ps1
```

İkisi de bittiğinde: `python scripts/run_web.py` ile web arayüzünü
başlatın. Detaylı/manuel kurulum adımları ve tüm CLI komutları için
aşağıdaki bölümlere bakın.

---

Türk trafiğinde araç plakalarını okuyan (yalnızca Türk formatına özel)
ve aynı görüntüden araç tipini (otomobil/motosiklet/otobüs/kamyon)
tespit eden bir bilgisayarlı görü sistemi. Fotoğraf, video dosyası ve
tarayıcıdan canlı kamera girişini destekler.

Kapsam, mimari kararları ve gerekçeleri için bkz. `docs/`.

## Özellikler

- Araç tespiti (COCO-pretrained YOLO, ek eğitim gerekmez)
- Plaka bölgesi tespiti (özel eğitilmiş YOLO checkpoint)
- Plaka OCR (PaddleOCR, Türk plaka karakter setiyle sınırlı)
- Plaka format doğrulama (il kodu + harf/rakam grubu kuralları)
- Video/canlı kamerada kareler arası araç takibi + oylama ile daha
  güvenilir plaka okuma
- Web arayüzü: fotoğraf yükleme, video yükleme, tarayıcıdan canlı kamera

> **Not:** Araç marka/model tanıma denendi ama yeterli/etiketli veri
> olmadığından (bkz. `docs/decisions.md` karar #25-#29) rastgele tahmin
> seviyesini geçemedi ve kapsamdan çıkarıldı; güncel hedef araç tipi +
> plaka. Alt yapı kodda duruyor (`configs/pipeline.yaml` →
> `classification.enabled: false`), ileride yeterli veri toplanırsa tek
> satırlık bir ayarla geri açılabilir.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"      # temel + test araçları

# İhtiyaca göre ağır bağımlılık grupları:
pip install -e ".[detection]"       # ultralytics
pip install -e ".[ocr]"             # paddleocr
pip install -e ".[classification]"  # torch, timm (devre dışı özellik için altyapı)
pip install -e ".[serving]"         # web arayüzü (fastapi, uvicorn)

# Plaka dedektörü checkpoint'i (models/plate_detector/best.pt) git'e
# dahil değil — GitHub Release'den indir (bkz. docs/decisions.md #44):
python scripts/download_weights.py
```

## Kullanım

```bash
# Tek fotoğraf (isteğe bağlı --annotate ile kutulu çıktı outputs/'a yazılır)
python scripts/run_inference.py path/to/image.jpg --annotate

# Video dosyası
python scripts/run_inference_video.py path/to/clip.mp4 --output outputs/annotated.mp4

# Canlı kamera
python scripts/run_inference_video.py 0

# Web arayüzü (fotoğraf/video yükleme + tarayıcıdan canlı kamera)
python scripts/run_web.py
```

### Ortam değişkenleri (opsiyonel, sadece veri hazırlama script'leri için)

Çıkarım/pipeline'ın kendisi herhangi bir API anahtarı gerektirmez.
Sadece veri indirme/etiketleme araçları için gerekir:

| Değişken | Kullanıldığı yer |
|---|---|
| `ROBOFLOW_API_KEY` | `scripts/download_datasets.py` (Roboflow veri seti indirme) |
| `ANTHROPIC_API_KEY` | `scripts/label_plates_with_claude.py` (Claude vision ile yarı otomatik plaka etiketleme, `pip install -e ".[labeling]"` gerekir) |

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
  classification/   marka/model sınıflandırma (devre dışı, bkz. yukarıdaki not)
  validation/       Türk plaka format doğrulayıcı (model gerektirmez)
  pipeline/         şemalar, protokoller, tek-kare orkestrasyon, video/kamera
                     için kareler arası takip, görselleştirme
  web/              FastAPI web arayüzü (foto/video/canlı kamera)
  data/             veri seti düzeni yardımcıları
  evaluation/       metrikler: CER, tam eşleşme, top-k doğruluk
  utils/            logging
configs/            aşama başına YAML config (ağırlık yolları, eşikler, hiperparametreler)
scripts/            CLI giriş noktaları (veri indirme, eğitim, çıkarım, web sunucusu)
tests/unit/         her src modülü için, harici bağımlılık gerektirmeyen testler
tests/integration/  gerçek ağırlık/veri gerektiren testler için (şimdilik boş)
data/               raw/ external/ processed/ — git'e dahil değil, bkz. data/README.md
models/             eğitilmiş checkpoint'ler — git'e dahil değil
```

Detaylı mimari haritası: `docs/architecture.md`
Karar günlüğü (gerekçelerle): `docs/decisions.md`
Yol haritası ve ilerleme durumu: `docs/roadmap.md`

## Durum

Plaka tespiti + OCR + araç tipi tespiti uçtan uca çalışıyor (foto, video,
web arayüzü dahil). Devam eden çalışma: plaka dedektörünün açılı/uzak
kameralardaki recall'ünü iyileştirmek — detaylar `docs/roadmap.md`'de.

## Lisans

`pyproject.toml` şu an `Proprietary` olarak işaretli — kod lisansı
kararı hâlâ netleştirilmeyi bekliyor.

`models/plate_detector/best.pt` checkpoint'inin eğitim verisi (Roboflow,
CC BY 4.0 + `data/external/user_plates/`, kullanıcının internetten
topladığı görüntüler) yayınlanabilir olarak netleşti — bkz.
`docs/decisions.md` #46. **Kullanıcının kendi çektiği test video/
fotoğrafları (`data/external/test_videos/`, `test_foto/`, `speed_eval/`)
hiçbir zaman yayınlanmıyor** — bunlar zaten git'e hiç girmiyor
(`.gitignore`), sadece bu sınır burada açıkça not düşülüyor.
