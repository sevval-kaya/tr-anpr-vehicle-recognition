# Yol Haritası

Proje dokümanının 7. bölümündeki aşamalar, bu repodaki somut ilerlemeyle.

- [x] **1. Hazırlık** — Kapsam netleştirme, repo iskeleti, config yapısı,
  test altyapısı. *(bu oturumda tamamlandı)*
- [ ] **2. Temel Model (Baseline)** — Açık kaynak veri setleriyle ilk
  eğitim, uçtan uca pipeline'ın çalışır hale getirilmesi.
  - [x] Orkestrasyon kodu (`InferencePipeline`) ve arayüzler yazıldı, sahte
    (fake) bağımlılıklarla test edildi.
  - [x] Türk plaka format doğrulayıcı tam implementasyon + testler.
  - [x] Değerlendirme metrikleri (CER, tam eşleşme, top-k) implementasyonu.
  - [x] VMMRdb indirildi ve çıkarıldı: `data/external/vmmrdb/` (9.170 sınıf,
    285.086 görüntü — doküman 291.752 diyor, küçük fark kaynağı henüz
    araştırılmadı). Sınıf listesi `data/processed/vmmrdb_classes.txt`e
    yazıldı (`discover_class_names` ile yeniden üretilebilir).
    Türkiye-ilgili marka kapsamı empirik olarak doğrulandı: 1 Renault sınıfı
    (`renault_captur_2015`), 15 Fiat, 0 Togg (2016 sonrası olduğu için
    beklenen), 150 Hyundai — 4.2 bölümündeki dengesizlik riski gerçek.
  - [ ] Stanford Cars: torchvision indiricisi kırık (bkz. `docs/decisions.md`
    #11); Kaggle/HuggingFace mirror'ı gerekiyorsa daha sonra eklenir.
  - [x] Roboflow Türk plaka veri seti indirildi: `data/external/roboflow_plates/`
    (v8, 3.458 görüntü, tek sınıf `license_plate`, YOLO bbox formatı, CC BY
    4.0). İndirme sırasında üç ayrı Windows/OneDrive kaynaklı hata bulunup
    düzeltildi (bkz. `docs/decisions.md` #12).
  - [x] `scripts/prepare_plate_data.py` ile `data/processed/plates/`e
    normalize edildi: 2766/345/347 train/val/test, `data.yaml` yazıldı.
    Plaka dedektörü eğitimine hazır.
  - [ ] Türk plaka veri seti (Kaggle) — CC0 lisanslı, YOLO formatında bbox
    etiketli olduğu sayfa metadata'sından doğrulandı (indirmeden), ama
    henüz indirilmedi; Kaggle kimlik bilgisi kurulumu kullanıcıya bırakıldı.
  - [x] Plaka dedektörü — **gerçek ölçekli baseline eğitildi**:
    `scripts/train_detector.py`, `yolo26n` (Ultralytics'te gerçekten mevcut
    olduğu doğrulandı, bkz. `docs/decisions.md` #7 çözüldü), GPU'da 100
    epoch (~1 saat). **Sonuç: mAP50 %82.0, mAP50-95 %50.6, precision %93.2,
    recall %74.2.** Yol boyunca `data/processed/plates/`teki etiketlerin
    ~%7'sinin bbox/poligon karışımı yüzünden Ultralytics tarafından sessizce
    atıldığı fark edildi ve düzeltildi (bkz. `docs/decisions.md` #15) — düzeltme
    sonrası val'de 0 kayıp. Checkpoint `models/plate_detector/best.pt`,
    `PlateDetector.detect()` ile gerçek test görüntülerinde (tekli ve
    çoklu-plaka sahnelerinde) doğru tespitler verdiği teyit edildi.
  - [x] Marka/model sınıflandırıcı — **gerçek ölçekli baseline eğitildi**:
    `scripts/train_classifier.py --turkey-subset`, 200 sınıf (Türkiye'de
    yaygın 20 marka, `plaka.data.select_target_classes` ile markalar
    arası dengeli round-robin seçim — bkz. `docs/decisions.md` #13),
    ~36.800 görüntü, `efficientnet_b0` (dondurulmuş omurga, sadece
    classifier head eğitiliyor, 256.200 parametre), 160x160 girdi boyutu,
    25 epoch. **Sonuç: val_top1 %28.3, val_top5 %63.2.** Oturum içinde
    NVIDIA RTX 4060 Laptop GPU (8.6GB) keşfedilip CUDA'ya geçildi (bkz.
    `docs/decisions.md` #14); `DataLoader`'a paralel worker eklenince
    epoch süresi ~7-9 dakikadan ~30-37 saniyeye düştü (veri yükleme
    darboğazı GPU'yu %10-48 kullanımda bırakıyordu). Toplam eğitim süresi
    ~17 dakika. Checkpoint `models/vehicle_classifier/` altında,
    `VehicleClassifier.predict()` ile gerçek bir görüntüde doğru tahmin
    verdiği teyit edildi.
  - [ ] Baseline pipeline'ın (`InferencePipeline`: araç tespiti → plaka
    tespiti → OCR → format doğrulama → marka/model sınıflandırma) gerçek
    görüntülerle uçtan uca smoke test'i — artık her iki model de hazır.
  - [ ] Plaka **OCR**'ı henüz eğitilmedi ve eğitilemez durumda: ne
    Roboflow ne Kaggle veri seti plaka metnini (örn. "34 ABC 123")
    etiketliyor, ikisi de yalnızca bbox tespiti için. Şimdilik
    `PlateOcr`, PaddleOCR'ın hazır (bizim eğitmediğimiz) ağırlıklarıyla
    çalışacak; gerçek ince ayar için plaka-metni etiketli veri toplanmalı
    (3. aşama).
- [ ] **3. Veri Toplama ve Etiketleme** — Gerçek Türk trafik/plaka
  görüntülerinin toplanması ve etiketlenmesi.
- [ ] **4. İnce Ayar (Fine-tuning)** — Toplanan veriyle yeniden eğitim,
  Türkiye'de yaygın marka/modellere ağırlık verme.
- [ ] **5. Zorlu Senaryo Testi** — Gece, yağmur, açılı çekim testleri, hata
  analizi. Perspektif düzeltme ihtiyacı burada değerlendirilecek (bkz.
  `docs/decisions.md` #4).
- [ ] **6. Dağıtım Optimizasyonu** — Deployment hedefi netleştiğinde
  ONNX/TensorRT ile hızlandırma.
- [ ] **7. İzleme ve Sürekli İyileştirme** — Üretim performans izleme,
  periyodik yeniden eğitim.

## Şu an nerede duruyoruz

Aşama 1 tamamlandı. Aşama 2'de: hem plaka dedektörü (mAP50 %82.0) hem
marka/model sınıflandırıcı (val_top1 %28.3) GPU'da eğitildi ve ayrı ayrı
gerçek görüntülerle doğrulandı. Sırada: `InferencePipeline` üzerinden
ikisini birlikte uçtan uca test etmek, ve plaka OCR'ı için — hangi kaynakta
olursa olsun — metin-etiketli bir veri kaynağı bulmak/toplamak (şu an
PaddleOCR'ın hazır ağırlıklarıyla çalışıyor, bizim eğittiğimiz bir model
değil).
