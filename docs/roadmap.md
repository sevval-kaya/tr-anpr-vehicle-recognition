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
  - [ ] Plaka dedektörü ilk eğitimi (veri hazır, eğitim script'i
    (`scripts/train_detector.py` benzeri) henüz yazılmadı).
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
  - [ ] Baseline pipeline'ın gerçek görüntülerle uçtan uca smoke test'i
    (plaka dedektörü eğitilince mümkün olacak).
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

Aşama 1 tamamlandı. Aşama 2'de: plaka tespiti için veri hazır (Roboflow,
2766/345/347 train/val/test) ama dedektör henüz eğitilmedi; marka/model
sınıflandırıcı için 200 sınıflık Türkiye-odaklı bir baseline GPU'da
eğitildi (val_top1 %28.3). Sırada: plaka dedektörü eğitim script'i
(`scripts/train_detector.py` benzeri, `ultralytics` ile) ve iki modelin
`InferencePipeline` üzerinden gerçek görüntülerle uçtan uca smoke test'i.
