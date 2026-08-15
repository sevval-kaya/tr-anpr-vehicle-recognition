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
  - [x] Plaka dedektörü — **416px baseline'dan 640px'e yeniden eğitildi**:
    aynı `yolo11n`, aynı 1750 görüntülük dengeli alt küme (Roboflow +
    kullanıcı verisi), sadece `image_size` 416→640, `epochs` 35→100
    (patience=10 ile epoch 65'te erken durdu) (bkz. `docs/decisions.md`
    #22). **Sonuç: mAP50 %79.1, mAP50-95 %53.6, precision %88.7, recall
    %61.9→%75.4.** OCR pilotunda (karar #21) tekrarlanan hata sınıfının
    (küçük/düşük-pikselli kırpımlarda karakter kaçırma) çözünürlük
    artışıyla düzelip düzelmeyeceği hipotezi doğrulandı — recall
    iyileşmesi veri hacminden değil tek başına çözünürlükten geldi (veri
    seti değişmedi). Checkpoint `models/plate_detector/best.pt` (5.4MB)
    güncellendi.
    Not: `yolo26n` + tam veri (5.413 görüntü) + 100 epoch ile daha önce bir
    kez mAP50 %82.0 elde edilmişti, ama o checkpoint yeniden eğitim
    denemeleri sırasında silindi ve `cache='ram'` denemesi makinenin 16GB
    RAM'ini aşıp `MemoryError`'a yol açtı (bkz. `docs/decisions.md` #17).
    Tam birleşik veri setiyle 640px'te eğitim henüz denenmedi — çözünürlük
    kazancının üstüne veri hacminin ek katkısı olup olmadığı hâlâ açık.
  - [x] **Kapsam değişikliği (karar #29):** marka/model sınıflandırma
    varsayılan akıştan çıkarıldı (üç ayrı gerçek-veri denemesi
    rastgele-tahmin seviyesini geçemedi — #25, #26, #27), yerine **araç
    tipi (car/motorcycle/bus/truck) + plaka** kondu. Kod silinmedi,
    `configs/pipeline.yaml` → `classification.enabled: false` ile
    tek satırlık geri açma bırakıldı. Aynı kararla
    `scripts/run_inference_video.py`'ye canlı kamera için threaded
    (üretici/tüketici) yakalama eklendi — mekanizması ölçülüp
    doğrulandı (bkz. #29).
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
  - [x] Baseline pipeline'ın (`InferencePipeline`: araç tespiti → plaka
    tespiti → OCR → format doğrulama → marka/model sınıflandırma) gerçek
    görüntülerle uçtan uca smoke test'i — `scripts/run_inference.py`
    yazıldı, 5 gerçek görüntüde çalıştırıldı. İlk çalıştırmada iki gerçek
    boru hattı hatası bulundu ve düzeltildi: sıfır-dolgulu plaka
    kırpımları PaddleOCR'ın dedektörünü tamamen susturuyordu, ve düşük
    mutlak piksel yüksekliğindeki kırpımlar dolgudan bağımsız olarak
    kaçırılıyordu (bkz. `docs/decisions.md` #23). Düzeltmelerden sonra
    5/5 görüntüde geçerli plaka okundu.
  - [x] Plaka **OCR** — `PlateOcr`, `paddleocr` paketi ilk kez gerçekten
    kurulup çalıştırıldığında tamamen bozuk çıktı (3.x'e geçişte kırılan
    API); yeniden yazıldı + bölge-birleştirme mantığı eklendi (bkz.
    `docs/decisions.md` #20). Kullanıcının Claude-vision ile ücretsiz elle
    etiketlediği 42 gerçek plaka üzerinde **ince ayar yapılmadan** ölçülen
    baseline: **tam eşleşme %97.6, CER %0.64**. **Zorlu senaryo (gece/karanlık/
    bulanık/küçük) 10 örnekte de aynı sonuç: tam eşleşme %90, CER %5.4**
    (bkz. `docs/decisions.md` #21) — OCR *tanıma* doğruluğu zaten yeterli;
    tekrar eden tek hata sınıfı (küçük/düşük-çözünürlüklü kırpımlarda
    karakter kaçırma) plaka dedektörünün sorumluluğunda. **Sonuç: OCR ince
    ayarı için veri toplamaya devam edilmeyecek, öncelik plaka dedektörü
    recall'ını (%61.9) iyileştirmeye kaydı.**
- [ ] **3. Veri Toplama ve Etiketleme** — Gerçek Türk trafik/plaka
  görüntülerinin toplanması ve etiketlenmesi.
- [ ] **4. İnce Ayar (Fine-tuning)** — Toplanan veriyle yeniden eğitim,
  Türkiye'de yaygın marka/modellere ağırlık verme.
- [ ] **5. Zorlu Senaryo Testi** — Gece, yağmur, açılı çekim testleri, hata
  analizi. Perspektif düzeltme ihtiyacı burada değerlendirilecek (bkz.
  `docs/decisions.md` #4). **Altyapı hazır** (karar #28-#38): video
  döndürme desteği, saniye-bazlı örnekleme, kare-arası araç izleme +
  plaka/tip konsensüsü, web arayüzü (fotoğraf/video/canlı kamera) — hepsi
  `arac2.mp4`/`arac3.mp4` üzerinde doğrulandı. Sıradaki adım: elde
  yeni bir gerçek trafik videosu (farklı açı/mesafe/ışık) çıkarsa aynı
  altyapıyla hızlı bir tur atıp sürpriz bulgu var mı bakmak.
- [ ] **6. Dağıtım Optimizasyonu** — Deployment hedefi netleştiğinde
  ONNX/TensorRT ile hızlandırma.
- [ ] **7. İzleme ve Sürekli İyileştirme** — Üretim performans izleme,
  periyodik yeniden eğitim.

## Şu an nerede duruyoruz

Aşama 1 tamamlandı. Aşama 2 fiilen tamamlandı: plaka dedektörü (640px'e
yeniden eğitildi, mAP50 %79.1, recall %61.9→%75.4 — karar #22), marka/model
sınıflandırıcı (val_top1 %28.3) ve plaka OCR'ı (ince ayarsız baseline,
kolay sette tam eşleşme %97.6, zorlu sette %90) hem ayrı ayrı hem de
`InferencePipeline` üzerinden uçtan uca gerçek görüntülerle doğrulandı —
bu süreçte iki gerçek boru hattı hatası (crop padding, küçük kırpımlarda
dedektör kaçırması) bulunup düzeltildi (karar #23). **Sıradaki adım:**
Aşama 3 (gerçek Türk trafik verisiyle daha geniş ölçekli toplama/etiketleme)
veya tam birleşik veri setiyle (5.413 görüntü, henüz denenmedi) 640px'te
dedektör yeniden eğitimi — çözünürlük kazancının üstüne veri hacminin ek
katkısı olup olmadığını görmek için.

**Marka/model sınıflandırıcı — kapsam dışına alındı (karar #29):**
VMMRdb kaynaklı ABD-pazarı önyargısı (karar #13) çözülmedi. Üç veri
seti/yöntem denendi, üçü de rastgele-tahmin seviyesini geçemedi: Kaggle
katalog seti (33 marka, stüdyo/render görüntüleri — karar #25, tam
başarısızlık), VMMRdb'nin marka-seviyesine indirgenmiş hali (gerçek
ikinci-el ilan fotoğrafları, kendi alanında iyi ama 6 held-out Türkiye
fotoğrafında 0/6 — karar #26), ve elle etiketlenmiş 80 gerçek Türkiye
fotoğrafı/18 marka (karar #27, 16 held-out'ta 1/16 ve 0/16). **Sonuç:
üçünde de asıl darboğaz aynı — sınıf başına yetersiz görüntü sayısı,
farklı bir mimari/veri kaynağı denemenin getirisi düşük.** Proje kapsamı
bu nedenle marka/model'den **araç tipi (car/motorcycle/bus/truck,
zaten güvenilir) + plaka**'ya kaydırıldı (karar #29); classifier kod
olarak duruyor, `configs/pipeline.yaml`'da tek satırla geri açılabilir,
ama artık varsayılan hedef değil. Etiketleme hacmi ileride büyürse
(Claude-vision, karar #21) yeniden değerlendirilebilir.

**Web arayüzü + video/kamera altyapısı — uçtan uca doğrulandı (karar
#30-#38):** Fotoğraf/video/canlı kamera için FastAPI tabanlı bir web
arayüzü yazıldı (`scripts/run_web.py`), gerçek sunucuda test edildi.
Bu süreçte bulunup düzeltilen gerçek sorunlar: canlı kameranın event-loop
bloklanması yüzünden donması (karar #30), PaddleOCR'ın CPU'da aşırı yavaş
olması (medium→tiny model, ~13x hızlanma, karar #31), plaka dedektörünün
açılı/uzak kameralarda düşük recall'ı (karar #28/#32 — kısmen çözüldü:
`arac2.mp4` fiziksel bir çözünürlük tabanına takıldı, yazılımla
çözülemez; `arac3.mp4` rotasyon hatası düzeltilince gerçek sinyal ortaya
çıktı, karar #33/#34), video döndürme + saniye-bazlı örnekleme desteği
(karar #34), kare-arası araç izleme + plaka/tip konsensüsü (karar #37 —
tek kareye güvenmek yerine aynı aracın birden fazla karedeki okumasını
çoğunluk oyuyla birleştiriyor, "23 ACM 638" gibi gürültülü tek-kare
okumaları tek bir güvenilir sonuca indiriyor).

**Plaka dedektörü ince ayarı denendi, geri alındı (karar #35/#38):** 19
pseudo-label görüntüyle (`arac3.mp4`'ten) hafif ince ayar, hedeflenen
held-out pencerede iyileşti ama tam videoda geçerli okunan araç sayısını
2'den 1'e düşürdü — net kayıp, üretime alınmadı. Bu ölçekte (19 görüntü)
self-training'in riskli olduğu, tam-video doğrulaması olmadan
üretime alınmaması gerektiği kayıt altında. `models/plate_detector/best.pt`
(karar #22) hâlâ üretimdeki checkpoint.
