# Mimari Kararlar Günlüğü

Bu iskeleti kurarken alınan kararlar ve gerekçeleri. Kullanıcı onayı gereken
kararlar ayrıca işaretlendi; henüz onay bekleyenler için bkz. plan özeti.

## 1. `src/` layout + gruplu opsiyonel bağımlılıklar (pyproject.toml)

**Karar:** Ana paket sadece hafif bağımlılıklarla (`numpy`, `opencv-python-headless`,
`pydantic`, `pyyaml`) kurulabiliyor; `torch`/`timm`/`ultralytics`/`paddleocr`
gibi ağır (ve GPU'ya özel) paketler `[detection]`, `[ocr]`, `[classification]`
opsiyonel gruplarına ayrıldı.

**Gerekçe:** Format doğrulayıcı, metrik hesaplama gibi model gerektirmeyen
modüller kurulum yapmadan test edilebilsin; eğitim makinesinin CUDA sürümü
netleşmeden `torch` sürümünü kilitlemek riskli.

**Etki:** `pip install -e .` ile temel modüller çalışır; her aşama için
`pip install -e '.[detection,ocr,classification]'` gerekir.

## 2. Araç tespiti için sıfır eğitimle COCO-pretrained YOLO

**Karar:** `VehicleDetector`, COCO'nun zaten tanımladığı car/motorcycle/bus/truck
sınıflarını kullanan hazır bir YOLO checkpoint'i sarmalıyor; özel eğitim
gerektirmiyor.

**Gerekçe:** Araç tespiti Türk trafiğine özgü bir problem değil; COCO'nun
genel kapsamı büyük ihtimalle yeterli. Bu, baseline pipeline'ın veri
toplamadan önce uçtan uca çalışır hale gelmesini sağlıyor (dokümandaki 2.
aşama hedefi).

**Risk:** Zor senaryo testinde (gece, açılı çekim) COCO-pretrained recall
yetersiz kalırsa, Türkiye trafiğine özel ince ayar gerekebilir — bu durumda
plaka dedektörüyle aynı eğitim döngüsüne girer.

## 3. Plaka↔araç eşleştirmede IoU değil `containment_ratio`

**Karar:** `BoundingBox.containment_ratio(other)` eklendi: `other` kutusunun
ne kadarının `self` içinde kaldığını döndürür. `InferencePipeline`, doğru
plaka-araç eşleşmesi için IoU yerine bunu kullanıyor.

**Gerekçe:** Plaka kutusu araç kutusuna göre çok küçük; birleşim (union)
alanı araç alanına domine olduğundan, plaka tamamen araç içinde olsa bile
IoU neredeyse sıfıra yakın çıkar (bkz. `tests/unit/test_schemas.py`). IoU
bu soruyu yanlış soruyor; containment doğru soruyu ("plakanın ne kadarı bu
aracın içinde?") doğrudan yanıtlıyor.

## 4. Perspektif düzeltme baseline'da yok, sadece kontrast artırma var

**Karar:** `plaka.ocr.preprocessing.enhance_plate_crop` yalnızca CLAHE
kontrast artırma uyguluyor; dokümanda bahsedilen perspektif düzeltme
uygulanmadı.

**Gerekçe:** Perspektif düzeltmenin doğru çalışması için plakanın 4 köşesi
ya da yönlü (oriented) kutu gerekir; eksen-hizalı (axis-aligned) YOLO
kutusu bunu sağlamıyor. Bunu şimdiden ekleyip yanlış varsayımlarla
karmaşıklık eklemek yerine, 5. aşamadaki (zorlu senaryo/açılı çekim) hata
analizinde gerçekten darboğaz olduğu doğrulanırsa, ya OBB (oriented bounding
box) modeli ya da 4-köşe keypoint modeliyle ele alınacak.

**Onay gerekli mi:** Hayır, ama roadmap stage 5'te tekrar gündeme gelecek.

## 5. mAP yeniden implemente edilmedi

**Karar:** Tespit metrikleri (mAP@0.5, mAP@0.5:0.95) için özel kod
yazılmadı; Ultralytics'in `model.val()` fonksiyonu kullanılacak.

**Gerekçe:** COCO-style mAP (IoU eşleştirme, precision-recall interpolasyonu)
iyi test edilmiş, standart bir hesaplama; yeniden yazmak hata riski taşır ve
doğruluk açısından hiçbir kazanç sağlamaz. CER, tam eşleşme oranı ve top-k
doğruluk gibi OCR/sınıflandırma metrikleri proje-özel oldukları ve harici
kütüphanelerde hazır bulunmadıkları için `plaka.evaluation.metrics` içinde
implemente edildi.

## 6. `InferencePipeline`, somut sınıflar yerine Protocol'lere bağımlı

**Karar:** `plaka.pipeline.protocols` içinde `VehicleDetectorProtocol` vb.
yapısal arayüzler tanımlandı; `InferencePipeline` bunlara bağımlı.

**Gerekçe:** Orkestrasyon mantığını `torch`/`ultralytics`/`paddleocr`
kurulu olmadan, hafif sahte (fake) nesnelerle test edebilmek; ayrıca her
aşamanın arka ucunu (örn. PaddleOCR → özel CRNN) orkestrasyon kodunu
değiştirmeden değiştirebilmek.

## 7. Doküman "YOLO26" öneriyor — koddaki varsayılan YOLO11 olarak bırakıldı

**Karar:** `configs/*.yaml` ve modül docstring'lerinde varsayılan olarak
`yolo11n` kullanıldı; doküman Ocak 2026'da yayımlandığı belirtilen "YOLO26"yı
öneriyor.

**Gerekçe:** YOLO26'nın Ultralytics tarafındaki güncel durumunu
(kararlılık, dokümantasyon, NMS-free iddiaları) bu ortamdan doğrulayamadım.
YOLO11 kanıtlanmış şekilde olgun ve geniş topluluk desteğine sahip. Baseline'ı
doğrulanamamış bir varsayıma kilitlemek yerine, `docs.ultralytics.com`
üzerinden YOLO26'nın gerçek durumu teyit edildikten sonra tek satırlık config
değişikliğiyle geçiş yapılabilir (`weights_path` alanı zaten bunun için
parametrik).

**Onay gerekli mi:** Evet — plan özetinde soruldu.

## 8. Veri seti lisans notu

**Karar:** `scripts/download_datasets.py`, CompCars'ı otomatikleştirmiyor.

**Gerekçe:** VMMRdb ve Stanford Cars programatik olarak indirilebilir
durumda; CompCars ise kurum bazlı manuel lisans talebi gerektiriyor. Üretime
geçmeden önce VMMRdb/Stanford Cars/CompCars/Kaggle Türk plaka veri setinin
lisans şartlarının ticari kullanıma uygunluğu ayrıca doğrulanmalı — bu proje
iskeletinin kapsamı dışında, hukuki/lisans incelemesi gerektirir.

## 9. Sınıf isimlerinin sıralanması

**Karar:** `discover_class_names`, alt klasörleri alfabetik sıralıyor.

**Gerekçe:** `torchvision.datasets.ImageFolder`'ın sınıflara atadığı indeks
sırasıyla birebir eşleşmesi için; böylece eğitimde kullanılan
sınıf-indeks eşlemesi ile çıkarımda okunan `classes.txt` her zaman tutarlı
kalır.
