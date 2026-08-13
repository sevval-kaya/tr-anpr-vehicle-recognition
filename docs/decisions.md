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

## 7. Doküman "YOLO26" öneriyor — [ÇÖZÜLDÜ] doğrulandı, gerçekten var

**Karar (güncellendi):** `configs/detection.yaml`'da plaka dedektörü artık
`yolo26n` kullanıyor. İlk kararda (aşağıda korunuyor, tarihsel referans için)
YOLO11'de kalınmıştı çünkü YOLO26'nın varlığı doğrulanamamıştı.

**Güncelleme gerekçesi:** Plaka dedektörü eğitim script'i yazılırken
kurulu `ultralytics` paketi (8.4.118) incelendi:
`ultralytics/cfg/models/26/yolo26*.yaml` dosyaları ve `default.yaml`'daki
`end2end` (YOLO26/YOLOv10 için NMS-free head) alanı YOLO26'nın gerçekten
mevcut ve pakete gömülü olduğunu doğruluyor — dokümanın iddiası doğruymuş.
Artık `yolo26n` kullanılıyor.

---

*Orijinal karar (12 Ağustos 2026, doğrulanamadığı için):* `configs/*.yaml`
ve modül docstring'lerinde varsayılan olarak `yolo11n` kullanıldı; doküman
Ocak 2026'da yayımlandığı belirtilen "YOLO26"yı öneriyor. YOLO26'nın
Ultralytics tarafındaki güncel durumunu (kararlılık, dokümantasyon,
NMS-free iddiaları) o ortamdan doğrulayamadım. YOLO11 kanıtlanmış şekilde
olgun ve geniş topluluk desteğine sahip. Baseline'ı doğrulanamamış bir
varsayıma kilitlemek yerine, gerçek durum teyit edildikten sonra tek
satırlık config değişikliğiyle geçiş yapılabilir denmişti — nitekim öyle
oldu.

## 12. Roboflow indirmesinde Windows'a özgü üç ayrı hata bulundu ve düzeltildi

Roboflow Türk plaka veri setini (v8, 3.458 görüntü) indirirken art arda üç
farklı hatayla karşılaşıldı; hepsi Windows + OneDrive-senkronize derin
klasör yapısının bir sonucu, kodun mantığında değil:

1. **`overwrite=False` sessiz no-op:** `download_turkish_plates_roboflow`,
   roboflow'u çağırmadan önce `destination.mkdir(...)` çağırıyordu.
   roboflow'un `Version.download()` metodu, hedef klasör zaten varsa
   (`overwrite=True` verilmedikçe) hiçbir şey indirmeden sessizce
   `Dataset` nesnesi döndürüyor (`roboflow/core/version.py:301`). Sonuç:
   komut "başarılı" görünüyordu ama klasör boş kalıyordu. **Düzeltme:**
   `mkdir` çağrısı kaldırıldı, `overwrite=True` geçildi.
2. **roboflow'un kendi zip extraction'ı MAX_PATH'e takılıyor:** Roboflow
   görüntü dosya adları kaynak Instagram/sosyal medya başlıklarından
   türetildiği için 100+ karakter olabiliyor; OneDrive içindeki derin proje
   yoluyla birleşince Windows'un ~260 karakter MAX_PATH sınırını aşıyor,
   `zipfile.extract()` içeride `FileNotFoundError` fırlatıyor. **Düzeltme:**
   `_extract_zip_windows_long_path_safe` eklendi — hedef yolu `\\?\` öneki
   ile genişletilmiş-uzunluk Win32 API'sine yönlendiriyor (sistem geneli
   "uzun yol" ayarı/admin yetkisi gerektirmez). roboflow'un indirdiği zip
   dosyası duruyor, sadece extraction bizim tarafımızdan tekrarlanıyor.
3. **Aynı MAX_PATH sorunu `materialize_split`'te de çıktı:** Bu kez
   `plaka.data.yolo_dataset.materialize_split`'in kendi `shutil.copy2`
   çağrısında. **Düzeltme:** Aynı `\\?\` önek tekniği `_long_path()` olarak
   genelleştirildi, hem kaynak hem hedef yola uygulanıyor.

**Gerekçe (neden hemen fark edilmedi):** Her üç hata da "sessiz başarı"
şeklinde tezahür etti — komut sıfır çıkış koduyla bitiyor ama klasör boş
kalıyordu (1. hata) ya da kısmi/traceback'li ama "arka planda çalıştı,
tamamlandı" bildirimiyle karışıyordu (2. ve 3. hata arka plan görev takibi
üzerinden ilk denemelerde gözden kaçtı). Doğrulama adımı olarak her indirme
sonrası dosya sayısını diskte fiilen saymak (`find ... | wc -l`) zorunlu
hale getirildi — sadece exit code'a güvenmek yeterli değil.

**Etki:** `data/external/roboflow_plates/` (3.458 görüntü) ve
`data/processed/plates/` (2766/345/347 train/val/test) artık doğru ve
tam durumda.

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

## 10. VMMRdb kaynak adresi ve dağıtım şekli düzeltildi

**Karar:** `scripts/download_datasets.py`, VMMRdb'yi `github.com/faezetta/VMMRdb`
üzerinden barındırılan **~11.5GB'lık tek bir Dropbox zip dosyası** olarak
indiriyor (git clone değil, doğrudan HTTP indirme + extract).

**Gerekçe:** Proje dokümanının referans listesindeki `github.com/lgov/VMMRdb`
adresi mevcut değil (404). Doğru repo GitHub arama API'siyle bulundu
(`faezetta/VMMRdb`); repo'nun kendisi sadece kod/metadata içeriyor, asıl
291.752 görüntü Dropbox'ta tek zip olarak barındırılıyor. Bu nedenle ilk
yazdığım `git clone` tabanlı indirici çalışmıyordu — HTTP streaming
indirmeyle değiştirildi.

**Not:** Dropbox linkleri kişisel hesaplara bağlı barındırma olduğundan
gelecekte kaldırılma riski taşır; üretim öncesi veri setinin kalıcı bir
kopyasının kurum içi depolamaya (S3/GCS vb.) alınması önerilir.

## 11. Stanford Cars (torchvision) kırık, kullanılmıyor

**Karar:** `download_stanford_cars` fonksiyonu kodda duruyor ama şu an
kullanılmıyor; VMMRdb ile devam edildi.

**Gerekçe:** torchvision'ın `StanfordCars.download()` metodu, orijinal
Stanford host'unun çevrimdışı olması nedeniyle koşulsuz `ValueError`
fırlatıyor (torchvision'ın kendi bilinen sorunu). Kaggle mirror'ı
(`jessicali9530/stanford-cars-dataset`) veya HuggingFace mirror'ları
alternatif olarak kullanılabilir, ama kullanıcı şu an için Kaggle kimlik
bilgisi kurmayı ertelemeyi tercih etti.

## 13. Türkiye-odaklı sınıf alt kümesi: round-robin seçim, VMMRdb dengesizliğine karşı

**Karar:** Tam VMMRdb (9.170 sınıf) yerine, `plaka.data.select_target_classes`
ile Türkiye'de yaygın 20 markadan (bkz. `configs/classification.yaml`
`target_makes_subset`) en fazla 200 sınıf seçiliyor. Seçim, her markadan
sırayla bir sınıf alan round-robin algoritmasıyla yapılıyor (marka içinde
görüntü sayısına göre azalan sırayla); basit "en çok görüntüsü olan N
sınıfı al" yaklaşımı kullanılmadı.

**Gerekçe:** VMMRdb marka dağılımı son derece dengesiz — empirik olarak
ölçüldü: Ford 870 sınıf, Toyota 584, BMW 442, Mercedes-Benz 474 (not: veri
setinde "mercedes benz" boşlukla ayrılmış, alt çizgi değil — ayrı bir
parsing kuralı gerekti) varken **Renault sadece 1 sınıf, Opel 4, Citroën 2,
Peugeot 2**; Dacia/Skoda/SEAT ise **hiç yok** (VMMRdb ABD pazarına göre
toplanmış). Basit bir "top-N by image count" seçimi bu nedenle neredeyse
tamamen Ford/Toyota/BMW/Mercedes'ten oluşurdu ve Renault gibi Türkiye'de
kritik markaları büyük ihtimalle tamamen dışarıda bırakırdı. Round-robin,
her markanın (veri kısıtı ölçüsünde) temsil edilmesini garanti ediyor.

**Bilinen sınırlama:** Renault yalnızca 1 sınıfla (`renault_captur_2015`)
temsil ediliyor — sınıflandırıcı gerçekte diğer Renault modellerini hiç
görmeden eğitiliyor. Bu, VMMRdb'nin temel bir kapsam boşluğu; 4. aşamadaki
(ince ayar) gerçek Türkiye verisi toplanmadan düzelmeyecek (bkz. proje
dokümanı 4.2, "Türkiye'de yaygın marka/modeller için ek görüntü toplanmalı").

## 14. GPU keşfedildi, eğitim CPU'dan CUDA'ya taşındı

**Karar:** Kullanıcının "GPU varsa kullan" talimatı üzerine `nvidia-smi` ile
donanım kontrol edildi: **NVIDIA GeForce RTX 4060 Laptop GPU, 8.6GB VRAM**,
boşta durumda bulundu. CPU-only kurulu `torch`/`torchvision` kaldırılıp
sürücünün desteklediği CUDA 13.3'e uygun `torch==2.13.0+cu130` /
`torchvision==0.28.0+cu130` kuruldu (`--index-url
https://download.pytorch.org/whl/cu130`). `configs/classification.yaml`
`device: cuda` olarak güncellendi.

**Gerekçe:** Aynı 20 sınıf/444 görüntülük ölçüm testinde GPU, CPU'ya göre
~2.5x daha hızlı çıktı (küçük veri setinde sabit maliyetler — kernel
başlatma, Python döngü ek yükü — baskın olduğundan bu oran alt sınır;
36.833 görüntülük tam koşuda fark daha belirgin olması bekleniyor). Bu
sayede `--max-images-per-class` sınırlaması kaldırılıp 200 sınıfın **tüm**
görüntüleriyle (~36.833 görüntü, CPU planındaki 50/sınıf sınırından çok
daha fazla) 25 epoch'luk tam bir koşu çalıştırıldı.

**Not:** `freeze_backbone: true` korundu — GPU bu kısıtı gereksiz kılmıyor,
sadece hızlandırıyor; dondurulmuş omurga hâlâ makul bir baseline stratejisi
(daha az overfitting riski, daha hızlı yakınsama). Tam ince ayar (omurga
dahil) GPU ile artık mümkün, ama bu oturumun kapsamı dışında bırakıldı —
gelecekte `--no-freeze-backbone` ile denenebilir.

## 15. Roboflow etiketlerinde bbox/poligon karışımı — `materialize_split` normalize ediyor

**Karar:** `plaka.data.yolo_dataset.normalize_yolo_label_text`, her etiket
satırını (5 değer → zaten bbox, geçer; >5 değer → poligon, sınırlayıcı
dikdörtgene çevrilir) düz `class x_center y_center width height` formatına
dönüştürüyor. `materialize_split`, artık etiket dosyalarını ham kopyalamak
yerine bu normalizasyondan geçiriyor.

**Gerekçe:** Plaka dedektörü eğitim script'ini ilk kez çalıştırırken
Ultralytics'in `train: ... 2766 images, 39 backgrounds, 192 corrupt`
uyarısı fark edildi — `data/processed/plates/`teki etiket dosyalarının bir
kısmı (train'de %6.9, val'de %8.1) hem düz bbox hem poligon satırı
içeriyordu (muhtemelen Roboflow'da bazı örnekler bbox, bazıları poligon
aracıyla etiketlenmiş). Ultralytics böyle "karışık" dosyaları resmi olarak
desteklemiyor ve **tüm görüntüyü sessizce atıyor** — hata vermeden ~%7
veri kaybı anlamına geliyordu. Poligonu sınırlayıcı dikdörtgenine
çevirmek (görev zaten object detection, segmentation değil) bu görüntüleri
kayıpsız geri kazandırıyor.

**Etki:** `scripts/prepare_plate_data.py` yeniden çalıştırıldı,
`data/processed/plates/` yeniden üretildi — artık `0 corrupt` bekleniyor.

## 16. Kullanıcının plaka veri seti eklendi + eğitim hızlandırıldı (batch, cache)

**Karar:** Kullanıcının sağladığı 1.955 gerçek Türk plakası fotoğrafı
(`data/external/user_plates/`, zaten temiz bbox formatında) Roboflow
setiyle birleştirildi (`prepare_plate_data.py data/external/roboflow_plates
data/external/user_plates` → 5.413 görüntü, 4330/541/542 train/val/test).
Ayrıca eğitim, doğruluktan ödün vermeyen iki değişiklikle hızlandırıldı:
`batch_size` 16→64, `cache: ram` eklendi.

**Gerekçe:** İlk 100-epoch koşusunda (5.413 görüntülük yeni veriyle
başlamadan önce, 3.458 görüntülük Roboflow-only sette) GPU VRAM kullanımı
sadece ~3/8GB'ta kalıyordu (batch=16 ile GPU yeterince doldurulmuyordu) ve
her epoch kaynak görüntüleri (bazıları 4608x2592 gibi yüksek çözünürlüklü)
diskten yeniden okuyup yeniden decode ediyordu. `cache=ram`, görüntüleri
bir kere decode/resize edip bellekte tutuyor. Epoch sayısını azaltmak
**tercih edilmedi** — ilk koşuda mAP50 epoch 50'den (%77.1) epoch 100'e
(%82.0) kadar gerçek anlamda iyileşmeye devam etti, yani 100 epoch
gereksiz değildi.

**Kullanıcı veri seti hakkında not:** Kaynağı/lisansı kullanıcı tarafından
belirtilmedi ("benim eklediğim/indirdiğim"); harici paylaşım/yayın öncesi
bu netleştirilmeli (bkz. `data/README.md`).

## 17. `cache="ram"` Windows'ta `MemoryError` ile çöktü — `cache="disk"`e geçildi

**Karar:** `configs/detection.yaml`'da `cache: "ram"` yerine `cache: "disk"`
kullanılıyor.

**Gerekçe:** #16'daki `cache="ram"` denemesi, val seti önbelleklemesi
bittikten hemen sonra worker süreçleri başlatılırken şu hatayla çöktü:

```
File "...multiprocessing\spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
MemoryError
```

Sebep Windows'a özgü: `multiprocessing` Windows'ta `fork()` değil `spawn()`
kullanıyor, yani her worker süreci ana süreçten **kopya değil, sıfırdan bir
Python yorumlayıcısı** olarak başlıyor ve ana sürecin durumunu (RAM'deki
tüm görüntü önbelleği dahil) pickle ile yeniden inşa ediyor. `workers=8` ile
~5.5GB'lık önbellek 8 kez pickle'lanmaya çalışılınca bellek taştı. Linux'ta
`fork()` belleği copy-on-write paylaştığı için bu sorun oluşmaz — tamamen
Windows'a özgü bir tuzak. Ultralytics'in kendi uyarısı da zaten
`cache='disk'`i "deterministic alternative" olarak öneriyordu; `disk`
modu her worker'ın kendi başına, önbelleği pickle'lamadan diskteki
yeniden-boyutlandırılmış dosyaları okumasını sağladığı için bu sorunu
yaşamıyor.

## 18. Yerel makineye göre hafifletilmiş baseline (nano model, 416px, alt küme, az epoch)

**Karar:** `configs/detection.yaml`: `yolo26n`→`yolo11n`, `input_size`/`image_size`
640→416, `epochs` 100→35, `patience` 20→10, `batch_size` 64→16, `workers` 8→3.
`plaka.data.sample_balanced_subset` eklendi:
`prepare_plate_data.py ... --max-examples 1750` ile 5.413 görüntüden
kaynaklar arası dengeli (Roboflow ~875, kullanıcı verisi ~875) 1750'lik bir
alt küme türetildi (1400/175/175 train/val/test).

**Gerekçe:** Önceki denemeler (batch=64, cache=ram) 16GB RAM'lik makinede
`MemoryError`'a yol açtı ve sistemi kullanılamaz hale getirdi (kullanıcı
"bilgisayarım çok yavaşladı" dedi). Kullanıcı, doğruluktan bilinçli olarak
ödün verip hız + sistem kullanılabilirliğini önceliklendirmeyi tercih etti:
tam ölçekli (mAP50 %82) sonuç yerine, pipeline'ı uçtan uca çalışır hale
getirecek daha mütevazı bir baseline yeterli; veri/model/epoch büyütme
daha uygun bir zamanda (veya daha güçlü donanımda) tekrar ele alınabilir.

**`sample_balanced_subset` tasarımı:** Kaynak boyutuyla orantılı değil,
kaynaklar arası **eşit** paylaşım yapıyor (`select_target_classes`'daki
round-robin mantığıyla aynı ilke — bkz. karar #13); bir kaynak payını
dolduramazsa (örn. çok küçükse) açık kalan kotayı diğer kaynaklara
devrediyor, kaybetmiyor.

**İşlem önceliği:** Eğitim süreci Windows'ta "Below Normal" önceliğiyle
başlatıldı — makine eğitim sürerken kullanılabilir kalsın diye.
