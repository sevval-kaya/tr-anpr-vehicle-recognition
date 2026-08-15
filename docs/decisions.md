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

## 19. Claude vision ile yarı-otomatik OCR etiketleme + `TurkishPlateValidator` grup tablosu düzeltmesi

**Karar:** `data/external/user_plates/`teki bbox-only görüntüleri OCR
eğitimi için metin-etiketlemek amacıyla `scripts/label_plates_with_claude.py`
yazıldı: mevcut YOLO bbox'ıyla plakayı kırpar (`plaka.data.plate_crop`),
kırpımı Claude API'ye (vision, `claude-opus-5`, düşük effort/thinking
kapalı — basit transkripsiyon işi) gönderip metni okutur
(`plaka.ocr.claude_labeler`), `TurkishPlateValidator` ile doğrular.
Kullanıcı API maliyetinden kaçınmak için pilot etiketlemeyi (42 görüntü)
API yerine kendi Claude arayüzünü elle kullanarak ücretsiz yaptı; script
API anahtarı sağlandığında daha büyük ölçekte aynı işi otomatik yapabilir
durumda bekliyor.

**Yan bulgu — doğrulayıcı hatası düzeltildi:** Bu 42 örneği doğrularken 10
tanesi (`66 AAP 914`, `19 ABT 122` dahil) "invalid_group_combination" ile
reddedildi. Kırpım görüntülerini doğrudan inceleyince (Read tool ile) bu
plakaların gerçek, doğru okunmuş plakalar olduğu görüldü — sorun OCR'da
değil, `plate_format.py`'deki `VALID_GROUP_COMBINATIONS` tablosunda: 3
harfli gruplar için sadece 2 rakam izin veriliyordu, 3 rakam
(`3: frozenset({2})`) hiç yoktu. Gerçek TR plaka formatında 3 harf + 3
rakam da geçerli bir kombinasyon; tablo `3: frozenset({2, 3})` olarak
düzeltildi, `tests/unit/test_plate_format.py` buna göre güncellendi.

**Gerekçe:** Kullanıcının isteği üzerine etiketleme adımına
`TurkishPlateValidator` doğrulaması dahil edildi — bu, sadece etiketleri
filtrelemekle kalmadı, doğrulayıcının kendisindeki bir kapsam boşluğunu da
ortaya çıkardı. Düzeltilmeden bırakılsaydı, gerçekte doğru okunmuş
plakalar sessizce "geçersiz" sayılıp hem bu pilotta hem üretimde
reddedilecekti.

## 20. `PlateOcr`, `paddleocr` 3.x'e göre yeniden yazıldı (ilk gerçek kurulum + çalıştırma)

**Karar:** `plaka.ocr.plate_ocr.PlateOcr`, `paddleocr` paketini ilk kez
gerçekten kurup çalıştırınca (bu pilot değerlendirmesi sırasında)
tamamen bozuk çıktı — `PaddleOCR.ocr(image, cls=False)` API'si paddleocr
3.x'te kaldırılmış (`predict()` ile değişmiş, `cls` argümanı yok).
`_ensure_model_loaded`/`read()` `predict()`e ve onun sözlük-tabanlı sonuç
şekline (`rec_texts`/`rec_scores`/`rec_boxes`) göre yeniden yazıldı; ayrıca
bu makinede CPU çıkarımı sırasında oneDNN/PIR kaynaklı bir
`NotImplementedError` ile çöktüğü görüldü, `enable_mkldnn=False` ile
atlatıldı.

**Yan değişiklik — en büyük bölgeyi seçme:** `predict()`, amaçlı kırpılmış
bir plaka görüntüsünde bile plaka dışı metinleri (bayi çerçevesi markası,
il adı, "TR" AB şeridi) ayrı bölgeler olarak tespit edebiliyor. Eski kod
tüm parçaları soldan sağa birleştiriyordu — bu artık plaka dışı metni de
sonuca karıştırırdı. `read()` artık `rec_boxes`'tan alan hesaplayıp en
büyük bölgeyi seçiyor (plakanın crop içindeki en baskın metin olduğu
varsayımıyla), birleştirmiyor.

**Gerekçe:** `pyproject.toml`deki `ocr` opsiyonel grubu (`paddleocr>=2.8`)
proje boyunca hiç kurulup test edilmemişti (dokümanın kendisi de roadmap'te
"PaddleOCR'ın hazır ağırlıklarıyla çalışacak, ama henüz doğrulanmadı"
diyordu). Bu pilot, `PlateOcr`'ı ilk kez gerçek bir kurulumla çalıştırma
fırsatı oldu ve kod, o zamandan beri değişmiş bir üçüncü parti API'ye göre
yazılmış olduğu için baştan çalışmıyordu.

**Not:** `pyproject.toml`deki alt sınır `paddleocr>=3.0` olarak sıkılaştırıldı
— yeni `read()` implementasyonu 2.x'in `.ocr()` API'siyle çalışmaz, sadece
3.x'in `predict()`'iyle çalışır.

**Bölge birleştirme algoritması (`_select_plate_text`):** `predict()`,
plaka satırını bazen tek bölge (örn. "34 ABD 987"), bazen birden fazla
bölgeye bölünmüş olarak (örn. il kodu "38" ile geri kalanı "PD369" ayrı
tespit edilmiş) döndürüyor; ayrıca "TR" rozeti ve bayi/il çerçeve yazıları
da ayrı bölgeler olarak geliyor. Üç adımlı bir sezgisel kural yazıldı: (1)
en büyük alanlı bölge çapa (anchor) alınır, (2) çapayla aynı satırda
olduğu tespit edilen (dikey merkezi çapanın y-aralığında + yüksekliği
çapanın en az yarısı kadar + güven skoru eşik üstü) diğer bölgeler soldan
sağa birleştirilir, (3) harfi harfine "TR" olan bölge (yükseklik/güven
kontrollerini geçse bile) hariç tutulur. `tests/unit/test_plate_ocr.py`,
bu pilot sırasında karşılaşılan gerçek başarısızlık örneklerini (satır
bölünmesi, kısa çapa üzerinde küçük punto bayi yazısı, düşük güvenli
logo/amblem yanlış okuması) birebir regresyon testi olarak kullanıyor.

**42 örnek üzerinde ölçülen sonuç (ince ayar YAPILMADAN, sadece hazır
PP-OCRv6 ağırlıkları + yukarıdaki düzeltmeler):** tam eşleşme %97.6
(41/42), CER %0.64, `TurkishPlateValidator`den geçen 41/42. Tek kalan hata
(`s_197.jpg`, "38" hiç tespit edilmedi) bir seçim/birleştirme sorunu değil,
gerçek bir dedektör-recall kaçırması — sonrasında düzeltilemedi.

## 21. PaddleOCR ince ayarı şimdilik yapılmadı — veri hacmi + altyapı maliyeti gerekçesiyle

**Karar:** Kullanıcının 42 örnekle (Claude vision, ücretsiz, elle) istediği
"OCR ince ayar denemesi" adımı, gerçek bir derin öğrenme fine-tune'u olarak
çalıştırılmadı. Bunun yerine #20'deki iki gerçek hata (kırık API,
zayıf bölge birleştirme) düzeltilip aynı 42 örnek üzerinde **ince ayarsız**
baseline ölçüldü (%97.6 tam eşleşme).

**Gerekçe:**
1. **Altyapı maliyeti:** `paddleocr` pip paketi (3.x) sadece çıkarım
   (inference) içeriyor; gerçek fine-tune için PaddleX'in ayrı eğitim
   eklentisinin kurulması gerekiyor (`paddlex --install PaddleOCR` —
   büyük bir eğitim deposu klonlanıyor, önceden eğitilmiş checkpoint
   indiriliyor, PaddleX'in kendi config/label formatına veri dönüştürmek
   gerekiyor). Bu, sadece "işe yarıyor mu?" sorusuna cevap aramak için
   orantısız bir altyapı yatırımı.
2. **Veri hacmi istatistiksel olarak anlamsız:** 42 örnek sadece 22
   benzersiz plakaya karşılık geliyor (4 plaka 5-7 kez tekrarlanıyor, 18
   plaka sadece 1 kez var). Derin bir CRNN tanıma modelini anlamlı
   şekilde iyileştirmek genelde binlerce örnek gerektirir; bu ölçekte
   train/val ayrımı (örn. 34/8) hem sızıntıya (aynı plaka hem train hem
   val'de) hem de "iyileşme"nin gürültüden ayırt edilememesine yol açardı
   — ölçülen herhangi bir fark güvenilir bir sonuç olmazdı.
3. **Baseline zaten güçlü çıktı:** İki gerçek hata (API uyumsuzluğu, bölge
   seçimi) düzeltildikten sonra hazır (Türkiye'ye özel hiç eğitilmemiş)
   ağırlıklarla %97.6 tam eşleşme elde edildi. Bu, mevcut darboğazın
   "model Türk plakası karakterlerini tanımıyor" olmadığını gösteriyor —
   darboğaz artık düzeltilen pipeline hatalarıydı. Net, iyi aydınlatılmış,
   önden çekilmiş fotoğraflarda ince ayarın kazandıracağı marjinal fayda
   şüpheli.

**Sonuç/tavsiye (kullanıcının "işe yararsa devam, yetersizse farklı yol"
çerçevesine yanıt):** Claude-vision ile elle etiketleme yaklaşımı **işe
yarıyor** — 42/42 geçerli etiket üretti, doğrulayıcıdaki gerçek bir hatayı
da ortaya çıkardı. Ama bir sonraki adım "aynı yöntemle daha fazla etiket"
olmalı — ama rastgele/kolay örnekler değil, roadmap 5. aşamanın hedeflediği
**zorlu senaryolar** (gece, yağmur, açılı çekim, bulanık, düşük çözünürlük)
önceliklendirilmeli. Hazır model bu tür görüntülerde muhtemelen zorlanıyor;
gerçek ince ayar yatırımı (PaddleX eğitim altyapısının kurulması dahil)
yeterli hacimde (yüzlerce benzersiz plaka) ve zor-senaryo ağırlıklı bir
veri seti toplandığında tekrar gündeme alınmalı.

**Güncelleme — zorlu senaryo ölçümü de yapıldı, sonuç aynı yönde:**
Kullanıcı, otomatik parlaklık/bulanıklık/boyut taramasıyla seçilen 270
zorlu adaydan ilk 10'unu (dark/dark+blur/dark+small) yine Claude vision ile
ücretsiz etiketledi. Aynı ince-ayarsız `PlateOcr` bu 10 örnekte de **tam
eşleşme %90 (9/10), CER %5.4** verdi — kolay sette (%97.6) ölçülenle aynı
mertebede. Tek hata (`h_1470.jpg`) incelendiğinde nedeni yine bir
seçim/birleştirme sorunu değil: 400x104 piksellik çok küçük bir kırpımda
"41 VF" kısmı dedektör tarafından hiç tespit edilmedi (kolay setteki
`s_197.jpg` ile aynı hata sınıfı — düşük çözünürlüklü kırpımlarda dedektör
recall kaçırması).

**Sonuç:** Zorlu senaryolarda da OCR *tanıma* doğruluğu yüksek çıktı;
tekrarlanan tek hata sınıfı (küçük/düşük-çözünürlüklü kırpımlarda dedektör
karakter kaçırması) OCR modelinin değil, **plaka dedektörünün** (mAP50
%68.7, recall %61.9 — bkz. karar #18) sorumluluğunda. Bu, kullanıcının ön
gördüğü gibi, kalan 260 zorlu adayı OCR ince ayarı için etiketlemeye devam
etmenin düşük getirili olacağını gösteriyor — **öncelik plaka dedektörü
recall'ını iyileştirmeye kaymalı** (bkz. karar #22).

## 22. Plaka dedektörü 640px'te yeniden eğitildi — recall %61.9 → %75.4

**Karar:** `configs/detection.yaml`: `input_size`/`image_size` 416→640,
`epochs` 35→100 (patience=10 ile epoch 65'te erken durdu). **Mimari
(`yolo11n`) ve veri seti değişmedi** — hâlâ karar #18'deki aynı 1750
görüntülük dengeli alt küme (1400/175/175 train/val/test) kullanıldı.

**Önemli düzeltme:** Bu retraining'i raporlarken ilk paylaşılan özette
"640px, tam veri" (5.413 görüntü) deniyordu; diskteki
`data/processed/plates/` doğrudan sayılınca (`.npy` disk-cache dosyaları
hariç tutularak) gerçek görüntü sayısının hâlâ 1750 olduğu, yani tam
birleşik setin (5.413) hiç kullanılmadığı görüldü. Bu, sonucun
yorumlanışını değiştiriyor: recall iyileşmesi (%61.9→%75.4) **veri
hacminden değil, tek başına çözünürlük artışından** kaynaklanıyor — hatta
daha temiz bir bulgu, çünkü tek değişken izole edilmiş oldu.

**Sonuç (test seti, `best.pt`, epoch 65 checkpoint):** mAP50 %79.1,
mAP50-95 %53.6, precision %88.7, **recall %75.4** (önceki %61.9'dan).
Precision hafif düştü (%93.2'den, kaybolan Roboflow-only checkpoint'e
göre) — daha fazla plaka yakalanırken birkaç yanlış pozitif de arttı,
kabul edilebilir bir denge.

**Gerekçe:** OCR pilot değerlendirmesinde (karar #21) hem kolay hem zorlu
sette tekrarlanan tek hata sınıfının küçük/düşük-pikselli kırpımlarda
dedektörün karakter/plaka kaçırması olduğu tespit edilmişti — bu doğrudan
"görüntü çözünürlüğü artırılırsa recall düzelir mi?" hipotezini test etti.
Hipotez doğrulandı.

**Yan not:** Eğitim çıktısında ~93 "corrupt JPEG restored and saved"
uyarısı vardı (train ~70, val ~23) — Ultralytics otomatik onardı, eğitimi
engellemedi. Kaynağı doğrulanmadı; OneDrive senkronizasyon kaynaklı kısmi
indirme şüphesi var (kullanıcı notu). Tekrarlarsa, özellikle yeni veri
eklenirken araştırılmalı.

**Sıradaki adım:** Tam birleşik veri setiyle (5.413 görüntü, hâlâ
denenmedi) 640px'te yeniden eğitim, resim çözünürlüğünün getirdiği
kazancın üstüne veri hacminin ek bir kazanç sağlayıp sağlamadığını
görmek için — ama önce `InferencePipeline` uçtan uca testi (roadmap'te
zaten bekleyen adım) önceliklendirildi.

## 23. `InferencePipeline` uçtan uca ilk kez çalıştırıldı — iki gerçek hata bulundu (crop padding, küçük kırpımlarda dedektör kaçırması)

**Karar:** `scripts/run_inference.py` yazıldı (`configs/pipeline.yaml`
+ alt config'lerden gerçek `VehicleDetector`+`PlateDetector`+`PlateOcr`+
`VehicleClassifier`'ı kurup `InferencePipeline.process_frame()`'i
çalıştırıyor, `--annotate` ile kutulanmış çıktı görüntüsü yazıyor). Bu,
projenin dört bileşeninin GERÇEKTEN birlikte ilk kez çalıştırılmasıydı —
5 gerçek görüntüde test edildi, ilk çalıştırmada 5'ten 4'ü doğru plaka
okudu, biri ("66 LN 948", tamamen okunaklı bir plaka) boş metin döndürdü.

**Bulgu 1 — sıfır dolgulu kırpım, PaddleOCR'ın dedektörünü tamamen
susturuyor:** `InferencePipeline._crop()`, dedektörün kutusunu hiç
dolgusuz (padding=0) kırpıyordu. `models/plate_detector/best.pt`
(karar #22) plaka sınırlarına çok sıkı kutular üretiyor; bu sıkı
kırpımlar `PlateOcr.read()`'e verilince PaddleOCR'ın metin dedektörü
**hiçbir bölge bulamıyordu** (boş liste, kötü okuma değil). 10px dolgu
eklemek bile güveni 0'dan %96.7'ye çıkardı. `_crop()`'a opsiyonel
`padding_ratio` parametresi eklendi; `InferencePipeline`, plaka
kırpımı için `PLATE_CROP_PADDING_RATIO = 0.15` kullanıyor artık.

**Bulgu 2 — düşük mutlak piksel yüksekliği, arka plan dolgusundan
bağımsız olarak dedektörü kaçırtıyor:** %15 dolgu eklendikten sonra bile
aynı görüntü hâlâ boş dönüyordu. Kök neden dolgu değil, mutlak boyutmuş:
aynı kırpımı 2x/3x büyütünce (gerçek görüntü içeriğiyle, beyaz kenarlık
olmadan) okuma anında düzeldi (%99.8 güven). Yaklaşık 113px kırpım
yüksekliğinde başarısız, ~170px'te düzeliyor — dolgu içeriği (gerçek
arka plan vs. yapay beyaz kenarlık) sonucu değiştirmiyor, sadece mutlak
piksel ölçeği önemli. `PlateOcr.read()`'e `_upscale_if_small()` eklendi:
kırpım yüksekliği 200px'in altındaysa, en-boy oranı korunarak
büyütülüyor (kübik interpolasyon).

**Gerekçe:** Bu iki hata, tek başına birim testlerle (fake bağımlılıklı
`test_inference_pipeline.py`, gerçek model gerektirmeyen) veya izole
OCR pilotlarıyla (kırpımlar zaten önceden makul boyutlu/dolgulu üretilmiş
elle etiketleme akışından geliyordu) **hiç ortaya çıkmazdı** — sadece
gerçek dedektörün ürettiği sıkı kutularla, gerçek bir görüntü üzerinde
uçtan uca çalıştırınca görünür oldu. Bu, "her bileşen ayrı ayrı çalışıyor"
ile "boru hattı uçtan uca çalışıyor" arasındaki farkın neden ayrı bir
doğrulama adımı gerektirdiğinin somut kanıtı.

**Sonuç:** Düzeltmelerden sonra 5/5 test görüntüsünde geçerli plaka
okundu. `tests/unit/test_inference_pipeline.py` ve
`tests/unit/test_plate_ocr.py`'a gerçek örnek verilerle regresyon
testleri eklendi (99 test, hepsi geçiyor).

## 24. Video/kamera sarmalayıcısı — `scripts/run_inference_video.py`

**Karar:** Kullanıcı, sınıflandırıcı ince ayarının önüne video/kamera
canlı testini aldı ("asıl hedefim video/kamera ile canlı test"). Pipeline
kurulum kodu (`build_pipeline`) ve çizim kodu (`_annotate`), önceden
`scripts/run_inference.py`'ye gömülüydü; ikinci bir script'in aynı
mantığı kopyalamaması için `src/plaka/pipeline/builder.py`
(`build_pipeline_from_config`) ve `src/plaka/pipeline/visualization.py`
(`annotate_frame`) olarak `src/`e taşındı — her iki script de artık
bunları import ediyor. `scripts/run_inference_video.py` yazıldı:
`cv2.VideoCapture` ile dosya yolu veya kamera indeksi (`0` gibi) açıyor,
her kareyi (veya `--frame-stride` ile her N'inci kareyi) pipeline'dan
geçiriyor, `--output` ile MP4'e yazıyor ve/veya `--no-display`
verilmedikçe `cv2.imshow` ile canlı gösteriyor. **İzleme/zamansal oylama
yok** — kullanıcının açık isteğiyle her kare bağımsız işleniyor, "hızlı
çalışan bir v1" hedefi. Düşük güvenli (`<%30`, `DEFAULT_LOW_CONFIDENCE_THRESHOLD`)
marka/model tahminleri turuncu renkte "?" ve yüzde ile işaretleniyor,
gizlenmiyor — VehicleClassifier'ın VMMRdb kaynaklı ABD-pazarı önyargısı
(karar #13) nedeniyle bu, gerçek Türk trafiğinde beklenen normal durum.

**Doğrulama:** Ajan olarak etkileşimli bir pencereyi görüp 'q' ile
kapatamadığımdan, canlı gösterim/webcam kısmını bizzat test edemedim.
Bunun yerine 3 gerçek görüntüden (720p'ye küçültülmüş) sentetik bir test
videosu üretip `--no-display --output` ile başsız modda uçtan uca
çalıştırdım — 24 kare işlendi, çıktı videosu doğru kutulanmış/etiketlenmiş
kareler içeriyordu (görsel olarak teyit edildi). Kullanıcının kendi
makinesinde gerçek kamera/pencere davranışını (GUI: WIN32UI destekli
`opencv` kurulu, `.venv`'de) doğrulaması gerekiyor.

**Performans notu:** Test videosunda ölçülen hız ~0.5-1 kare/saniye —
gerçek "canlı" kullanım için yavaş (dört modelin sırayla, aynı süreçte
çalışması + PaddleOCR'ın CPU'da koşması baskın maliyet). `--frame-stride`
ile azaltılabilir; asıl hızlandırma (ONNX/TensorRT, roadmap aşama 6)
kapsam dışı bırakıldı — öncelik "çalışan bir v1" idi.

## 25. Kaggle marka veri seti (ahmedelsany/car-brand-classification-dataset) — ciddi alan (domain) uyuşmazlığı nedeniyle kullanılamaz

**Karar:** `data/external/car_brand_dataset/` (33 marka, 349 görüntü/sınıf,
`train/val/test` bölünmüş) indirildi (bu oturumun dışında, kullanıcı
tarafından). Kullanıcının 3 CSV'sinden (23 görüntü, gerçek Türkiye
trafiğinden) 21 kullanılabilir satır, `scripts/build_classifier_dataset.py`
'ye eklenen `--make-only --merge` moduyla `train/`e eklendi: 4 yeni marka
klasörü (Renault=5, Opel=2, Citroen=1, Peugeot=1) + 7 mevcut markaya takviye
(Toyota+2, Hyundai+2, Volkswagen+3, Honda+1, FIAT+2, MINI+1, BMW+1).
Birleştirme **hiçbir mevcut dosyayı silmedi/üzerine yazmadı** — büyük/küçük
harf duyarsız eşleştirmeyle mevcut klasör adları korundu (örn. "FIAT",
"Toyota"), sadece gerçekten yeni markalar Title-Case ile oluşturuldu.

`models/vehicle_classifier_brand/` altına ayrı bir çıktı olarak (üretim
checkpoint'i `models/vehicle_classifier/`e dokunulmadı) 37 sınıflı,
dondurulmuş omurga, 30 epoch'luk bir marka sınıflandırıcı eğitildi
(val_top1 %24.3, val_top5 %51.8 — ama bu agregat sayı 33 büyük Kaggle
sınıfının etkisi altında).

**Kritik bulgu:** `scripts/eval_brand_classifier.py` ile modelin
kendisinin 21 gerçek Türkiye etiketimiz üzerindeki performansına
bakıldığında sonuç **%19.0 (4/21)** — ve daha önemlisi, **bol örnekli
markalar bile tamamen başarısız**: Volkswagen 0/3, Toyota 0/2, Hyundai
0/2, Fiat 0/2 (bunların Kaggle setinde 349-352'şer eğitim görüntüsü var).
Bu, "Renault'da az örnek var, düşük doğruluk beklenir" (kullanıcının
zaten öngördüğü, doğru bir kaygı) sorunundan tamamen ayrı, çok daha temel
bir sorun işaret ediyor.

**Kök neden (görsel olarak doğrulandı):** `car_brand_dataset`'teki
görüntüler gerçek sokak fotoğrafı değil — bir araç yapılandırıcı/katalog
sitesinden alınmış **stüdyo çekimleri** (bazıları tam araç, düz/beyaz
arka planla; bazıları far/tampon gibi **yakın plan detay kırpımları**).
Dosya adları da bunu doğruluyor (`Toyota_4Runner_2011_40_20_270_...`
gibi kodlanmış özellik/konfigürasyon ID'leri, gerçek fotoğraf
meta verisi değil). Bu, gerçek trafik fotoğraflarından (değişken açı,
arka plan karmaşası, aydınlatma, mesafe) o kadar farklı bir görsel alan
ki, dondurulmuş bir ImageNet omurgası bu alan kaymasını (domain shift)
aşamıyor — örnek sayısı ne olursa olsun.

**Değerlendirme yöntemi notu (dürüstçe belirtilmeli):** `eval_brand_classifier.py`,
21 etiketli görüntünün TAMAMINI (train/val ayrımı yapılmadan) test etti.
Eğitim, bu görüntülerin çoğunu muhtemelen `train` bölümüne aldığından
(rastgele %90/%10 global bölünme), "doğru" çıkan 4 tahmin (Renault 2/5,
Peugeot 1/1, Honda 1/1) gerçek genelleme değil, büyük ihtimalle ezber
olabilir — ayrı bir held-out kontrol yapılmadı. Bu, sonucu daha da
kötüleştiriyor, iyileştirmiyor: 0'a düşen sınıflar zaten kesin başarısız,
"başarılı" görünenler de şüpheli.

**Sonuç/tavsiye:** Bu Kaggle veri seti bu haliyle **kullanılamaz** —
marka kapsamı (33 marka) yeterli görünse de, görsel alan uyuşmazlığı
sorunu çözüyor değil, gizliyordu (ilk seçim kriteri sadece "Türkiye'de
yaygın markaları kapsıyor mu" idi, "gerçek fotoğraflara mı, katalog
görsellerine mi benziyor" hiç kontrol edilmemişti — bu, gelecekteki
veri seti değerlendirmelerinde eklenecek bir kriter). Önerilen adımlar:
1. İkinci aday (`alirezaatashnejad/over-20-car-brands-dataset`) hiç
   indirilmedi/incelenmedi — indirilmeden önce birkaç örnek görüntüsü
   gerçek fotoğraf mı katalog/render mı diye görsel olarak kontrol
   edilmeli, aksi halde aynı sorunla karşılaşılır.
2. İki aday da katalog/render çıkarsa, en güvenilir yol muhtemelen
   Claude-vision ile ücretsiz etiketlemeyi (zaten kanıtlanmış çalışan
   yöntem) ölçek büyütmek — yabancı, alan-uyumsuz bir veri setinden çok
   daha az ama gerçek Türkiye verisi, muhtemelen daha iyi genelleme
   sağlar.

**models/vehicle_classifier_brand/ durumu:** Deneysel, üretim pipeline'ı
(`configs/classification.yaml`) hâlâ eski 200-sınıflı `models/vehicle_classifier/`i
kullanıyor — bu yeni checkpoint hiçbir yere bağlanmadı, silinmedi de;
kullanıcı isterse ikinci aday değerlendirilene kadar referans olarak kalabilir.

## 26. VMMRdb marka-seviyesi deneme — gerçek fotoğraf ama yine de sıfır held-out doğruluk

**Karar:** Kullanıcının önerisiyle, Kaggle katalog seti yerine zaten
indirilmiş VMMRdb (291.752 gerçek ikinci-el ilan fotoğrafı, 9.170
`<marka>_<model>_<yıl>` sınıfı) marka seviyesine indirgendi.
`plaka.data.datasets.extract_brand()` eklendi — ilk `_` karakterinde
bölme, "mercedes benz" dahil tüm durumları doğru ayırıyor (boşluk
içeriyor ama alt çizgi içermiyor, tek özel durum gerekmedi).
`aggregate_images_by_brand()`, her markanın tüm alt sınıflarındaki
görüntüleri tek havuzda toplayıp marka başına `max_images_per_brand`
(300) ile sınırlıyor — Ford gibi 870 alt sınıflı bir markanın Renault
gibi 1 alt sınıflı bir markayı gölgelememesi için (bkz. karar #13'teki
aynı dengesizlik deseni, bu kez model değil marka seviyesinde).

**Görsel doğrulama (Kaggle hatasını tekrarlamamak için):** Rastgele bir
VMMRdb görüntüsü (`toyota_corolla_2010`) incelendi — gerçek bir
sokak/araba yolu fotoğrafı (ağaçlar, evler, plaka, doğal ışık), katalog
değil. Bu adım güvenle geçildi.

**Kritik keşif — hedef markalar VMMRdb'de de son derece az:**
`configs/classification.yaml`'daki 20 hedef markadan `renault` (3
görüntü, 1 alt sınıf: `renault_captur_2015`), `peugeot` (5, 2 alt
sınıf), `citroen` (6, 2 alt sınıf), `opel` (8, 4 alt sınıf) VMMRdb'de de
neredeyse yok — kullanıcının "VMMRdb'nin kapsamadığı markalar" listesi
kısmen yanlıştı (bu markalar VMMRdb'de VAR, sadece çok az); Togg
gerçekten hiç yok (2022 sonrası kuruldu, VMMRdb ondan eski).

**Birleştirme:** `vehicle_labels_pilot*.csv`'lerdeki 21 kullanılabilir
Türkiye satırından 6'sı (≥2 örneği olan her marka — volkswagen, renault,
fiat, toyota, hyundai, opel — için 1'er görüntü, dosya adı numarasına
göre deterministik seçim) **eğitime hiç girmeyecek şekilde ayrıldı**
(`vehicle_labels_TEST_holdout.csv`); kalan 15 satır (`vehicle_labels_TRAIN.csv`)
`build_classifier_dataset.py --make-only --merge` ile
`data/processed/vmmrdb_by_brand/`e eklendi. Tek örneği olan markalar
(citroen, peugeot, honda, bmw, mini) tamamen eğitime gitti — bunlar için
ayrı bir held-out test yok, bu açıkça not edildi.

**Sonuç:** `models/vehicle_classifier_brand_v2/` (20 sınıf, 4.244 eğitim
görüntüsü, dondurulmuş omurga, 30 epoch) eğitildi. İçsel val_top1 %22.6,
val_top5 %57.4 — bu, VMMRdb'nin kendi alanı içinde şansın (%5) belirgin
üzerinde, modelin gerçek fotoğraflar arasında marka ayırt etmeyi bir
ölçüde öğrendiğini gösteriyor. **Ama daha önce hiç görmediği 6
Türkiye fotoğrafında (`scripts/eval_brand_classifier.py`,
`vehicle_labels_TEST_holdout.csv`): 0/6 (%0).** En yakın ıskalama
volkswagen (gerçek), tahmin sırasıyla audi (%35)/volkswagen (%25) —
en azından ilişkili bir marka, tamamen rastgele değil; ama net sonuç
yine de tam başarısızlık.

**Yorum (dürüstçe, aşırı yorumlamadan):** n=6 çok küçük bir örneklem —
bu tek başına "VMMRdb işe yaramıyor" demek için yeterli istatistiksel
güç sağlamıyor. Ama örüntü açık: VMMRdb "gerçek fotoğraf" olsa da
(Kaggle'ın stüdyo/katalog sorununu çözdü), muhtemelen kendi görsel
alt-alanı var — ikinci el ilan sitesi fotoğrafları (poz verilmiş,
yakın, park yeri/araba yolu, satıcı tarafından çekilmiş) — ve bu,
Türkiye trafik/kamera görüntülerinin (değişken açı, mesafe, trafik
kamerası çözünürlüğü/sıkıştırması) görsel dağılımından hâlâ yeterince
farklı. Yani "gerçek fotoğraf mı" tek başına yeterli bir kriter değil;
"hangi TÜR gerçek fotoğraf" da önemli.

**Sonuç/tavsiye:** İki ayrı harici veri seti denemesi (Kaggle katalog:
tam başarısızlık, VMMRdb marka-seviyesi: içsel iyi ama held-out'ta yine
başarısızlık) aynı yöne işaret ediyor: yabancı kaynaklı veri, ne kadar
"gerçek fotoğraf" olursa olsun, bizim özel görsel alanımıza (Türkiye
trafik/kamera görüntüleri) güvenilir şekilde aktarılamıyor. En güçlü
sinyal hâlâ karar #21/#25'te vurgulanan yoldan geliyor: **Claude-vision
ile ücretsiz etiketlemeyi ölçek büyütmek** — hacim şu an çok düşük
(toplam 21 etiketli görüntü) ama tek doğru-alan (in-domain) kaynak bu.
`models/vehicle_classifier_brand_v2/` deneysel olarak duruyor, üretim
pipeline'ına bağlanmadı.

## 27. Saf gerçek-Türkiye verisiyle (80 görüntü, 18 marka) marka sınıflandırıcısı — hacim yetersiz kaldı

**Karar:** Kullanıcının Claude-vision ile etiketlediği, `data/external/user_plates/labels_manual/vehicle_labels_pilot1-10.csv`
altında toplanan (10 dosya, gerçek Türkiye trafiği, önceki 3 dosyalık
pilotun devamı) etiketler kullanılarak **hiçbir harici veri seti
karıştırılmadan** (ne Kaggle ne VMMRdb) saf bir marka sınıflandırıcısı
denendi — üçüncü, bağımsız bir karşılaştırma noktası olarak.

**Veri hazırlığı:** `plaka.data.manual_labels` eklendi
(`merge_and_dedupe_labels`, `stratified_holdout_split`) +
`scripts/prepare_manual_vehicle_labels.py`. 91 ham satırdan 85 benzersiz
`image_file`, 80'i kullanılabilir (5'i `not_visible`/boş marka).
**Çakışma kontrolü:** 0 gerçek çakışma bulundu (aynı görüntüye farklı
pilotlarda verilen etiketler hep tutarlıydı) — 6 görüntü birden fazla
CSV'de tekrar ediyordu, hepsi aynı markayı veriyordu. 18 farklı marka;
en büyüğü Renault (15), en küçükleri Audi ve MINI (1'er, held-out testi
yok — açıkça raporlandı). Sabit seed (42) ile marka-katmanlı (stratified)
split: her ≥2 örnekli markadan 1 görüntü held-out teste ayrıldı (16 test,
64 train), tam liste `data/processed/vehicle_labels_manual_split/SPLIT_MANIFEST.csv`'de.

**Model tarafı eklemeler (`scripts/train_classifier.py`):**
- `--strong-augmentation`: timm'in varsayılan (sadece crop+flip)
  `create_transform`'u yerine açık `RandomResizedCrop` + `ColorJitter` +
  `RandomHorizontalFlip` + `RandomRotation(15°)` — küçük veri setinde
  overfitting'i geciktirmek için.
- `--freeze-mode {full,partial,none}`: `partial`, EfficientNet-B0'ın son
  MBConv bloğu + `conv_head`/`bn2` + classifier'ı çözüyor (1.15M eğitilebilir
  parametre, `full`'daki 23K'ya karşı).
- `--patience`: erken durdurma. **Bulunan ve düzeltilen hata:** ilk
  implementasyon `val_top1 >= best_val_top1`'i hem checkpoint kaydetme
  hem de "iyileşme" sayacını sıfırlama için kullanıyordu — 6 görüntülük
  minik bir val setinde (sadece 7 olası skor değeri: 0/6...6/6) aynı
  değerde uzun süre takılı kalmak çok olası, ve `>=` bunu her seferinde
  "iyileşme" sayıp patience'ı hiç tetiklemiyordu. `>` (kesin iyileşme) ile
  `>=` (checkpoint kaydetme) ayrıştırıldı.
- **Operasyonel not:** İlk çalıştırmayı düzeltmek için süreci
  PowerShell'den `Get-CimInstance Win32_Process` ile bulup
  `Stop-Process` ile durdururken, filtre (`*train_classifier.py*`) çok
  geniş kaldı ve komut dizisinde bu metni barındıran **sarmalayıcı
  bash.exe/powershell.exe süreçlerini de** öldürdü (kendi çalıştığım
  PowerShell süreci dahil — komut "Exit code 255" ile kendini
  sonlandırdı). Zarar yoktu (arka plan görevleri bağımsız yeniden
  başlatılabilir), ama gelecekte süreç sonlandırırken filtreye
  `$_.Name -eq 'python.exe'` gibi ek bir daralma eklemek gerekiyor.

**Eğitim:** İki varyant, aynı 64 görüntü/18 sınıf üzerinde, "Below
Normal" öncelikle, GPU'da, 40 epoch bütçesi + patience=10:
- **Tam dondurulmuş** (`full`): epoch 10'da erken durdu, iç val_top1
  hep %0 kaldı (6 görüntülük iç val setinde gürültülü/anlamsız), train_loss
  yüksek kaldı (~4.0-4.6) — model neredeyse hiç öğrenmedi.
- **Kısmi çözülmüş** (`partial`): epoch 10'da erken durdu, train_loss
  hızla düştü (4.07→0.88) — küçük eğitim setini ezberlediğinin açık
  işareti.

**Held-out test sonucu (16 görüntü, hiç eğitime girmemiş):**

| Varyant | Doğruluk |
|---|---|
| Tam dondurulmuş | 1/16 (%6.2) |
| Kısmi çözülmüş | 0/16 (%0.0) |

18 sınıfta rastgele tahmin başarı oranı ~%5.6 — **her iki model de bu
seviyede veya altında**, yani hiçbiri anlamlı bir şey öğrenmedi. Kısmi
çözülmüş model, eğitim setini ezberleyip (train_loss 0.88) held-out'ta
tam dondurulmuştan bile daha kötü çıktı — klasik overfitting imzası.
Confusion çiftlerinde belirgin bir örüntü yok (dağınık, rastgele
görünüyor), bu da "kısmen doğru ama karıştırıyor" değil "hiç
öğrenmedi" tablosuna işaret ediyor.

**Yorum — bu, önceki iki denemeden FARKLI bir başarısızlık nedeni:**
Kaggle ve VMMRdb denemeleri **alan uyumsuzluğundan** başarısız oldu
(yanlış türde görüntü). Bu deneme **doğru alanda** (gerçek Türkiye
trafiği) ama **yetersiz hacimde** (sınıf başına ortalama 3.5 görüntü,
18 sınıf) başarısız oldu — istatistiksel öğrenme açısından temel bir
sınır, teknikle (augmentation, kısmi fine-tune) aşılamayacak kadar az
veri. Bu üç deneme birlikte netleştiriyor: **hem doğru alan HEM yeterli
hacim gerekiyor, biri diğerini telafi etmiyor.**

**Sonuç/tavsiye:** `models/vehicle_classifier_manual_frozen/` ve
`_partial/` deneysel, üretime bağlanmadı. Doğru alan zaten elde var
(Claude-vision etiketleme kanıtlanmış şekilde çalışıyor) — eksik olan
tek şey hacim. Sınıf başına en az ~30-50 görüntü (kaba bir kural, kesin
değil) olmadan bu yaklaşımın gerçekten işe yarayıp yaramadığını
söylemek zor; şu anki 80 görüntü/18 sınıf ölçeğinde daha fazla epoch,
farklı mimari veya başka bir augmentation denemenin getirisi düşük —
asıl kaldıraç etiketleme hacmini büyütmek.

## 28. İlk gerçek video testi — pipeline mekanik olarak sağlam, ama plaka dedektörü bu videoda neredeyse hiç plaka bulamadı

**Karar:** Kullanıcının sağladığı gerçek bir trafik videosuyla
(`data/external/test_videos/arac2.mp4`, 1024×576, 11 saniye, 328 kare,
~30fps) `scripts/run_inference_video.py` ilk kez gerçek bir video
dosyasıyla uçtan uca çalıştırıldı (`--no-display --output`, Below
Normal öncelik, GPU).

**Hız:** Isındıktan sonra ~16 fps'e ulaştı (ilk birkaç kare model
yükleme yüzünden yavaştı), 328 kare ~33 saniyede bitti — bu çözünürlükte
gerçek zamanlının bile üzerinde. Video I/O, kare-kare annotasyon,
`--no-display` modu sorunsuz çalıştı.

**Ama plaka tespiti bu videoda ciddi şekilde başarısız oldu:** 328
kareden yalnızca **1 tanesinde** (kare 0) herhangi bir plaka tespit
edildi, ve o tek tespit de 40×40 piksellik bir kırpım üretip anlamsız
bir OCR okuması ("ELLIE") verdi. Diğer 327 karede plaka dedektörü hiçbir
şey bulamadı.

**Kök neden (ölçüldü, tahmin değil):** Kare 0'da en yakın araç, kare
genişliğinin **%56'sını** kaplıyor (büyük/yakın, iyi görünür durumda) —
ama tespit edilen plaka kutusu sadece kare genişliğinin **%3.1'i**
(31×31 piksel ham boyut). Aracın kendisi büyük olsa da, kamera açısı
(yandan/ön-çeyrek açı, düz karşıdan değil) plakayı öngörünüm
(foreshortened) küçük bir dikdörtgen haline getiriyor — bizim eğitim
verimizin (Roboflow + kullanıcının kendi fotoğrafları) çoğunlukla daha
cepheden/arkadan çekimlere dayandığı düşünülürse, farklı bir çekim
geometrisi.

**Bu, önceki "küçük kırpım" bulgusundan (karar #23) farklı bir
katman:** #23'te dedektör plakayı BULUYORDU, sorun bulunan kırpımın
OCR için çok küçük olmasıydı (düzeltildi: dolgu + büyütme). Burada
dedektörün kendisi plakayı çoğu karede **bulamıyor** — kırpım/büyütme
düzeltmelerinin devreye bile girmediği, daha önceki bir aşama.

**Sonuç/tavsiye:** Bu tam olarak roadmap'in 5. aşamasının ("zorlu
senaryo testi") beklediği türden gerçek bir bulgu — kod hatası değil,
gerçek bir kapsam boşluğu: dedektör, plakaların küçük/açılı göründüğü
kamera geometrilerinde zayıf. Marka/model karşılaştırması bu klipte
anlamsız kaldı çünkü karşılaştıracak hiç okunmuş plaka yoktu. Sonraki
adım aday: plaka dedektörünü bu tür (uzak/açılı) görüntüler içeren ek
veriyle güçlendirmek — ya da en azından bu videodan alınan zorlu
kareleri gelecekteki bir dedektör yeniden eğitimine dahil etmek.

## 29. Kapsam değişikliği: marka/model kaldırıldı (araç tipi + plaka), threaded kamera yakalama, ölçülebilir gecikme raporu

**Karar:** `VehicleClassifier` üç ayrı gerçek-veri denemesinde de
(Kaggle katalog #25, VMMRdb-marka #26, elle etiketlenmiş Türkiye
fotoğrafları #27) rastgele-tahmin seviyesini geçemedi — sorun teknik
değil istatistiksel (sınıf başına 30-50 görüntü gerekiyor, mevcut
etiketleme hızıyla haftalar sürer). Kullanıcı kararıyla marka/model
kapsamdan çıkarıldı; yeni hedef **araç tipi (car/motorcycle/bus/truck,
zaten güvenilir COCO-pretrained `VehicleDetector`'dan geliyor) + plaka**.

**Uygulama:**
- `VehicleDetection.vehicle_type: str` alanı eklendi (`schemas.py`) —
  `RawDetection.class_name`'den geliyor, ek model/eğitim gerekmiyor.
- `InferencePipeline.__init__`'te `vehicle_classifier` artık
  `VehicleClassifierProtocol | None = None` — `None` ise classification
  adımı hiç çalışmıyor (crop bile alınmıyor, ekstra hız). Kod silinmedi;
  `configs/pipeline.yaml`'a eklenen `classification.enabled` flag'i
  (varsayılan `false`) ile `builder.py` sadece `true` olduğunda
  `VehicleClassifier`'ı kurup pipeline'a veriyor — ileride hacim
  yeterince büyürse tek satırlık bir geri açma.
- `annotate_frame`: kutu etiketi artık `"{vehicle_type} | {plate}"`
  (örn. `"car | 23 AFS 937"`); `make_model` hâlâ şemada var ve set
  edilmişse (classification yeniden açılırsa) eskisi gibi düşük-güven
  turuncu işaretlemesiyle gösteriliyor.
- `scripts/run_inference.py` / `run_inference_video.py` çıktı/log
  formatları aynı şekilde sadeleştirildi.

**Doğrulama (gerçek görüntü):** `data/external/test_foto/foto.jpeg`
üzerinde `--annotate` ile çalıştırıldı — 2 araç, ikisi de doğru tipte
(`car`, `truck` — COCO detector'ın "truck" etiketi burada aslında bir
Opel Corsa hatchback'e yanlış verilmiş, ama bu `VehicleDetector`'ın
kendi COCO-pretrained sınırlaması, bu PR'ın kapsamı dışında), ikisinde
de plaka doğru okundu (`23 FE 251`, `23 AFS 937`, ikisi de VALID).

**Threaded kamera yakalama (`scripts/run_inference_video.py`):**
`ThreadedFrameGrabber` — arka planda sürekli `capture.read()` çağıran
ayrı bir thread, tek-slotlu bir buffer'a (lock korumalı) en güncel
kareyi + bir generation sayacı yazıyor. Ana döngü (işleme+gösterim)
`read_latest()` ile en güncel kareyi alıyor; generation değişmediyse
pipeline'ı hiç çağırmadan son sonucu yeniden çiziyor. Kamera kaynağı
(`source` tam sayıysa) varsayılan olarak bu mod; `--no-threaded` ile
eski sıralı akışa dönülebiliyor. Video dosyası kaynaklarında hep sıralı
akış kullanılıyor (kasma sorunu yalnızca canlı kamerada var).

**Ölçüm — sentetik yük testi (gerçek `data/external/test_videos/arac2.mp4`
altında ölçülen ~65-90ms/kare gerçek pipeline maliyetini, gerçek
webcam'de tekrarlanabilir kılmak için `process_frame`'e yapay 150ms
gecikme eklenerek izole edildi — laptop webcam'i boş bir odaya
baktığından gerçek trafik sahnesi yoktu, VehicleClassifier hiç tetiklenmedi
ve doğal yükte iki mod da fark göstermedi, bu yüzden mekanizmayı ayrı
test etmek gerekti):
- **Kare-başı işlem gecikmesi** (art arda iki `process_frame` çağrısı
  arası, `LatencyTracker`): sıralı p50=170.8ms / p95=176.0ms, threaded
  p50=191.1ms / p95=197.0ms — threaded burada **%10-12 daha yavaş**,
  beklenmedik ama dürüst bir sonuç: Python GIL altında sürekli
  `capture.read()` çağıran arka plan thread'i, ana thread ile
  zamanlama/lock rekabetine giriyor. Threading, modelin kendisini
  hızlandırmıyor — bunu iddia etmiyoruz.
- **Asıl mekanizma kanıtı — yakalama hiç bloklanmıyor:** aynı 6 saniyelik
  pencerede threaded modun arka plan thread'i toplam **252 kare**
  yakaladı, ana döngü bunlardan yalnızca **45'ini** işleyebildi (207'si
  "pipeline hâlâ öncekiyle meşgulken" atlandı — yeni eklenen skip-count
  log satırı). Yani kamera hiçbir zaman ana döngüyü beklemedi; gösterilen
  kare her zaman en fazla ~1 kamera-karesi (~33ms) bayat, pipeline ne
  kadar geride kalırsa kalsın. Sıralı modda bu garanti yok: her
  `capture.read()` çağrısı bir önceki işlemenin bitmesini bekliyor, yani
  gösterilen kare pipeline ne kadar gerideyse o kadar bayat olabilir
  (büyüyen gecikme, "kasma"nın kendisi).
- **Gerçek (yapay olmayan) kamera yükü altında** (laptop webcam'i, boş
  oda, sadece tespit — sınıflandırma yok): sıralı ve threaded mod
  istatistiksel olarak ayırt edilemezdi (ikisi de p50≈31ms, 0 sıçrama)
  — beklenen, çünkü pipeline zaten kameranın 30fps'ini rahatça
  yetiştiriyor; threading'in faydası yalnızca pipeline kameradan
  YAVAŞ olduğunda ortaya çıkıyor (yukarıdaki sentetik test).

**Video dosyası testleri (`arac2.mp4`, `arac3.mp4`, sıralı mod, GPU,
Below Normal):** `arac2.mp4` — 328 kare, 272'sinde ≥1 araç, toplam 445
araç tespiti (439 car, 6 bus), **0 geçerli plaka** (karar #28'deki açı
sorunu hâlâ çözülmedi — ayrı, devam eden bir görev). `arac3.mp4` — 901
kare ama yalnızca 6'sında araç tespit edildi; kök neden video dosyasının
**464×832 dikey (portre) çekilmiş ve OpenCV tarafından döndürülmeden**
okunması — COCO-pretrained dedektör 90° yan yatmış bir sahneyi tanıyamıyor.
Bu, karar #28'deki "plaka açısı" sorunundan tamamen farklı, yeni bir
bulgu (video I/O/rotasyon, dedektör kapasitesi değil); bu prompt'un
kapsamı dışında bırakıldı, düzeltilmedi — kullanıcıya bildiriliyor.

**Sonuç:** Mimari değişiklik doğru çalışıyor ve mekanizması (skip-count
üzerinden) somut kanıtlandı; ama iki bağımsız açık uç var — (1) plaka
dedektörünün açılı/uzak kameralardaki recall sorunu (#28, ayrı görevde
sürüyor) tam video doğrulamasını hâlâ engelliyor, (2) `arac3.mp4`'ün
döndürülmemiş okunması yeni, küçük bir video-I/O bulgusu.

## 30. Web arayüzü — fotoğraf/video yükleme + tarayıcıdan canlı kamera (FastAPI)

**Karar:** Projeyi "profesyonel" ve sunulabilir hale getirmek için
`InferencePipeline`'ı bir web arayüzüne bağladık — kod tekrar yazılmadı,
`src/plaka/pipeline/builder.py` + `visualization.py` aynen kullanıldı.

**Mimari:** `src/plaka/web/app.py` (FastAPI, `create_app(pipeline=None, ...)`
fabrika fonksiyonu — gerçek kullanım için `configs/pipeline.yaml`'dan
pipeline kurar, testler sahte pipeline enjekte eder), `src/plaka/web/jobs.py`
(`JobManager` — video işleme ayrı bir daemon thread'de, tek HTTP isteği
içinde bitmesi beklenemeyecek kadar yavaş olduğu için, bkz. #28/#29),
`src/plaka/web/static/` (vanilla HTML/CSS/JS, build aracı yok — kurulum
tek adım). Üç sekme:
- **Fotoğraf:** `POST /api/infer/image` — senkron, annotate edilmiş
  görüntüyü base64 + araç/plaka JSON'ı döndürüyor.
- **Video:** `POST /api/infer/video` iş kuyruğa alıyor (`job_id`),
  istemci `GET /api/jobs/{id}`'yi 1sn'de bir yokluyor (ilerleme çubuğu),
  bitince özet (araç tipi sayıları, benzersiz okunan plakalar +
  küçük resimler) + tam videoyu indirme linki gösteriliyor.
- **Canlı Kamera:** `WS /ws/camera` — kamera **tarayıcıda**
  (`getUserMedia`) açılıyor, her kare `<canvas>` üzerinden JPEG'e
  kodlanıp WebSocket'e gönderiliyor, sunucu annotate edilmiş kareyi +
  JSON sonuçları geri yolluyor. İstemci bir sonraki kareyi ancak
  önceki yanıt tam döndükten sonra gönderiyor (doğal geri basınç —
  masaüstü threaded yakalamadaki "bayat kare biriktirme" sorunu burada
  zaten yok, çünkü tarayıcının kendi video önizlemesi sunucudan
  bağımsız akıyor).

**Video kodek notu:** Bu makinede OpenCV'nin H.264 (`avc1`/`H264`/`X264`)
kodlayıcıları çalışmıyor (`openh264-1.8.0-win64.dll` sürüm uyuşmazlığı,
boş/geçersiz dosya üretiyor — doğrulandı, bkz. test). `mp4v` (MPEG-4
Part 2) kullanılmaya devam edildi — tarayıcı `<video>` etiketi bunu
her zaman satır içi oynatamayabilir, bu yüzden video sonucu satır içi
oynatıcıya güvenmek yerine (a) indirilebilir dosya + (b) geçerli plaka
bulunan karelerin JPEG küçük resimleri (kodeksiz, her tarayıcıda çalışır)
olarak sunuluyor.

**Doğrulama:** Gerçek sunucu başlatılıp gerçek pipeline'la uçtan uca
test edildi (birim testleri sahte pipeline ile, `tests/unit/test_web_app.py`,
6 test): `foto.jpeg` → `/api/infer/image` iki aracı da doğru tip+plaka ile
döndürdü; `arac2.mp4` → `/api/infer/video` iş kuyruğu 328 kareyi ~6
yoklamada bitirdi, sonuç (439 car, 6 bus, 0 geçerli plaka) #29'daki
video-dosyası testiyle birebir eşleşti; `/ws/camera`'ya gerçek bir kare
gönderilip doğru JSON + annotate edilmiş JPEG alındığı doğrulandı.

**Çalıştırma:** `python scripts/run_web.py` (varsayılan `http://127.0.0.1:8000`).
`uvicorn`'un factory modu (`plaka.web.app:create_app_from_env`) kullanıldı
ki `plaka.web.app`'i sadece import etmek (örn. testlerde) gerçek modelleri
tetiklemesin.

## 31. Canlı kamera donması + OCR'ın gerçek darboğaz olduğu bulundu — PaddleOCR "tiny" modele geçildi (~13x hızlanma)

**Şikayet:** Kullanıcı web arayüzünü denedikten sonra: "video kısmı çok
başarısız, kamera kısmı da aşırı kasılıp donuyor." Sorulan netleştirme
sorusunda video için "plaka/tip yanlış ya da hiç bulunamıyor" seçildi —
yani video tarafı bir web hatası değil, bilinen dedektör-açısı sorunu
(#28) ile aynı; ayrı bir görevde ele alınacak. Kamera tarafı ise gerçekten
yeni bir bulgu.

**Bulgu 1 — event loop bloklanıyordu (gerçek ama ikincil neden):**
`/ws/camera` ve `/api/infer/image` route'ları `async def` olmasına rağmen
içeride `pipeline.process_frame()`'i doğrudan (senkron) çağırıyordu — bu,
FastAPI'nin tek asyncio event loop'unu her kare için bloke ediyor,
o sırada sunucudaki **her** bağlantı donuyordu. `asyncio.to_thread()` ile
ağır kısım (`_infer_and_annotate`) bir thread pool'a taşındı;
`JobManager`'ın kendi arka plan thread'iyle aynı pipeline nesnesine aynı
anda erişebilme riski için paylaşılan bir `threading.Lock` eklendi
(ultralytics/paddleocr model nesneleri thread-safe olarak dokümante
edilmemiş).

**Bulgu 2 — asıl darboğaz: PaddleOCR "medium" modeli CPU'da çok yavaş
(ölçüldü, tahmin değil):** Bu makinede `paddlepaddle` CUDA'sız (CPU-only)
kurulu (`paddle.device.is_compiled_with_cuda() == False`) — GPU'ya
geçirilemiyor risksizce (CUDA sürüm eşleştirme, mevcut kurulumu bozma
riski). Ham OCR çağrısı tek başına (dedektörler hariç) **2.2-3.2
saniye** sürüyordu (`PP-OCRv6_medium_det/rec`, varsayılan). Boş bir
karede (araç yok, OCR hiç tetiklenmiyor) aynı istek sadece ~40ms —
yani tüm gecikme OCR'dan geliyordu, dedektörlerden değil.

**Düzeltme — ölçülmüş, doğrulanmış model değişikliği:** `PlateOcr`,
`PP-OCRv6_tiny_det`/`PP-OCRv6_tiny_rec` kullanacak şekilde güncellendi
(`src/plaka/ocr/plate_ocr.py`). Orijinal OCR pilotunun (karar #21) aynı
42+10 insan-doğrulamalı gerçek plaka kırpımı üzerinde önce/sonra
karşılaştırması:

| | tam eşleşme (42, kolay) | CER (kolay) | tam eşleşme (10, zorlu) | CER (zorlu) | ort. gecikme |
|---|---|---|---|---|---|
| medium (eski) | %97.6 | %0.64 | %100 | %0.00 | 2.2-3.2s |
| **tiny (yeni)** | **%100** | **%0.00** | %90 | %1.35 | **~200ms (~13x)** |

Kolay sette "tiny" aslında daha iyi; zorlu 10 örnekte 1 ek hata (10
örnekte %10, istatistiksel olarak gürültülü). Bu takas kabul edildi —
canlı kamerayı kullanılabilir kılmanın değeri, 10 örneklik zorlu sette
tek bir ek hatadan çok daha büyük.

**Sonuç (gerçek sunucuda ölçüldü):** Tek bağlantı, 640px kare, 2
araç+plaka: **~2000ms → ~176ms/kare** (kamera artık ~5-6 fps'te akıyor,
öncesinde ~0.5 fps'ti — bu "kasılıp donma" hissinin birebir açıklaması).
Event-loop düzeltmesi ayrıca doğrulandı: kamera akarken eşzamanlı HTTP
isteklerinin gecikmesi p95=37ms'ye düştü (düzeltmeden önce event loop
bloklandığı için saniyelerce beklerlerdi).

**Video tarafı için hâlâ açık:** Bu değişiklik video/kamera'yı *hızlı*
yapıyor ama karar #28'deki plaka dedektörü recall sorununu çözmüyor —
`arac2.mp4` hâlâ 0 geçerli plaka veriyor (dedektör plakayı bulamıyor,
OCR'a hiç sıra gelmiyor). Kullanıcının "video kısmı başarısız" şikayeti
büyük olasılıkla bu — ayrı, daha büyük bir görev (yeni açılı/uzak plaka
verisiyle dedektör yeniden eğitimi).

## 32. Karar #28 teşhis Aşama 1 — dedektörde zayıf ama gerçek bir sinyal var (config değiştirilmeden, doğrudan `confidence_threshold=0.05` ile)

**Yöntem:** `configs/detection.yaml` değiştirilmedi — geçici bir teşhis
script'i `PlateDetector`'ı doğrudan `confidence_threshold=0.05` ile
kurup `arac2.mp4`'ün 328 karesinin tamamında çalıştırdı. Her araç için,
o karedeki tüm düşük-güven plaka kutuları arasından en iyi "containment"
(aracın içinde kalma oranı, pipeline'ın kendi eşleştirme mantığıyla
aynı ölçüt) hesaplandı; ≥%50 containment olan aday "yerelleşmiş sinyal"
sayıldı (konum/boyut/güven kaydedildi). 9 örnek kare, araç (yeşil) +
tüm düşük-güven plaka kutuları (turuncu <%50, kırmızı ≥%50 üretim eşiği)
işaretlenmiş halde `outputs/detector_diagnosis_samples/`e yazıldı ve
görsel olarak doğrulandı.

**Sonuç — "zayıf sinyal var" (birinci senaryo):** 445 araç örneğinden
**181'inde (%40.7)** aracın içinde, plakanın olması beklenen konumda
(rel_y medyan 0.73 — aracın alt/arka üçte biri) ve makul boyutta
(genişlik medyan çerçevenin %3.1'i — karar #28'deki 31px bulgusuyla
tutarlı) bir kutu bulundu; 264'ünde (%59.3) 0.05'e kadar inildiğinde
bile hiçbir kutu yoktu. Bulunan 181 adayın güven dağılımı: 0.05-0.10
arası 44, 0.10-0.20 arası 38, 0.20-0.30 arası 30, 0.30-0.50 arası 46,
ve **23'ü zaten üretim eşiği ≥0.50'de** (yani halihazırda tespit
ediliyor ama muhtemelen kırpım çok küçük olduğu için OCR sonrasında
geçersiz/anlamsız metne dönüşüyor — bu, karar #23'teki bilinen küçük-
kırpım sorunuyla örtüşüyor, dedektörün kendisiyle değil).

**Görsel doğrulama (9 örnek kareden 6'sı incelendi):** Plaka kareye
açıkça göründüğünde (kare 0, 40, 80, 320), turuncu/kırmızı kutu neredeyse
her zaman plakanın tam üzerine oturuyor — dedektör doğru yeri buluyor,
sadece güveni düşük. Plaka kareye görünmediğinde (kare 120, 240 — araç
arkadan/yandan kesilmiş, plaka fiziksel olarak kadraj dışı veya
kapalı), hiçbir kutu yok — bu, insan gözüyle de okunamayacak bir kare
olduğu için beklenen, model eksikliği değil.

**Yorum:** %59.3'lük "sinyal yok" grubunun ne kadarının gerçekten
"plaka fiziksel olarak görünmüyor" (düzeltilemez) ve ne kadarının
"görünüyor ama dedektör kaçırıyor" (düzeltilebilir) olduğu, kare bazlı
izleme (tracking) olmadan tam ayrıştırılamadı — 445 örneğin çoğu muhtemelen
aynı 1-2 aracın art arda karelerinden geliyor (arac2.mp4'te frame başına
ortalama 1.35 araç var, sahne yavaş/yoğun değil), yani bu sayı bağımsız
445 farklı örnek değil. Ama örnek kare incelemesi net bir örüntü
gösterdi: plaka görünürse dedektör neredeyse hep bir şeyler buluyor.

**Sonuç → tavsiye:** Bu "zayıf sinyal" senaryosu — tam yeniden eğitim
şart değil, önce ucuz seçenekler (eşik kalibrasyonu + varsa az miktarda
açılı örnekle ince ayar) denenmeli. Aşama 2 için kullanıcıdan onay
bekleniyor.

## 33. Aşama 2 — eşik kalibrasyonu tek başına yetmiyor: `arac2.mp4`'te çözünürlük tabanı bulundu; ama `arac3.mp4`'te gizli bir rotasyon hatası düzeltilince gerçek pozitif sinyal ortaya çıktı

**Eşik taraması (`configs/detection.yaml` değiştirilmedi, gerçek
`InferencePipeline` 5 farklı eşikle kuruldu):**

| eşik | kare eşleşen | %valid | toplam eşleşme |
|---|---|---|---|
| 0.15 | 108/328 (%32.9) | **0** | 117 |
| 0.20 | 90/328 (%27.4) | **0** | 99 |
| 0.25 | 76/328 (%23.2) | **0** | 83 |
| 0.30 | 64/328 (%19.5) | **0** | 69 |
| 0.35 | 53/328 (%16.2) | **0** | 56 |

Eşiği düşürmek daha çok kutu buluyor ama **hiçbir eşikte tek bir geçerli
plaka bile okunmuyor** — Aşama 1'in "zayıf sinyal" teşhisi konum
açısından doğruydu, ama bu sinyal OCR'a hiç dönüşmüyor.

**Kök neden (ölçüldü): çözünürlük tabanı, eşik sorunu değil.** 0.15
eşikte bulunan 117 eşleşmenin ham kutu boyutları: genişlik medyan 32px
(min 15, **max 42**), yükseklik medyan 10px (min 6, max 33). **Videodaki
EN BÜYÜK kutu bile 32×31px.** 117 eşleşmeden sadece 13'ü OCR'dan HERHANGİ
bir metin (çoğu tek karakter, hepsi geçersiz) aldı. Kare 0'daki en
büyük/en güvenli (conf=0.79) kutuyu gerçek `_crop`+`_upscale_if_small`
zinciriyle işleyip görsel olarak inceledim
(`outputs/tiny_plate_upscaled.png`): 40×40 → 200×200 büyütme sonrası
bile plaka bir bulanık lekeye dönüşüyor, insan gözüyle de okunamıyor —
bilgi kaynakta hiç yok, büyütme icat edemiyor. **Sonuç: `arac2.mp4`'te
threshold kalibrasyonu, ince ayar, hatta daha iyi bir dedektör bile bu
videodaki plakaları okutamaz — bu, kameranın bu mesafeden/zoom'dan
çekilmiş olmasından kaynaklanan, yazılımla çözülemeyecek bir taban.**

**Görsel false-positive kontrolü (30 örnek, 5 eşikten 6'şar, video
boyunca dağıtılmış):** `outputs/threshold_qa_montages/`. 0.15-0.30
arası **29/30 örnekte kutu tam plakanın üzerinde**, hiç yanlış nesne
yok — düşük eşikte precision kaybı bu videoda gözlenmedi. Tek şüpheli
örnek 0.35 eşikte (kare 196, kaputun ön kenarına yakın bir kutu). Bu,
eşiği 0.5'ten düşürmenin (genel olarak, başka görüntülerde) düşük riskli
olduğuna dair olumlu ama küçük-örneklemli bir kanıt.

**Padding/upscale zincirinin düşük-güvenli kutulara da uygulandığı
doğrulandı (kod + ampirik):** `InferencePipeline._read_matching_plate`
her eşleşen kutuya güvenden bağımsız `_crop(..., padding_ratio=0.15)`
uyguluyor (inference_pipeline.py:101); `PlateOcr.read()` her çağrıda
koşulsuz `_upscale_if_small()` çağırıyor (plate_ocr.py:198). Yukarıdaki
görsel kanıt zaten bu zincirden geçirilerek üretildi — mekanizma çalışıyor,
sorun mekanizmanın eksikliği değil, kaynaktaki piksel/bilgi yokluğu.

**Beklenmedik yan bulgu — `arac3.mp4` aslında iyi bir video, rotasyon
hatası yüzünden kullanılamaz görünüyordu:** Karar #29'da "dikey çekilmiş,
döndürülmeden okunuyor, 901 kareden sadece 6'sında araç" denmişti.
Doğru yönü (`cv2.ROTATE_90_COUNTERCLOCKWISE`) bulup uyguladığımda: **araçlı
kare sayısı 6/901 → 897/901**, eşleşen plaka kutusu sayısı 623, kutu
genişliği **12px'ten 122px'e** kadar değişiyor (yani bu videoda hem çok
uzak/küçük hem de yeterince yakın/okunabilir plakalar var — `arac2.mp4`
gibi tek-tip-çok-uzak değil). Tam pipeline'ı (eşik 0.5 ve 0.25) döndürülmüş
kareyle çalıştırınca: **901 karenin 10-11'inde geçerli plaka okundu**,
en az 2 farklı gerçek plaka (biri muhtemelen "23 ACM 638" — birden çok
karede tutarlı varyantlarla okunmuş, diğeri "23 T 0445"/"23 TC 145").
Bu, `arac2.mp4`'ün aksine, **gerçek, kullanılabilir pozitif sinyal**.

**Sonuç → plan revizyonu:** Aşama 2'nin 3. adımı (`arac2.mp4`'ün
zayıf/uzak kutularından pseudo-label ince ayar verisi çıkar) orijinal
öncülü artık geçersiz — 30x10px'lik kutulardan pseudo-label üretip ince
ayar yapmak, okunabilirlik sıfır olduğu için hedeflenen metriği
("kaç karede geçerli plaka okundu artışı") hiç iyileştirmeyecek, üstelik
dedektörü aşırı küçük/belirsiz kutulara "güvenmeye" alıştırma riski taşıyor.
Kullanıcıya raporlandı, ince ayar kaynağının `arac2.mp4` yerine (rotasyonu
düzeltilmiş) `arac3.mp4` olması ve video-döndürme desteğinin pipeline'a
eklenmesi öneriliyor — onay bekleniyor.

## 34. Video döndürme desteği (web + script, ortak modül) + saniye-bazlı örnekleme — `arac3.mp4` web'de doğrulandı

**Durum netleştirmesi:** Önceki turda döndürme + `arac3.mp4` ince ayarı
için onay *istenmişti* ("Devam edeyim mi?"), ama kullanıcıdan yanıt
gelmeden bu prompt geldi — yani o iş hiç başlamamıştı. Bu prompt döndürmeyi
açıkça istedi, onu yapıldı; `arac2.mp4` pseudo-label ince ayar kararı
**hâlâ askıda**, henüz hiçbir eğitim çalıştırılmadı.

**Otomatik rotasyon algılama denendi, güvenilmez çıktı (ölçüldü):**
`cv2.CAP_PROP_ORIENTATION_META`: `arac2.mp4` (zaten doğru) → 180°
(!), `arac3.mp4` (gerçekten döndürülmüş) → 0° (metadata hiç yok).
İkisi de yanlış yönde yanıltıcı — metadata'ya güvenmek doğru
görüntüyü bozar ya da bozuk olanı düzeltmez. Bu yüzden döndürme
**açık, elle verilen bir parametre** (`--rotate 0/90/180/270`), otomatik
algılama değil.

**Uygulama:** `src/plaka/pipeline/video_io.py` (yeni, paylaşımlı) —
`apply_rotation()`, `rotates_dimensions()` (90/270 genişlik/yükseklik
değiştirir, `VideoWriter` boyutu buna göre ayarlanmalı),
`resolve_frame_stride(fps, sample_interval_seconds, frame_stride)`
("N saniyede bir" isteğini `round(fps*N)` ile mevcut frame_stride
mekanizmasına çeviriyor, en az 1), `FrameSamplingPlan` (ikisini birlikte
taşıyan küçük bir sarmalayıcı). Hem `scripts/run_inference_video.py`
(`--rotate`, `--sample-interval-seconds`, ikisi çakışırsa saniye
kazanır ve loglanır) hem `src/plaka/web/jobs.py` (`JobManager.submit`
artık `rotate_degrees`/`sample_interval_seconds` alıyor) bu modülü
kullanıyor — döndürme mantığı tek yerde. Web'de `/api/infer/video`
form alanları (`rotate`, `sample_interval_seconds`) ve arayüzde iki
açılır menü eklendi (`index.html`/`app.js`). Kalan tahmini süre
(`estimated_seconds_remaining`, son 20 işlenmiş karenin hareketli
ortalamasından) ve her plaka için `timestamp_seconds` de eklendi.

**Testler:** `tests/unit/test_video_io.py` (19 test, saf mantık) +
`tests/unit/test_web_app.py`'ye 4 yeni test (geçersiz `rotate` reddi,
`rotate=90`'ın hem pipeline'a giden karede hem çıktı videosunda
genişlik/yükseklik gerçekten değiştirdiğinin doğrulanması, `sample_interval_seconds`'ın
gerçekten `pipeline.process_frame` çağrı sayısını azalttığının
doğrulanması, geçersiz aralık reddi). 185 test, hepsi geçiyor.

**Gerçek sunucuda doğrulama — `arac3.mp4` web'den, önce/sonra:**

| | araç tespiti (901 kare) | geçerli plaka | süre |
|---|---|---|---|
| `rotate=0` (önce) | 6 | 0 | — |
| `rotate=270` (sonra) | 2325 | **8** (2 farklı gerçek plaka) | 43s |

Bir örnek kareyi indirip görsel doğruladım
(`outputs/arac3_web_plate_sample.jpg`) — döndürme doğru, kutu plakanın
tam üzerinde, metin okunaklı ("23 ACM 638"). Script-tabanlı önceki
testle (karar #33) birebir tutarlı; artık web yolu da aynı sonucu
veriyor.

**Saniye-bazlı örnekleme karşılaştırması — kabul edilebilirlik kullanıcıya
bırakıldı:**

| mod | süre | geçerli plaka |
|---|---|---|
| tam kare (`frame_stride=1`) | 43s | **8** (2 gerçek plaka, t=1.72-1.87s ve t=13.45-13.56s) |
| 1 saniyede bir (`frame_stride=60`) | **5s** (~8.6x hızlı) | **0** |

Bulunan **her iki gerçek plaka da yalnızca ~0.15 saniyelik dar bir
pencerede** geçerli okunuyor (aracın doğru mesafe/açıya geldiği birkaç
ardışık kare) — 1 saniyelik örnekleme bu pencerelerin ikisini de
kaçırdı, %100 kayıp. **Bu, saniye-bazlı örneklemenin bu tür "dar
pencereli" tespitler için bedelsiz bir hızlanma olmadığının somut
kanıtı** — hız kazancı gerçek ama "her plakayı bul" hedefiyle doğrudan
çelişiyor. Kullanıcıyla karar verilmesi gereken bir ödünleşim, kod
tarafında zaten netleştirildi (her iki mod da doğru çalışıyor, seçim
kullanıcıda).

**Açık kalan:** `arac2.mp4` ince ayar/pseudo-label kararı hâlâ
bekliyor — hiçbir eğitim çalıştırılmadı.

## 35. Web varsayılanı tam kare olarak sabitlendi + `arac3.mp4` pseudo-label ince ayarı çalıştırıldı (küçük ama gerçek iyileşme)

**(a) Web varsayılanı:** `index.html`/`app.js` — video panelinde
örnekleme dropdown'ının varsayılanı "Tam kare" oldu (döndürme
varsayılanı zaten "Yok"du, değişmedi). Diğer seçenekler artık
"⚠ hızlı önizleme, plaka kaçırabilir" etiketiyle işaretli; seçilince
karar #34'teki somut bulguya atıfla dinamik bir uyarı metni beliriyor
("Bu videoda gerçek plakalar sadece ~0.15 saniyelik dar bir pencerede
okunabilir çıktı..."). Gerçek sunucuda doğrulandı: form alanı
dokunulmadan gönderilen bir video `frame_stride=1` ile işlendi.

**(b) `arac3.mp4` pseudo-label ince ayarı — sonuç:**

*Veri:* Train aralığı (kare 0-720, held-out son %20 hariç) düşük eşikte
(0.15) taranıp araç-içi (containment≥%50) adaylar toplandı (518 aday).
Kutu genişliği çoğunlukla küçüktü (medyan 28px) ama iki gerçek, okunaklı
araç geçişi vardı: kare 73-114 ("23 TC 445", Toyota, 47-71px) ve kare
518-556 ("01 ZJ 651", koyu araç, 45-80px). Görsel montajla ikisi de
doğrulandı (`outputs/arac3_candidate_montages/`) — kutular hep plakanın
tam üzerinde, yanlış pozitif yok. Bunlardan seyrek örneklenmiş **19
kare** (her iki geçişten de aralıklı) pseudo-label olarak YOLO formatına
çevrildi, unutmayı önlemek için mevcut dengeli eğitim setinden rastgele
**300 görüntüyle** karıştırıldı (`data/processed/plate_finetune_arac3/`).

*Eğitim:* `scripts/train_detector.py`'a `--base-weights`/`--lr0`/`--freeze`
eklendi (sıfırdan değil, `best.pt`'den ince ayar yapabilmek için).
`lr0=0.0005` (varsayılanın ~20'de 1'i), `freeze=10` (backbone dondu),
`epochs=15`, GPU, Below Normal — **50 saniyede bitti**. Genel val setinde
(175 görüntü, `data/processed/plates/val`) sonuç orijinal `best.pt`'ye
(karar #22: mAP50 %79.1, precision %88.7, recall %75.4) çok yakın —
**mAP50 %77.9, precision %90.9, recall %72.0** — unutma/çökme yok,
küçük ve kabul edilebilir bir kayma. `models/plate_detector_arac3_finetune/`e
yazıldı, **üretim config'ine bağlanmadı**.

*Held-out karşılaştırma (kare 721-900, ince ayara hiç girmedi):*

| | geçerli plaka okuması | pencere (kare aralığı) | ham eşleşme (üretim eşiği ≥0.5) |
|---|---|---|---|
| önce (orijinal `best.pt`) | 8 | 806-813 (8 kare, ~0.13s) | 20 eşleşme / 18 kare |
| sonra (ince ayarlı) | **10** | **802-813 (12 kare, ~0.2s)** | 16 eşleşme / 14 kare |

Geçerli okuma sayısı arttı (8→10) ve pencere **4 kare (~67ms) daha
erkenden** başladı — aracın gerçek plakası muhtemelen "23 ACM 638"
(her iki durumda da en güvenli/tutarlı okuma bu). Ham eşleşme sayısı
aslında biraz DÜŞTÜ (20→16) — yani ince ayar daha FAZLA kutu üretmedi,
üretilen kutuların OCR'a çevrilme **kalitesi** biraz arttı (daha az ama
daha isabetli kutu). Bir örnek karede (802) görsel karşılaştırma
yapıldı — iki model de plakayı aynı sıkılıkta buluyor, aradaki fark
büyük/dramatik değil, küçük ve marjinal.

**Yorum — dürüst değerlendirme:** Bu, beklenen ölçekte, **küçük ama
gerçek** bir iyileşme — 19 pseudo-label görüntüyle "mucize" beklenmiyordu,
beklenmedi de. Plaka hâlâ yalnızca dar bir zaman/mesafe penceresinde
okunabiliyor; OCR metni kareler arası hâlâ tutarsız (AI/AID/MCI/MEL/MC/
MCM/ACM gibi farklı yanlış okumalar) — bu detector fine-tune'unun
çözebileceği bir şey değil, OCR/çözünürlük sınırı (karar #33) hâlâ
geçerli. Sonuç: yön doğru, ölçek küçük — daha fazla pseudo-label
(daha fazla gerçek video/geçiş) kümülatif olarak daha fazla kazanç
sağlayabilir, ama tek video/2 araçlık bu ilk denemeden büyük bir sıçrama
beklenmemeli.

**Açık kalan (değişmedi):** `arac2.mp4`'ün 30px'lik okunamaz kutularından
pseudo-label ince ayarı hâlâ ele alınmadı, bilinçli olarak askıda.

## 36. Fine-tune checkpoint üretime alındı + kare-arası plaka konsensüsü eklendi

**(1) Üretime alma:** `configs/detection.yaml` → `plate_detector.weights_path`
artık `models/plate_detector_arac3_finetune/best.pt` (karar #35).
Orijinal checkpoint `models/plate_detector/best.pt`'de referans/rollback
için duruyor, silinmedi.

**(2) Kare-arası araç izleme + plaka konsensüsü — yeni modül
`src/plaka/pipeline/tracker.py`:** `run_inference_video.py`'nin kendi
docstring'inde "roadmap stage 3+" olarak planlanmış adım. Basit, açıkça
"library değil" bir tasarım:

- `VehicleTracker.update(frame_index, vehicles)`: her aracı, mevcut
  track'lerin en çok örtüştüğü (IoU) kutusuyla açgözlü (greedy, en yüksek
  IoU önce) eşleştiriyor; eşleşme yoksa yeni track açılıyor. Bir track,
  `max_frames_since_seen` (varsayılan 15 kare) boyunca eşleşme almazsa
  emekliye ayrılıyor — tek sonraki kareye değil, kısa boşluklara (frame-stride
  atlamaları, kısa oklüzyon) tolerans için.
  `VehicleDetection.track_id` (şemada zaten önceden ayrılmış, boştu) artık
  gerçekten dolduruluyor.
- `VehicleTrack.consensus_text`: track'in gördüğü tüm geçerli-formatlı
  okumalar üzerinde **çoğunluk oyu** (normalize metin), eşitlik durumunda
  **en yüksek OCR-confidence'lı** okuma kazanıyor.
- `apply_consensus(result, tracker, frame_index)`: `FrameResult`'ı
  track_id + (varsa) konsensüs metniyle güncelleyip döndürüyor — kutu/
  güven değerleri dokunulmadan kalıyor, sadece metin (muhtemelen daha
  güvenilir bir metinle) değişiyor.
- `scripts/run_inference_video.py` ve `src/plaka/web/jobs.py` ikisi de
  her işlenen karede `pipeline.process_frame()` sonrası `apply_consensus()`
  çağırıyor — annotate edilen görüntü ve loglar artık kare-bazlı değil,
  konsensüs metnini gösteriyor. Web'in `plate_sightings` listesi de
  metne göre değil **track_id'ye göre** dedupe ediliyor (`track_sightings`
  dict, upsert) — aynı aracın erken bir karede farklı okunan geçici metni
  artık kalıcı, ayrı bir "sighting" olarak kalmıyor, track'in en güncel
  konsensüsüyle sürekli güncelleniyor.

**Testler:** `tests/unit/test_tracker.py` (13 test — IoU eşleştirme, sıra-
bağımsız en-iyi-eşleşme, track emekliliği, çoğunluk oyu, eşitlik bozma,
geçersiz-format okumaların oya katılmaması), `test_web_app.py`'a 1 yeni
test (aynı araç 4 karede 2 farklı metinle okunuyor → tek sighting, çoğunluk
metniyle). 199 test, hepsi geçiyor.

**Gerçek doğrulama — `arac3.mp4` held-out (kare 721-900), üretim pipeline'ı
(ince ayarlı checkpoint) ile:**

*Önce (konsensüssüz, tek kare):* 10 geçerli-format okuma, **7 farklı
metin** — "23 AI 638", "23 AID 08", "23 MCI 698", "23 MEL 638",
"23 MC 633"(×2), "23 MCM 638", "23 ACM 638"(×3). Elle bakmadan hangisi
doğru belli değil.

*Sonra (konsensüs):* Basit tracker bu 10 okumayı 2 track'e böldü (araç
tipi car/bus arasında salınan sınıflandırma, aynı fiziksel aracın kutusu
— tracker'ın bilinen, küçük bir kusuru) — ama **her iki track de
bağımsız olarak aynı sonuca vardı: "23 ACM 638"** (7 gözlemden 1 track,
3 gözlemden diğeri). Bu metin, kolay sette (foto.jpeg gibi) yapılan
görsel doğrulamayla da örtüşüyor — daha önce (karar #34) bu plakanın net
bir kırpımı (`outputs/arac3_web_plate_sample.jpg`) gözle "23 ACM 638"
olarak okunmuştu; bağımsız bir doğrulama.

**Sonuç:** 7 farklı gürültülü okumadan → 1 tutarlı, bağımsız olarak iki
kez doğrulanmış okumaya geçiş — bu, tek-kareye güvenmenin gerçek
maliyetini ve konsensüsün gerçek değerini somut olarak gösteriyor.
Tracker'ın araç-tipi-değişince-track-bölme kusuru bilinen bir sınırlama
(daha sofistike bir eşleştirme —örn. tip'i yok sayıp sadece IoU'ya
bakmak zaten yapılıyor, ama görünüşe göre iki kutu bazı karelerde IoU
eşiğinin altına düşüyor— ileride iyileştirilebilir), sonucu bu örnekte
bozmadı ama genel olarak "aynı aracı hep tek track say" garantisi vermiyor.

**Açık kalan (değişmedi):** `arac2.mp4`'ün 30px'lik kutuları hâlâ kapsam
dışı — fiziksel taban sorunu, konsensüs de dahil hiçbir yazılım
düzeltmesiyle çözülmüyor (10 okumadan hiçbiri geçerli formatta bile
değildi ki konsensüse girsin). Kurulum tarafında (kamera mesafesi/zoom)
telafi edilecek, dokümantasyonda kayıtlı.

**Dürüst bir yan bulgu — ince ayar her yerde eşit iyileşme sağlamadı:**
Tam video (`arac3.mp4`, 901 kare, üretim pipeline'ı) çalıştırılınca
`plate_sightings` yalnızca 2 kayıt (ikisi de aynı held-out ACM638 aracı)
döndürdü — train aralığındaki iki bilinen araç (kare ~103-113 "23 T 0445"/
"23 TC 145", kare ~518-556 "01 ZJ 651"; ikisi de karar #34'te ince ayardan
ÖNCE geçerli okunmuştu) artık hiç geçerli-format okuma üretmiyor. Ham OCR
metnini inceledim: kare 106-110'da "231015"/"231045"/"23TO4S" gibi
karaktere çok yakın ama format olarak geçersiz çıktılar var — dedektörün
kutu sınırları ince ayarla hafifçe kaymış olabilir (bu kareler pseudo-label
setinin kendisinde), OCR'ın karakter bölütlemesini bozacak kadar. Yani
ince ayar **held-out'ta net iyileşti ama train aralığındaki (pseudo-label
kaynağı) iki araçta hafifçe geriledi** — net etki videoda pozitif
(1 tutarlı okuma eskisi 8 gürültülü okumaya karşı hâlâ bir kazanç) ama
"her yerde düzelme" iddia edilmiyor. Kök nedeni tam doğrulamak (kutu
sınırı karşılaştırması, kare kare) bu turun kapsamı dışında bırakıldı —
gelecekte daha fazla pseudo-label eklenirse tekrar kontrol edilmeli.

## 37. Tracker kimlik hatası düzeltildi (asıl neden: aynı karede car+bus çifte tespit) + kamera CSS bug'ı + tam video regresyon ölçümü

**(1) Tracker düzeltmesi — teşhis kullanıcının hipotezinden farklı çıktı:**
Kod incelemesi: `VehicleTracker.update()` zaten `vehicle_type`'ı eşleştirme
kriteri olarak KULLANMIYORDU (yalnızca IoU). Gerçek nedeni bulmak için
held-out'taki bölünme noktasını (kare 798-816) ham çıktısıyla inceledim:
COCO araç dedektörü **aynı karede, aynı fiziksel araç için hem "car" hem
"bus" kutusu üretiyor** (örn. kare 809: `car (135,252)-(528,461)` VE
`bus (132,252)-(527,461)` — neredeyse özdeş koordinatlar). Kök neden bu:
YOLO'nun NMS'i sınıf-bazlı çalışıyor, aynı nesnenin iki farklı sınıf
etiketli kutusunu birbirine bastırmıyor. Bu çift kutular her ikisi de
aynı anda track için "yarışınca", biri track'i alıyor, diğeri yeni bir
track açıyor — fiziksel tek araç 2 track'e bölünüyor, oylama havuzu
yarıya iniyor.

**Düzeltme (`src/plaka/pipeline/tracker.py` yeniden yazıldı):**
`_merge_same_frame_duplicates()` — aynı karede birbirine çok yüksek IoU'lu
(≥0.7, eşleştirme eşiğinden çok daha sıkı) kutuları union-find ile
kümeliyor, eşleştirme için kümenin en yüksek-güvenli temsilcisini
kullanıyor ama plaka/tip gözlemlerini kümenin TÜM üyelerinden topluyor.
`VehicleTrack.vehicle_type` artık sabit değil, `consensus_vehicle_type`
(plaka metniyle aynı çoğunluk oyu + confidence-tie-break mantığı)
— gösterilen araç tipi de artık kare-arası tutarsızlıktan arınıyor.

**Testler:** 6 yeni test (aynı karede car+bus → tek track; tip
salınımı kare-arası → track bölünmez; gerçekten ayrı yakın araçlar
birleşmez; kümenin herhangi bir üyesinden gelen plaka okuması kaybolmaz;
tip çoğunluk oyu; `apply_consensus` gösterilen tipi günceller) —
toplam 19 tracker testi, 205 test genelinde hepsi geçiyor.

**Held-out doğrulama (kare 721-900), düzeltme öncesi/sonrası:**

| | track sayısı | oylama havuzu | sonuç |
|---|---|---|---|
| önce | 2 (car, bus) | 7+3 (bölünmüş) | ikisi de "23 ACM 638" (rastlantıyla aynı) |
| **sonra** | **1** | **10 (birleşik)** | "23 ACM 638", daha güçlü destekle |

**(2) Kamera CSS bug'ı — düzeltildi ve gerçek tarayıcıda doğrulandı:**
`style.css`'teki `.camera-view video, .camera-view img { display: block; }`
kuralı, tarayıcının `[hidden] { display: none }` varsayılan kuralını
eziyordu (yazar stilleri her zaman UA stillerini geçersiz kılar, `hidden`
seçicisinin nominal specificity'si önemli değil). `:not([hidden])` ile
düzeltildi. **Playwright kurulu değildi; kurmak yerine** mevcut Chrome +
zaten kurulu `websockets` paketiyle ham Chrome DevTools Protokolü
üzerinden headless Chrome'u (`--use-fake-device-for-media-stream` ile
sahte kamera) gerçekten başlatıp, "Canlı Kamera" sekmesine tıklayıp,
"Kamerayı Başlat"a tıklayıp `getComputedStyle` ile doğruladım: kamera
başlamadan önce ikisi de `display:none`; başladıktan sonra ham video
**hâlâ `none`** (doğru, gizli kalmaya devam ediyor), işlenmiş görüntü
**`block`** (görünür). Ekran görüntüsü de tek panel gösterdi.

**(3) Tam video regresyon ölçümü — kritik bulgu, kullanıcıya bildirildi:**
Düzeltilmiş tracker'la `arac3.mp4`'ün tamamı (901 kare), iki checkpoint
karşılaştırıldı:

| | geçerli okunan farklı araç sayısı | detay |
|---|---|---|
| önce (orijinal `best.pt`) | **2** | "23 T 0445" (train aralığı, 2 gözlem) + "23 AC 638"≈ACM638 (held-out, 8 gözlem) |
| sonra (ince ayarlı) | **1** | yalnızca ACM638 kaldı (10 gözlem) — "23 T 0445" tamamen kayboldu |

Held-out'taki tek araç daha sağlam okunuyor ama train aralığındaki araç
tamamen kayboldu — **net etki videoda 2 araçtan 1 araca düşüş, net kazanç
değil**. Kullanıcının kendi belirlediği eşiğe göre ("net kayıpsa haber
ver") durum bildirildi, config henüz geri alınmadı — karar bekleniyor.

**(4) yolo11n vs yolo11s (düşük öncelik, ek güvence amaçlı):** Aynı
videoda araç dedektörü karşılaştırıldı — **yolo11s tip-çift-tespit
sorununu düzeltmiyor, aksine hafifçe kötüleştiriyor** (%1.0 → %2.3
karede aynı-nesne farklı-sınıf çifte tespit) ve %34 daha yavaş (77→57
fps), üstelik bu videoda hiç "bus" sınıfı üretmiyor (yolo11n 6 kez
üretiyor). **Sonuç: daha büyük stok model gerekmiyor — tracker
düzeltmesi (madde 1) zaten yeterli ve doğru çözüm**, model değişikliği
önerilmiyor.

## 38. Fine-tune checkpoint'i geri alındı — 19 örneklik self-training bu ölçekte riskli bulundu

**Karar:** `configs/detection.yaml` → `plate_detector.weights_path`
orijinal `models/plate_detector/best.pt`'e döndürüldü. Gerekçe: karar
#37'nin tam-video ölçümü, ince ayarlı checkpoint'in geçerli okunan
**farklı araç sayısını 2'den 1'e düşürdüğünü** gösterdi (held-out'taki
kazanç — 8→10 gözlem, aynı araç — bunu telafi etmiyor). Bir demo için
"garantili 2 doğru okuma" > "1 daha güçlü ama diğerini kaybeden okuma".

**Sonuç/ders — bu ölçekte self-training riskli:** 19 pseudo-label
görüntüyle (2 gerçek araç geçişinden, görsel olarak doğrulanmış kutularla)
yapılan hafif ince ayar, genel val setinde çökme yaratmadı (mAP50
%79.1→%77.9) ve hedeflenen held-out penceresinde gerçek bir iyileşme
sağladı (8→10 kare, 4 kare daha erken) — ama **aynı videonun başka bir
yerinde, ince ayarın kendisiyle ilgisiz görünen bir araçta** (muhtemelen
pseudo-label kaynağı karelerdeki kutu sınırı kaymasının OCR karakter
bölütlemesini bozması yüzünden) tam bir regresyona yol açtı. 19 örnek,
modelin genel davranışını öngörülemeyen şekillerde değiştirmeye yetecek
kadar veri — bu ölçekte pseudo-label self-training'in **üretime alınmadan
önce mutlaka tam-video (sadece hedef pencere değil) önce/sonra
karşılaştırmasını gerektirdiği** kayıt altına alınıyor. Daha büyük ve
çeşitli bir pseudo-label kümesi (birden fazla video/geçiş) bu riski
azaltabilir ama şimdilik bu yol durduruldu.

`models/plate_detector_arac3_finetune/` **silinmedi**, deneysel bir
artefakt olarak duruyor — gelecekte daha fazla pseudo-label birikirse
yeniden değerlendirilebilir bir başlangıç noktası.

**Son doğrulama (orijinal checkpoint + düzeltilmiş tracker, tam video,
gerçek üretim pipeline'ı üzerinden — `build_pipeline_from_config`):**

```
processed 901 frames
  track   2 (car): '23 T 0445'  (2 obs, kare 114'e kadar)
  track  22 (car): '23 AC 638'  (8 obs, kare 813'e kadar)
TOTAL: 2 distinct vehicle(s) with a valid consensus reading
```

İki araç da tek, bölünmemiş track olarak okunuyor — karar #37'deki
tracker düzeltmesi bu senaryoda da doğru çalışıyor, checkpoint'ten
bağımsız bir kazanım olduğu doğrulandı.

**yolo11n vs yolo11s kararı (karar #37 madde 4'ün özeti):** `yolo11n`
kalıyor — `yolo11s` tip-çift-tespit sorununu düzeltmedi (aksine %1.0→%2.3
kötüleşti) ve %34 daha yavaştı, hiçbir net fayda sağlamadı.

## 39. Web özet kartı yanlış sayıyordu (kare-bazlı, track-bazlı değil) + döndürme önizlemesi + döndürme/mesafe ayırt edici teşhis

**(1) `vehicle_type_counts` düzeltildi — kök neden gerçekten basitti:**
`JobManager._run` her karedeki her tespiti tek tek sayıyordu; `arac3.mp4`
(901 kare, 2 gerçek araç) için bu "2325 toplam araç" gibi anlamsız bir
sayı üretiyordu — plaka tarafı (track-bazlı, karar #35) düzeltilirken bu
kısım unutulmuştu. Düzeltme: `track_types: dict[track_id, vehicle_type]`
her karede upsert ediliyor (plaka olsun olmasın, her track için), kart
`Counter(track_types.values())`'tan hesaplanıyor — artık "kaç kare-tespiti"
değil "kaç farklı araç" sayıyor. Kart etiketi de "Toplam Araç Tespiti"
→ **"Tespit Edilen Araç Sayısı"** yapıldı. 2 yeni test (statik tek-araç
senaryosu artık `{"car": 1}` bekliyor, 2-farklı-araç senaryosu
`{"car": 1, "motorcycle": 1}` — önceden `{"car": 6, "motorcycle": 6}`
olurdu). 206 test, hepsi geçiyor.

**Gerçek tarayıcıda uçtan uca doğrulama** (headless Chrome + CDP,
gerçek dosya seçimi + gerçek upload + gerçek 901 kare işleme):
`arac3.mp4` sonucu artık **"28 Tespit Edilen Araç Sayısı" / "2 Benzersiz
Plaka" / "26 Otomobil" / "2 Motosiklet"** — makul, inanılır bir sayı
(videodaki gerçek arka plan trafiği dahil), 2325 değil.

**(2) Döndürme görsel önizlemesi eklendi:** Kullanıcı artık "270° sola
çevir" gibi metinler arasında tahmin yapmak zorunda değil — video
seçilir seçilmez tarayıcıda (`<video>`+`<canvas>`, yükleme yok) ilk
kareden 4 döndürme (0/90/180/270) küçük önizlemesi üretiliyor, doğru
duran birine tıklanıyor. `app.js`'teki `drawRotated()` merkez-taşı-döndür
tekniğini kullanıyor (90/270 için kutu boyutu otomatik takas ediliyor).

**Kritik doğrulama — JS döndürmesi backend'le piksel piksel aynı mı?**
Playwright kurulu değildi; gerçek çalışan sunucuyu headless Chrome +
ham DevTools Protokolü ile açıp, sayfadaki **gerçek `drawRotated`
fonksiyonunu** (yeniden yazılmış bir kopyasını değil) bir test görüntüsü
üzerinde 4 değer için çalıştırdım, çıktıları `apply_rotation()`'ın
(backend, cv2) aynı görüntü üzerindeki çıktısıyla görsel olarak karşılaştırdım
— **4 rotasyonun tamamı birebir eşleşti**. Bu, önizlemede "doğru duran"
seçilince gerçekten sunucunun da aynı şekilde döndüreceğinin garantisi.

**(3) Döndürme vs mesafe ayırt edici teşhis — yeni `scripts/diagnose_rotation.py`:**
Tek seferlik CLI aracı (video job pipeline'ına gömülmedi) — plaka
dedektörünü 0.05 eşikte 4 döndürmenin tümüyle deniyor, hangisinde
araç-içi plaka kutusu daha çok/daha büyük çıktığını raporluyor; kazanan
döndürmenin kutuları hâlâ küçükse (< 50px medyan — `arac2.mp4`'ün
doğrulanmış çözünürlük tabanıyla aynı büyüklük, karar #32/#33) ayrıca
uyarıyor. **İki bilinen videoda doğrulandı:** `arac3.mp4` → rotate=270
açık ara kazanıyor (149 vs 0 kare); `arac2.mp4` → rotate=0 kazanıyor
(105 vs 0 kare) ama kutular hâlâ küçük (medyan 32.2px) — script doğru
şekilde "döndürme sorunu değil, muhtemelen mesafe/çözünürlük sorunu"
uyarısını tetikliyor. Kullanıcının kendi videosu için de aynı komutu
çalıştırması yeterli: `python scripts/diagnose_rotation.py <video> --max-frames 200`.

## 40. Video galerisi artık her tespit edilen aracı gösteriyor, sadece plakası okunanları değil

**Karar:** `job.plate_sightings` yalnızca `plate.is_format_valid` olan
track'leri kaydediyordu — kullanıcı sisteme hiç yakalanmayan araçlarla,
yakalanıp da sadece plakası okunamayan araçları ayırt edemiyordu.
`VideoJob.vehicle_sightings` (yeniden adlandırıldı) artık her track
için bir kart tutuyor, `plate_status` alanıyla üç durumu ayırıyor:
"read" (geçerli format, konsensüs metni var), "unreadable" (plaka
kutusu en az bir kez bulundu ama hiç geçerli formata ulaşmadı — varsa
raw_ocr_text ham OCR denemesini taşıyor, gözle çapraz kontrol için),
"no_plate" (bu track için plaka kutusu hiç bulunamadı).

**Temsilci kare seçimi:** Bir track hiç plaka kutusu görmüşse, o kutuların
en yüksek OCR-confidence'lı karesi temsilci seçiliyor (geçerli/geçersiz
fark etmez — en iyi OCR denemesinin göründüğü kare, plakanın gerçekten
okunamaz mı yoksa yakın mı olduğunu görsel kontrol için en faydalısı).
Hiç plaka kutusu bulunamamışsa araç kutusunun en büyük olduğu kareye
düşülüyor (insan gözüyle plakayı yine de fark etme şansı en yüksek kare).

**Kapasite:** MAX_SAMPLE_FRAMES=12 -> MAX_TRACKED_VEHICLES=40 — artık
sadece başarılı okumalar değil tüm track'ler sayıldığı için
(arac3.mp4 tek başına ~28 gerçek track içeriyor, sıradan arka plan
trafiği dahil) daha fazla alana ihtiyaç vardı.

**Dedupe — aynı araç iki kart olarak görünmesin:** Tracker bazen aynı
fiziksel aracı 2 ayrı track'e böler (kısa bir kopukluk, karar #37'deki
gibi). Geçerli okunan plakalar için güvenilir bir anahtar var (normalize
metin) — _dedupe_sightings() aynı metne sahip birden fazla track'i tek
karta indiriyor (daha çok gözlemi olan kazanıyor). Plakası okunamayanlarda
böyle bir anahtar yok — mükemmel dedupe garanti edilemiyor, ayrı kartlar
kalabilir (bilinen, kabul edilmiş bir sınır). Galeri sırası: önce
okunanlar, sonra okunamayanlar/görünmeyenler.

**Frontend:** Kartlar iki görsel duruma ayrıldı — okunan plakalar mevcut
yeşilimsi "plate-pill", okunamayanlar nötr gri "status-pill" (+ varsa
soluk italik ham OCR metni). Her kartta araç tipi + zaman damgası her
zaman gösteriliyor. "Benzersiz Plaka" özet sayısı artık sadece
plate_status=="read" olanları sayıyor (toplam kart sayısını değil).

**Testler:** 3 yeni senaryo — plaka kutusu bulunup hiç doğrulanmayan
track (raw_ocr_text doğru taşınıyor), plaka kutusu hiç bulunmayan
track, ve aynı plakaya yakınsayan 2 ayrı track'in tek karta indiği
(dedupe). 208 test, hepsi geçiyor.

**Gerçek doğrulama (arac3.mp4, üretim pipeline'ı, gerçek tarayıcı):**
28 track -> 2 okunan ("23 T 0445", "23 AC 638"), 8 okunamayan
(hepsinde anlamlı ham OCR parçaları var: "ZWU3", "6NR", "UCM", "2A0" gibi
— plakanın kısmen görüldüğünü ama tam çözülemediğini gösteriyor), 18
plaka görünmeyen (çoğu uzak/arka plan araç ve motosiklet). Galeri
doğru sırada (önce 2 okunan), her kartta doğru araç tipi + zaman damgası,
thumbnail'ler doğru kareyi gösteriyor.

## 41. Web arayüzü görsel/estetik yeniden tasarım — "kontrol paneli" konsepti

**Karar:** Fonksiyonel taraf sağlamlaştıktan sonra (tracker+konsensüs,
galeri, döndürme önizlemesi, kamera), kullanıcı arayüzün de "resmi bir
plaka tanıma/trafik izleme sistemi" izlenimi vermesini istedi — sunumda
amatör görünmemesi için. Değişiklik sadece `style.css`/`index.html`/`app.js`
üzerinde, backend'e dokunulmadı, build tool eklenmedi (proje felsefesi
gereği — Python + düz HTML/CSS/JS, sadece CDN'den font/ikon çekiliyor).

**Renk sistemi:** Tek renkli kırmızı yerine iki tonlu bir sistem —
`--red` (TR bayrağı) birincil aksiyon rengi (butonlar, aktif tab)
olarak kalıyor, yeni `--blue` (plakanın mavi Euroband şeridinden
esinlenilmiş, `#1a3fa0`) ikincil/bilgilendirici aksan oldu (ikincil
butonlar, araç tipi ikonları, özet istatistiklerin bir kısmı,
ilerleme çubuğunun gradyanı). Nötr gri tonlar genişletildi
(`--ink-soft`, `--ink-softer`, `--border`, `--border-strong`) kart
derinliği/hiyerarşisi için.

**Koyu/açık kontrast:** Üst bar (`--ink` tabanlı koyu gradyan, alt
kenarda kırmızı 3px çizgi) zaten kısmen koyuydu — güçlendirildi, "resmi
sistem" hissini pekiştiriyor. Kenar çubuğu eklenmedi — 3 sekmelik yatay
navigasyon için gereksiz karmaşıklık olurdu. Kart içerikleri temiz beyaz
kaldı (mevcut kontrast korunuyor).

**Tipografi:** Google Fonts CDN üzerinden Inter (400-800) genel arayüz
metni için, JetBrains Mono (500/700) plaka numaraları için eklendi
(`Consolas` fallback olarak korundu — kullanıcının özellikle "kaybetme"
dediği detay). Başlıklarda daha güçlü ağırlık/hiyerarşi.

**İkonografi:** Lucide icon seti CDN üzerinden (`unpkg.com/lucide`)
eklendi — düz karakterler ("＋") ve emoji uyarı işaretleri ("⚠") SVG
ikonlarla değiştirildi. Araç tipi ikonları eklendi (otomobil→car,
kamyon→truck, otobüs→bus, motosiklet→bike — Lucide'de "motorcycle" yok,
en yakın karşılık "bike", CDN'den `-L` ile 200/404 kontrol edilerek
doğrulandı). Tüm ~35 ikon adı kullanılmadan önce gerçek CDN'e karşı
tek tek doğrulandı (ilk denemede `-L` olmadan curl her isim için 302
döndürdüğü için — unpkg var olmayan yollarda da yönlendirme yapıyor —
bu yanlış pozitif fark edilip düzeltildi).

**Bulunan ve düzeltilen gerçek hata:** Döndürme önizleme kartlarında
(video sekmesi) seçili olmayan 3 küçük resimde de checkmark ikonu
görünüyordu — sadece seçili olanda görünmesi gerekirken. Kök neden:
Lucide'in `createIcons()` fonksiyonu, kaynak `<i data-lucide="check"
hidden>` elemanındaki `hidden` HTML attribute'unu üretilen `<svg>`'ye
güvenilir şekilde taşımıyor (sadece `class` gibi birkaç standart
attribute taşınıyor). Düzeltme: görünürlük tamamen CSS'e taşındı
(`.rotate-thumb-check { display: none; } .rotate-thumb.selected
.rotate-thumb-check { display: inline; }`), `app.js`'teki
`setSelectedRotation()` sadece `.selected` class'ını toggle'lıyor,
`index.html`'deki gereksiz `hidden` attribute'ları kaldırıldı. Gerçek
tarayıcıda (headless Chrome + CDP) computed style kontrolüyle
doğrulandı: sadece seçili thumb'da `display:block`.

**Doğrulama:** Fotoğraf/video/kamera sekmelerinin tümü gerçek veriyle
(`foto.jpeg`, `arac3.mp4`, fake kamera cihazı) headless Chrome
üzerinden ekran görüntüsü alınarak kontrol edildi — dropzone, döndürme
önizlemesi + seçim, örnekleme uyarısı, iş ilerleme çubuğu, galeri
kartları (durum ikonlu), kamera canlı görünümü dahil hiçbir mevcut
işlev bozulmadı. `pytest tests/unit` 208 test, hepsi geçiyor (bu
değişiklikler frontend-only olduğu için beklenen sonuç).
