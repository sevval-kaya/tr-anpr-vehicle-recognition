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
  - [ ] Türk plaka veri seti (Kaggle) — kimlik bilgisi kurulumu kullanıcıya
    bırakıldı, henüz indirilmedi.
  - [ ] Plaka dedektörü ilk eğitimi (VMMRdb'de plaka bbox etiketi yok;
    Türk/çok-ülkeli plaka veri setleri gerekiyor).
  - [ ] Marka/model sınıflandırıcı ilk eğitimi (VMMRdb ile başlanabilir).
  - [ ] Baseline pipeline'ın gerçek görüntülerle uçtan uca smoke test'i.
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

Aşama 1 tamamlandı, Aşama 2'nin kod tarafı (pipeline, doğrulayıcı, metrikler)
hazır ve test edilmiş durumda. Aşama 2'nin veri tarafına (indirme + ilk
eğitim) geçmeden önce, kullanıcıdan büyük kararlarda onay isteniyor — bkz.
oturum sonundaki plan özeti.
