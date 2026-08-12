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
  - [ ] Veri setleri indirilip `data/external/`e yerleştirilecek (onay
    bekleniyor — bkz. plan özeti).
  - [ ] Plaka dedektörü ilk eğitimi (VMMRdb'de plaka bbox etiketi yok;
    Türk/çok-ülkeli plaka veri setleri gerekiyor).
  - [ ] Marka/model sınıflandırıcı ilk eğitimi (VMMRdb + Stanford Cars).
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
