# SONIC kol hedefi neden yükseliyor? (23 Eylül 2026)

Bu not, mevcut BlockStacking deneylerinin kanıtlarını bir araya getirir. Yeni bir canlı
rollout sonucu değildir. İncelenen dört eski rollout ile sonraki kontrollü deneyler aynı
görevi kullanır; başka göreve genelleme henüz ölçülmedi.

## Ayrılması gereken dört seviye

1. **Demonstrasyon:** 78D eğitim aksiyonunun ilk 64 değeri SONIC hareket latentıdır.
   Demonstrasyonun ölçülmüş eklemleri latentın tek ve sabit bir pozu değildir.
2. **VLA çıktısı:** Ψ₀ veya GR00T latent üretir. Ψ₀ köprüsü bunu FSQ gridine yuvarlar.
3. **SONIC hedefi:** SONIC v1.1 decoder latenta ek olarak 10 karelik robot durumu,
   hız, yerçekimi yönü ve önceki aksiyon geçmişi alır; mutlak eklem hedefi üretir.
4. **Isaac ölçümü:** Bu hedef PD ve tork sınırıyla uygulanır. Hedef iskeleti (sarı/camgöbeği)
   üçüncü seviyeyi gösterir; robotun gerçek pozu dördüncü seviyedir.

## Doğrudan ölçülenler

- BlockStacking eğitim kümesinin 286 bölümündeki geçerli karelerde medyan palm-z
  (pelvis çerçevesi) sol/sağ **0,197/0,210 m**. İki hareketli demonstrasyonun kayıtlı
  latentları deployment decoderında kendi palm hareketini izledi; plant projeksiyonlu
  palm-z MAE iki elde de **5 cm altında** kaldı. Bu, “eğitim latentı doğal olarak
  elleri yukarı istiyor” açıklamasını desteklemiyor.
- Eski dört canlı rolloutta hedef palm, ölçülen palmın medyan **0,208–0,281 m**
  üstündeydi. Eklem hedefi gerçekten uygulanan PD'nin `q` alanı: kaydedilen
  tork, `tau_ff + kp(q_target-q) + kd(dq_target-dq)` ile **1,1e-5 N·m**
  en büyük artıkla yeniden elde edildi. Gecikme veya koordinat dönüşümü bu farkı
  açıklamadı.
- Varsayılan kolda `tau_ff=0`, `kp=14,25 N·m/rad`. Kaydedilen P torku
  yerçekimi momentiyle dengeleniyor; uzatılmış kolda hata büyüyor. Sabit yüksek
  hedef deneyinde yerçekimi telafisi palm-z MAE'yi **0,4269→0,0114 m** düşürdü.
- Aynı Ψ₀ checkpointiyle üç eşleşmiş kapalı çevrim gravity A/B koşusunda tracking
  hatası **%70–87** azaldı. Modelin *ürettiği hedef* de **0,09–0,30 m** aşağı indi.
  Bu, fazla yüksek hedefin önemli bölümünün yanlış plant geri bildirimine bağlı
  olduğunu gösteren en güçlü doğrudan kanıt. Ancak hiçbir koşuda kavrama oluşmadı.
- Token destek testi, iki checkpointin canlı latentlarını BlockStacking
  demonstrasyonlarının merkezinden uzakta buldu (Ψ₀ medyan L1 **3,00**,
  GR00T **2,27**, held-out demonstrasyon **1,25**). Bu bir genel policy/dağılım
  sorunu olasılığını artırıyor, fakat yüksek palm bölümleriyle tutarlı bir ilişki
  yalnız dört rollouttan birinde görüldü; tek başına nedensel açıklama değil.
- İlk policy gözlemi dört eski rolloutun üçünde, policy başlamadan önce
  demonstrasyon üst gövde desteğinin dışındaydı. Bu, kötü başlangıç durumunun
  ek bir katkı olabileceğini gösterir; warmstart/handshake deneyleri yüksek hedefi
  güvenilir biçimde gidermedi.

## Sonuç ve sınır

**Hedefin yüksek olması ile robotun hedefe yetişememesi ayrı olgular.** İkincisinin
nedeni düşük kazanç ve yerçekimi telafisinin yokluğu için güçlü mekanik kanıt var.
Birincisi Ψ₀'da aynı plant düzeltilince belirgin azaldı; dolayısıyla tüm yüksekliği
fine-tune etiketlerine yüklemek yanlış. GR00T'un eski sol omuz komutları demonstrasyon
p05–p95 aralığının dışındaydı; iyi tracking ile yapılan sonraki GR00T koşusunda da
hedefler küplerden uzaktı. Eğitim verisi için kanıtlanan şey masa seviyesine inişin
mevcut olduğu; iki modelin bu sahnede onu kapalı çevrim uygulayabildiği değil.

Kesin görev kök nedeni açık: eğitim kamerasının kesin kalibrasyonu, masa/obje
dönüşümü ve aynı sahne/plant üzerinde başarılı referans replay yok. Sıradaki
ayırt edici ölçüm, yeni görevler dahil aynı başlangıç ve görüntüyle şu dört
diziyi aynı sim zamanında saklamak: demonstrasyon latentı, canlı VLA latentı,
SONIC `q_target`, Isaac `q_measured`. Palm pozları pelvis çerçevesinde ve
nesneye mesafeyle karşılaştırılmalı. Yerçekimi telafisi açık/kapalı koşuları
ayrı, eşleşmiş tekrarlar olarak tutulmalı; mevcut varsayılan hâlâ kapalıdır.

## Kanıtlar

- `data/outputs/blockstacking-debug/dataset-comparison/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/01-demo-token-decoder/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/02-state-support/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/03-tracking-validation/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/04-gravity-comp-ab/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/05-policy-token-support/REPORT.md`
- `data/outputs/blockstacking-debug/experiments/10-psi0-gravity-rollout-ab/REPORT.md`
- `data/outputs/blockstacking-debug/followup-diagnosis/REPORT.md`
