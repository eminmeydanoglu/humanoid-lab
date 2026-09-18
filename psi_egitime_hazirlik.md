# Psi0 SONIC Eğitime Hazırlık Planı

**Tarih:** 17 Eylül 2026  
**Durum:** Tasarım kararları tamamlandı; veri dönüştürücü henüz uygulanmadı  
**Psi0 commit:** `4f3720d45e102b36d7c3e9465ab8062274170518`  
**Hedef:** Unitree G1 Dex3 demonstrasyonlarını, üretilmiş SONIC v1.1 latent aksiyonlarıyla birleştirerek Psi0 fine-tuning dataseti hazırlamak; tek RTX 5090 üzerinde smoke testten sonra gerçek eğitimi başlatmak.

---

## 1. Kısa karar özeti

| Konu | Karar |
|---|---|
| Kullanılacak veri | Yalnızca 13 resmi Unitree G1 Dex3 koleksiyonu |
| Kullanılabilir episode | 3.150 |
| Hariç tutulan episode | `ObjectPlacement/0`, `ToastedBread/396` |
| Fruits / Apple-to-Plate | Bu eğitime dahil edilmeyecek |
| Kaynak görüntü frekansı | 30 Hz |
| SONIC kaynak frekansı | 50 Hz; kaynak olarak aynen korunacak |
| Final Psi0 dataset frekansı | 30 Hz |
| Kamera | Her zaman `cam_left_high` |
| Model kamera anahtarı | `observation.images.egocentric` |
| Action | 64D SONIC token + 14D Dex3 el + 2D sıfır padding = 80D |
| State | 15D sentetik standing alt gövde/bel + 28D ölçülen kol/el = 43D; modelde 45D'ye padding |
| State'in ölçülen kısmı | Ham `observation.state`; ham `action` observation olarak kullanılmayacak |
| Action horizon | 30 adım @ 30 Hz, yaklaşık 1 saniye |
| Kamera çözünürlüğü | İlk hedef `240×320` |
| Warm-start | `psi0/postpre.sonic1.0.unifolm.2609092156.40k` |
| Eğitim stratejisi | VLM frozen, action expert trainable |
| İlk microbatch | 1 |
| İlk gradient accumulation hedefi | 32; gerçek değer VRAM/throughput smoke testinden sonra kesinleşecek |
| Logging | W&B offline |
| Görselleştirme | Hazır LeRobot + Rerun; gerekirse Psi0 SONIC visualizer |
| Validation | Episode bazında yaklaşık %5; her 13 görev temsil edilecek |

---

## 2. Değiştirilmeyecek kaynaklar

Aşağıdaki veri kaynakları dondurulmuş girdi kabul edilir. Dönüşüm bunları yerinde değiştirmeyecek.

### 2.1 Ham Unitree G1 Dex3

```text
data/datasets/first_tur_ham/unitree-g1-dex3/
```

Özellikler:

- 13 koleksiyon
- 3.152 episode
- 2.587.515 frame
- 30 Hz
- yaklaşık 49 GiB
- `observation.state`: 28D `float32`
- `action`: 28D `float32`
- iki baş kamerası:
  - `observation.images.cam_left_high`
  - `observation.images.cam_right_high`
- kameralar 640×480, 30 FPS, AV1

28D kaynak düzeni:

```text
0:14   sol ve sağ kol eklemleri
14:28  sol ve sağ Dex3 el eklemleri
```

`observation.state` ile `action` aynı şey değildir:

- `observation.state`: ölçülen mevcut kol/el eklem pozisyonu
- `action`: robota gönderilen istenen kol/el pozisyonu

Psi0 observation state hazırlanırken **ham `observation.state` kullanılacaktır**.

### 2.2 SONIC v1.1 78D latent korpusu

```text
data/datasets/unitree-sonic-v1.1-78d/
```

Özellikler:

- 13 koleksiyon
- 3.150 episode
- 4.307.871 adet 50 Hz frame
- 4.166.121 geçerli training frame
- yaklaşık 1,4 GB
- production QC: `PASS`

Her episode:

```text
episodes/episode_XXXXXX/
├── action.npz
└── manifest.json
```

`action.npz` alanları:

| Alan | Şekil | Anlam |
|---|---:|---|
| `action` | `[T, 78]` | 64D SONIC token + 14D Dex3 hedefi |
| `timestamp` | `[T]` | 50 Hz zaman çizgisi |
| `frame_index` | `[T]` | Episode-local frame |
| `training_valid_mask` | `[T]` | Future window gerçekse `true` |

78D düzeni:

```text
0:64   SONIC v1.1 whole-body motion token
64:71  sol Dex3 hedefi
71:78  sağ Dex3 hedefi
```

Bu final korpusun içinde observation state veya video yoktur. Bunlar ham datasetten alınacaktır.

### 2.3 Elenen iki episode

| Dataset | Episode | Neden |
|---|---:|---|
| `G1_Dex3_ObjectPlacement_Dataset` | 0 | 49 kaynak frame; SONIC future window üretimi için fazla kısa |
| `G1_Dex3_ToastedBread_Dataset` | 396 | Upstream episode metadata/parquet offset tutarsızlığı |

Bu iki episode yeniden dahil edilmeyecektir. Final beklenen sayı tam olarak **3.150 episode** olacaktır.

---

## 3. Yeni final dataset

Yeni eğitim dataseti mevcut kaynaklardan ayrı üretilecektir.

Önerilen kök:

```text
data/datasets/psi0-unitree-dex3-sonic-v1/
```

Train ve validation, Psi0'ın `train_repo_ids` / `val_repo_ids` düzenine uygun ayrı LeRobot datasetleri olacaktır:

```text
data/datasets/psi0-unitree-dex3-sonic-v1/
├── train/
│   ├── data/
│   ├── videos/
│   └── meta/
├── val/
│   ├── data/
│   ├── videos/
│   └── meta/
└── split_manifest.json
```

Kaynak 50 Hz SONIC korpusu bu dönüşümden sonra da aynen korunacaktır.

---

## 4. Neden final dataset 30 Hz olacak?

Psi0'ın yayımlanmış SONIC reçetesi:

- 30 Hz kontrol zaman ölçeğine,
- `action_chunk_size=30`,
- yaklaşık bir saniyelik gelecek aksiyon penceresine

uygundur.

Dataset 50 Hz ilan edilirse 30 aksiyon yalnızca 0,6 saniyeyi kapsar ve warm-start checkpoint'in öğrendiği zaman ölçeği değişir.

Bu nedenle:

1. Kamera 30 Hz kalacak.
2. Observation state ham 30 Hz zaman çizgisinde kalacak.
3. Her 30 Hz hedef timestamp için 50 Hz SONIC korpusundaki karşılık seçilecek.
4. Video 50 FPS'e dönüştürülmeyecek.
5. `action_chunk_size=30` korunacak.

İlk yöntem, her 30 Hz timestamp'e en yakın 50 Hz örneği deterministik seçmektir. Latent uzayında lineer interpolation'ın güvenli olduğu doğrulanmadığı için başlangıçta nearest-timestamp kullanılacaktır.

Dönüşümün küçük örnek üzerindeki görsel ve sayısal eşleşmesi Kapı 1'de doğrulanacaktır.

---

## 5. Kamera sözleşmesi

Karar kesindir:

```text
kaynak: observation.images.cam_left_high
hedef:  observation.images.egocentric
```

Kurallar:

- Her episode'da sol kamera kullanılacak.
- Rastgele sol/sağ seçimi yapılmayacak.
- Sağ kamera training input'una eklenmeyecek.
- Ham sağ kamera dosyaları korunacak.
- İlk model input çözünürlüğü `240×320` olacaktır.
- Kaynak 640×480 görüntünün 4:3 en-boy oranı korunacaktır.

Hazır görselleştirme sırasında sağ kamera istenirse karşılaştırma amacıyla ayrıca incelenebilir; bu durum training contract'ını değiştirmez.

---

## 6. Observation state sözleşmesi

### 6.1 Önemli ayrım

SONIC latent üretiminde kullanılan 50 Hz bütün-gövde referansının kol ve el bölümü ham `action` alanından türetilmiştir. Bu referans istenen hareketi temsil eder; gerçek observation değildir.

Bu nedenle bütün `reference.npz` dosyası Psi0 `observation.state` olarak kullanılmayacaktır.

### 6.2 Final 43D state

Ham dataset bacak ve bel observation'ı kaydetmediği için final state şu şekilde oluşturulacaktır:

```text
state43 = lower_body_and_waist_15
        + measured_arms_14
        + measured_hands_14
```

Kaynaklar:

| Bölüm | Boyut | Kaynak | Niteliği |
|---|---:|---|---|
| Bacaklar + bel | 15D | `deployment_standing_pose()` | Sentetik, sabit standing proxy |
| Kollar | 14D | Ham `observation.state[0:14]` | Ölçülen observation |
| Eller | 14D | Ham `observation.state[14:28]` | Ölçülen observation |

Model tarafında:

```text
state45 = state43 + [0, 0]
```

Son iki kanal kullanılmayan boyun padding'idir.

### 6.3 Standing değerlerinin anlamı

15D alt-gövde/bel değerleri:

- SONIC kontrolcünün çalışması sonucu ölçülmüş gerçek qpos değildir.
- Dataset boyunca varsayılan resmi SONIC standing hedefinde sabit kabul edilen sentetik değerlerdir.
- SONIC token üretimindeki alt-gövde varsayımıyla aynıdır.
- Dataset stationary manipulation için kullanıldığı için uygun bir proxy'dir.

Metadata bu durumu açıkça kaydedecektir:

```text
arms/hands: measured observation
legs/waist: synthetic constant SONIC standing proxy
```

Bu kanallar dataset boyunca sabit olduğundan normalization sırasında sıfıra dönüşecek ve model alt-gövde hareketi öğrenmeyecektir. Bu, kayıtta bulunmayan hareket bilgisini uydurmaktan daha doğrudur.

### 6.4 Joint sırası

Beklenen grup düzeni:

```text
0:15   bacaklar + bel
15:29  kollar
29:43  Dex3 eller
43:45  model-side zero padding
```

Tam joint isim sırası tahmin edilmeyecektir. SONIC'in `SONIC_REFERENCE_JOINT_ORDER`, Psi0'ın referans metadata'sı ve ham Unitree feature isimleri üzerinden name-based eşleme yapılacaktır.

Dönüştürücü başlangıçta en az şu kontrolleri yapacaktır:

- Kaynak `observation.state` 28D olmalı.
- Kaynak state joint isimleri beklenen Unitree isimleriyle birebir eşleşmeli.
- Final 43D isim listesi beklenen canonical sırayla birebir eşleşmeli.
- Kol ve el değerleri ham `observation.state` ile frame bazında aynı olmalı.
- Alt-gövde/bel kanalları episode boyunca sabit olmalı.

---

## 7. Action sözleşmesi

### 7.1 Dataset içinde saklanacak alanlar

Mevcut 78D birleşik array doğrudan `action` adıyla yazılmayacaktır. Psi0 SONIC reçetesindeki `action[:14]` ifadesi el aksiyonu anlamına geldiği için birleşik 78D array sessizce yanlış yorumlanabilir.

Final parquet alanları:

```text
action.body_token_v1_1 = source_action[:, 0:64]
action                 = source_action[:, 64:78]
```

### 7.2 Modelin gördüğü 80D action

Psi0 repack/padding sonrası:

```text
action80 = body_token_64
         + dex3_hands_14
         + neck_padding_2
```

Yani:

```text
0:64   SONIC v1.1 body token
64:78  Dex3 hedefleri
78:80  [0, 0]
```

Bu düzen, yayımlanmış SONIC warm-start checkpoint'inin `64 + 14 + 2 = 80D` sözleşmesine uygundur.

### 7.3 Action horizon

```text
dataset fps:          30
action chunk size:    30
action exec horizon:  30
training RTC:         kapalı
max delay:            8
```

RTC training sırasında kullanılmayacak; gerektiğinde deployment tarafında uygulanacaktır.

---

## 8. Geçerli training anchor'ları

SONIC encoder her token için gelecekteki referans penceresine bakar. Episode sonundaki future-clamped token'lar replay için geçerlidir fakat VLA supervision için uygun değildir.

30 Hz'e aktarım sırasında 50 Hz `training_valid_mask` de taşınacaktır.

Bir training anchor'ı ancak gelecekteki bütün 30 action target geçerliyse kullanılacaktır:

```text
anchor_valid[i] = all(valid_30[i : i + 30])
```

Kurallar:

- Yalnızca mevcut frame'in geçerli olması yeterli değildir.
- 30 hedef içinden biri geçersizse anchor training sample olmayacaktır.
- Geçersiz tail loss'a girmeyecektir.
- Mask bilgisi final datasette korunacaktır.

---

## 9. Psi0 ve LeRobot ilişkisi

Psi0'ın bu fine-tuning yolu LeRobot biçiminde dataset bekler. Yeni bir veri standardı icat edilmeyecektir; LeRobot'un hazır modality kutuları doldurulacaktır.

### 9.1 Temel alanlar

| Rol | Final alan |
|---|---|
| Video | `observation.images.egocentric` |
| State | `observation.state` |
| Body action | `action.body_token_v1_1` |
| Hand action | `action` |
| Instruction | `task_description` veya metadata üzerinden eşdeğer canonical alan |
| Zaman | `timestamp`, `frame_index` |
| Episode | `episode_index`, `index`, `next.done` |
| Görev | `task_index` |
| Validity | `training_valid_mask` ve/veya dönüştürülmüş anchor filtresi |

### 9.2 Metadata

Final datasetin en az şu LeRobot metadata dosyaları olacaktır:

```text
meta/info.json
meta/modality.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/stats.json
meta/stats_psi0.json
```

Rolleri:

| Dosya | Rol |
|---|---|
| `info.json` | FPS, episode/frame sayıları, feature şekilleri ve joint isimleri |
| `modality.json` | Video/state/action/language alan eşlemeleri |
| `tasks.jsonl` | Canonical görev metinleri |
| `episodes.jsonl` | Episode uzunluğu, görev ve kaynak bağlantısı |
| `stats.json` | Genel sayısal istatistikler |
| `stats_psi0.json` | Psi0 action/state normalization istatistikleri |

Train split istatistikleri normalization için kullanılacak; validation aynı `stats_psi0.json` sözleşmesini kullanacaktır.

---

## 10. Görev metinleri

Özel bir text-editing uygulaması yapılmayacaktır. Görev metinleri dataset üretim config'i veya küçük bir metadata mapping içinde tutulacaktır.

Bilinen upstream hata mutlaka düzeltilecektir:

```text
G1_Dex3_GraspSquare_Dataset
eski: camera packaging
yeni: grasp square nesnesini doğru anlatan canonical İngilizce instruction
```

13 görev için kısa, tutarlı İngilizce canonical metinler conversion başlamadan önce tek listede sabitlenecektir. Bunlar deployment sırasında verilecek instruction'larla aynı anlamı taşımalıdır.

Metin varyasyonu/paraphrase ilk eğitim için kullanılmayacaktır.

---

## 11. Train / validation ayrımı

İlk hedef:

```text
train: yaklaşık %95
validation: yaklaşık %5
```

3.150 episode için yaklaşık:

```text
train: 2.992 episode
validation: 158 episode
```

Kesin sayı stratified split sonucunda belirlenecektir.

Kurallar:

- Split episode bazında yapılacak.
- Aynı episode train ve validation'a bölünmeyecek.
- Her 13 koleksiyon validation içinde temsil edilecek.
- Seed sabitlenecek ve `split_manifest.json` içine yazılacak.
- İki hariç episode hiçbir split'e girmeyecek.

### Validation'ın amacı

Training loss, modelin ağırlıklarını güncellediği örneklerdeki hatadır. Validation loss ise modelin training sırasında ağırlık güncellemek için görmediği episode'lardaki hatadır.

Temel yorum:

| Training loss | Validation loss | Yorum |
|---|---|---|
| Düşüyor | Düşüyor | Öğrenme ve genelleme sağlıklı |
| Düşüyor | Yükseliyor | Overfitting riski |
| Değişmiyor | Değişmiyor | Veri, config veya optimizer sorunu olabilir |
| Kararsız/NaN | Kararsız/NaN | Eğitim sağlık kapısı başarısız |

Validation gerçek robot başarısının yerine geçmez; eğitim sırasında temel genelleme ve sağlık sinyalidir.

---

## 12. Görselleştirme

Özel Gradio/Streamlit uygulaması ilk aşamada yapılmayacaktır.

Mevcut hazır araçlar:

- LeRobot dataset visualizer
- Rerun
- Psi0 `scripts/viz/viz_episode_g1_sonic.py`
- Vuer tabanlı Psi0 araçları

Öncelikli araç LeRobot + Rerun olacaktır. Final LeRobot dataset üzerinden:

- sol kamera videosu,
- state zaman serileri,
- action zaman serileri,
- episode timeline,
- görev metadata'sı

incelenecektir.

Hazır araçlar yetersiz kalmadıkça özel arayüz geliştirilmeyecektir.

---

## 13. Model ve eğitim stratejisi

### 13.1 Warm-start

```text
data/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k/
```

Bu checkpoint:

- SONIC için post-train edilmiş VLM ve action expert içerir.
- 50 saat UniFolM verisiyle hazırlanmıştır.
- Yeni 80D SONIC action contract'ına uygundur.
- Eski 36D AMO checkpoint'lerinin yerine kullanılacaktır.

### 13.2 Base recipe

Yapısal başlangıç:

```text
third_party/Psi0/scripts/train/psi0/
finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh
```

Script olduğu gibi kullanılmayacak; tek GPU ve bizim dataset yolları için sade bir wrapper/config hazırlanacaktır.

### 13.3 Tek GPU stratejisi

Donanım:

- NVIDIA GeForce RTX 5090
- yaklaşık 32 GB VRAM
- PyTorch `2.7.0+cu128`
- Python 3.11

İlk eğitim:

```text
VLM:                    frozen
action expert:          trainable
mixed precision:        BF16
microbatch:             1
gradient accumulation:  başlangıç hedefi 32
effective batch:        başlangıç hedefi 32
gradient checkpointing: açık
resolution:             240×320
```

Gradient accumulation ve learning rate, gerçek forward/backward VRAM ve throughput ölçümünden sonra kesinleştirilecektir.

İlk learning-rate adayı `2.5e-5` olabilir; bu kesin kabul edilmiş değer değil, batch 128 upstream reçetesinden batch 32'ye lineer ölçeklenmiş başlangıç hipotezidir.

### 13.4 Neden VLM frozen?

Full VLM tuning tek 32 GB GPU üzerinde action header, gradient, optimizer state ve activation'larla birlikte güvenli görünmemektedir.

Frozen VLM'de:

- Görüntü ve instruction kullanılmaya devam eder.
- VLM görsel/metinsel özellik üretir.
- Yalnız action expert bu özelliklerden bizim SONIC/Dex3 aksiyonlarımızı üretmeyi öğrenir.

İlk action-only eğitim görsel domain'e uyum sağlayamazsa daha sonra sınırlı VLM bileşenlerini açmak değerlendirilebilir. Bu ilk yolun parçası değildir.

---

## 14. Augmentation ve robustness

Smoke test deterministic başlayacaktır:

```text
state drop:       kapalı
state jitter:     kapalı
state noise:      kapalı
image augmentation: kapalı
view augmentation:  kapalı
```

Temel veri ve model yolu doğrulandıktan sonra tam eğitime aday upstream değerleri:

```text
state_drop_prob:                 0.1
state_temporal_jitter:           10 frame @ 30 Hz
state_temporal_jitter_prob:      0.5
state_noise_std:                 0.05
view augmentation:              açık
view_aug_min_scale:              0.85
```

Dataset finalde 30 Hz olduğu için 10-frame jitter yaklaşık 0,33 saniyedir ve upstream SONIC fine-tune zaman ölçeğini korur.

---

## 15. Logging

İlk tercih W&B offline:

```bash
export WANDB_MODE=offline
export WANDB_PROJECT=psi0-unitree-dex3
```

Neden:

- `wandb` ortamda kurulu.
- API key gerektirmez.
- Psi0'ın mevcut özel logging yolu W&B ile uyumludur.
- İlk aşamada TensorBoard eklemek için kod değişikliği gereksizdir.

İzlenecek minimum metrikler:

- training loss
- validation loss
- learning rate
- action gradient norm
- mevcutsa bileşen gradient normları
- state drop fraction
- step süresi
- GPU peak allocated/reserved memory

Son iki metrik smoke helper veya training wrapper tarafından ayrıca ölçülebilir.

---

## 16. Uygulama yol haritası ve kapılar

Her kapı geçmeden sonraki aşamaya geçilmeyecektir.

### Kapı 0 — Sözleşme ve config sabitleme

**Amaç:** Dönüştürücü yazılmadan önce bütün semantik kararları makinece okunabilir hâle getirmek.

Yapılacaklar:

1. Canonical 43D joint isim sırasını doğrula.
2. 13 canonical task metnini sabitle.
3. `%95/%5` episode split manifestini üret.
4. Sol kamera kararını config'te sabitle.
5. Hariç episode listesini config'te sabitle.

**Geçiş kriterleri:**

- [ ] 43 state joint ismi ve sırası kaynağa bağlı olarak doğrulandı.
- [ ] İlk state adı ve hand başlangıç konumu Psi0 SONIC contract'ıyla uyuşuyor.
- [ ] 13 task için tek canonical instruction listesi var.
- [ ] `GraspSquare` yanlış etiketi düzeltildi.
- [ ] Split'te 3.150 episode var; başka kayıp yok.
- [ ] Train ve validation kesişimi boş.
- [ ] Her 13 task validation içinde mevcut.

**Başarısızlık halinde:** Converter yazımına başlanmaz; mapping veya metadata düzeltilir.

---

### Kapı 1 — Mini dataset dönüşümü

**Amaç:** Tüm veri yerine her görevden 1–2 episode ile 30 Hz contract'ını doğrulamak.

Yapılacaklar:

1. Ham episode'u `observation.state`, metadata ve sol kamera ile oku.
2. Eşleşen 50 Hz `action.npz` dosyasını bul.
3. 50 Hz action'ı 30 Hz timestamp'lere deterministik örnekle.
4. 43D state oluştur.
5. 64D body token ve 14D hand action'ı ayrı alanlara yaz.
6. Strict 30-target validity kuralını uygula.
7. Mini LeRobot dataset ve metadata üret.

**Geçiş kriterleri:**

- [ ] Her seçilen episode'un video/state/action zaman çizgisi tutarlı.
- [ ] Final dataset FPS tam olarak 30.
- [ ] Yalnız `cam_left_high` kullanılıyor.
- [ ] `observation.state` 43D.
- [ ] İlk 15 state kanalı standing proxy ve episode boyunca sabit.
- [ ] Sonraki 28 kanal ham `observation.state` ile aynı.
- [ ] `action.body_token_v1_1` 64D.
- [ ] `action` 14D.
- [ ] Bütün sayısal değerler finite.
- [ ] Geçersiz tail hiçbir geçerli 30-action anchor'a sızmıyor.
- [ ] Video hazır LeRobot/Rerun aracıyla açılıyor.

**Başarısızlık halinde:** Tüm dataset dönüştürülmez; yalnız mini converter düzeltilir.

---

### Kapı 2 — Psi0 loader kapısı

**Amaç:** Üretilen mini datasetin Psi0'ın gerçek loader ve transform zinciriyle açıldığını kanıtlamak.

Beklenen model girdileri:

```text
observations: tek sol kamera view
states:       [B, 1, 45]
actions:      [B, 30, 80]
actions_mask: action ile uyumlu
instruction:  canonical text
```

**Geçiş kriterleri:**

- [ ] Train loader batch üretiyor.
- [ ] Validation loader `no_aug` ile batch üretiyor.
- [ ] State 43D'den 45D'ye doğru pad ediliyor.
- [ ] Action `64 + 14`ten 80D'ye doğru pad ediliyor.
- [ ] Son iki state/action kanalı sıfır.
- [ ] Joint ve action sırası assertion'lardan geçiyor.
- [ ] `stats_psi0.json` yükleniyor.
- [ ] Normalize edilen tensorlarda NaN/Inf yok.
- [ ] Geçersiz anchor batch'e girmiyor veya loss maskesi kesin olarak kapalı.

**Başarısızlık halinde:** Model forward çalıştırılmaz; dataset contract/metadata düzeltilir.

---

### Kapı 3 — Checkpoint forward kapısı

**Amaç:** Warm-start checkpoint ile gerçek batch'in forward pass yaptığını ve VRAM'e sığdığını doğrulamak.

Ayarlar:

```text
batch: 1
VLM: frozen
BF16
augmentation: kapalı
no_grad forward
```

**Geçiş kriterleri:**

- [ ] `model.safetensors` ve `action_header.safetensors` yükleniyor.
- [ ] Checkpoint shape mismatch yok.
- [ ] Forward loss finite.
- [ ] Predicted action şekli 30×80 contract'ıyla uyumlu.
- [ ] GPU peak allocated/reserved memory kaydedildi.
- [ ] OOM yok.

**Başarısızlık halinde:** Checkpoint/action-state sözleşmesi veya memory config düzeltilir; backward yapılmaz.

---

### Kapı 4 — Backward ve optimizer kapısı

**Amaç:** Tek gerçek optimizer step'in sağlıklı olduğunu doğrulamak.

**Geçiş kriterleri:**

- [ ] VLM parametreleri frozen.
- [ ] Action expert parametreleri trainable.
- [ ] Optimizer yalnız trainable parametreleri içeriyor.
- [ ] Loss finite.
- [ ] Gradientler finite.
- [ ] Gradient normu aşırı/NaN değil.
- [ ] Optimizer step tamamlanıyor.
- [ ] Peak VRAM 32 GB sınırında güvenli pay bırakıyor veya config küçültülüyor.

**Başarısızlık halinde:** Microbatch, checkpointing veya trainable parametre seçimi düzeltilir.

---

### Kapı 5 — 10–50 step kısa eğitim

**Amaç:** Eğitim döngüsü, validation, logging, checkpoint ve resume davranışını doğrulamak.

İlk 10 step:

- augmentation kapalı
- gradient accumulation düşük veya 1
- sık logging

Sonraki 50 step:

- aday gerçek gradient accumulation
- seçilen augmentation ayarları
- validation açık

**Geçiş kriterleri:**

- [ ] Training loss finite ve makul hareket ediyor.
- [ ] Validation loss üretiliyor.
- [ ] W&B offline run oluşuyor.
- [ ] Checkpoint yazılıyor.
- [ ] Checkpoint boyutu ve disk tüketimi ölçüldü.
- [ ] Resume aynı run/global step'ten devam ediyor.
- [ ] Resume optimizer/scheduler durumunu sıfırlamıyor.
- [ ] Step süresi ölçüldü.
- [ ] Tahmini tam eğitim süresi gerçek ölçümden hesaplandı.

**Başarısızlık halinde:** Full 3.150-episode dönüşüm veya gerçek eğitim başlatılmaz.

---

### Kapı 6 — Tam dataset üretimi

**Amaç:** Doğrulanmış converter ile bütün 3.150 episode'u üretmek.

**Geçiş kriterleri:**

- [ ] Tam 3.150 episode üretildi.
- [ ] Hariç episode listesi tam olarak beklenen iki episode.
- [ ] 13 task mevcut.
- [ ] Train/validation sayıları split manifestiyle aynı.
- [ ] Bütün parquet dosyaları okunuyor.
- [ ] Bütün sol kamera videoları decode ediliyor.
- [ ] Video/state/action zaman hizası geçiyor.
- [ ] State/action boyutları bütün framelerde doğru.
- [ ] NaN/Inf yok.
- [ ] `training_valid_mask` ve anchor filtresi doğru.
- [ ] Train stats üretildi.
- [ ] Validation aynı Psi0 normalization stats sözleşmesini kullanıyor.
- [ ] LeRobot/Rerun ile her 13 tasktan örnek episode görsel olarak incelendi.

**Başarısızlık halinde:** Gerçek eğitim başlatılmaz; hatalı episode/dönüşüm düzeltilip validation yeniden çalıştırılır.

---

### Kapı 7 — Gerçek eğitim başlangıç kapısı

**Amaç:** Ölçülmüş donanım sınırlarıyla gerçek run config'ini sabitlemek.

Run başlamadan önce kaydedilecekler:

- Git commit ve Psi0 submodule commit
- Dataset conversion version/hash
- Split manifest
- Task text listesi
- Joint order manifesti
- Warm-start checkpoint yolu/hash bilgileri
- Batch ve accumulation
- Learning rate
- Augmentation ayarları
- Validation/checkpoint aralıkları
- W&B offline run adı
- Tahmini süre ve disk ihtiyacı

**Geçiş kriterleri:**

- [ ] Kapı 0–6 tamamen geçti.
- [ ] Tek GPU config OOM olmadan 50 step tamamladı.
- [ ] Validation finite.
- [ ] Resume doğrulandı.
- [ ] Diskte checkpointler için yeterli alan var.
- [ ] Kullanılacak final config tek dosya/script içinde kayıtlı.

Bu kapı geçince gerçek eğitim başlatılabilir.

---

## 17. İlk gerçek eğitim için başlangıç hipotezi

Bu tablo smoke test sonucu değişebilir:

| Ayar | Başlangıç değeri |
|---|---|
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| VLM | Frozen |
| Action expert | Trainable |
| Precision | BF16 |
| Microbatch | 1 |
| Gradient accumulation | 32 |
| Effective batch | 32 |
| Learning rate | `2.5e-5` başlangıç adayı |
| Scheduler | Cosine |
| Warmup | Yaklaşık 1.000 step adayı |
| Resolution | 240×320 |
| Action chunk | 30 |
| Action dim | 80 |
| State dim | 45 |
| Training RTC | Kapalı |
| Max delay | 8 |
| State drop | 0.1 |
| State jitter | 10 frame, olasılık 0.5 |
| State noise | 0.05 |
| Validation | İlk aşamada sık; sonra seyrekleştirilebilir |
| Checkpoint | Gerçek checkpoint boyutu ölçüldükten sonra aralık sabitlenecek |
| Max steps | 40.000 upstream referansı; kısa run sonuçlarına göre kesinleşecek |

`40.000 step` henüz kesin karar değildir. Dataset büyüklüğü, effective batch, step süresi, validation eğrisi ve checkpoint davranışı görüldükten sonra sabitlenecektir.

---

## 18. Bilinçli olarak yapılmayacaklar

İlk sürümde kapsam dışı:

- Mevcut 50 Hz SONIC korpusunu değiştirmek veya silmek
- Ham Unitree datasetini yerinde değiştirmek
- Fruits veya Apple-to-Plate verisini eklemek
- İki kamerayı aynı anda modele vermek
- Sağ kamerayı training input'u yapmak
- Frame bazında rastgele kamera seçmek
- Videoları gereksiz yere 50 FPS'e çevirmek
- Bütün dataset için doğrulama olmadan H.264 transcode yapmak
- Full VLM tuning ile başlamak
- LoRA/QLoRA, FSDP, DeepSpeed veya CPU offload eklemek
- W&B ve TensorBoard'u aynı anda kurmak
- Özel kapsamlı dataset-editing web uygulaması yazmak
- Validation olmadan doğrudan uzun eğitim başlatmak
- Ham `action`ı observation state olarak kullanmak
- 43D joint sırasını tahmin etmek

---

## 19. Uygulama sırasında doğrulanacak fakat tasarımı değiştirmeyen noktalar

Bunlar karar bekleyen ürün konuları değil, teknik smoke kontrolleridir:

1. Mevcut FFmpeg/TorchCodec kurulumunun kaynak AV1 videoları sorunsuz decode edip etmediği.
2. Sol kamera episode kesitlerinin LeRobot video contract'ına nasıl bağlanacağı veya yeniden paketleneceği.
3. Frozen-VLM action-only training'in gerçek peak VRAM'i.
4. Batch 1 + accumulation 32'nin gerçek step süresi.
5. İlk checkpointin gerçek disk boyutu.
6. Nearest 50→30 Hz action seçiminin küçük örnekte timestamp hatası ve sürekliliği.
7. Psi0 loader'da strict validity filtresinin en küçük ve güvenli bağlanma noktası.

Bu kontrollerin sonucu tasarım sözleşmesini değiştirmeden implementasyon ayrıntısını belirleyecektir.

---

## 20. Son hedef akış

```text
Dondurulmuş ham Unitree Dex3 30 Hz
    ├── cam_left_high
    ├── observation.state[28]
    └── task/episode metadata

Dondurulmuş SONIC v1.1 korpusu 50 Hz
    ├── body token[64]
    ├── Dex3 target[14]
    └── training_valid_mask

                ↓  doğrulanmış 50→30 Hz dönüşüm

Yeni Psi0 LeRobot dataset
    ├── observation.images.egocentric = sol kamera
    ├── observation.state[43]
    ├── action.body_token_v1_1[64]
    ├── action[14]
    ├── canonical instruction
    ├── strict valid anchors
    └── train/validation metadata + stats

                ↓  Psi0 repack

Model input/target
    ├── image: 240×320
    ├── state: 43 → 45D
    ├── action: 64 + 14 → 80D
    └── horizon: 30 @ 30 Hz

                ↓

Frozen VLM + trainable action expert
    ↓
Kapı 3 forward
    ↓
Kapı 4 backward
    ↓
Kapı 5 kısa eğitim/resume
    ↓
Kapı 6 tam dataset
    ↓
Kapı 7 gerçek eğitim
```

---

## 21. Tamamlanma tanımı

“Psi0 eğitimine hazırız” denebilmesi için:

- 3.150 episode'luk final LeRobot dataset üretilmiş,
- bütün structural ve zaman hizası testleri geçmiş,
- her 13 task görsel olarak örneklenmiş,
- Psi0 loader gerçek batch üretmiş,
- warm-start checkpoint forward ve backward geçmiş,
- tek RTX 5090 üzerinde 50-step run tamamlanmış,
- validation, W&B offline logging, checkpoint ve resume doğrulanmış,
- final training config kayıt altına alınmış

olmalıdır.

Bu koşullar sağlanmadan uzun eğitim başlatılmayacaktır.
