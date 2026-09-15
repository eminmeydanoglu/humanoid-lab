# First Tur Dataset → SONIC v1.1 → GR00T Uygulama Planı

## 1. Amaç ve başarı tanımı

Amaç, `first_tur_ham` altındaki üç veri koleksiyonunu SONIC tabanlı
VLA eğitiminde kullanılacak ortak action space'e dönüştürmektir:

```text
action.motion_token[64]
+ action.left_hand_joints[7]
+ action.right_hand_joints[7]
= action[78]
```

Kaynaklar:

1. Unitree G1 Dex3: 13 alt dataset, 30 Hz, 14D kol + 14D el.
2. NVIDIA Fruits: 4 alt dataset, 20 Hz, 29D body + 14D el.
3. NVIDIA AppleToPlate: 1 dataset, 30 Hz, 29D body + 14D el.

Ham veri değiştirilmeden korunur:

```text
data/datasets/first_tur_ham/
```

Bütün üretilmiş veri, ara çıktı ve doğrulama kanıtı şuraya yazılır:

```text
data/datasets/first_tur_processed/
```

Başarı, yalnızca encoder'dan 64 sayı almak değildir. Aşağıdaki zincir
kanıtlanmalıdır:

```text
kaynak desired trajectory
  → kanonik 50 Hz SONIC reference
  → SONIC v1.1 encoder
  → saklanan 64D latent
  → resmî SONIC v1.1 decoder/controller
  → mevcut DDS hattı
  → Isaac G1'de beklenen hareket
```

## 2. Ana kurallar

### 2.1 Kullanıcı doğrulama kapısıdır

Her fiziksel replay aşamasında ajan:

1. Isaac replay videosunu kaydeder.
2. Kaynak episode videosunu aynı zaman ekseninde hazırlar.
3. Kaynak ve Isaac görüntüsünü yan yana senkron video yapar.
4. Joint grafikleri, metrik JSON'u, manifest ve logları aynı evidence
   klasörüne koyar.
5. Sonucu kullanıcıya sunar ve o kapıda durur.

Kapı sonucu:

```text
pending_human_review
accepted
accepted_with_notes
rejected
```

Kullanıcı kabul etmeden sonraki koleksiyona veya bulk conversion'a geçilmez.
Otomatik metrikler görsel kabulün yerini almaz; yanlış mapping, zamanlama,
NaN/Inf, düşme, drift ve tracking sorunlarını erkenden yakalar.

### 2.2 SONIC sürümü kilitlidir

```text
SONIC source commit:
a0732b642c0333077e127a2f56ab0014c196bca4

GEAR-SONIC artifact revision:
9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2

variant:
sonic_v1_1
```

Encoder, decoder ve `observation_config.yaml` aynı revision'dan gelir.
SHA-256 değerleri dataset ve her replay manifest'ine yazılır. Farklı SONIC
checkpointlerinden token karıştırılmaz.

### 2.3 Mevcut Isaac–SONIC mimarisi korunur

Isaac fizik süreci ile SONIC controller ayrı terminal/süreç olarak kalır.
Replay için ikinci body controller yazılmaz:

```text
SONIC decoder/controller
  → Unitree DDS LowCmd + HandCmd
  → mevcut SonicDdsController
  → CompleteRobotCommand
  → Isaac PhysX
```

Keyboard SONIC kullanımı bozulmaz. Dataset replay için aynı resmî binary'nin
`zmq_manager` input modu ayrı launcher seçeneği olur. Bağımsız MuJoCo SONIC
süreçlerine dokunulmaz; global `pkill` kullanılmaz.

### 2.4 Ham veri immutable ve her çıktı izlenebilirdir

Converter `first_tur_ham` altına yazmaz. Her processed episode şunları taşır:

- Kaynak repo/revision, dataset ve episode ID.
- Converter git commit/config sürümü.
- Joint ve hand mapping ID'leri.
- SONIC model/config checksum'ları.
- Sentetik ve measured alanlar.
- QC ve insan kabul durumu.

## 3. Basit anlatım: standing + Unitree arms nasıl yapılacak?

Unitree verisinde sadece iki kol ve iki el vardır. SONIC G1 encoder 29 body
joint'inin tamamını bekler. Kayıp gerçek bacak hareketini tahmin etmeyeceğiz.
Şu reference'ı açıkça tanımlayacağız:

> G1 resmî SONIC IDLE trajectory'sini izlesin; Unitree episode'undaki
> kolları ve elleri oynatsın.

Her 20 ms'lik frame:

```text
SONIC IDLE trajectory legs[12]  ─┐
SONIC IDLE trajectory waist[3] ─┼→ full body reference[29]
Unitree desired arms[14]       ─┘

Unitree desired left hand[7]  ──→ ayrı Dex3 hedefi
Unitree desired right hand[7] ──→ ayrı Dex3 hedefi
```

Bu sayıları toplamak değildir. Standing reference'taki 14 arm joint kolonu,
isimleriyle Unitree desired arm kolonlarıyla değiştirilir. Legs ve waist
IDLE trajectory değerinde kalır. Tek bir IDLE karesi episode boyunca
dondurulmaz; planner'ın zaman serisi korunur.

1. Unitree 30 Hz action trajectory timestamp tabanlı 50 Hz'e lineer
   interpolate edilir.
2. IDLE `q/qd/root orientation` aynı 50 Hz zaman tabanında korunur. Demo kol
   velocity'si, 50 Hz'e resample edilmiş kol pozisyonundan türetilip yalnız
   arm kolonlarına yazılır.
3. Anchor orientation resmî IDLE planner trajectory'sinden gelir ve SONIC'in
   heading-relative dönüşümüyle encoder observation'a girer.
4. Başa ve sona standing pre-roll/post-roll eklenir; bir anda ilk poza
   atlanmaz ve final velocity sıfıra iner.
5. El hedefleri aynı 50 Hz timeline'a interpolate edilir, fakat 64D body
   latent'inin parçası olmaz.

SONIC token tek pose'u değil `t...t+0.9 s` future trajectory penceresini temsil
eder. Offline composition gerçek gelecek kol trajectory'sini korur; live
upper-body override gibi aynı pozu bütün pencereye kopyalamaz.

## 4. Kalıcı kod yapısı

Tek kullanımlık notebook veya kopuk scriptler yerine:

```text
src/humanoid_lab/datasets/sonic/
├── schema.py
├── joints.py
├── timeline.py
├── reference.py
├── encoder_observation.py
├── encoder.py
├── provenance.py
├── quality.py
├── evidence.py
├── replay.py
└── adapters/
    ├── unitree_dex3.py
    ├── nvidia_fruits.py
    └── nvidia_apple_to_plate.py

configs/datasets/sonic/
├── sonic_v1_1.yaml
├── unitree_dex3.yaml
├── nvidia_fruits.yaml
├── nvidia_apple_to_plate.yaml
└── first_tur_v1.yaml

scripts/
├── convert-sonic-dataset.py
├── validate-sonic-dataset.py
├── replay-sonic-reference.py
├── replay-sonic-latent.py
└── compose-sonic-training-view.py
```

Scriptler ince CLI entrypoint olur; davranış import edilebilir/test edilebilir
paket kodunda kalır. Hedef kullanım:

```bash
./dev.sh sonic-replay-reference \
  --source unitree-dex3/object-placement --episode 0 --record

./dev.sh sonic-tokenize \
  --source unitree-dex3/object-placement --episode 0

./dev.sh sonic-replay-latent \
  --source unitree-dex3/object-placement --episode 0 --record --metrics
```

Kesin komut adları mevcut `dev.sh` stiline uyarlanabilir; direct-reference ile
stored-latent replay kullanıcıya ayrı ve açık komutlar olarak görünür.

## 5. Bölüm 1 — Mevcut Isaac–SONIC standing baseline

### Amaç

Dataset koduna dokunmadan mevcut Isaac, SONIC v1.1 decoder/controller, DDS ve
Dex3 hattının temiz Idle/standing koşusunda çalıştığını kanıtlamak. Bu test
sonraki bölümlerin kontrol grubudur.

### Akış

```text
Terminal 1: Isaac G1 + Dex3 + DDS simulator
Terminal 2: resmî SONIC v1.1 controller + Idle planner
```

1. Yalnızca Isaac-targeted eski controller mevcut ownership mekanizmasıyla
   temizlenir.
2. Isaac temiz başlatılır.
3. SONIC model-ready olana kadar startup support robotu tutar.
4. Idle/standing control açılır.
5. Test penceresi boyunca robot ve telemetry kaydedilir.
6. Kontrollü stop uygulanır. Normal launcher unbounded kalır; süre sınırı
   sadece test koşusuna özeldir.

### Otomatik test ve metrikler

- Model/observation checksum eşleşmesi.
- Tek Isaac-targeted controller ve doğru DDS izolasyonu.
- İlk geçerli command, support release nedeni/zamanı.
- Command tick sayısı, NaN/Inf ve saturation.
- Pelvis orientation, root height/XY drift.
- Foot contact/slip ve fall detection.
- Stop sonrası passive davranış.

### Evidence ve kapı

```text
validation/standing/<run_id>/
├── isaac.mp4
├── metrics.json
├── run_manifest.json
├── controller.log
├── simulator.log
├── joint_state.parquet
├── plots/
└── human_review.json
```

Robot düşmüyor/flail etmiyor, support doğru zamanda bırakılıyor ve
kullanıcı videoyu kabul ediyorsa **Human Gate 1** geçilir.

## 6. Bölüm 2 — Bir Unitree Dex3 episode'unu standing reference ile oynatma

### Amaç ve teknik ayrım

Unitree arm-only kaynağından full-body reference kurup Isaac'te stabil
oynatmak. Henüz saklanmış latent kullanılmaz: SONIC'in resmî C++ observation
builder'ı reference'tan tokenı kendi içinde üretir ve decoder'a verir.

### SONIC IDLE trajectory asset

Resmî SONIC Idle planner/reference hattından startup transient sonrası gerçek
zaman serisi 50 Hz'de kaydedilerek deterministik asset oluşturulur. Tek kare
seçilip çoğaltılmaz:

```text
references/unitree_idle_trajectory_v2/
├── timestamps.csv
├── joint_pos.csv
├── joint_vel.csv
├── body_pos.csv
├── body_quat.csv
├── metadata.txt
└── provenance.json
```

Startup transient'i seçilmez. Provenance SONIC/planner revision ve checksum,
kayıt run ID'si, segment süresi ve 12 leg+3 waist hareket aralığını içerir.
Asset, episode süresini karşılamalıdır; kısa asset wrap edilmez ve fail-closed
olur.

### Source adapter

Kullanıcı herhangi bir Unitree dataset/episode seçebilir. İlk fixture adayı:

```text
G1_Dex3_ObjectPlacement_Dataset / episode 0
```

Adapter:

1. Episode'u timestamp ve ID ile seçer.
2. 14 arm+14 hand alanını metadata isimleriyle çözer.
3. Unitree collector semantiğine göre aynı row desired action'ı kullanır;
   shift uygulamaz. Ölçülen yaklaşık iki-frame state lag normal fiziksel
   tracking gecikmesidir.
4. 30→50 Hz dönüşüm yapar.
5. IDLE lower body+Unitree arms ile 29D IsaacLab-order reference kurar.
6. IDLE lower-body qdot'unu korur; resampled kollardan arm qdot üretir ve
   canonical Dex3 hedeflerini aynı timeline'da tutar.

### Direct-reference replay

Resmî SONIC ZMQ Protocol v1 kullanılır:

```text
joint_pos[N,29]      resmî SONIC reference / IsaacLab order
joint_vel[N,29]      resmî SONIC reference / IsaacLab order
body_quat[N,4]       wxyz
frame_index[N]       monotonik
left_hand_joints[7]
right_hand_joints[7]
```

Protocol v1 encoder'ı bypass etmez; encode mode 0/G1'i çalıştırır. Eller
ayrı Dex3 hedefidir. Encoder `t...t+0.9 s` geleceği istediği için publisher
tek geçmiş frame göndermez; upstream batch/stream-merger semantiğiyle en az
gereken 46-frame lookahead frame-index'li olarak stage edilir. Yeni protokol
yazılmaz.

Lifecycle: fresh Isaac/SONIC readiness → future window stage → standing
pre-roll → explicit start → 50 Hz replay → stationary post-roll → explicit
stop/reset. Stale-last-target davranışına güvenilmez.

### Testler

- 29D body ve iki Dex3 permutation unit testleri.
- Duplicate/missing joint için fail-closed davranış.
- Timestamp ve 30→50 Hz interpolation round-trip.
- q/qdot, pre/post-roll ve joint-limit/velocity/jerk testleri.
- Protocol v1 packet, lookahead, clamp, frame gap/duplicate testleri.
- Underrun/NaN halinde explicit stop.

### Görsel kanıt

```text
validation/unitree_reference/<dataset>/<episode>/<run_id>/
├── source.mp4
├── isaac.mp4
├── side_by_side.mp4
├── reference.parquet
├── tracking.parquet
├── metrics.json
├── plots/body_tracking.png
├── plots/hand_tracking.png
├── run_manifest.json
└── human_review.json
```

Bu kapı task/obje başarısını değil şunu sorar: lower body stabil mi ve
kol/el hareketinin zamanı, yönü ve pozu kaynak demonstrasyona benziyor mu?
Kullanıcı kabul ederse **Human Gate 2** geçilir.

Gate paketi her örnekte üç senkron görünüm içerir: kaynak kamera, sabit tabanlı
doğrudan joint replay (SONIC bypass; mapping oracle) ve serbest tabanlı SONIC
replay. Doğrudan replay yanlışsa dataset→robot joint semantiği; yalnız SONIC
yanlışsa encoder/decoder veya denge hattı incelenir. MuJoCo/Unitree motor sırası
ile SONIC reference/IsaacLab sırası ayrı isimli sözleşmelerdir ve aralarındaki
29D permutation round-trip test edilir.

## 7. Bölüm 3 — Unitree episode'unu kalıcı latent'e dönüştürme

### Exact encoder sözleşmesi

SONIC v1.1 encoder input'u daima 1751D'dir:

| Slice | Boyut | Anlam |
|---|---:|---|
| `[0:4]` | 4 | `encoder_mode_4`; G1 mode 0 için `[0,0,0,0]` |
| `[4:294]` | 290 | 10×29 joint position, step5 |
| `[294:584]` | 290 | 10×29 joint velocity, step5 |
| `[584:644]` | 60 | 10×6 heading-relative root orientation, step5 |
| `[644:1751]` | 1107 | Diğer mode alanları; G1 için sıfır |

Future indeksleri 50 Hz'de:

```text
t + {0,5,10,15,20,25,30,35,40,45} frame
= t + {0.0,0.1,...,0.9} saniye
```

Episode sonunu aşan indeksler son frame'e clamp edilir. Final velocity de
tekrarlandığı için stationary tail contract'ın parçasıdır ve
`future_clamp_fraction` raporlanır.

Minimum G1 reference `joint_pos[29]`, `joint_vel[29]`, root `body_quat[4]`,
root `body_pos[3]` ve metadata body index `[0]` ister; full FK gerekmez. Root
orientation IDLE planner trajectory'sinden gelir. Quaternion
doğrudan encoder'a yazılmaz; resmî heading-relative quaternion → rotation
matrix → ilk iki kolon → row-wise 6D işlemi uygulanır.

### Golden C++ oracle

Aynı reference iki yoldan geçer:

```text
A. Resmî C++ observation builder + encoder/TensorRT
B. Python exact observation builder + aynı model_encoder.onnx
```

C++ `token_state` veya decoder policy input'un ilk 64 değeri golden çıktıdır.
`max_abs`, RMSE, p99 abs ve cosine similarity raporlanır. ONNX Runtime ile
TensorRT için uydurma `1e-6` eşiği konmaz; önce resmî reference'ta backend
fark tabanı ölçülür.

Python ONNX Runtime mevcut GR00T ortamında inceleme anında yoktu. Geçici host
install yerine dependency persistent `/opt/venvs` bootstrap katmanına pinlenir;
backend ve sürüm manifest'e yazılır.

### Pilot çıktı ve testler

```text
processed/pilots/unitree_object_placement_ep000/
├── data/episode_000000.parquet
├── videos/episode_000000.mp4
├── meta/
├── reference/
├── encoder_input_sample.npz
├── token_parity.json
└── provenance.json
```

Testler: exact 1751D shape/order, inactive-zero, mode0, step5 future window,
tail clamp, quaternion/6D, ONNX shape, C++/Python parity, token finite/norm ve
deterministik yeniden üretim. Yalnızca pilot episode üretilir; bulk conversion
başlamaz. Golden parity geçerse **Automated Gate 3** tamamlanır.

## 8. Bölüm 4 — Saklanan Unitree latent'ini oynatma

### Amaç

Artık reference veya encoder göndermeden Bölüm 3'te saklanan
`motion_token64 + left7 + right7` action'larını oynatmak ve Bölüm 2 ile aynı
davranışı koruduğunu görmek.

### Transport ve launcher

Resmî ZMQ Protocol v4:

```text
token_state[64]
frame_index[1]
left_hand_joints[7]
right_hand_joints[7]
body_quat_w[4]       # heading gerekiyorsa
```

V4 encoder'ı bypass eder; decoder/controller'ı etmez. Decoder token yanında
Isaac'ten measured 10-frame robot history kullanır. Bu nedenle önceden decoder
29D command saklamak canonical replay değildir.

Mevcut `./dev.sh sonic-controller` keyboard davranışı korunur. Replay modu
aynı model paths, PID marker, takeover lock ve ownership kodunu ortaklaşa
kullanır; yalnızca `--input-type zmq_manager` seçer. Upstream v4 packer yeniden
kullanılır; yeni wire protocol yazılmaz.

Fresh state/model-ready beklenir; ilk action stage edilir; 50 Hz ve monotonik
index kullanılır. NaN/Inf, underrun veya deadline kaçırma explicit stop'tur.
Episode sonunda kısa final hold ve explicit stop/reset uygulanır.

### Asıl round-trip kanıtı

Üç video senkronlanır:

```text
A. Orijinal Unitree video
B. Direct-reference Isaac replay
C. Stored-latent Isaac replay
```

B–C için per-joint q/qdot, arm/hand tracking, root orientation/XY drift,
foot slip/contact, fall, saturation ve replay jitter/stale sayısı karşılaştırılır.
Stateful decoder/PhysX nedeniyle bit-identical state aranmaz; aynı hareket
niyeti ve makul tracking aranır.

```text
validation/unitree_latent/<dataset>/<episode>/<run_id>/
├── source.mp4
├── reference_replay.mp4
├── latent_replay.mp4
├── three_way_comparison.mp4
├── reference_vs_latent_metrics.json
├── tracking.parquet
├── plots/
├── run_manifest.json
└── human_review.json
```

Kullanıcı üçlü videoyu kabul ederse **Human Gate 4** geçilir ve Unitree
bulk tokenization'a izin verilir.

## 9. Bölüm 5 — NVIDIA kaynaklarında aynı round-trip

Her kaynak için sıra aynıdır:

```text
semantic parse
→ Protocol v1 direct-reference replay
→ human gate
→ encoder parity/tokenization
→ Protocol v4 stored-latent replay
→ human gate
→ source bazında bulk conversion izni
```

### Fruits

Fruits 43D sırası:

```text
left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand
```

Apple ile aynı değildir. Adapter `modality.json` blokları ve `info.json` joint
isimleriyle body29/hand14 çıkarır; positional ortak 43D parser kullanmaz.
Desired action 20→50 Hz interpolate edilir, qdot aynı trajectory'den gelir,
video causal last-frame hold ile hizalanır. Root/projected gravity olmadığı
için sentetik upright olarak işaretlenir.

Yerel analiz arm action'ın measured state'i yaklaşık iki 20 Hz frame
(100 ms) önden takip ettiğini gösterdi; bu servo lag ile uyumludur ve row shift
uygulanmaz. Pilot direct/latent replay kabul edilmeden dört Fruits alt dataset'i
bulk convert edilmez. Bu **Human Gate 5A**'dır.

### AppleToPlate body

Apple sırası:

```text
left_leg, right_leg, waist, left_arm, right_arm, left_hand, right_hand
```

Dataset card `action` alanını desired joint positions olarak tanımlar. Body29
action'dan; VLA proprioception measured `observation.state`'ten gelir. Dataset
stationary manipulation ve root trajectory içermediği için root/gravity v1'de
sentetik upright'tır.

### AppleToPlate hand blocker'ı

Apple full 78D dataset henüz hazır sayılmaz:

- 7 kanalın iç joint isimleri metadata'da yok.
- Sayısal dağılım Unitree motor-radyan verisinden farklı.
- 171.625 satırın tamamında right-hand action yedi kanal da sıfır.
- Yalnızca sol elde belirli kanallar aktif.

SONIC bu 7 değeri normalize etmeden doğrudan motor `q` olarak yazar. Bu nedenle
Apple hand action olduğu gibi gönderilmez. Blocker şöyle kapanır:

1. NVIDIA collector/config kodundan kanal sırası ve birimi aranır.
2. Action, measured hand state ve effort ilişkisi kanal bazında incelenir.
3. Gerekirse Isaac'te düşük hız/genlikli tek-kanal hand schema discovery
   testi yapılır.
4. `source_index → motor_id`, sign, scale, offset ve limit tablosu çıkarılır.
5. Hand-only video kullanıcıya sunulur.

Apple body pilotu ilerleyebilir; fakat full replay ve final action bu blocker
kapanmadan `valid` olmaz. Full Apple round-trip kabulü **Human Gate 5B**'dir.

Kanonik Dex3 motor sırası:

```text
left:  thumb0, thumb1, thumb2, middle0, middle1, index0, index1
right: thumb0, thumb1, thumb2, index0, index1, middle0, middle1
```

Unitree zaten bu per-side sıradadır; Fruits isimle remap edilir; Apple
doğrulanana kadar unresolved kalır.

NVIDIA replay kapısı source objelerinin Isaac'te birebir reconstruction'ı veya
task success değildir. Full-body motion yapısı, kol/wrist zamanı, lower-body
stabilitesi, root drift/foot slip/fall ve mapping sonrası hand tracking incelenir.

## 10. Bölüm 6 — Bulk conversion ve eğitim corpus'u

### Bulk geçiş kuralı

Bir kaynak ancak semantic test, pilot direct replay human gate, golden parity,
pilot latent replay human gate ve source-specific blockerlar tamamlandığında
bulk convert edilir. Unitree kabulü NVIDIA'yı; Fruits kabulü Apple'ı otomatik
onaylamaz.

### Format ve timeline

Pinli Isaac-GR00T LeRobot v2 beklediği için processed format:

```text
GR00T LeRobot v2 + meta/modality.json
```

Ham Unitree v3 kalır. Processed timeline 50 Hz'dir; stock
`UNITREE_G1_SONIC` action horizon `range(40)` olduğu için 40 frame = 0,8 s olur.
20/30 Hz'de bırakılırsa aynı row offsetleri yanlış fiziksel süreye dönüşür.

Nihai state:

```text
left_leg[6] + right_leg[6] + waist[3]
+ left_arm[7] + right_arm[7]
+ left_hand[7] + right_hand[7]
+ projected_gravity[3]
= observation.state[46]
```

NVIDIA'da body/hands measured state, gravity sentetik `[0,0,-1]`; Unitree'de
legs/waist IDLE reference, arms/hands measured, gravity sentetiktir. Metadata
her state grubunu measured/synthetic olarak ayrı ayrı belirtir.

Nihai action:

```text
motion_token[64] + left_hand_joints[7] + right_hand_joints[7] = action[78]
```

Üç action grubu GR00T'ta `ABSOLUTE / NON_EEF / DEFAULT` olur. Normalizasyon
fiziksel schema değil statistics/processor katmanıdır.

### Kamera politikası

Stock config tek `ego_view` bekler:

- Fruits/Apple: kaynak `ego_view`.
- Unitree v1 önerisi: `cam_left_high → ego_view`.
- `cam_right_high` hamda kalır ve catalog'da auxiliary availability olarak
  işaretlenir; varsayılan training view'e kopyalanmaz.

```yaml
camera_policy:
  primary: cam_left_high
  auxiliary: cam_right_high
  include_auxiliary_in_training: false
  resample: causal_last_frame_hold
```

Yapay ara pixel üretilmez. Her 50 Hz row, timestamp'i policy time'dan ileri
olmayan en yeni source frame'i kullanır; source frame/time, `video_age_ms` ve
repeat bilgisi saklanır.

### Veri yerleşimi ve filtrelenebilir view'lar

```text
first_tur_processed/sonic_v1_1/
├── references/unitree_idle_trajectory_v2/
├── pilots/
├── converted/
│   ├── unitree-g1-dex3/<13 datasets>/
│   ├── nvidia-g1-fruits-1k/<4 datasets>/
│   └── nvidia-apple-to-plate/
├── catalog/
│   ├── episodes.parquet
│   ├── frames.parquet
│   └── transformations.json
├── view_specs/
│   ├── all_valid.yaml
│   ├── unitree_only.yaml
│   ├── nvidia_only.yaml
│   └── custom/
├── training_views/all_valid/
└── validation/<evidence runs>/
```

Her source bağımsız processed shard olur; tek contiguous LeRobot v2 training
view sonradan compose edilir. Catalog filtresiyle source/task/QC grubu
çıkarılabilir, token yeniden hesaplanmaz. Video tekrarında mümkünse hardlink
kullanılır.

### Metadata ve QC

Episode catalog: processed/source episode ID, collection/dataset/repo/revision,
task original/canonical, source/processed FPS ve frame count, split/session,
converter commit, SONIC revision/checksum, joint/hand schema, reference/lower
body/root/gravity/camera source, quality/exclusion ve human-review status.

QC: NaN/Inf/timestamp/joint-limit sayıları; velocity/acceleration/jerk p99;
video age p50/p95/max; future clamp mean/max; token norm p50/p99; encoder
parity; replay fall/root drift/body-arm-hand RMSE ve stale packet count.

```text
quality_status = unvalidated | valid | warning | invalid
```

Veri sessizce silinmez. Invalid episode catalog'da kalır, default view'e girmez
ve gerekçesi kayıtlıdır.

### Split ve training-readiness

- Split frame değil episode seviyesindedir.
- Session varsa aynı session farklı splitlere dağıtılmaz.
- Source/task oranları stratified tutulur; seed/config kaydedilir.
- 50 Hz conversion sonrası `stats.json` ve `relative_stats.json` yeniden
  hesaplanır.
- Her episode offline QC'den geçer; her alt dataset'ten pilotlar, stratified
  örnekler, outlier ve warning episode'lar Isaac replay görür.
- Replay aracı herhangi bir episode'u tek komutla oynatabilir.

Tam eğitimden önce LeRobot/GR00T validator, state46/action78/horizon40,
video+language decode, statistics, birkaç dataloader batch, kısa
forward/backward ve checkpoint save/load smoke yapılır. Bu smoke convergence
veya task success kanıtı değildir.

## 11. Test piramidi

### Unit testler

- Source body/hand mappings, missing/duplicate fail-closed.
- Semantic block extraction ve 20/30→50 Hz resampling.
- qdot, pre/post-roll, endpoint ve root 6D rotation.
- Encoder 1751D layout, future window/clamp.
- Protocol v1/v4 packet round-trip.
- Metadata, view filter ve deterministic split.

### Contract testleri

- Model/config checksum ve ONNX input/output dimension.
- IsaacLab 29D order.
- State46/action78 `modality.json` slice'ları.
- `UNITREE_G1_SONIC` horizon/field uyumluluğu.
- Source metadata beklenmedik değişirse fail-closed.

### Integration testleri

- Unitree fixture → reference.
- Reference → C++ ve Python token → parity.
- Processed LeRobot episode write/read.
- V1 publisher → SONIC receiver.
- V4 publisher → SONIC receiver.
- Underrun/NaN/stop/reset lifecycle.

### Isaac E2E + insan kabulü

- Standing baseline.
- Unitree direct reference.
- Unitree stored latent.
- Fruits direct reference ve stored latent.
- Apple body direct reference ve stored latent.
- Apple hands schema kabulünden sonra full replay.

Her koşu yeni run ID ve evidence bundle ile kanıtlanır; eski video yeni
koşunun kanıtı sayılmaz.

## 12. Milestone sırası

1. **Milestone 0 — Contract/iskelet:** model lock, schema registry, kalıcı
   CLI/config/provenance/evidence ve fixture testleri.
2. **Milestone 1 — Standing:** mevcut yolu değiştirmeden video+metrics.
   **Human Gate 1.**
3. **Milestone 2 — Unitree direct reference:** standing asset, adapter,
   resampler, Protocol v1 ve side-by-side. **Human Gate 2.**
4. **Milestone 3 — Unitree tokenization:** 1751D builder, persistent ONNX
   dependency, C++ oracle ve pilot 78D. **Automated Gate 3.**
5. **Milestone 4 — Unitree stored latent:** Protocol v4 ve üçlü karşılaştırma.
   **Human Gate 4.**
6. **Milestone 5A — Fruits:** adapter, direct replay, parity, latent replay.
   **Human Gate 5A.**
7. **Milestone 5B — Apple:** body round-trip, hand schema discovery, full
   replay. **Human Gate 5B.**
8. **Milestone 6 — Corpus:** kabul edilen kaynakları bulk convert et, QC,
   split, all-valid view, GR00T stats ve training smoke. **Gate 6.**

## 13. Kilitli kararlar ve açık noktalar

| Konu | Karar/durum |
|---|---|
| SONIC | v1.1, tek encoder/decoder/config/checksum |
| Encoder | Tam 1751D; G1 alanları dolu, diğerleri sıfır |
| Future | `t + {0,0.1,...,0.9}s` |
| Body order | IsaacLab 29D, isim tabanlı |
| Timeline | Processed dataset 50 Hz |
| Horizon | 40 frame = 0,8 s |
| Action | `[64 token, 7 left hand, 7 right hand]` |
| State | 43 body+hands + 3 projected gravity = 46D |
| Format | GR00T LeRobot v2 + `modality.json` |
| Unitree timing | Aynı-row desired target; shift yok |
| Unitree lower body | Resmî SONIC IDLE planner trajectory zaman serisi |
| Root/gravity | IDLE anchor orientation; provenance zorunlu |
| Fruits | Source-specific name mapping |
| Apple body | Desired action body29 |
| Apple hand | Sıra/birim blocker |
| Unitree camera | V1 önerisi `cam_left_high` |
| Direct replay | Protocol v1; encoder SONIC içinde |
| Latent replay | Protocol v4; encoder bypass, decoder korunur |
| Replay kabulü | Kullanıcı videoları birlikte inceler |
| Bulk conversion | Source round-trip kabulünden sonra |

Implementation sırasında ölçümle/seçimle kapanacak noktalar:

1. IDLE trajectory asset'inin kayıt süresi/segment'i.
2. Pilot episode ID'leri.
3. TensorRT↔ONNX parity toleransı.
4. Tracking/root drift eşikleri; resmî baseline'da kalibre edilir.
5. AppleToPlate hand motor contract'ı.
6. Unitree primary camera kararının pilot videoda onayı.

## 14. Kapsam sınırı

Bu plan source→reference, SONIC latent ve parity, mevcut Isaac hattında
reference/latent replay, kullanıcı evidence'i, filtrelenebilir training dataset'i
ve GR00T training smoke'u kapsar.

Şu aşamada kaynak sahne/objelerin birebir Isaac reconstruction'ı, full VLA
training convergence, gerçek robot deployment, apple/fruit task success ve
concurrent Isaac+SONIC+GR00T VRAM kabulü kapsam dışıdır. Bunlar temiz
dataset/body-control yolu kanıtlandıktan sonraki ayrı entegrasyon kapılarıdır.
