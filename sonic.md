





Selam. İnceleme sonucunda ana yön doğru, fakat bulk conversion’a başlamadan önce birkaç kritik düzeltmeyi tasarıma eklememiz gerekiyor. En önemlisi şu:

> Body tarafındaki dönüşüm büyük ölçüde netleşti. Kalan gerçek belirsizlik SONIC değil; AppleToPlate’in el action sözleşmesi, sentetik standing reference’ın hangi kesin değerlerden üretileceği ve kamera seçimi.

Henüz kod veya veri değiştirmedim.

## Netleşen nihai mimari

Ham veri değişmeden kalacak:

```text
data/datasets/first_tur_ham/
```

Bütün üretilmiş veri, ara çıktılar, QC sonuçları ve eğitim görünümü şurada olacak:

```text
data/datasets/first_tur_processed/
```

Kanonik işleme hattı:

```text
immutable raw source
  ↓
source-specific semantic parser
  ↓
canonical timestamps + named joints
  ↓
50 Hz desired body reference
  ↓
missing-body/root augmentation
  ↓
exact SONIC v1.1 1751D encoder observation
  ↓
model_encoder.onnx
  ↓
motion_token[64]
  +
canonical Dex3 left[7] + right[7]
  ↓
GR00T LeRobot v2 action[78]
  ↓
ZMQ protocol v4 replay
  ↓
official SONIC decoder/controller
  ↓
existing DDS → Isaac G1
```

Mevcut ham set zaten düzenli ve doğrulanmış: 3 koleksiyon, 18 alt dataset, 4.649 episode ve yaklaşık 52 GB. Yerel envanter burada: [dataset.md](/home/aksoy-msi/code/humanoid-lab-main/data/datasets/first_tur_ham/dataset.md:1).

## Kesinlikle kilitlememiz gereken kararlar

### 1. SONIC sürümü

Tek sürüm:

```text
SONIC source commit:
a0732b642c0333077e127a2f56ab0014c196bca4

GEAR-SONIC model revision:
9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2

variant:
sonic_v1_1
```

Üç kaynak da aynı:

```text
model_encoder.onnx
observation_config.yaml
```

ile tokenize edilecek. Replay/deployment aynı revision’ın:

```text
model_decoder.onnx
observation_config.yaml
```

dosyalarını kullanacak.

Dosya adını kaydetmek yeterli olmayacak; SHA-256 değerlerini de dataset provenance’a yazacağız. Yerelde model dosyalarının checksum’ları zaten `MODEL_PROVENANCE.json` içinde bulunuyor.

### 2. Encoder girdisi 644D değil, tam 1751D

G1 modu yalnızca ilk 644 anlamlı değeri kullansa da ONNX modeline her zaman tam 1751D tensor verilmesi gerekiyor:

| Aralık | Boyut | İçerik |
|---|---:|---|
| `[0:4]` | 4 | `encoder_mode_4` |
| `[4:294]` | 290 | 10 × 29 joint position |
| `[294:584]` | 290 | 10 × 29 joint velocity |
| `[584:644]` | 60 | 10 × 6 heading-relative root orientation |
| `[644:1751]` | 1107 | Diğer encoder modları; G1 modunda sıfır |

G1 modunda `encoder_mode_4`, klasik one-hot değil. Resmi C++ gatherer mode ID’yi ilk elemana yazıp kalanını sıfırlıyor. Mode ID `0` olduğu için sonuç fiilen:

```text
[0, 0, 0, 0]
```

oluyor. Python encoder implementasyonunda bunu “one-hot G1” yaparsak sessiz ama büyük bir parity hatası üretiriz.

SONIC’in resmi observation sözleşmesi ve mode-specific zero-fill davranışı [encoder observation dokümanında](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/a0732b642c0333077e127a2f56ab0014c196bca4/docs/source/references/observation_config.md) tanımlanıyor.

### 3. Bir token yaklaşık 0,9 saniye ileriye bakıyor

Her `motion_token[t]` şu referans örneklerinden üretiliyor:

```text
t
t + 0.1 s
t + 0.2 s
...
t + 0.9 s
```

Çünkü SONIC:

```text
10 frames × step5 × 20 ms
```

kullanıyor.

Episode sonunda future index’ler son frame’e clamp ediliyor. Bunun iki sonucu var:

- Son position tekrar edilir.
- Son velocity de tekrar edilir; otomatik olarak sıfırlanmaz.

Bu nedenle converter episode sonuna fiziksel olarak stationary bir tail koyacak veya en azından terminal velocity’yi sıfırlayacak. Ayrıca her satır için:

```text
future_clamp_fraction
```

hesaplanacak. Böylece ileride tail’e fazla bağımlı örnekler filtrelenebilir.

### 4. Canonical body order IsaacLab 29D olacak

Reference trajectory’nin kanonik joint sırası:

```text
left_hip_pitch
right_hip_pitch
waist_yaw
left_hip_roll
right_hip_roll
waist_roll
left_hip_yaw
right_hip_yaw
waist_pitch
left_knee
right_knee
left_shoulder_pitch
right_shoulder_pitch
left_ankle_pitch
right_ankle_pitch
left_shoulder_roll
right_shoulder_roll
left_ankle_roll
right_ankle_roll
left_shoulder_yaw
right_shoulder_yaw
left_elbow
right_elbow
left_wrist_roll
right_wrist_roll
left_wrist_pitch
right_wrist_pitch
left_wrist_yaw
right_wrist_yaw
```

Upstream C++ header’ında reference motion için MuJoCo order kullanıldığına dair eski/genel bir yorum var; fakat gerçek CSV interchange dokümanı açıkça IsaacLab order diyor. Bu nedenle serialization sözleşmesinde [resmi motion-reference dokümanını](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/a0732b642c0333077e127a2f56ab0014c196bca4/docs/source/references/motion_reference.md) esas alacağız.

Hiçbir converter:

```python
body = action[:29]
```

gibi varsayımlarla çalışmayacak. Her kaynak için ayrı isim tabanlı adapter ve unit-basis permutation testi olacak.

### 5. Derived eğitim dataset’i 50 Hz olacak

Ham Fruits 20 Hz, Unitree ve AppleToPlate 30 Hz olarak korunacak. Fakat nihai GR00T–SONIC eğitim dataset’i 50 Hz olacak.

Bunun nedeni yalnız SONIC encoder değil. Pinli Isaac-GR00T `UNITREE_G1_SONIC` config’i:

```python
delta_indices=list(range(40))
```

kullanıyor. Bunlar saniye değil dataset satır offsetleri. 50 Hz’de 40 action:

```text
40 × 20 ms = 0,8 saniye
```

anlamına geliyor. 20 Hz’de bırakırsak aynı config yanlışlıkla 2 saniyelik horizon, 30 Hz’de ise 1,33 saniyelik horizon oluşturur. Resmi embodiment config’i bunu açıkça tanımlıyor: [Isaac-GR00T `unitree_g1_sonic`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/data/embodiment_configs.py#L1037-L1128).

Video için gelecekteki frame seçilmeyecek:

```text
policy time t
→ timestamp ≤ t olan en yeni kaynak video frame’i
```

kullanılacak. Yani causal last-frame hold. Ne optical-flow ne de yapay ara görüntü üretilecek.

Her satırda veya QC sidecar’ında şu alignment bilgileri bulunacak:

```text
source_video_frame_index
source_video_timestamp
video_age_ms
video_repeated
```

### 6. Nihai format LeRobot v3 değil, LeRobot v2 olacak

Burada gelen araştırma yanıtındaki öneriyi düzeltiyorum.

Ham Unitree datasetleri LeRobot v3 olarak kalabilir. Fakat yerelde pinli Isaac-GR00T:

- Doğrudan LeRobot v2 bekliyor.
- V3 datasetler için önce `convert_v3_to_v2.py` kullanılmasını istiyor.
- `tasks.jsonl` ve `episodes.jsonl` sözleşmesini kullanıyor.
- SONIC benchmark README’si de verinin LeRobot v2 olmasını istiyor.

Dolayısıyla `first_tur_processed` çıktısı:

```text
GR00T LeRobot v2 + meta/modality.json
```

olacak. Yerel upstream’in veri hazırlama dokümanı bunu açıkça belirtiyor: [Isaac-GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md).

Bu, “v3 daha yeni” tartışmasından daha önemli: hedefimiz eldeki pinli training stack ile gerçekten çalışan format.

## Canonical eğitim şeması

### Observation state: 46D

Stock `UNITREE_G1_SONIC` ile uyumlu state:

```text
left_leg[6]
right_leg[6]
waist[3]
left_arm[7]
right_arm[7]
left_hand[7]
right_hand[7]
projected_gravity[3]
────────────────────────────
46D
```

NVIDIA full-body kaynaklarında ilk 43D measured state’ten semantik olarak çıkarılacak.

Unitree arm-only kaynaklarında:

```text
legs[12]              = synthetic standing nominal
waist[3]              = synthetic standing nominal
arms[14]              = measured Unitree state
hands[14]             = measured Unitree state
projected_gravity[3]  = synthetic upright [0, 0, -1]
```

olacak.

Bunu metadata’da yalnız bir boolean ile değil, alan bazında belirteceğiz:

```json
{
  "state_sources": {
    "left_leg": "synthetic_static_standing",
    "right_leg": "synthetic_static_standing",
    "waist": "synthetic_static_standing",
    "left_arm": "measured",
    "right_arm": "measured",
    "left_hand": "measured",
    "right_hand": "measured",
    "projected_gravity": "synthetic_upright"
  }
}
```

NVIDIA verilerinde de root/IMU olmadığı için `projected_gravity` measured değil, sentetik `[0,0,-1]` olacak. Bu yalnız Unitree’ye özel bir eksiklik değil.

### Action: 78D

```text
motion_token[64]
left_hand_joints[7]
right_hand_joints[7]
────────────────────
78D
```

`meta/modality.json`:

```json
{
  "action": {
    "motion_token": {
      "start": 0,
      "end": 64
    },
    "left_hand_joints": {
      "start": 64,
      "end": 71
    },
    "right_hand_joints": {
      "start": 71,
      "end": 78
    }
  }
}
```

Üç action alanı da GR00T tarafında:

```text
ABSOLUTE
NON_EEF
DEFAULT
```

olacak. “Normalized hand action” diye fiziksel şema adı koymayacağız; normalizasyon, GR00T statistics/processor katmanında yapılacak.

## Kaynak bazında dönüşüm

### Unitree Dex3

Unitree collector kodunu da doğruladım. Aynı kontrol döngüsünde:

1. Measured arm state okunuyor.
2. IK ile `sol_q` desired arm target üretiliyor.
3. `sol_q` robota gönderiliyor.
4. Aynı `sol_q`, aynı satırın action alanına kaydediliyor.

Dolayısıyla Unitree için `action[t]` satırını ileri veya geri kaydırmayacağız. Collector kodu burada görülebilir: [Unitree `teleop_hand_and_arm.py`](https://github.com/unitreerobotics/avp_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/teleop_hand_and_arm.py#L362-L518).

Yerel veri üzerinde ölçtüğüm action→measured-state takip gecikmesi yaklaşık iki 30 Hz kare, yani yaklaşık 67 ms. Bu beklenen fiziksel kontrol gecikmesi; action indeksleme hatası değil.

Unitree body reference:

```text
static validated standing legs/waist
+
Unitree desired arms
+
synthetic upright root quaternion
```

olacak.

Burada araştırma yanıtındaki öneriye katılıyorum: her episode için yeniden planner çalıştırıp farklı idle trajectory üretmemeliyiz. Tek bir resmi standing referansı çıkarılacak ve sürümlenecek:

```text
unitree_static_lower_body_v1
```

Kaynağı:

```text
official SONIC idle/planner recording
+ selected stable frame/segment
+ SONIC source revision
+ selected frame index
```

olarak kaydedilecek.

### Fruits

Fruits 43D sırası AppleToPlate’ten farklı. Yerel `modality.json` bunu doğruluyor:

```text
left_leg
right_leg
waist
left_arm
left_hand
right_arm
right_hand
```

Burada body 29D şu bloklardan oluşacak:

```text
left_leg + right_leg + waist + left_arm + right_arm
```

ve sonra isimle IsaacLab order’a çevrilecek.

Fruits `info.json`, el joint isimlerini içeriyor. Dolayısıyla el motor sırası güvenilir biçimde isimle remap edilebilir.

Measured veride Fruits kol action’ı measured state’i yaklaşık iki kare, yani yaklaşık 100 ms önden götürüyor. Bu desired-position servo davranışıyla uyumlu. State’i reference yapmak yerine action trajectory kullanılmalı.

### AppleToPlate

Body tarafı temiz:

```text
left_leg
right_leg
waist
left_arm
right_arm
```

ve action dataset card’da açıkça “desired joint positions” olarak tanımlanıyor. [AppleToPlate dataset card](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate).

Ancak el tarafında açık bir blokaj var:

- `info.json`, 43 kanalın tek tek isimlerini vermiyor.
- `modality.json` yalnız `left_hand[29:36]`, `right_hand[36:43]` diyor.
- El bloklarının iç joint sırası belirtilmemiş.
- El action sayısal dağılımı Unitree’nin motor-radyan dağılımından farklı.
- Bütün 171.625 satırda sağ el action’ının 7 kanalının tamamı sıfır.
- Sol elde yalnız belirli kanallar aktif.

Bu yüzden:

> AppleToPlate el action’ını doğrudan SONIC Dex3 motorlarına gönderemeyiz.

Body token üretimini bu durum engellemiyor. Fakat final `[64+7+7]` dataset ve hand replay kabulü için Apple el sırası/birimi doğrulanmalı. Bunu şu üç yoldan kapatacağız:

1. Dataset’i üreten NVIDIA collector/config kodunu bulmak.
2. Dataset içindeki `action.effort_left_hand` ve measured state ilişkisini analiz etmek.
3. Kontrollü Isaac hand-only replay ile kanal bazında hangi joint’in hareket ettiğini görmek.

Üçüncü test gerektiğinde son doğrulama değil, schema discovery testi olacak; joint limitleri ve düşük hızla yapılacak.

## Dex3 kanonik el sırası

SONIC C++ herhangi bir dönüşüm yapmıyor; ZMQ’dan gelen 7 sayıyı doğrudan motor ID `0..6` olarak DDS command’a yazıyor. Bu nedenle canonical sıra gerçek motor sırası olacak.

Sol:

```text
thumb0
thumb1
thumb2
middle0
middle1
index0
index1
```

Sağ:

```text
thumb0
thumb1
thumb2
index0
index1
middle0
middle1
```

Unitree dataset zaten tam bu per-side sırada.

Fruits farklı metadata sırasından geliyor ve isimle buna remap edilecek.

AppleToPlate sırası şu an doğrulanmamış.

Ayrıca yalnız isim değil, her adapter şunları tanımlayacak:

```text
source name
canonical name
source index
canonical motor id
unit
sign
scale
offset
joint limits
```

İlk sürümde `scale=1`, `offset=0` ancak kanıtlandığı kaynaklarda kullanılacak.

## Root orientation

G1 encoder full FK istemiyor. Minimum reference için yeterli alanlar:

```text
joint_pos[29]
joint_vel[29]
body_quat root[4]  # wxyz
body_pos root[3]
metadata body indexes = [0]
```

Bütün kaynaklarımız stationary manipulation olduğu ve root quaternion içermediği için v1’de:

```text
body_quat = [1, 0, 0, 0]
body_pos  = [0, 0, 0]
root_source = synthetic_upright_fixed
```

kullanabiliriz.

Ancak encoder’a verilen 6D orientation doğrudan bu dört quaternion sayısı değildir. Resmi C++:

```text
reference root orientation
relative to current/assumed robot heading
→ rotation matrix
→ first two columns
→ row-wise 6D
```

hesaplıyor. Python builder bu matematiği resmi C++ ile birebir eşleştirecek.

Offline conversion’da recorded base heading olmadığı için:

```text
assumed_robot_heading = identity
reference_heading = identity
```

olacak. Bu da provenance’a yazılacak.

## Kalıcı kod yapısı

Bunu birkaç bağımsız one-off script olarak bırakmayacağım. Önerdiğim yapı:

```text
src/humanoid_lab/datasets/sonic/
├── schema.py
├── joints.py
├── timeline.py
├── reference.py
├── encoder.py
├── provenance.py
├── quality.py
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
└── replay-sonic-episode.py

tests/
├── test_sonic_joint_mappings.py
├── test_sonic_timeline.py
├── test_sonic_encoder_observation.py
├── test_sonic_dataset_schema.py
├── test_sonic_provenance.py
└── test_sonic_replay_protocol.py
```

Scriptler ince CLI entrypoint olacak; asıl davranış import edilebilir paket kodunda bulunacak. Böylece daha sonra:

- Yeni dataset adapter’ı ekleyebiliriz.
- Aynı encoder’ı başka corpus üzerinde kullanabiliriz.
- Bir episode’u tekrar tokenize edebiliriz.
- Dataset’in belirli kaynak/task/QC bölümlerinden yeni bir eğitim görünümü oluşturabiliriz.
- Replay aracını hem manuel doğrulamada hem CI fixture’ında kullanabiliriz.

## İşlenmiş veri düzeni

Tek dev dataset içine her şeyi geri dönüşsüz biçimde ezmek yerine iki katman öneriyorum:

```text
first_tur_processed/
└── sonic_v1/
    ├── corpus/
    │   ├── data/
    │   ├── videos/
    │   └── meta/
    ├── catalog/
    │   ├── episodes.parquet
    │   ├── frames.parquet
    │   └── transformations.json
    ├── references/
    │   └── unitree_static_lower_body_v1/
    ├── golden/
    │   ├── official_reference/
    │   ├── unitree_object_placement_ep000/
    │   └── apple_to_plate_ep000/
    ├── validation/
    │   └── runs/
    └── views/
        ├── all-valid.json
        ├── unitree-only.json
        ├── nvidia-only.json
        └── custom/
```

`corpus/` GR00T LeRobot v2 eğitim dataset’i olacak.

`catalog/episodes.parquet` ayrıntılı, filtrelenebilir kaynak/QC tablosu olacak.

`views/*.json` yalnız episode seçimleri içerecek. Örneğin ileride Fruits’i çıkarmak istediğinizde token veya video yeniden üretmeyeceğiz:

```text
view filter:
source_collection != "nvidia_fruits"
```

ile yeni training view materialize edilecek. Video dosyalarında mümkün olduğunda hardlink kullanılabilir; 50 Hz videoları tekrar tekrar kopyalamayız.

## Metadata’da tutulacak alanlar

Episode seviyesinde en az:

```text
processed_episode_index
source_collection
source_dataset
source_repo
source_revision
source_episode_index
source_task_index
task_text_original
task_text_canonical
source_fps
processed_fps
source_frame_count
processed_frame_count
source_duration_s
split
session_id                # varsa
converter_version
converter_git_commit
sonic_source_commit
sonic_model_revision
encoder_sha256
decoder_sha256
observation_config_sha256
joint_schema_id
hand_schema_id
reference_trajectory_source
lower_body_source
root_source
projected_gravity_source
camera_policy
quality_status
exclusion_reasons
```

Episode QC özetinde:

```text
nan_count
inf_count
timestamp_violation_count
joint_limit_violation_count
velocity_p99
acceleration_p99
jerk_p99
video_age_ms_p50
video_age_ms_p95
video_age_ms_max
future_clamp_fraction_mean
future_clamp_fraction_max
token_norm_p50
token_norm_p99
encoder_parity_max_abs
encoder_parity_rmse
replay_fell
replay_root_xy_drift_m
replay_body_tracking_rmse
replay_arm_tracking_rmse
replay_hand_tracking_rmse
replay_stale_packet_count
```

Frame seviyesinde gerekli olan kaynak hizaları ayrıca tutulabilir, fakat training loader’ın kullanmadığı ağır debug alanlarını ana `action/state` vektörlerine karıştırmayacağız.

`quality_status` yalnız:

```text
unvalidated
valid
warning
invalid
```

gibi sınırlı bir enum olacak; gerçek nedenler `exclusion_reasons` listesinde duracak. Veriyi sessizce silmeyeceğiz.

## Kamera kararı

Stock `UNITREE_G1_SONIC` yalnız bir:

```text
ego_view
```

bekliyor. NVIDIA kaynaklarında zaten tek kamera var. Unitree’de iki kafa kamerası bulunuyor.

V1 için önerim:

```text
Unitree ego_view = cam_left_high
```

ve sağ kamera ham dataset’te korunmaya devam etsin. Processed training corpus’a ikinci kamerayı koymak şu aşamada gereksiz disk maliyeti ve custom modality config getirir.

Ancak converter config’inde seçim açık olacak:

```yaml
camera_policy:
  primary: cam_left_high
  auxiliary: cam_right_high
  include_auxiliary_in_training: false
```

Böylece daha sonra stereo veya sağ kamera deneyi yapmak için converter kodunu değiştirmemiz gerekmez.

## Golden encoder doğrulaması

Bulk conversion öncesi zorunlu kapı:

1. Bir resmi SONIC reference.
2. Bir Unitree ObjectPlacement episode.
3. Bir AppleToPlate body episode.

için iki yol çalışacak:

```text
A. Official C++ observation builder + TensorRT encoder
B. Bizim Python observation builder + ONNX Runtime encoder
```

Karşılaştırılacak değerler:

```text
max absolute error
RMSE
p99 absolute error
cosine similarity
```

Doğrudan `1e-6` gibi uydurma threshold koymayacağız. Önce aynı resmi reference için ONNX Runtime–TensorRT doğal fark tabanı ölçülecek; kabul eşiği bu tabana göre belirlenecek.

Python encoder implementation bulk conversion için kalıcı yol olacak. C++ yolu oracle/parity test olarak kalacak.

## Isaac latent replay

Burada tahmininiz doğru: ana kontrol hattı hazır.

Upstream ZMQ protocol v4 zaten şunları kabul ediyor:

```text
token_state[64]
frame_index[1]
left_hand_joints[7]   # optional
right_hand_joints[7]  # optional
```

ve resmi `run_vla_inference.py` içinde paketleme kodu mevcut. Yeni wire protocol yazmayacağız. [Upstream VLA inference publisher](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/a0732b642c0333077e127a2f56ab0014c196bca4/gear_sonic/scripts/run_vla_inference.py).

Mevcut yerel launcher hâlâ keyboard input ile açılıyor:

```text
--input-type keyboard
```

Bu [dev.sh](/home/aksoy-msi/code/humanoid-lab-main/dev.sh:400) içinde görülüyor. Dolayısıyla C++ yeteneği hazır olsa da kalıcı episode replay komutu henüz yok.

Temiz ekleme şöyle olacak:

```bash
# Terminal 1
./dev.sh isaac-g1-sonic dex3

# Terminal 2
./dev.sh sonic-controller replay

# Terminal 3
./dev.sh sonic-replay \
  --dataset first_tur_processed/sonic_v1/corpus \
  --episode 123 \
  --record \
  --metrics
```

Default keyboard davranışını bozmayacağız. Launcher’ın mevcut SONIC model/ownership kısmını kopyalamak yerine ortak helper’a ayırıp yalnız input modunu değiştireceğiz:

```text
keyboard → mevcut kullanım
zmq_manager → VLA ve dataset latent replay
```

Replay sender:

- Isaac state hazır olmadan playback başlatmayacak.
- İlk geçerli token’ı önceden stage edecek.
- Monotonik `frame_index` kullanacak.
- Tam 50 Hz yayınlayacak.
- NaN/Inf veya zamanlama kaçırılırsa explicit stop yapacak.
- SONIC’in stale token’ı sonsuza kadar tutmasına güvenmeyecek.
- Episode sonunda kısa, config ile belirlenen final hold yapacak.
- Ardından explicit stop/reset uygulayacak.
- Sonraki episode’a fallen robot state’iyle geçmeyecek; temiz restart/reset kullanılacak.

Decoder çıktısını önceden hesaplayıp saklamayacağız. SONIC decoder’ın girdisi token yanında measured 10-frame robot history içeriyor; dolayısıyla replay mutlaka resmi decoder ile closed-loop yapılmalı.

## Isaac doğrulama sırası

### Kapı 1 — Resmi baseline

Önce değiştirilmemiş bir resmi SONIC reference:

```text
reference → official encoder → token → official decoder → Isaac
```

hattından geçecek.

Bu başarısızsa converter’ı suçlamayacağız; Isaac/controller entegrasyonu araştırılacak.

### Kapı 2 — Unitree episode

Beklenen:

- Ayaklar yerde ve stable.
- Waist/legs synthetic standing’e yakın.
- Kollar Unitree desired trajectory’yi takip ediyor.
- Hands canonical action’ı takip ediyor.
- Fall, joint saturation ve stale packet yok.

### Kapı 3 — NVIDIA full-body episode

Beklenen:

- SONIC body execution original desired body29 trajectory’yi kabul edilebilir hata ile izliyor.
- Root heading stabil.
- Ayağın contact sırasında kayması sınırlı.
- Hands ayrıca takip ediliyor.

### Kapı 4 — Corpus QC

Her episode için replay yapmak pahalı olabilir. Dolayısıyla:

- Her episode offline schema/kinematic/token QC’den geçecek.
- Her source/task grubundan stratified örnekler Isaac replay görecek.
- Warning/outlier episode’lar ayrıca replay edilecek.
- Kullanıcı istediği herhangi bir episode’u tek komutla tekrar oynatabilecek.
- İsterseniz daha sonra bütün corpus için overnight replay modu da eklenebilir.

## Şu anda açık kalan gerçek kararlar

| Konu | Durum | Önerim |
|---|---|---|
| SONIC v1.1 ve checksum pinleri | Kapalı | Aynen kilitle |
| Encoder 1751D construction | Kapalı | Resmi G1 zero-fill |
| 50 Hz derived timeline | Kapalı | Stock 40-step/0,8 s horizon |
| Output action 78D | Kapalı | 64 + 7 + 7 |
| Output state 46D | Kapalı | 43 + projected gravity |
| Output format | Kapalı | GR00T LeRobot v2 |
| Unitree action zamanlaması | Kapalı | Aynı satır, shift yok |
| Unitree missing body | Tasarım kapalı, değer seçilecek | Tek frozen official standing reference |
| Fruits body/hand order | Kapalı | Name-based source adapter |
| Apple body order | Kapalı | Modality block + canonical names |
| Apple hand order/unit | Açık, bulk hand conversion blocker | Collector/config veya kontrollü hand replay ile çöz |
| Root ve gravity | Kapalı | Synthetic upright, açık provenance |
| Unitree primary camera | Karar önerildi | Sol head camera → `ego_view` |
| Replay transport | Kapalı | Mevcut ZMQ v4 |
| Replay lifecycle | Tasarım kapalı | Explicit readiness/start/stop/reset |
| Golden parity threshold | Ölçümle belirlenecek | Official C++ baseline’a göre |

## Önerdiğim ilk implementation dilimi

Bulk conversion’a hemen girmem. İlk kalıcı engineering dilimi şu olmalı:

1. Canonical joint/state/action schema ve üç source adapter.
2. Joint permutation unit testleri.
3. 20/30→50 Hz timestamp/resampling katmanı.
4. Resmi standing reference extraction.
5. Tam 1751D SONIC observation builder.
6. C++ golden-token oracle ve Python parity testi.
7. Bir Unitree ObjectPlacement episode body+hand conversion.
8. Bir AppleToPlate episode body conversion; el yalnız schema çözülürse eklenir.
9. Kalıcı ZMQ v4 episode replay CLI.
10. İki episode için Isaac replay metrics/evidence.
11. Bunlar geçince bulk corpus conversion.

Bu sırayla gidersek ilk episode üzerinde bütün sözleşmeyi kanıtlamadan 2,8 milyon ham frame’i 50 Hz’e büyütüp yanlış latent üretme riskimiz olmaz.

Kısacası: planın body/SONIC kısmı artık yeterince net. Ben implementasyonda `first_tur_ham`ı immutable tutup, kalıcı adapter–encoder–QC–replay bileşenlerini kurarım. Bulk conversion öncesindeki iki zorunlu deney ise frozen standing reference’ın çıkarılması ve AppleToPlate el sözleşmesinin çözülmesidir.