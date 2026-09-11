# Modüler G1 Simülasyon, SONIC, GR00T ve CloudWalk Uygulama Planı

## 1. Amaç

Bu planın amacı, Raider üzerinde çalışan sistemleri tek ve anlaşılır bir bileşen
ağına dönüştürmektir:

```text
Isaac veya MuJoCo
    robot state + kamera
             ↓
GR00T policy
    zamanlanmış policy action
             ↓
SONIC body controller + seçilen el adapter'ı
    düşük seviye robot komutu
             ↓
Isaac veya MuJoCo
             ↺
```

Sistem şu temel davranışları sağlamalıdır:

- Simulator tek başına başlatılabilmelidir.
- Controller bağlı değilse G1'e gizli bir hold/position target uygulanmamalı;
  serbest kök, gravity ve passive actuator'larla robot doğal olarak düşmelidir.
- SONIC bağlanınca aynı simulator API'sinden state almalı ve aynı API'ye geçerli
  body joint komutları vermelidir.
- GR00T bağlanınca seçilen model paketine göre observation almalı ve SONIC ile
  el adapter'larının kabul ettiği typed action üretmelidir.
- CloudWalk ayrı bir runtime veya ikinci bir SONIC–Isaac yolu olmamalıdır.
  CloudWalk; checkpoint, embodiment, scene, prompt, adapter ve kabul ölçütlerini
  seçen bir profil olmalıdır.
- Inspire FTP ve Dex3 robot varyantlarının ikisinde de robot üzerine bağlı,
  öne/yere doğru bakan head RGB camera bulunmalıdır.
- GUI, head camera ve video kaydı fizik döngüsünün sahibi olmamalı; bunların
  açılıp kapanması robot trajectory'sini değiştirmemelidir.
- Isaac Lab, Isaac-GR00T ve SONIC upstream checkout'ları pinli ve mümkünse
  tamamen temiz kalmalıdır.

Plan bütün nihai tasarımı baştan tanımlar; uygulama ise dört büyük kapıda
ilerler. Bir kapı Raider'da kendi kabul kriterlerini geçmeden sonraki kapı
tamamlanmış sayılmaz.

---

## 2. Değişmez Mimari İlkeler

### 2.1 Tek kanonik yol

Aynı iş için birden fazla runtime yolu bulunmayacaktır. Özellikle:

- “genel SONIC–Isaac” ve “CloudWalk SONIC–Isaac” diye iki entegrasyon olmayacak;
- interactive ve headless modlar farklı fizik döngüleri kullanmayacak;
- Inspire ve Dex3 farklı simulator API'leri oluşturmayacak;
- test controller, SONIC ve gelecekteki controller'lar aynı `JointCommand`
  sözleşmesini kullanacak;
- CloudWalk kendi actuation, watchdog veya process graph implementasyonunu
  taşımayacaktır.

### 2.2 Upstream kod bütünlüğü

Şunlar normal çalışma sırasında değiştirilmeden kullanılacaktır:

- Isaac Lab runtime ve asset kodu;
- Isaac-GR00T `Gr00tPolicy`, `PolicyServer`, `PolicyClient` ve inference kodu;
- SONIC body encoder/decoder, model semantiği ve controller matematiği.

Bizim yazacağımız kod şunlarla sınırlı olacaktır:

- typed sözleşmeler;
- ince simulator/controller/policy/hand adapter'ları;
- declarative model ve run profilleri;
- küçük bir orchestrator;
- güvenlik/lifecycle kontrolleri;
- kanıt ve kabul testleri.

Farklı GR00T fine-tune'ları upstream GR00T kodunu değiştirmeyecektir. Dataset,
modality, statistics, processor, embodiment, checkpoint ve action schema bir
`ModelPackage` içinde bulunacaktır.

SONIC upstream koduna değişiklik gerçekten kaçınılmaz olursa değişiklik:

1. body latent semantiğini ve decoder'ı değiştirmemeli;
2. küçük ve tek amaçlı olmalı;
3. repo içinde ayrı bir patch olarak tutulmalı;
4. upstream commit'ine bağlanmalı;
5. neden adapter ile çözülemediği belgelenmeli;
6. ayrı test ve provenance kanıtı taşımalıdır.

Hedef durumda şu kontroller boş sonuç vermelidir:

```bash
git -C /opt/src/isaaclab status --porcelain
git -C /opt/src/isaac-groot status --porcelain
git -C /opt/src/sonic status --porcelain
```

### 2.3 Fizik döngüsünün tek sahibi simulator'dır

Playing durumunda bir fizik iterasyonu yalnızca şu sırayı izler:

```text
command al ve geçerliliğini kontrol et
→ command varsa uygula, yoksa passive yap
→ scene.write_data_to_sim()
→ sim.step()
→ scene.update(sim_dt)
→ yeni state/camera/evidence yayınla
```

Kurallar:

- Loop başına tek `sim.step()` vardır.
- `sim.step()` sonrasında ekstra `simulation_app.update()` yoktur.
- Paused durumda fizik ilerletmeden UI için `sim.render()` kullanılır.
- Başlangıçta tek hard reset vardır.
- Pause→Play sırasında refreeze, teleport veya `settle_frames` yoktur.
- Recorder veya camera consumer fizik adımı atamaz.
- Tekrarlanan reset ancak açık bir `Reset` lifecycle isteği ile yapılabilir.

Bu çekirdek, Raider'daki pinli Isaac Lab sürümünün resmî
`InteractiveScene`/`SimulationContext` demo kalıbından türetilecektir:

- <https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/02_scene/create_scene.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html>

### 2.4 Controller yoksa gerçek passive davranış

Başlangıç pozu robotu spawn eder; ayakta tutmaz. Simulator şu durumlarda body
actuator stiffness/damping ve effort hedefini passive hâle getirir:

- hiç controller bağlanmadıysa;
- geçerli command henüz gelmediyse;
- command TTL geçmişse;
- controller kapandıysa;
- episode/reset kimliği uyuşmuyorsa;
- command bozuk, non-finite veya joint schema ile uyumsuzsa.

Default sonuç açıkça şudur:

```text
controller yok + Play + free root + gravity → G1 düşer
```

Simulator controller yüklenene kadar robotu gizlice tutmaz. Gerekirse SONIC'in
`HOLDING` durumu doğrulanmış standing latent/reference üreterek robotu tutar;
bu davranış simulator'a ait değildir.

### 2.5 Body ve el sahipliği ayrıdır

SONIC body controller'ın sorumluluğu:

```text
64D body latent + robot state/history → 29 body joint command
```

El adapter'ının sorumluluğu:

```text
typed left/right hand action → seçilen elin gerçek joint command'ı
```

Komutlar simulator sınırında birleştirilir:

```text
BodyJointCommand[29]     ← SONIC
LeftHandJointCommand     ← HandAdapter
RightHandJointCommand    ← HandAdapter
              ↓
CompleteRobotCommand
```

Inspire/Dex3 değişimi SONIC body decoder'ını değiştirmez. Bununla birlikte el
asset'inin kütlesi, ataleti ve wrist bağlantısı body dinamiğini etkilediği için
her body+hand+SONIC birleşimi ayrıca simülasyonda doğrulanır.

### 2.6 Şema uyumluluğu başlatmadan kontrol edilir

Boyut eşitliği semantik uyumluluk sayılmaz. Her producer ve consumer şema
kimliğini ilan eder. Örnekler:

```text
g1_body_state_v1
g1_body_lowcmd_v1
sonic_g1_body_latent_v1
g1_inspire_policy_action_v1
g1_dex3_policy_action_v1
inspire_ftp_joint_command_v1
dex3_joint_command_v1
```

Orchestrator uyumsuz policy/controller/hand birleşimini process'leri
başlatmadan reddeder. Sessiz truncate, pad, reshape veya yaklaşık joint-order
eşlemesi yapılmaz.

---

## 3. Hedef Bileşen Ağı

```text
                              RunProfile
                                  │
                                  ▼
                            Orchestrator
                 ┌────────────────┼─────────────────┐
                 ▼                ▼                 ▼
        SimulatorService   ControllerService   PolicyService
          Isaac/MuJoCo          SONIC              GR00T
                 │                ▲                 ▲
                 │ RobotState     │ RobotState      │ RobotState
                 ├────────────────┤                 │
                 ├──────────────────────────────────┤
                 │ CameraFrame                      │
                 └──────────────────────────────────┤
                                                   │
                              PolicyActionChunk ───┘
                                      │
                              ActionScheduler
                               ┌──────┴──────┐
                               ▼             ▼
                         SONIC body       HandAdapter
                               │             │
                               └──────┬──────┘
                                      ▼
                             CompleteRobotCommand
                                      │
                                      ▼
                                SimulatorService
```

CloudWalk bu grafikte ayrı bir servis değildir. Şunları seçen `RunProfile`dır:

- CloudWalk checkpoint/model package;
- `grab the bottle` prompt'ı;
- G1 embodiment ve el varyantı;
- head camera;
- masa, bottle, ışık ve oda scene'i;
- controller/policy oranları;
- görev ve kanıt metrikleri.

---

## 4. Kanonik Input/Output Sözleşmeleri

Bütün mesajlar en az şu ortak envelope'u taşır:

```text
schema_version
run_id
episode_id
source_component
sequence
source_timestamp_ns
physics_tick (varsa)
```

`run_id` farklı oturumları, `episode_id` reset öncesi/sonrasını ayırır.
Sequence her producer için monoton artar. Eski run/episode mesajları koşulsuz
reddedilir.

### 4.1 Simulator input: `CompleteRobotCommand`

Body bölümü:

```text
BodyJointCommand
- joint_schema
- valid_from_tick
- valid_until_tick
- q[29]
- dq[29]
- tau[29]
- kp[29]
- kd[29]
```

El bölümleri seçilen `HandSpec`e göre typed mesajlardır:

```text
LeftHandJointCommand
RightHandJointCommand
- hand_schema
- joint_names
- position/effort alanları
- valid_from_tick
- valid_until_tick
```

Isaac body effort'i SONIC/MuJoCo parity formülü ile hesaplar ve joint effort
limitlerine göre sınırlar:

```text
effective_tau = tau
              + kp * (q_command - q_measured)
              + kd * (dq_command - dq_measured)
```

### 4.2 Simulator output: `RobotState`

```text
RobotState
- robot_schema
- physics_tick
- simulated_time
- root position/orientation
- root linear/angular velocity
- IMU/projected gravity
- body q/dq/tau[29]
- selected hand state
- last applied command sequence
- control mode: passive|controlled|fault
```

Units, frame adları ve quaternion sırası şemada sabittir:

```text
position: metre
angles: radyan
time: saniye/nanosaniye
quaternion: wxyz
world/body/sensor frame: açıkça isimli
```

### 4.3 Simulator output: `CameraFrame`

```text
CameraFrame
- camera_schema
- camera_id
- physics_tick
- simulated_time
- width/height/channels
- encoding
- intrinsics
- camera-to-parent transform
- pixel payload veya buffer reference
```

Head camera ile debug viewport ayrıdır:

- `g1_head_rgb`: GR00T observation sensor'ıdır ve robot link'ine bağlıdır.
- `debug_viewport`: insanın dışarıdan izlediği penceredir.
- Recorder ikisini de gözlemleyebilir ama hiçbirini step edemez.

### 4.4 GR00T input: `PolicyObservation`

Model package hangi alanların zorunlu olduğunu ilan eder:

```text
PolicyObservation
- observation_schema
- RobotState veya seçilmiş state alanları
- bir veya daha fazla CameraFrame
- prompt/task
- source physics tick
```

CloudWalk modeli için beklenen içerik:

```text
head RGB
G1 + seçilen el state'i
"grab the bottle"
```

### 4.5 GR00T output: `PolicyActionChunk`

```text
PolicyActionChunk
- action_schema
- source_observation_sequence/tick
- horizon
- target_action_rate_hz
- body actions
- left hand actions
- right hand actions
- inference latency
```

CloudWalk `UNITREE_G1_SONIC` checkpoint'i özelinde:

```text
40 × 78
├── 64D SONIC motion token
├── 7D sol el action
└── 7D sağ el action
```

Bu boyut evrensel değildir. Başka fine-tune farklı horizon, body action veya el
şeması üretebilir. Şema model paketinin parçasıdır.

### 4.6 SONIC input

SONIC iki akış alır.

Robot state/history:

```text
SonicRobotState
- physics_tick
- base angular velocity
- projected gravity
- body q[29]
- body dq[29]
- previous body action[29]
```

Zamanlanmış body latent:

```text
SonicAction
- schema: sonic_g1_body_latent_v1
- source policy sequence
- valid_from_tick
- valid_until_tick
- motion_token[64]
```

GR00T zorunlu latent kaynağı değildir. Aynı API'yi şunlar besleyebilir:

- doğrulanmış standing reference;
- canned latent test fixture;
- upstream SONIC keyboard/planner teleop;
- başka bir policy.

Keyboard, Isaac'e doğrudan pose veya effort yazan ayrı bir controller değildir.
Gerçek SONIC input/lifecycle yolunu sürer; Isaac yalnız SONIC'in ürettiği
`BodyJointCommand`u görür:

```text
gerçek terminal/TTY keyboard input
→ upstream SONIC keyboard/planner input
→ SONIC state machine ve decoder/controller
→ BodyJointCommand
→ Isaac
```

Bu sayede keyboard testi SONIC'i bypass eden sahte bir teleop testi olmaz.

### 4.7 SONIC output

```text
BodyJointCommand
- schema: g1_body_lowcmd_v1
- sequence
- produced_at
- valid_from_tick
- valid_until_tick
- q/dq/tau/kp/kd[29]
```

SONIC output'u geçersiz, non-finite, sıra dışı veya stale ise simulator passive
moda geçer.

### 4.8 Action scheduler

GR00T düşük frekansta chunk üretir; SONIC daha yüksek frekansta tek action step
alır. Scheduler:

- chunk'ın hangi observation'dan üretildiğini korur;
- yalnız geçerli episode'daki en yeni chunk'ı kabul eder;
- her control tick için doğru body/el action step'ini seçer;
- eski inference cevabını reddeder;
- chunk biterse açık hold/timeout politikasını uygular;
- yeni chunk geldiğinde tanımlı biçimde replace/blend eder;
- log'da seçilen chunk ve index'i görünür kılar.

Başlangıç hedef oranları profile'da açıkça yazılır:

```text
Isaac physics: 200 Hz hedefi
SONIC/action: 50 Hz hedefi
GR00T inference: 2.5 Hz hedefi
CloudWalk action horizon: 40
```

Wall-clock ve simulated-time birbirine karıştırılmaz. Kabul metrikleri her
ikisinin değerini ayrı raporlar.

---

## 5. Robot, El ve Kamera Tanımları

### 5.1 Robot bileşimi

```text
RobotSpec = RobotBodySpec + HandSpec + CameraSpec
```

İlk body:

```text
RobotBodySpec: unitree-g1-29dof-v1
```

İlk el varyantları:

```text
HandSpec: none-v1
HandSpec: inspire-ftp-v1
HandSpec: dex3-v1
```

`none` varyantı body fiziği ve SONIC entegrasyonu için iç referans olarak
tutulur. Kullanıcı tesliminde Inspire ve Dex3 birinci sınıf varyantlardır.

Her `HandSpec` şunları tanımlar:

- asset kaynağı ve SHA/provenance;
- link ve joint adları/sırası;
- wrist attachment;
- joint limits;
- coupling/mimic kuralları;
- actuator modeli;
- collision meshes;
- mass/inertia doğrulaması;
- policy action schema;
- simulator command schema;
- gerçek robot driver desteği ve kanıt seviyesi.

### 5.2 Inspire FTP

Raider'daki Isaac Lab 2.3.2, resmî `G1_INSPIRE_FTP_CFG` ve
`g1_29dof_inspire_hand.usd` asset'ini içerir. Ancak resmî config üst-gövde
manipülasyonu için varsayılan olarak fixed-base ve gravity-disabled'dır. Bizim
adapter kopya config üzerinde açıkça:

```text
fix_root_link = false
disable_gravity = false
```

ayar ve bunu runtime evidence içinde raporlar. Upstream config değiştirilmez.

### 5.3 Dex3

Pinli SONIC kaynağında Dex3/`g1_29dof_with_hand` URDF, USD ve MJCF parçaları
bulunsa da pinli Isaac Lab asset modülünde Inspire gibi hazır ve açık bir
`G1_DEX3_CFG` bulunmamaktadır. Bu nedenle Dex3 hazır kabul edilmeyecektir.

Dex3 adapter şu doğrulamaları tamamlamalıdır:

- kaynak asset ve lisans/provenance;
- mesh dosyalarının tam çözülmesi;
- G1 pelvis articulation root;
- 29 body joint parity;
- Dex3 joint/link sırası;
- wrist parent-child ilişkisi;
- collision ve self-collision davranışı;
- joint limits, actuator ve mimic/coupling;
- mass ve inertia;
- ground/contact davranışı;
- head camera parent link uyumu.

Proxy/fake hand görseli Dex3 physics kabulü olarak raporlanamaz.

### 5.4 Ortak head camera

Arşivlenmiş ve mevcut tasarımdaki camera korunur ve CloudWalk scene kodundan
çıkarılarak ortak `CameraSpec` yapılır:

```yaml
id: g1-head-rgb-v1
resolution: [640, 480]
focal_length_mm: 15.1159925892
horizontal_aperture_mm: 20.955
parent_link: torso_link
position_rel_parent_m: [0.0576235, 0.01753, 0.41987]
rotation_wxyz_parent: [0.9149596678, 0.0, 0.4035452964, 0.0]
```

Bu camera hem Inspire hem Dex3 varyantında bulunmalıdır. Her varyant için:

- parent link mevcut ve doğru olmalı;
- camera robotla beraber hareket etmeli;
- aşağı/öne bakan görüş yönü world-space ray veya kalibrasyon target'ı ile
  doğrulanmalı;
- intrinsics aynı kalmalı;
- ardışık physics tick'lerinde bağımsız RGB buffer alınmalı;
- frame hash ve changed-pixel metrikleri fizik hareketini göstermeli;
- camera frame'i doğru physics tick ile eşlenmelidir.

---

## 6. Model Package ve Profile Tasarımı

### 6.1 GR00T model package

Her fine-tune self-describing bir dizindir:

```text
model-package/
├── checkpoint/
├── policy.yaml
├── modality.json
├── processor_config.json
├── statistics.json
├── embodiment config
└── provenance.json
```

Örnek CloudWalk manifest'i:

```yaml
id: cloudwalk-g1-inspire-v10
provider: groot-n1.7
checkpoint: checkpoint-30000
checkpoint_revision: f980fd880c92b984d65dcc3f9dc82b98bc8009c
embodiment: unitree_g1_sonic
observation_schema: g1_inspire_head_rgb_state_v1
action_schema: sonic_g1_inspire_action_v1

action:
  horizon: 40
  rate_hz: 50
  body:
    type: sonic_latent
    size: 64
  left_hand:
    type: normalized_hand
    size: 7
  right_hand:
    type: normalized_hand
    size: 7
```

Yeni robot/el için upstream GR00T kodu değil; yeni dataset, modality,
statistics, embodiment ve model package oluşturulur.

### 6.2 Run profile

Çalışan topoloji tek YAML ile tanımlanır:

```yaml
profile: cloudwalk-bottle-v1

components:
  simulator:
    provider: isaac
    robot_body: unitree-g1-29dof-v1
    hand: inspire-ftp-v1
    camera: g1-head-rgb-v1
    scene: cloudwalk-bottle-v1

  controller:
    provider: sonic
    model: sonic-v1.1

  policy:
    provider: groot
    model_package: cloudwalk-g1-inspire-v10

connections:
  - simulator.robot_state -> controller.robot_state
  - simulator.robot_state -> policy.robot_state
  - simulator.head_rgb -> policy.head_rgb
  - policy.body_action -> scheduler.body_action
  - scheduler.body_action -> controller.sonic_action
  - policy.left_hand_action -> hand_adapter.left_action
  - policy.right_hand_action -> hand_adapter.right_action
  - controller.body_command -> simulator.body_command
  - hand_adapter.hand_command -> simulator.hand_command
```

Profile JSON Schema ile doğrulanır; editor autocomplete ve hata mesajı verir.

### 6.3 Görünür graph ve status

Çalışan config'in kendisinden graph üretilir:

```bash
humanoid validate configs/profiles/cloudwalk-bottle-v1.yaml
humanoid graph configs/profiles/cloudwalk-bottle-v1.yaml
humanoid run configs/profiles/cloudwalk-bottle-v1.yaml
humanoid status
humanoid stop
```

`graph` dokümantasyonda elle çizilmiş tahmini bir diyagram değil, validate edilen
profile'ın gerçek bağlantılarını gösterir. `status` her component için şunları
raporlar:

```text
state
ready/health
input/output schema
last sequence
last physics tick
message age
mean rate / max gap
passive|controlled|fault
model/asset provenance
```

---

## 7. Transport, Lifecycle ve Güvenlik

### 7.1 Referans transport

İlk sürümde process ve Python/C++ environment ayrımı için tek referans transport
kullanılır:

- data plane: ZMQ PUB/SUB;
- control plane: ZMQ REQ/REP;
- payload: versioned Protobuf veya açıkça sürümlenmiş MsgPack;
- endpoint'ler: orchestrator tarafından run başında dinamik üretilen manifest.

Sabit portlar business logic içine gömülmez. ZMQ adapter arkasında kalır;
simulator `CommandSource`/`StateSink`, SONIC ve GR00T typed interface'leri bilir.

Gerçek G1 için sonradan Unitree DDS transport adapter'ı eklenebilir. Bu, üst
seviye `RobotState`/`JointCommand` sözleşmesini değiştirmez. Simulator DDS'si
gerçek robot ağıyla aynı domain/interface'i paylaşamaz.

### 7.2 Lifecycle

Ortak lifecycle:

```text
CREATED
→ STARTING
→ READY
→ RUNNING
→ STOPPING
→ STOPPED
veya FAULT
```

Simulator ek durumları:

```text
PAUSED | PLAYING
```

Controller ek durumları:

```text
PASSIVE | HOLDING | CONTROLLED | STALE | FAULT
```

Kurallar:

- `READY`, input almak için hazır olduğunu kanıtlar; davranış başarısı değildir.
- Controller start, simulator timeline'ını otomatik değiştirmez.
- Simulator normal profillerde doğrudan Play başlar; paused başlangıç Kapı 1
  komut sözleşmesinin parçası değildir.
- Reset yeni `episode_id` üretir ve bütün cached state/action/command'ları
  geçersiz kılar.
- Stop önce command üretimini keser, simulator'ı passive yapar, sonra process'leri
  ters dependency sırasıyla kapatır.
- PID adına göre geniş `pkill` normal lifecycle yöntemi değildir; orchestrator
  yalnız kendi child process kimliklerini yönetir.

### 7.3 Failure ve backpressure

- Simulator policy inference bekleyerek fizik loop'unu durdurmaz.
- GR00T worker yalnız en yeni gerekli observation üzerinde çalışır.
- Eski inference reply episode/sequence kontrolünde reddedilir.
- SONIC state veya latent stale olursa yeni body command üretmez.
- Simulator command TTL geçince aynı fizik iterasyonunda passive olur.
- Hand command stale olduğunda profile'ın açık `open|hold|passive` politikası
  uygulanır; default gizli değildir.
- Bir consumer yavaşsa unbounded queue oluşmaz; bounded latest-value veya açık
  drop policy kullanılır.

### 7.4 Kanıt sınırları

- Static test runtime kanıtı değildir.
- Import veya HTTP/ZMQ ready mesajı model inference kanıtı değildir.
- Isaac tensor hareketi viewport hareketiyle karıştırılmaz.
- Viewport videosu physics kanıtının tek kaynağı değildir.
- Simulator başarısı fiziksel robot başarısı değildir.
- Inspire görünmesi exact collision/mass/mapping kanıtı değildir.
- Dex3 proxy asset exact Dex3 kabulü değildir.

---

## 8. Dört Uygulama Kapısı

## Kapı 1 — Temiz Isaac G1 Platformu

**Durum: TAMAMLANDI — 11 Eylül 2026'da yeniden doğrulandı**

Kapı 1, Raider üzerinde hem Inspire FTP hem Dex3 varyantı için ayrı Isaac
süreçlerinde doğrulandı. Controller olmadan robotlar passive actuator, serbest
kök, gravity ve ground collision ile doğal olarak düştü; fizik tick'leri monoton
ilerledi, robotlar zeminde sınırlandı ve ortak 640×480 head camera kesintisiz
frame üretti. Normal profil doğrudan Play başlar; paused başlangıç seçeneği Kapı
1 kapsamından çıkarıldı. Son kabul koşuları:

- `isaac-g1-inspire-ftp --test passive-fall --headless`: PASS, 2.383 tick,
  0,727 m root düşüşü, 297 adet 640×480 kamera frame'i ve 296 frame değişimi;
- `isaac-g1-dex3 --test passive-fall --headless`: PASS, 2.698 tick, 0,730 m
  root düşüşü, 337 adet 640×480 kamera frame'i ve 336 frame değişimi;
- `isaac-g1-dex3` GUI: aktif X display otomatik seçilerek doğrudan Play,
  görünür düşüş ve akan kamera kullanıcı tarafından doğrulandı; normal koşu
  25.544 fizik tick'iyle `COMPLETED` oldu;
- `isaac-g1-no_hands --test passive-fall --headless`: PASS, 1.384 tick,
  0,721 m root düşüşü ve 173 adet 640×480 kamera frame'i;
- `./dev.sh smoke`: 13 PASS, 0 WARN/BLOCKED/FAIL;
- `./doctor.sh`: PASS; Python testleri 10/10 ve üç shell test betiği PASS.

#### Kapanışta öğrenilen sorunlar ve kararlar

- Isaac Lab standalone fizik döngüsü ile Isaac Sim editor toolbar'ının aynı anda
  timeline sahibi yapılması Play/Pause davranışını güvenilmez hale getirdi.
  Normal çalışma bu nedenle doğrudan Play başlar; fizik döngüsünün tek sahibi
  `SimulatorService`tir.
- Reset callback'i içinde erken `timeline.pause()` yapmak PhysX tensor state'ini
  güncellerken Fabric ana viewport'un teleport edilen articulation pozunu
  kaçırmasına yol açtı. Callback artık yalnız lifecycle isteğini kuyruğa koyar.
  Reset, resmî Isaac Lab demo sırasıyla çalışan simulator döngüsünde state'i
  yazar, `scene.reset()` çağırır, aynı tek fizik adımında pozu yayınlar ve sonra
  pause eder.
- Ek `simulation_app.update()`, `sim.forward()`, settle-frame ve sürekli
  refreeze/hold denemeleri reddedildi; final runtime'da bulunmaz.
- Head camera için ikinci bir Isaac viewport açmak gereksiz render/timeline
  etkileşimi ve yanıltıcı kamera davranışı oluşturdu. Normal viewer bu nedenle
  yalnız ana viewport'u açar. Kamera profilde korunur; `--head-camera-window`
  verildiğinde veya `--test passive-fall` kabulünde robotun `torso_link`ine bağlı
  640×480 Isaac Lab sensörü kurulur. İkinci pencere GPU-backed viewport'tur;
  texture-to-host kopyası yapılmaz.
- Resmî Isaac Lab ortam döngüsü gibi fizik `sim.step(render=False)` ile ilerler ve
  dış sayaç her `render_interval=8` adımda `sim.render()` çağırır. Bu özel servis
  dış render çizelgesinin sahibi olduğundan `SimulationContext` iç aralığı `1`dir;
  aynı `8` değerinin içte de verilmesi Kit update çağrısını yaklaşık 40 ms bekletip
  GUI'yi `RTF≈0,5 / 13 FPS` düzeyine indiriyordu. Raider GUI ölçümünde düzeltme
  kararlı örnekleri `RTF=1,00–1,04 / 25–26 FPS` düzeyine çıkardı.
- Milestone adı runtime tasarımına taşınmadı. Genel yol
  `humanoid_lab.simulators.isaac`, komut `./dev.sh isaac-g1 ...`, kabul davranışı
  ise yalnız açık `--test passive-fall` seçeneği altındadır.
- Raider masaüstü yeniden açıldığında aktif X display `:0`dan `:1`e geçebilir.
  GUI komutu çağıran terminalin canlı `DISPLAY` değerini korur; yapılandırılmış
  socket yoksa hosttaki tek aktif X11 socket'ini seçer ve seçimi görünür biçimde
  bildirir.
- Isaac Sim 5.1 `simulation_app.close()` çağrısı bu hostta hem ana thread hem
  bounded worker denemesinde dönmeyebilir. CLI GUI'de en fazla 2 saniye, headless
  koşuda 20 saniye bekler; sonra `forced_exit=true` bildirip süreci sonlandırır.
  `dev.sh` tekil çalışma kilidi ile sinyal/HUP sonrasında doğrulamalı process-group
  temizliği uygular. Bu davranış graceful Kit shutdown olarak yorumlanmaz.
- Eski CloudWalk/SONIC ağırlıklı implementasyon aktif runtime ile karışmaması
  için `archive/eski_kotu/` altında saklandı.

### Hedef

SONIC, GR00T, CloudWalk, DDS ve özel controller kodu olmadan; resmî Isaac Lab
demo kalıbından türetilmiş küçük, tek sahipli ve güvenilir G1 simulator servisi.

### Kapsam

- Yeni `SimulatorService` ve temel typed contract'lar.
- Ground plane, ışık ve tek G1.
- G1 29DoF body.
- Inspire FTP varyantı.
- Dex3 varyantı.
- İkisinde de ortak head camera.
- Dış debug viewport.
- Headless ve GUI'nin aynı fizik loop'unu kullanması.
- Bağımsız evidence/recording sink.
- Declarative robot/hand/camera profile'ları.

### Uygulama şartları

- Mevcut büyük runner kopyalanmaz.
- Resmî `InteractiveScene` demo yapısı temiz bir modülde yeniden kurulur.
- Sahne topology'si ilk reset öncesinde tamamlanır.
- Bir hard reset yapılır.
- Playing loop'ta tek `sim.step()` bulunur.
- Paused loop'ta yalnız `sim.render()` bulunur.
- Body controller yokken bütün body drive'ları gerçekten passive olur.
- El actuator davranışı `HandSpec`te açıkça tanımlanır; body'yi tutamaz.
- Kabul sırasında kamera çözünürlüğü ve frame değişimi bounded sayaç/hash ile
  ölçülür; frame dizisi RAM'de veya diskte biriktirilmez.

### Deliverable'lar

1. `SimulatorService` API ve şemaları.
2. `g1-29dof + inspire-ftp + head-camera` çalıştırılabilir profili.
3. `g1-29dof + dex3 + head-camera` çalıştırılabilir profili.
4. İç fizik referansı olarak `g1-29dof + none` profili.
5. GUI ve headless için aynı runner.
6. Saniyelik fizik/render/RTF durum çıktısı.
7. Bounded passive-fall ve head-camera kabul metrikleri.
8. Profile yükleme ile runtime joint/asset doğrulamaları.
9. Temiz start, doğrudan Play, Reset ve stop davranışları.

### Zorunlu kabul testleri

Her `inspire-ftp` ve `dex3` varyantı ayrı Isaac sürecinde:

```text
controller = absent
root = free
gravity = enabled
ground collision = enabled
body actuator = passive
```

koşullarıyla:

- robot başlangıç pozunda görünür;
- mesh path'leri tam çözülür;
- head camera frame'i dolu ve doğru boyuttadır;
- debug viewport arayüzü yanıt verir ve görüntü akar;
- fizik tick'i monoton artar;
- root yüksekliği ve/veya pelvis yönelimi değişir;
- robot belirlenen pencere içinde gerçekten düşer;
- floor collision sonrası world altına sınırsız kaçmaz;
- head-camera 640×480 RGB frame üretir;
- bounded frame hash metriği ardışık head-camera görüntülerinin değiştiğini
  gösterir;
- uygulama force-quit dialog veya orphan process bırakmadan kapanır.

Ana görünür kapı:

```text
Play'e basıldı veya normal run başladı.
Controller yok.
Robot düşüyor.
Görüntü akıyor.
```

Bu dörtlünün biri yoksa Kapı 1 geçmez.

### Kapı 1 dışı

- Robotu ayakta tutmak.
- SONIC yüklemek.
- GR00T inference.
- CloudWalk sahnesi.
- Bottle başarısı.
- Fiziksel robot kontrolü.

---

## Kapı 2 — Tek SONIC Controller Hattı

### Durum (2026-09-11)

**Entegrasyon kilometre taşı çalışıyor; Kapı 2 henüz resmen kapanmış
sayılmamalıdır.** Tek Isaac actuation yolu, deterministik test controller'ı,
Unitree DDS üzerinden SONIC adapter'ı, isim bazlı 29-DoF eşleme, effort
sınırlama, TTL/passive fallback, Dex3/Inspire profilleri ve gerçek TTY sahibi
upstream keyboard akışı uygulanmış ve Raider'da ölçülmüştür.

Kapıyı kapatmadan önce kalanlar:

1. `CompleteRobotCommand` ve controller state mesajlarını bölüm 4'teki tam
   envelope'a (`schema_version`, `run_id`, `source_component`, source timestamp,
   `valid_from_tick`) yükseltmek.
2. 64D standing/canned latent'i typed fixture olarak ve kaynak/hash provenance'ı
   ile repo sözleşmesine almak; mevcut upstream reference/planner dosya yolu tek
   başına bu deliverable'ı kapatmaz.
3. İki terminaldeki Isaac ve SONIC süreçlerini tek `start/status/stop/reset`
   lifecycle altında birleştirmek. Mevcut güvenli reset controller aktifken
   isteği reddeder; SONIC history'sini koordine ederek temizleyen reset henüz yoktur.
4. Inspire ve Dex3 için doğrudan hand probe'larını ve her iki varyantı kapsayan
   otomatik kabul matrisini tamamlamak.
5. Keyboard tuşu/state/command/hareket korelasyonunu, terminal kopmasını ve
   orphan/port temizliğini tek self-contained run dizininde otomatik PASS/FAIL
   kanıtına dönüştürmek; eski ikinci controller/actuation yollarının kaldırma
   listesini yayımlamak.

Bu ayrım mimari yönü değiştirmez: DDS robot-emulation sınırı, upstream SONIC'i
Isaac'e özel bir decoder fork'undan koruduğu için doğru ana tasarımdır. Kalan
işler esas olarak sözleşme tamamlama, orkestrasyon ve tekrarlanabilir kanıttır.

### Hedef

Kapı 1'deki tek simulator API'sine önce test controller, sonra upstream SONIC
bağlamak; controller var/yok/stale davranışını kesinleştirmek.

### Kapsam

- `JointCommand` ingest ve TTL.
- Deterministik test controller.
- SONIC service wrapper/adapter.
- SONIC state history.
- Doğrulanmış 64D standing/canned latent kaynağı.
- Upstream SONIC keyboard/planner teleop input'u.
- 29 body joint `q/dq/tau/kp/kd` komutu.
- Action scheduler'ın body bölümü.
- Inspire ve Dex3 için ayrı hand adapter iskeleti ve doğrudan hand probe.
- Tek lifecycle ve tek status yolu.

### Uygulama şartları

- Upstream SONIC body decoder ve model semantiği değiştirilmez.
- Custom “CloudWalk SONIC harness” üretim runtime'ı olmaz.
- SONIC'in simulator'a verdiği kanonik çıktı `BodyJointCommand` olur.
- Isaac SONIC `lowcmd`'ini aynı formülle effort'a dönüştürür.
- Joint mapping isimle kurulur; sayı/sıra uyuşmazlığı fail-closed'dur.
- Inspire ve Dex3 el action'ları SONIC body decoder'ına girmez.
- SONIC kapalıyken simulator davranışı Kapı 1 ile aynıdır.
- Keyboard modu gerçek TTY sahibi ayrı bir input process/session üzerinden
  upstream SONIC'in kendi keyboard/planner state machine'ini kullanır.
- Keyboard input Isaac API'sine hiçbir zaman doğrudan joint/effort yazmaz.
- Keyboard terminali kapanır, input process'i ölür veya command akışı stale
  olursa tanımlı stop/watchdog yolu simulator'ı passive moda döndürür.
- Isaac GUI focus'u ile terminal keyboard focus'u birbirine karıştırılmaz;
  tuş kabulü ve SONIC state geçişi status/evidence içinde görünürdür.

### Deliverable'lar

1. Tek `ControllerService` sözleşmesi.
2. Test controller ile simulator API integration testi.
3. Upstream SONIC service adapter'ı.
4. SONIC state/latent input ve body command output şemaları.
5. Standing/canned latent fixture ve provenance.
6. `isaac-g1-sonic-keyboard` çalıştırılabilir profili ve tek kullanıcı komutu.
7. Gerçek TTY üzerinde upstream SONIC keyboard teleop session'ı; başlatma,
   yeniden bağlanma/status ve temiz stop akışı.
8. Keyboard tuşu → SONIC input kabulü → planner/control state geçişi → body
   command → Isaac fiziksel hareket korelasyon kanıtı.
9. Inspire ve Dex3 hand adapter contract'ları.
10. `start/status/stop/reset` lifecycle.
11. Rate, gap, latency, sequence ve TTL evidence'i.
12. Eski ikinci actuation/controller yollarının kaldırılma listesi.

### Zorunlu kabul testleri

Her body+hand varyantında:

1. SONIC kapalıyken robot Kapı 1'deki gibi düşer.
2. Test controller geçerli command gönderince hedef jointlerde fiziksel tepki
   görülür.
3. Test controller kesilince TTL sonrası body passive olur.
4. Upstream SONIC doğrulanmış state ve 64D latent alır.
5. Upstream SONIC finite ve doğru sıralı 29-joint command üretir.
6. SONIC command Isaac'te doğru DOF'lara ve effort limitleri içinde uygulanır.
7. Doğrulanmış standing/hold koşusunda robot için açık stabilite metrikleri
   raporlanır; yalnız “process çalıştı” kabul değildir.
8. SONIC öldürüldüğünde watchdog/TTL tetiklenir ve robot yeniden passive olur.
9. Reset bütün SONIC history ve eski command'ları geçersiz kılar.
10. Head camera ve viewport kontrol sırasında akmaya devam eder.
11. Kullanıcı tek belgelenmiş komutla Isaac + SONIC keyboard profilini açar.
12. Gerçek terminal/SSH TTY üzerinden upstream SONIC'in start/stop ve hareket
    tuşları alınır; kabul edilen tuş ve controller state geçişi gözlenir.
13. Keyboard ile verilen ileri/geri/dönüş veya ilgili upstream teleop komutu
    SONIC body command'ına ve Isaac'te ölçülebilir doğru harekete dönüşür.
14. Keyboard bırakıldığında upstream'in tanımlı idle/zero-command davranışı
    görülür; terminal kopunca sistem sonsuz son-command uygulamaz.
15. Keyboard teleop oturumu kapanırken SONIC ve Isaac orphan process, bozuk TTY
    veya tutulmuş port bırakmadan temiz kapanır.

Ana görünür kapı:

```text
SONIC yok       → robot düşer
SONIC geçerli   → robot kontrol edilir
SONIC keyboard  → kullanıcı Isaac'teki G1'i SONIC üzerinden sürer
SONIC kayboldu  → timeout → robot passive olur/düşer
```

### Kapı 2 dışı

- GR00T model inference.
- CloudWalk scene/task başarısı.
- Aynı CloudWalk checkpoint'ini uyumsuz ele zorla bağlamak.
- Fiziksel robot hareket kanıtı.

---

## Kapı 3 — Generic GR00T ve Değiştirilebilir Model Paketleri

### Hedef

Upstream GR00T policy server'ı değiştirmeden; farklı fine-tune, embodiment,
observation ve action şemalarının aynı policy API üzerinden seçilebildiği hattı
SONIC ve hand adapter'larına bağlamak.

### Kapsam

- `ModelPackage` formatı.
- Generic `PolicyService` adapter'ı.
- `PolicyObservation` üretimi.
- `PolicyActionChunk` doğrulaması.
- Full action scheduler.
- Body/el action routing.
- Schema compatibility registry.
- Dataset observation replay.
- Nötr/boş Isaac sahnesinde gerçek GR00T→SONIC→Isaac closed loop.
- Model değiştirme ve provenance.

### Uygulama şartları

- Upstream `Gr00tPolicy`, `PolicyServer` ve `PolicyClient` değiştirilmez.
- Fine-tune farkları model package içindedir.
- CloudWalk sabitleri generic policy koduna gömülmez.
- Global `[40,78]` varsayımı yapılmaz; yalnız CloudWalk model manifest'i bunu
  ilan eder.
- Observation/action preprocess ve normalization model package'tan yüklenir.
- Uyumsuz hand/action schema başlamadan reddedilir.
- GR00T inference fizik loop'unu bloklamaz.

### Deliverable'lar

1. Versioned `ModelPackage` spec ve validator.
2. Generic upstream GR00T service adapter'ı.
3. En az bir gerçek fine-tune package ve bir küçük test/dummy package üzerinden
   model seçimi kanıtı.
4. Observation/action schema registry.
5. Dataset replay komutu ve evidence'i.
6. Prompt, camera, state, normalization ve output provenance kaydı.
7. Chunk scheduler ve stale reply reddi.
8. Inspire/Dex3 policy-hand compatibility kontrolü.
9. Nötr Isaac scene üzerinde gerçek GR00T→SONIC→Isaac teknik E2E.

### Zorunlu kabul testleri

- Datasetten gerçek observation upstream GR00T'a ulaşır.
- Prompt modele taşınır.
- Çıktı model manifest'indeki horizon ve action schema ile uyumludur.
- Shape, dtype ve finite değerler doğrulanır.
- Output body/el alanlarına typed biçimde ayrılır.
- Scheduler doğru sequence/index ile 50 Hz hedefinde action step üretir.
- SONIC body latent'i alıp gerçek body command üretir.
- Seçilen hand adapter doğru el komutunu üretir.
- Isaac aynı physics loop'ta body ve el komutlarını uygular.
- GR00T gecikirken fizik ilerlemeye devam eder.
- Eski episode'a ait inference cevabı uygulanmaz.
- Farklı model package seçildiğinde upstream GR00T kodu ve simulator kodu
  değişmez.
- Uyumsuz model/el birleşimi açıklayıcı hata ile start öncesi reddedilir.
- Head camera ve debug viewport akışı devam eder.

Ana görünür kapı:

```text
model package değişiyor
→ observation/action schema değişebiliyor
→ adapter seçimi değişebiliyor
→ upstream GR00T, SONIC ve Isaac kodu değişmiyor
```

### Kapı 3 dışı

- CloudWalk bottle görev başarısı.
- Her modelin her robot/elle otomatik uyumlu olduğu iddiası.
- Fiziksel G1 deployment kabulü.

---

## Kapı 4 — CloudWalk Closed-Loop Ürün Teslimi ve Legacy Temizliği

### Hedef

CloudWalk checkpoint, scene ve görev ölçütlerini Kapı 1–3'teki generic
bileşenlere yalnız declarative profile ve task adapter olarak eklemek; tek
komutla gerçek closed loop'u çalıştırmak ve eski paralel yolları kaldırmak.

### Kapsam

- CloudWalk v10 model package/checkpoint-30000.
- `UNITREE_G1_SONIC` embodiment.
- `grab the bottle` prompt.
- Dataset-kalibre masa, bottle, oda, ışık ve head camera scene'i.
- Uyumlu Inspire veya Dex3 hand seçimi.
- Gerçek GR00T inference.
- Gerçek SONIC body controller.
- Gerçek Isaac physics.
- Task/evidence/recording.
- Tek CLI, graph ve status.
- Legacy runner ve duplicate protokollerin kaldırılması.

### Uygulama şartları

- CloudWalk kodu simulator veya SONIC lifecycle sahibi olmaz.
- CloudWalk'a özgü kod yalnız scene, model package, observation adapter, hand
  adapter seçimi ve task metrikleridir.
- Checkpoint hangi el/action schema ile eğitildiyse profile bunu açıkça seçer.
- Aynı checkpoint uyumsuz ele sessizce bağlanmaz.
- Teknik E2E başarısı ile bottle görev başarısı ayrı raporlanır.

### Deliverable'lar

1. `cloudwalk-bottle-v1.yaml` run profile.
2. CloudWalk GR00T model package ve provenance.
3. Dataset-kalibre scene/camera config.
4. Tek `humanoid run|status|graph|stop` kullanıcı yolu.
5. Gerçek zamanlı component graph ve health görünümü.
6. Teknik E2E evidence paketi.
7. Bottle approach/contact/grasp/lift evidence paketi.
8. İki ardışık temiz-run kabul raporu.
9. Eski `run-sonic-isaac.py`, `run-cloudwalk-isaac.py`, duplicate controller,
   actuation, lifecycle ve hard-coded port yollarının kaldırılması.
10. Güncel README, operasyon ve sorun giderme rehberi.

### Teknik E2E kabulü

İki ardışık temiz süreçte:

- Isaac G1 + seçilen el + head camera doğru açılır.
- Tensor physics ve görüntü akışı ilerler.
- RGB/state/prompt gerçek upstream GR00T'a ulaşır.
- GR00T manifest'e uygun gerçek action chunk üretir.
- Scheduler doğru step'leri seçer.
- SONIC gerçek body command üretir.
- Hand adapter gerçek el command'ını üretir.
- Isaac doğru body/el jointlerini hareket ettirir.
- Sequence, rate, max-gap, latency ve TTL metrikleri eşikleri geçer.
- Reset ve stop temiz çalışır.
- Orphan process ve port bırakılmaz.
- Recorder fizik trajectory'sini değiştirmez.

### Görev kabulü

Nominal deterministic scene'de metrikler ayrı ayrı raporlanır:

```text
approach
contact
hand close
stable grasp
lift
bottle fell
max horizontal displacement
```

Başarı eşiği profile ve evidence validator içinde sürümlenir. “Video güzel
görünüyor” tek başına görev başarısı değildir; tensor/contact ve görsel kanıt
birlikte kullanılır.

### Ana görünür kapı

```text
humanoid run cloudwalk-bottle-v1

Isaac camera/state
→ upstream GR00T checkpoint-30000
→ typed 40×78 CloudWalk action
→ SONIC body + seçilen hand adapter
→ Isaac G1
→ bottle görevi
```

Bu sırada `humanoid graph` aynı gerçek bağlantıları, `humanoid status` ise
component health/rate/age durumlarını gösterir.

### Legacy silme koşulu

Legacy kod yalnız şu koşullarda kaldırılır:

1. Yeni karşılık aynı veya daha güçlü contract testlerine sahiptir.
2. İlgili yeni kapı Raider'da geçmiştir.
3. Eski yoldan kalan tekil davranış envanteri tamamlanmıştır.
4. Gerekli provenance/config/evidence yeni yapıya taşınmıştır.
5. Yeni core legacy modülleri import etmemektedir.

Kapı 4 teknik E2E iki kez geçmeden toplu legacy silme yapılmaz.

---

## 9. Önerilen Repo Yapısı

```text
humanoid-lab/
├── configs/
│   ├── robots/
│   ├── hands/
│   ├── cameras/
│   ├── controllers/
│   ├── policies/
│   ├── scenes/
│   └── profiles/
├── schemas/
│   └── humanoid.proto
├── src/humanoid_lab/
│   ├── contracts/
│   ├── runtime/
│   ├── simulators/
│   │   ├── base.py
│   │   ├── isaac/
│   │   └── mujoco/
│   ├── controllers/
│   │   └── sonic/
│   ├── policies/
│   │   └── groot/
│   ├── hands/
│   │   ├── inspire_ftp.py
│   │   └── dex3.py
│   ├── scheduling/
│   ├── transports/
│   ├── evidence/
│   └── cli.py
├── native/
│   └── sonic_service/
├── model-packages/
│   └── manifests/
├── tests/
│   ├── contracts/
│   ├── simulator/
│   ├── controller/
│   ├── policy/
│   └── e2e/
└── legacy/
```

`scripts/` yalnız ince operasyon wrapper'ları içerebilir; business logic büyük
`run-*.py` dosyalarında tutulmaz.

---

## 10. Doğrulama ve Evidence Standardı

Her Raider koşusu self-contained bir run dizini üretir:

```text
outputs/runs/<run-id>/
├── manifest.yaml
├── graph.mmd veya graph.svg
├── component-status.jsonl
├── simulator-metrics.jsonl
├── policy-metrics.jsonl
├── controller-metrics.jsonl
├── task-metrics.json
├── model-provenance.json
├── asset-provenance.json
├── head-camera.mp4
├── debug-viewport.mp4
└── summary.json
```

Her frame/tick için gerekli minimum korelasyon:

```text
run_id
episode_id
physics_tick
simulated_time
root pose
timeline state
camera frame hash
changed pixels from previous
latest policy sequence/chunk index
latest SONIC command sequence/age
control mode
```

Recorder düzeltme kabulü:

- Replicator verisi daha sonra kullanılacaksa açık kopya alınır;
- tercihen frame alındığı anda streaming writer'a yazılır;
- unbounded frame listesi yoktur;
- recording on/off aynı seed/config ile tensor trajectory karşılaştırılır.

Kapı raporu `PASS`, `FAIL`, `BLOCKED` ayrımı yapar. Eksik GPU/model/asset
`PASS` olamaz. Her failure ilk bozuk katmanı belirtir:

```text
asset
simulator physics
render/head camera
transport
SONIC state/decoder/command
GR00T observation/inference/action
hand adapter
scheduler
task physics
lifecycle/cleanup
```

---

## 11. Uygulama ve Migrasyon Stratejisi

1. Mevcut branch ve archive ref'leri korunur; başlangıç manifest'i alınır.
2. Yeni core, legacy runner'ları import etmeden ayrı dizinde kurulur.
3. Kapı 1 yalnız resmî Isaac demo ve doğrulanmış asset/config parçalarından
   oluşturulur.
4. Her kapıda önce saf contract testleri, sonra process integration, son olarak
   Raider E2E çalıştırılır.
5. Legacy'deki joint mapping, camera calibration, provenance ve acceptance
   bilgileri davranış bazında taşınır; büyük dosyalar kopyalanmaz.
6. Yeni karşılık kanıtlanınca ilgili legacy yol kaldırılır.
7. Her kapı sonunda repo lint/test, upstream cleanliness, process/listener ve
   evidence audit yapılır.
8. Fiziksel G1 yolu bu dört kapının simulator sonuçlarından ayrı bir güvenlik
   ve kabul planına tabi tutulur.

---

## 12. Nihai Başarı Tanımı

Plan tamamlandığında kullanıcı şunları yapabilmelidir:

```bash
# Isaac + Inspire camera; controller yok, robot düşer
humanoid run isaac-g1-inspire-passive

# Isaac + Dex3 camera; controller yok, robot düşer
humanoid run isaac-g1-dex3-passive

# Isaac + SONIC; doğrulanmış latent kaynağı
humanoid run isaac-g1-sonic

# Isaac + SONIC; gerçek terminal/TTY keyboard teleop
humanoid run isaac-g1-sonic-keyboard

# Model paketi seçilmiş generic GR00T + SONIC + Isaac
humanoid run groot-sonic-isaac --model-package <package>

# CloudWalk final profili
humanoid run cloudwalk-bottle-v1

# Gerçek çalışan ağı ve durumları gör
humanoid graph cloudwalk-bottle-v1
humanoid status
```

Bir bileşeni değiştirmek diğerlerinin kodunu değiştirmemelidir:

```text
Isaac → MuJoCo          : simulator provider değişir
Inspire → Dex3          : hand spec/adapter ve uyumlu model package değişir
CloudWalk → başka task  : scene/prompt/model package/metrics değişir
GR00T fine-tune A → B   : model package değişir
SONIC → başka controller: controller provider değişir
```

Son mimari tek cümlede şöyledir:

> Simulator state ve kamera üretir, policy niyet/action üretir, SONIC body
> komutu üretir, hand adapter seçilen eli sürer, orchestrator yalnızca uyumlu
> bileşenleri bağlayıp lifecycle'ı yönetir; CloudWalk ise bu parçaları seçen ve
> görev başarısını ölçen declarative bir profildir.
