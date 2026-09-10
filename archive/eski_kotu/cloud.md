Aşağıdaki hali ajana direkt verebilirsin. Önceki planın ana mimarisini korudum; sadece konuştuğumuz değişiklikleri işledim ve gereksiz ayrıntıları çıkardım.

## CloudWalk v10 GR00T N1.7 + SONIC → Isaac Closed-Loop Execution Plan

### Hedef

İlk teslim, Raider üzerinde Isaac Sim’de şu hattın gerçekten çalışmasıdır:

```text
Isaac scene
  G1 + Inspire FTP
  head RGB camera
  robot state
  table + bottle
       ↓
CloudWalk GR00T N1.7
checkpoint-30000
       ↓
40 × 78 action chunk
       ↓
upstream VLA client
       ↓
SONIC C++ deploy
       ↓
whole-body + Inspire hand commands
       ↓
Isaac G1
       ↺
```

Sabit artifact’ler:

```text
Dataset:
cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v10

Model:
cloudwalk-research/gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune

Checkpoint:
checkpoint-30000

Embodiment:
UNITREE_G1_SONIC

Prompt:
grab the bottle

Action:
40 × (64 motion_token + 7 left_hand + 7 right_hand)

SONIC/action rate:
50 Hz

GR00T inference:
2.5 Hz
```

Bu iterasyonda **Dex3 kullanılmayacak**. CloudWalk reproduction kendi Inspire FTP embodiment’ıyla yapılacak. Mevcut plan da hedefi aynı şekilde v10 checkpoint-30000 + `UNITREE_G1_SONIC` olarak sabitliyor.

### Ana mimari kuralı

GR00T, upstream VLA client ve upstream SONIC C++ mümkün olduğunca **değiştirilmeden** kullanılacak.

Bizim yazacağımız ana şey:

> **Isaac ↔ mevcut SONIC/VLA stack adapter’ı.**

Isaac için ayrı SONIC implementasyonu, özel policy veya farklı action semantics yazılmayacak. Mevcut planın en önemli mimari kararı da bu.

MuJoCo parity, generic simulator framework, multi-backend abstraction, gerçek robot deployment ve multi-task desteği **Isaac bottle task çalışana kadar yapılmayacak.**

---

## Çalışma şekli

Execution başlar başlamaz **5 worker subagent paralel launch edilmeli**.

**Tüm worker subagent modellerini `GPT-5.6 Terra`, thinking effort `High` olarak seç.**

Ana agent subagent sonuçlarını beklerken kendi entegrasyon işlerine devam etmeli. Subagent launch’ları işin başında gerçekten yapılmalı; yalnızca “delegate edeceğim” denmemeli.

Worker görevleri:

1. **GR00T worker**
   Checkpoint-30000, processor, statistics, embodiment ve exact I/O contract’ını doğrula. Datasetten gerçek observation replay ederek `[40,78]` output al. Prompt’ın gerçekten modele ulaştığını doğrula.

2. **SONIC worker**
   Doğru SONIC controller/config’i doğrula. Protocol v4, gerekli robot state’leri, 64D token + hand action → whole-body command zincirini test et.

3. **Inspire FTP worker**
   Gerçek asset, URDF/MJCF/CAD/mesh, joint ordering, limits, coupling ve `InspireFTPGripMapper` mapping’ini bul ve doğrula. Dex3 kullanma.

4. **Isaac worker**
   G1 + Inspire FTP articulation, actuator interface, head camera ve minimal masa+bottle scene’i hazırla. Isaac state read / SONIC command apply yolunu prototiple.

5. **CloudWalk dataset worker**
   Dataset videolarını inceleyip robot–masa mesafesi, camera view/FOV, bottle placement, grasp yaklaşımı ve scene görünüşü için yaklaşık parametreler çıkar.

Bu dağılım mevcut planın beş paralel agent ayrımını koruyor.

### Progress update kuralı

Bana yalnızca anlamlı gelişmeleri şu formatta bildir:

```text
GELİŞME: ...
```

Örneğin:

```text
GELİŞME: CloudWalk checkpoint gerçek dataset observation'ıyla başarıyla çalıştı ve [40,78] action üretiyor. Sıradaki entegrasyon SONIC protocol hattı.
```

veya:

```text
GELİŞME: Inspire FTP için düşündüğümüz asset'in joint ordering'i dataset mapping'iyle uyuşmuyor. Hipotezi değiştirdim; şu kaynak artık ana aday.
```

**GELİŞME:** mesajları checkpoint değildir. Benden onay bekleme; çalışmaya devam et.

Routine command, dosya okuma, download, build adımı gibi düşük seviyeli şeyleri anlatma. Yalnızca:

* önemli bir şey doğrulandığında,
* hipotez değiştiğinde,
* ciddi blocker çıktığında,
* bir milestone geçildiğinde

bildir.

---

# Execution sırası

### Aşama 1 — GR00T modelini izole doğrula

CloudWalk datasetinden gerçek bir observation al:

```text
camera
robot state
"grab the bottle"
```

→ checkpoint-30000’a ver.

Beklenen gerçek output:

```text
[40,78]

64 motion token
7 left hand
7 right hand
```

Shape, dtype, finite values, latency ve prompt transport doğrulansın. Mevcut plan bunu GR00T-only replay aşaması olarak tanımlıyor.

Bu çalışmadan Isaac’e geçme.

---

### Aşama 2 — SONIC’i izole doğrula

Known-good state + known-good/canned action ile:

```text
VLA client
   ↓
SONIC C++
   ↓
G1 body + hand commands
```

hattını doğrula.

Burada önemli değişiklik:

> **Arbitrary `zeros(64)` motion token’ını “standing/hold” olarak kabul etme.**

64D latent uzayında sıfır token’ın hareketsiz duruş anlamına geldiğini varsayma.

Standing/hold testi gerekiyorsa şunlardan doğrulanmış birini kullan:

```text
SONIC'in kendi initial/standing mode'u
veya
datasetten doğrulanmış standing reference/token
veya
bilinen güvenli reference motion
```

SONIC’in start/pause/initial-pose/stop/watchdog davranışını da doğrula. Mevcut planın SONIC-only aşaması bunun temelini zaten içeriyor.

---

### Aşama 3 — Isaac + SONIC

Isaac’te:

```text
G1
+
Inspire FTP
```

ayağa kalksın.

SONIC gerekli robot state’lerini Isaac’ten alsın ve ürettiği commands doğru Isaac DOF’larına uygulansın.

Testler:

```text
initial pose
gravity settle
validated standing/hold reference
hand open
hand close
pause
timeout
reset
```

Robot policy olmadan SONIC altında stabil durabilmeli.

---

### Inspire FTP asset işi diğer çalışmaları bloklamasın

Exact Inspire FTP geometry/mapping araştırması paralel devam etsin.

Eğer henüz bulunmadıysa:

> Camera, PolicyServer, process lifecycle ve observation plumbing gibi **physics’e bağlı olmayan** entegrasyon işleri açıkça `PROXY` olarak işaretlenmiş geçici bir el geometrisiyle devam edebilir.

Ancak:

> **Exact Inspire asset + mapping doğrulanmadan grasp başarısı veya physics başarısı raporlanamaz.**

Dex3 hiçbir şekilde CloudWalk reproduction için fallback olarak kullanılmayacak.

Bu ayrım önemli: asset worker diğer dört worker’ı bekletmesin.

---

### Aşama 4 — Kamera ve tam VLA loop

Head camera’yı bağla.

Sonra:

```text
Isaac RGB + state
        ↓
GR00T
        ↓
40×78
        ↓
VLA client
        ↓
SONIC
        ↓
Isaac
        ↺
```

tam closed-loop zinciri çalışsın.

Henüz bottle başarısı beklenmiyor.

Bu milestone’da yalnız şunların kanıtlanması yeterli:

```text
camera/state doğru geliyor
prompt modele ulaşıyor
GR00T [40,78] üretiyor
VLA client action scheduling çalışıyor
SONIC valid commands üretiyor
Isaac doğru jointleri hareket ettiriyor
reset/stop tekrar çalışıyor
```

Mevcut plandaki “birinci teknik kabul” de tam olarak bu sınırı tanımlıyor.

---

### Aşama 5 — CloudWalk-benzeri scene

Isaac’e minimal bir training-like scene koy:

```text
G1

      bottle
        |
   ┌──────────┐
   │  table   │
   └──────────┘
```

Dataset videolarından yaklaşık:

```text
camera FOV/pitch
robot-table distance
table height
bottle position
bottle dimensions
lighting/background
```

çıkar.

Ama **digital twin yapmaya çalışma**.

İlk hedef yalnızca modelin eğitim görüntülerinden çok uzak olmayan bir observation üretmek.

Önce tek deterministic nominal scene kullan. Domain randomization daha sonra.

---

### Aşama 6 — Bottle closed-loop test

Sırayla:

```text
empty scene

bottle nominal pose

bottle hafif kaydırılmış

aynı nominal koşulda tekrarlar
```

çalıştır.

Başarıyı yalnızca:

> “şişeyi aldı / alamadı”

diye ölçme.

Ayrı kaydet:

```text
approach
contact
hand close
stable grasp
lift
```

Ve burada bir değişiklik daha:

> **Tek başarılı rollout milestone’u kapatmaya yetmesin.**

Nominal aynı condition altında **en az 5 rollout** çalıştır.

Örneğin:

```text
Nominal:
4 / 5 successful lift
```

şeklinde raporla.

Tek şanslı grasp “model çalışıyor” sonucu olarak kabul edilmesin.

Bunun yanında en az bir başarılı closed-loop task görmek yine ilk milestone’un temel gereksinimi. Mevcut plan zaten task başarısını approach/contact/hand-close/grasp/lift şeklinde bölüyor.

---

# Önemli yorumlama kuralları

Model yalnızca:

```text
"grab the bottle"
```

promptuyla eğitildi.

Dolayısıyla farklı promptlara farklı action çıkması ilginç bir testtir ama bu milestone’da **general language understanding** iddia etme.

Ayrıca empty scene’de bile grasp hareketi yapabilir. Böyle bir davranışı gizleme; ayrı raporla.

Bir şey çalışmazsa doğrudan modeli yeniden eğitmeye atlama. Önce failure’ı şu katmanlardan birine indir:

```text
camera / visual domain
GR00T
SONIC
Inspire mapping
Isaac actuator
contact physics
```

Mevcut planın risk kuralları da özellikle bunu istiyor.

---

## Bu planın bittiği nokta

İş ancak şu noktada ilk milestone’u tamamlamış sayılır:

> **CloudWalk checkpoint-30000, gerçek upstream GR00T + SONIC stack’i kullanarak Isaac Sim’de Inspire FTP’li G1’i closed-loop kontrol ediyor; `"grab the bottle"` nominal scene’de test edilmiş; en az 5 tekrarlı sonuç, video ve failure breakdown mevcut.**

Bundan **sonra** MuJoCo parity, generic simulator abstraction ve kendi **G1 + Dex3 + Fruits** training pipeline’ımız için ayrı plan açılacak.
