# Implementation planı

## Net istek

Mevcut çalışan **Isaac + SONIC Y** sistemine, eğittiğimiz **Ψ₀ checkpoint’ini** bağlayacağız.

Kullanıcı minimal bir web arayüzünden canonical BlockStacking promptunu kullanarak policy’yi başlatacak, durduracak ve simülasyonu resetleyecek. Kullanıcı:

- policy’nin gördüğü head-camera görüntüsünü UI’da,
- simülasyonun tamamını mevcut Isaac WebRTC üzerinden

canlı izleyecek.

İlk sürüm yalnızca şu taskı destekleyecek:

> **“Stack the three cubic blocks on the black tape in the order red, yellow, blue.”**

Sahnede yalnızca:

- mevcut G1 + Dex3,
- uygun yükseklikte masa,
- kırmızı, sarı ve mavi üç küp,
- siyah hedef alan

olacak.

Otomatik başarı değerlendirmesi, reward, rollout skoru, paraphrase promptlar ve diğer 12 task **bu çalışmanın kapsamında değil**.

---

# Sabit kararlar

## Kullanılacak bileşenler

| Katman | Karar |
|---|---|
| Ψ₀ inference | Psi0’nun mevcut policy server’ı ve bizim checkpoint |
| Whole-body controller | Mevcut çalışan SONIC Y |
| State kaynağı | SONIC `g1_debug`, ZMQ `:5557` |
| Action hedefi | SONIC Protocol v4 `pose`, ZMQ `:5556` |
| Kamera | Mevcut Isaac `head_camera` |
| Kamera transport | Gerçek robot kamerasına benzeyen ZMQ endpoint, `:5558` |
| Simülasyon | Mevcut `SimulatorService` ve DDS domain 42 |
| Dünya assetleri | Unitree worktree’den yalnız masa/küp/sahne parçaları |
| İzleme | Head-camera UI preview + mevcut Isaac WebRTC |
| Prompt | Yalnız canonical BlockStacking promptu |

## Kullanılmayacak bileşenler

- Ψ₀ ekibinin SONIC X controller’ı.
- Unitree’nin controller veya RL environment döngüsü.
- Isaac’e doğrudan Ψ₀ joint action gönderimi.
- Otomatik task başarı değerlendirmesi.
- Reward/termination sistemi.
- Birden fazla task veya prompt paraphrase desteği.
- Yeni video streaming sistemi; mevcut WebRTC korunacak.
- Gereksiz framework veya genel amaçlı plugin mimarisi.

---

# Hedef mimari

```text
Minimal Web UI
  ├─ canonical prompt
  ├─ Start / Stop / Reset
  ├─ head-camera preview
  └─ Isaac WebRTC bağlantısı
             │
             ▼
     Minimal session service
             │
             ▼
       Ψ₀–SONIC client
        ├─ RGB   ← camera service :5558
        ├─ state ← g1_debug :5557
        ├─ prompt
        └─ action → pose :5556
             │
             ▼
       Mevcut SONIC Y
             │
       Unitree DDS domain 42
             │
             ▼
       Mevcut Isaac sistemi
        ├─ G1 + Dex3
        ├─ head_camera
        ├─ masa
        └─ üç küp + siyah hedef
```

---

# Uygulama aşamaları

## 1. Minimal BlockStacking sahnesini mevcut Isaac sistemine ekle

Unitree worktree’den yalnız gerekli dünya parçalarını taşı:

- masa/world asseti,
- kırmızı, sarı ve mavi küp,
- gerekiyorsa basit oda/zemin asseti,
- başlangıç pose’ları.

Bunları mevcut `SimulatorService` sahne oluşturma yoluna profile üzerinden bağla. Unitree’nin:

- controller’ını,
- `ManagerBasedRLEnv` yapısını,
- action manager’ını,
- physics loop’unu

taşıma.

Yeni profile yalnız BlockStacking sahnesini tanımlasın. Örneğin:

```text
configs/profiles/isaac-g1-sonic-blockstacking-dex3.json
```

Profile en azından şunları içersin:

```text
mevcut SONIC controller ayarları
mevcut G1/Dex3 robot asseti
mevcut head-camera ayarları
masa asseti ve pose’u
üç küpün asset/pose/material bilgisi
siyah hedef alan
```

### Masa yüksekliği

Rastgele bir masa yüksekliği girme.

1. Robotu mevcut `sonic_standing` pose’unda başlat.
2. Robotun settle olmasını bekle.
3. Palm linklerinin dünya koordinatındaki yüksekliğini oku.
4. Masa yüzeyini, robotun gövdesini öne eğmeden küplere erişebileceği seviyeye yerleştir.
5. Üç küpün head-camera görüntüsünde açıkça göründüğünü doğrula.
6. Masa yüksekliğini profile içinde açık bir değer olarak sakla.

Unitree sahnesindeki masa yüksekliği başlangıç değeri olabilir fakat nihai değer mevcut G1/Dex3 asseti ve `sonic_standing` pose’u üzerinde doğrulanmalı.

---

## 2. Isaac head-camera’yı policy endpoint’i olarak yayınla

Mevcut kamera tanımını değiştirme:

```text
head_camera
640×480
RGB
torso_link’e bağlı
mevcut intrinsics/extrinsics
```

Policy çalışırken kameranın UI ve WebRTC’den bağımsız olarak oluşturulmasını sağla. Bunun için mevcut kamera scene seçimini genel bir “camera enabled” koşuluna bağla; policy kamerasını acceptance-test veya viewport flag’ine bağımlı bırakma.

Isaac fizik/render döngüsü içinde en son RGB frame’i thread-safe bir tamponda tut. Ayrı, küçük bir ZMQ kamera service’i:

```text
tcp://*:5558
```

üzerinden gerçek robot kamera protokolüne benzer biçimde cevap versin:

```text
REQ: b"get_frame"
REP: multipart response
     ├─ JPEG RGB frame
     ├─ kullanılmayan IR placeholder
     └─ kullanılmayan depth placeholder
```

Policy client yalnız ilk parçayı kullanacak.

Kamera service’i:

- Isaac API’sine kendi thread’inden doğrudan erişmemeli,
- simülasyon döngüsünün ürettiği latest-frame tamponunu okumalı,
- eski frame’in yaşı görülebilir olmalı,
- UI ile aynı frame kaynağını kullanmalı.

### Görüntü dönüşümü

Client’a 640×480 görüntüyü ver. Client içinde ayrıca model boyutuna resize etme.

Eğitim yolu:

```text
observation.images.cam_left_high
    ↓
observation.images.egocentric
    ↓
640×480 RGB
    ↓
Ψ₀ transform: 240×320 resize
    ↓
Qwen3-VL processor
```

Deployment server’ın checkpoint config’inden aynı `240×320` transform’u yüklediğini doğrula. Preprocessing’i kamera service’inde veya client’ta tekrar etme.

---

## 3. İnce bir Ψ₀–SONIC Dex3 client yaz

SONIC X controller’ını veya submodule kodunu runtime bağımlılığı haline getirme.

Client için mevcut kodları referans ve yapı taşı olarak kullan:

- `third_party/Psi0/src/psi/deploy/serve_psi0_sonic.py`
- `third_party/Psi0/src/psi/deploy/mock_psi0_client_rtc.py`
- `third_party/Psi0/src/psi/deploy/helpers.py`
- mevcut SONIC `g1_debug` okuyucuları,
- `src/humanoid_lab/datasets/sonic/protocol_v4.py`
- `scripts/replay-sonic-latent.py`
- `scripts/replay-sonic-reference.py`

Client şu dört işi yapsın:

### A. Observation toplama

```text
RGB:
camera service :5558
key: observation.images.egocentric

State:
SONIC g1_debug :5557
trainingde beklenen state history düzeni

Instruction:
canonical BlockStacking prompt
```

State’i doğrudan Isaac’ten alma. SONIC’in yayınladığı `g1_debug` kaynağını kullan.

### B. Ψ₀ server’a gönderme

Mevcut Ψ₀ policy server’a WebSocket üzerinden bağlan:

```text
default: ws://localhost:8014
```

Checkpoint’in beklediği:

- image key,
- state şekli,
- history uzunluğu,
- action chunk boyutu

server metadata/config üzerinden doğrulansın. Bunlar mümkün olduğu yerde hardcode edilmesin.

### C. Action sözleşmesini açıkça dönüştürme

Checkpoint 80 boyutlu action üretiyorsa:

```text
0:64   SONIC token
64:71  left Dex3
71:78  right Dex3
78:80  neck/no-op
```

Client:

1. Action genişliğinin yalnız `78` veya `80` olduğunu kabul etsin.
2. `80` ise son iki neck boyutunun beklenen sıfır/no-op toleransında olduğunu doğrulasın.
3. Son iki boyut anlamlı hareket içeriyorsa yayın yapmayı durdursun.
4. Token’ı SONIC’in FSQ grid’ine uygun quantize etsin.
5. Token ve iki el action’ını named-field Protocol v4 mesajına dönüştürsün.
6. Mesajı `pose` topic’iyle `:5556` üzerinden SONIC Y’ye göndersin.

Sessiz `action[:78]` kesmesi yapılmasın.

### D. Güvenli çalışma durumu

Client’ın yalnız şu basit durumları olması yeterli:

```text
IDLE
RUNNING
STOPPED
ERROR
```

- `Start`: state ve kamera hazırsa action stream’ini başlatır.
- `Stop`: yeni action göndermeyi durdurur.
- `Reset`: önce stream’i durdurur, sonra simulator reset ister ve history’yi temizler.
- Bağlantı veya schema hatasında eski action’ı sonsuza kadar tekrar yayınlamaz.

Genel amaçlı workflow engine veya karmaşık task state machine yazma.

---

## 4. Minimal UI ve tek başlatma komutu ekle

Küçük bir local web service/UI yeterli.

UI’da yalnız şunlar olsun:

- task adı: `BlockStacking`,
- canonical prompt için text input,
- input canonical prompt ile hazır gelsin,
- backend yalnız exact canonical promptu kabul etsin,
- `Start`,
- `Stop`,
- `Reset`,
- Ψ₀ server bağlantı durumu,
- SONIC `g1_debug` bağlantı durumu,
- camera bağlantı durumu,
- son action zamanı,
- policy’nin gördüğü head-camera görüntüsü,
- mevcut Isaac WebRTC görünümünü açan bağlantı veya basit embed.

Otomatik başarı/başarısızlık göstergesi ekleme.

Kullanıcı sistemi tek komutla başlatabilsin. Örneğin:

```bash
./dev.sh psi0-isaac-eval \
  --checkpoint-dir <run-directory> \
  --checkpoint-step 40000
```

Bu launcher mevcut repo yaklaşımına uygun biçimde ayrı süreçleri başlatsın:

```text
Isaac
SONIC Y
Ψ₀ policy server
Ψ₀–SONIC client/session service
UI
```

Mevcut `dev.sh` komutlarını mümkün olduğunca tekrar kullansın; mevcut Isaac veya SONIC başlatma yollarını kopyalamasın.

---

# Makul test kapsamı

Test sayısını düşük tut. Yalnız hata yapma ihtimali yüksek sınırları test et.

## Küçük otomatik testler

### 1. Action adapter testi

Şunları doğrula:

- 78D action doğru şekilde 64+7+7 ayrılıyor.
- 80D action’da sıfır neck kabul ediliyor.
- Anlamlı neck action reddediliyor.
- Yanlış action genişliği reddediliyor.
- Protocol v4 mesajındaki field isimleri ve şekilleri doğru.

### 2. Kamera endpoint testi

Sentetik bir RGB frame kullanarak:

- `get_frame` isteğinin cevaplandığını,
- JPEG’in decode edilebildiğini,
- sonucun 640×480 HWC RGB olduğunu

doğrula.

Isaac’i bu birim testinde başlatma.

### 3. Scene/profile yükleme testi

BlockStacking profile’ının:

- mevcut SONIC controller config’ini,
- head-camera’yı,
- masa ve üç küp tanımını

başarıyla yüklediğini doğrula.

Bunun dışında kapsamlı mock testleri veya genel amaçlı evaluator testleri yazma.

---

# Manuel entegrasyon doğrulaması

Implementation tamamlandığında gerçek stack bir kez uçtan uca çalıştırılmalı:

1. Isaac BlockStacking sahnesi açılır.
2. Robot `sonic_standing` pose’unda dengede kalır.
3. Masa robotu eğilmeye zorlamayan yüksekliktedir.
4. Üç küp ve siyah hedef head-camera’da görünür.
5. SONIC Y `g1_debug` yayınlar.
6. Kamera service’i canlı frame verir.
7. Ψ₀ server checkpoint’i yükler.
8. UI canonical prompt ile açılır.
9. `Start` tıklanınca policy action üretir.
10. SONIC Y action’ı alır ve Isaac’e body/Dex3 komutu gönderir.
11. `Stop` action stream’ini keser.
12. `Reset` robotu ve üç küpü güvenli başlangıç durumuna getirir.
13. Mevcut Isaac WebRTC görünümü çalışmaya devam eder.

UI browser’da:

- Start/Stop/Reset gerçekten tıklanarak,
- head-camera frame’inin güncellendiği görülerek,
- desktop ve dar ekran boyutunda temel kullanılabilirlik kontrol edilerek

doğrulanmalı.

Taskın başarılı biçimde tamamlanması bu implementation’ın kabul kriteri değil. Bu aşamanın kriteri, policy’nin kapalı çevrimde temiz şekilde SONIC üzerinden simülasyonu kontrol etmesidir. Davranış kalitesini kullanıcı manuel değerlendirecek.

---

# Implementation agent’a sınırlar

Agent şu sınırlara uymalı:

1. **SONIC Y’yi değiştirme veya yeniden build etme.**
2. **SONIC X controller’ını sisteme ekleme.**
3. Mevcut Isaac–SONIC DDS bağlantısını değiştirme.
4. Unitree worktree’den yalnız gerekli scene/asset kodunu al.
5. Başarı evaluator’ı, reward veya task-stage sistemi yazma.
6. BlockStacking dışında task ekleme.
7. Canonical prompt dışında prompt çeşitlendirmesi yapma.
8. Yeni video streaming altyapısı yazma; mevcut WebRTC’yi kullan.
9. Ψ₀’yı Isaac fizik process’ine import etme.
10. Var olan launcher, protocol ve deployment kodlarını kopyalamak yerine tekrar kullan.
11. Mevcut kullanıcı değişikliklerine dokunma veya onları temizleme.
12. Çalışmayan/varsayımsal fallback yolları ekleme; gereken tek akışı tamamla.

## Tamamlanmış sayılması için

Aşağıdaki tek zincir çalışmalı:

```text
UI canonical prompt
  → Ψ₀ checkpoint inference
  → 80/78 action doğrulama
  → Protocol v4
  → mevcut SONIC Y
  → DDS domain 42
  → Isaac G1 + Dex3
```

ve kullanıcı hem policy kamerasını hem mevcut WebRTC simülasyonunu canlı görebilmeli. . use subagents: deepseek. At the end, for review and cleannes, run a gpt sol reviewer.