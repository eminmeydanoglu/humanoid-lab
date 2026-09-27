# Flux 3 / Dex3 — simülasyonda çalıştırma

Komutları depo kökünden çalıştırın. Varsayılan sahne **PickApple**, prompt
`Put the apple into the plate.`; robot 29 gövde + 14 Dex3 eklemiyle sabit
kök üzerinde çalışır. `flux-sim` Isaac Sim'i, ROS 2 kamera köprüsü ve
`flux_dex3` düğümünü, ayrıca GPU model sunucusunu başlatır. SONIC controller
çalıştırılmaz.

## 1. Hazırlık (makine/checkout başına bir kez)

Mevcut kurulumda aşağıdaki kontroller yeterlidir:

```sh
./dev.sh flux-model-server --check    # model ortamı, hazırlanmış adapter ve base model
./dev.sh flux-sim config              # profil, motor ayarı, portlar ve görüntü modu
```

Yeni kurulumda, eksik olanları sırayla hazırlayın:

```sh
docker compose --profile flux build flux-ros
./dev.sh flux-ros-build
./dev.sh flux-model-env --plan
./dev.sh flux-model-env              # indirip kalıcı GPU ortamını kurar
./dev.sh flux-checkpoint --source /path/to/trained/checkpoint \
  --base-model-dir /path/to/base-policy-export
./dev.sh flux-model-server --check
```

Eğitilmiş adapter ve base policy **harici dosyalardır**; otomatik indirilmezler.
`flux-checkpoint` adapter'ın salt-okunur kopyasını varsayılan olarak
`data/models/flux-dex3/checkpoint-2500` altına hazırlar. Base modeli kopyalamaz;
ona işaret eden yolu hazırlanan adapter içinde günceller. `--check` geçmiyorsa
`./dev.sh flux-model-server --plan` ile seçilen Python ve checkpoint yoluna bakın.
Hazır GPU sunucusu `./dev.sh flux-model-server --status` ile `READY` raporlar.

## 2. Simülasyonu aç, prompt ver, görevi çalıştır

Yeni bir **etkileşimli** oturum için her koşuda yeni bir `--tag` kullanın:

```sh
./dev.sh flux-sim up --tag pickapple-01 --duration 600 --gui
# Ayrı terminalde:
./dev.sh flux-sim status
./dev.sh flux-sim task --prompt "Put the apple into the plate." --run-s 15
```

`up` başlangıç kontrolünden sonra **aynı terminalde Isaac, ROS düğümü ve model
sunucusunun loglarını canlı gösterir**. Ctrl-C yalnız log takibinden çıkar;
çalışan simülasyonu kapatmak için `flux-sim stop` kullanın. Başka terminalde
sonradan loglara bağlanmak için `./dev.sh flux-sim logs`; log takibi olmadan
başlatmak için `./dev.sh flux-sim up --detach --tag pickapple-01 --duration 600`.
`task` ROS StartTask servisine prompt'u gönderir,
`--run-s` saniye boyunca komutları izler, ardından StopTask çağırır ve rapor
üretir. Varsayılan görev süresi **8 saniyedir**. `--duration` ise Isaac
simülasyonunun toplam süresidir (varsayılan **180 saniye**); görev süresinden
uzun seçin. `status` model, kamera ve eklem akışlarının sağlığını kontrol eder.

Prompt, `flux_dex3` düğümünün eğitim manifestindeki yazıyla **birebir aynı**
olmalıdır; rastgele metin reddedilir. PickGum için sahneyi ve prompt'u birlikte
değiştirin:

```sh
./dev.sh flux-sim up --tag pickgum-01 --duration 600 --gui \
  --profile configs/profiles/isaac-g1-flux-dex3-pickgum.json
./dev.sh flux-sim task --prompt "Put the gum into the plate." --run-s 15
```

Geçerli prompt'ların tamamı
`third_party/flux/flux-inference/ros2/flux_dex3/flux_dex3/node.py` içindeki
`PROMPTS` listesinde. Mevcut oturumun ayarları `data/outputs/flux-dex3/latest`
ile işaret edilen etikette saklanır; `status`, `task`, `verify` ve `stop` bu
oturumu seçer. Eski oturuma dönmek için `--tag ETIKET` verin. `up` etiketsiz
çağrılırsa `latest` oturumunu yeniden kullanır; yeni sonuçlar için yeni etiket
seçin. Aynı anda ikinci bir Isaac G1 koşusu başlatmayın.

Tek komutla otomatik görev + doğrulama istiyorsanız:

```sh
./dev.sh flux-sim e2e --prompt "Put the apple into the plate."
# PickGum alternatifi:
./dev.sh flux-sim e2e --profile configs/profiles/isaac-g1-flux-dex3-pickgum.json \
  --prompt "Put the gum into the plate." --duration 300 --run-s 15
```

`e2e` yeni etiket üretir, görevi çalıştırıp durdurur, Isaac'in toplam süresinin
bitmesini bekler, hareket kaydını doğrular ve oturumu kapatır. Devam ederken
logları ayrı terminalde `./dev.sh flux-sim logs` ile izleyin. Mevcut Isaac
koşusu varsa önce onun bitmesini bekleyin veya aşağıdaki `stop` ile kapatın.

## 3. ROS servisleriyle doğrudan kullanım

Önce `./dev.sh flux-sim up --tag YENI_ETIKET --duration 600` çalıştırıp
`./dev.sh flux-sim status` sonucunu kontrol edin. Başka terminalde, simülasyon
ve ROS düğümü çalışırken:

```sh
./dev.sh flux-ros ros2 service call /flux_dex3/get_status \
  flux_dex3_interfaces/srv/GetStatus '{}'
./dev.sh flux-ros ros2 service call /flux_dex3/start_task \
  flux_dex3_interfaces/srv/StartTask '{prompt: "Put the apple into the plate."}'
# Komutlar StopTask çağrılana kadar sürer; otomatik süre sınırı yoktur.
./dev.sh flux-ros ros2 service call /flux_dex3/stop_task \
  std_srvs/srv/Trigger '{}'
```

`start_task` yanıtında `accepted: true` yalnız başlangıç isteğinin kabul
edildiğini gösterir. Sonrasında `get_status` yanıtında `state: RUNNING` ve dolu
`session_id` görmelisiniz. `READY` ile boş `session_id` gelirse görev kendini
durdurmuştur; aşağıdaki hata ayıklama bölümündeki `ros-launch.log` dosyasına
bakın. `accepted: false` ise `reason` alanına bakın: model `READY` olmalı,
kamera/eklem ölçümleri güncel olmalı ve prompt eğitim manifestinde bulunmalı.
Görev sürerken yeni bir prompt göndermeden önce `stop_task` çağırın. ROS
servislerini doğrudan kullanırken `flux-sim task` komutunun süre sınırlı
raporlaması ve otomatik StopTask davranışı devrede değildir.

## 4. İzleme, kayıt ve kapatma

Ayrı terminalde RViz'i açın:

```sh
./dev.sh flux-rviz
```

RViz robot eklemlerini ve kafa kamerasını gösterir. Isaac'in görünümü
varsayılan olarak **WebRTC** ile yayınlanır (bağlantı adresi ve varsayılan
49100 portu koşunun `isaac.log` dosyasında); kurulu istemci için
`./dev.sh webrtc-client`. `--gui` yerel Isaac penceresini, `--headless` Isaac
penceresi olmadan koşuyu seçer. RViz her durumda ayrı yerel X11 penceresidir.

Normal süre sonunda Isaac `tracking.parquet` ve `summary.json` dosyalarını
yazar. Kayıt doğrulaması ve kapatma:

```sh
./dev.sh flux-sim verify
./dev.sh flux-sim stop --keep-model-server
```

`verify` için simülasyonun `--duration` süresini **normal biçimde** bitirmesini
bekleyin; erken `stop` bu dosyaları yazdırmayabilir. `--keep-model-server`
yüklenmiş GPU modelini korur; düz `./dev.sh flux-sim stop` modeli de kapatır.
Çıktılar `data/outputs/flux-dex3/<tag>/` altındadır: `isaac.log`, `ros-launch.log`,
`preflight.json`, `task-report.json` (yalnız `flux-sim task`), `tracking.parquet`,
`summary.json`, `tracking-report.json` (doğrulamadan sonra).

### StartTask kabul edildiği hâlde robot hareket etmiyorsa

```sh
tail -n 80 "data/outputs/flux-dex3/$(cat data/outputs/flux-dex3/latest)/ros-launch.log"
./dev.sh flux-ros ros2 service call /flux_dex3/get_status \
  flux_dex3_interfaces/srv/GetStatus '{}'
```

`prediction rejected: stale observation` çıktısı, tahmin döndüğünde kamera
ölçümünün simülasyon için **1,6 saniyelik** yaş sınırını aştığını ve motor
komutlarının kesildiğini gösterir. Robot düğümünün varsayılan sınırı 1,2 saniye
olarak kalır. Model `READY` kalsa da görev artık çalışmaz. GPU yükünü
azaltıp WebRTC/yerel Isaac penceresi olmadan yeniden denemek için mevcut koşu
işiniz bittikten sonra:

```sh
./dev.sh flux-sim stop --keep-model-server
./dev.sh flux-sim up --tag pickapple-headless-01 --duration 600 --headless
./dev.sh flux-sim task --prompt "Put the apple into the plate." --run-s 15
```

Gerekirse RViz penceresini de kapatıp deneyin; bu adım gecikmeyi azaltmayı
hedefler, sonucu `task` raporu ve `ros-launch.log` ile doğrulayın. Yaş sınırını
artırmak gecikmiş robot komutlarını kabul edeceği için sırf hareket elde etmek
amacıyla değiştirmeyin.

### RViz kamerası görünmüyorsa

Kamera köprüsü `/camera/color/image_raw` ile aynı zaman damgalı
`/camera/color/camera_info` yayımlar; RViz de `torso_link -> head_camera`
dönüşümünü ekler. Eski ROS/RViz süreçleri kod değişikliklerini otomatik
almaz. Aktif görevi bitirip eski RViz penceresini Ctrl-C ile kapattıktan sonra:

```sh
./dev.sh flux-ros-build
./dev.sh flux-sim stop --keep-model-server
./dev.sh flux-sim up --tag pickapple-02 --duration 600 --gui
./dev.sh flux-rviz                 # RViz'i de yeniden açın
```

Kamera çerçevesini kontrol etmek için
`./dev.sh flux-ros ros2 run tf2_ros tf2_echo world head_camera` kullanın
(Ctrl-C ile çıkılır). RViz **Camera** paneli 3B geometriyi görüntünün üzerine
çizebilir; yalnız ham kamera pikselleri için **Image** panelini kullanın.

Ek seçenekler için `./dev.sh flux-sim help`; motor komutları kapalı prova için
`./dev.sh flux-sim e2e --no-motor-commands --headless`. DDS bağlantı testi
`./dev.sh flux-dds-probe`, kamera köprüsü testleri `./dev.sh flux-ros-tests`.
