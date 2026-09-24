# VLA → Ethernet → Unitree SONIC: iş planı

## Hedef ve sınırlar

VLA çıkarımı Raider'da, SONIC ve robot kontrol döngüsü Unitree bilgisayarında çalışacak. Robot kamerası ile ölçülen durum Raider'a gelecek; seçilen modelin ürettiği SONIC v1.1 64 boyutlu latent ve 14 boyutlu Dex3 hedefi robota gidecek. Model sustuğunda, uygulama çöktüğünde veya bağlantı koptuğunda robot tarafı kontrollü biçimde ayakta duran `IDLE` durumuna geçmeli. Arayüz, kamerayı ve gerçek sistem durumunu gösterecek; Başlat, Durdur, Sıfırla, model ve veri kümesindeki 13 görev seçimini sağlayacak.

Bu çalışma kablo bağlanana kadar robotu hareket ettiren komut çalıştırmadan ilerler. Fiziksel güvenlik ve hareket davranışı yalnız gerçek robot üzerinde ölçülmüş koşuyla doğrulanmış sayılır.

## Şu an doğrulananlar

- Robotun `sonic` takma adı `policy/sonic_v1_1` + `--input-type gamepad`; `sonic-ll-zmq` ise `policy/low_latency` + `--input-type zmq` ve Raider'ın Tailscale adresini kullanıyor. İkisi aynı kontrol ikilisini ve planner'ı kullanıyor.
- Eğitim/dönüşüm sözleşmesi SONIC **v1.1** tokenlarına bağlı (`configs/datasets/sonic/sonic_v1_1.yaml` ve `configs/datasets/psi0/unitree_dex3_sonic_v1.yaml`). `low_latency` ayrı model/encoder/observation config'tir; token uyumu kanıtlanmadan alternatif seçilmez.
- NVIDIA VLA rehberi v1.1 için `zmq_manager` örneği veriyor. Bu giriş türü ağ üzerinden `command`, `planner`, `pose` konularını alır; `zmq` girişinin mod başlatma/geçişleri klavyeye dayanır.
- Robot kaynak kodunda planner komutları kesilince 1 saniye sonra `IDLE` yazılıyor. **Bu, dış latent akışının zaman aşımı değildir.** Dış token 200 ms kesildiğinde mevcut kod yalnız uyarıyor ve son tokenı kullanıyor. Bu güvenlik açığı kapatılmadan fiziksel VLA denemesi yapılmaz.
- Unitree `eth0` robotun `192.168.123.0/24` ağına bağlıdır. Raider'ın görünen fiziksel Ethernet arayüzü başka ağdadır. Robot kamera servisi inceleme sırasında etkin değildi.
- Raider'ın `enp129s0` arayüzü etkin kurumsal/default route'u ve `192.168.50.0/24` ek yolunu taşır; robota ayrılmamalı. Robotun `eth0` bağlantısı `192.168.123.164/24` ile etkin. Ayrı bir Raider Ethernet adaptörü ve robot switch'inde uygun bağlantı noktası fiziksel olarak doğrulanmalı.
- `composed_camera_server.service` robotta kurulu değil. Daha ayrıntılı aygıt incelemesi RealSense D435i (seri `401622070422`) buldu. `/dev/video2` 640×480 gri/IR akışıdır; tek başına `pyrealsense2` ile 640×480 RGB kare alındı. RGB kadrajı yalnız zemini görüyor. Eğitim girdisi `cam_left_high` olduğundan bu kamerayı eğitim görünümüyle eşdeğer varsayma. Sürekli RGB akışı sonraki denemede sensör zaman aşımı/aygıt meşgul hatası verdi; kamera hazır değil.
- Mevcut Isaac değerlendirme arayüzü kamera/Start/Stop/Reset/model seçimi sağlar, ancak Reset Isaac sahnesini sıfırlar. Veri kümesi yapılandırması 13 kanonik görev metni içerir; mevcut UI üç görev bilir.

## 1. Önce idle ve güvenlik sözleşmesi

- [x] `sonic_v1_1` + `zmq_manager` için kaynak kodundaki açılış, `PLANNER/IDLE` ve `STREAMED_MOTION` yollarını ayır. `stop=true` kontrolü sonlandırır; arayüzün normal Durdur işlemi planner moduna dönmelidir.
- [x] Robot kaynak kopyası için 300 ms geçerli Protocol v4 token zaman aşımı patch'i hazırla. İlk token hiç gelmezse veya akış durursa `zmq_manager` planner'a döner; geçersiz/alınmış ama çözümlenmemiş paketler saati yenilemez. Canlı robota uygulanmadı.
- [x] Saf zaman aşımı mantığının C++ testini ve robot mimarisinde kaynak dosyasının sözdizimi derlemesini çalıştır.
- [x] Gerçek `ZMQManager` sınıfını robot üzerinde donanım olmadan `127.0.0.1:16556` üzerinden sınayarak açık `planner=true` komutuyla Durdur geçişini ve bozuk `pose` paketleri sürerken geçerli token yoksa otomatik planner'a dönüşü doğrula (`tests/cpp/test_zmq_manager_idle.cpp`).
- [ ] Gerçek SONIC ikilisi ile donanımsız loopback/simülasyon koşusunda mod geçişini, eski el hedeflerinin temizlenmesini ve taze token zorunluluğunu ölç. Bu yapılmadan patch'i fiziksel robota kurma.
- [ ] Fiziksel denemeden önce kaydedilmiş tokenlarla yerel simülasyon/loopback koşusunda mode, token yaşı, hedef ve ölçülen hareket kanıtını üret. Gerçek robot için ayrıca gözetimli test kapısı bırak.

## 2. Kablolu ağ

- [x] Tailscale'in mevcut kalitesini ölç: 30/100/20 ICMP paketinde sırasıyla %16,7/%20/%25 kayıp görüldü. 17,7 KB'lık kaydedilmiş renk JPEG'i için 60 HTTP aktarımının tümü tamamlandı; ortanca gecikme 86,5 ms, P95 1290,2 ms, en kötü 1604,3 ms. Bu bağlantı mevcut hâliyle 30 Hz taze görüntü/kontrol varsayımını karşılamıyor. Ağ/erişim çalışıyor olması gerçek zamanlı yeterlilik kanıtı değildir.
- [ ] Robot switch'inde uygun portu ve Raider için ayrı Ethernet adaptörünü doğrula; `eth0` DDS robot ağını bozma.
- [ ] Kablo bağlandıktan sonra IP/route, gecikme, paket kaybı ve servis erişimini ölç. Tailscale'i yönetim/geri dönüş yolu olarak tut.
- [ ] Sabit Tailscale IP içeren takma ad yerine v1.1 + `zmq_manager` için açık, sürümlenmiş başlatma yapılandırması oluştur.

## 3. Kamera ve robot durumu

- [ ] D435i RGB akışını sürekli çalışır hâle getir; mevcut `scripts/robot-color-camera.py` yalnız renk akışını açıp HTTP üzerinden taze JPEG verir, fakat sensör denemesinde sürekli kare alınamadı. Servis bayat kareyi 503 ile reddeder. Kameranın fiziksel kadrajını eğitimdeki `cam_left_high` görünümüyle karşılaştır.
- [ ] SONIC `g1_debug` durumunu tek okuyucuyla al, eğitimdeki 43D gözlem sözleşmesine dönüştür; görüntü/durum yaşını sınırla ve senkronizasyonu kaydet.
- [ ] Kamera veya durum bayatsa modele yeni istek gönderme; robot tarafındaki idle geçişini doğrula.

## 4. Modelden bağımsız çıkarım

- [ ] Tek gözlem (`kamera + 43D durum + kanonik görev`) ve eylem (`64D latent + 14D Dex3`) sınırı tanımla; model kimliği, boyut, finite değer, zaman damgası ve sıra numarası doğrulansın.
- [ ] Mevcut Psi0 ve GR00T sunucularını ayrı adaptörler olarak bağla; model değişimini yalnız idle durumunda yap. Gelecek modeller aynı sözleşmeyi kullansın.
- [ ] Gecikme, action chunk süresi ve kontrol frekansını gerçek ölçümlerle ayarla; simülasyon saatine özgü varsayımları fiziksel robota taşıma.

## 5. Operatör arayüzü ve uçtan uca kapı

- [ ] Gerçek robot kamerasını, görüntü/durum/eylem yaşını, SONIC modunu ve hata nedenini göster.
- [ ] Başlat, Durdur → ayakta idle, Sıfırla → ayakta idle + oturum geçmişini temizle. Düğme yanıtı ile robotta gözlenen durum ayrı gösterilsin.
- [ ] Görev seçiciyi `configs/datasets/psi0/unitree_dex3_sonic_v1.yaml` içindeki 13 kesin metinden üret; model/checkpoint seçimini açık kimliklerle sınırla.
- [ ] Kademeli gerçek robot testi: latent yok, normal durdurma, modelin susması, uygulama çökmesi, kablo çıkarma, sonra tek model ve tek görev. Her testte SONIC hedefi, ölçülen robot tepkisi, son paket yaşı ve idle'a geçiş süresi kaydedilsin.

## Kodun yeri

Yeni gerçek robot oturumu `src/humanoid_lab/robot_runtime/` altında; çalıştırma girişi `scripts/` altında tutulmalı. Var olan `psi0_bridge` doğrulayıcıları, model adaptörleri ve SONIC protokol paketleyicileri yeniden kullanılmalı. Robot C++ değişikliği `patches/` altında ayrı tutulmalı. Isaac'e özel Reset ve UI davranışı gerçek robot oturumuna kopyalanmamalı.

## 24 Eylül 2026 doğrulama notu

`patches/sonic/0001-return-to-planner-idle-on-stale-v4-token.patch` ve `pose_watchdog.hpp` yalnız kaynak kopyasına uygulandı. `scripts/apply-sonic-idle-watchdog.sh --check` ve `--apply` geçici kopyada başarılı oldu. `tests/cpp/test_pose_watchdog.cpp` yerel `c++ -std=c++17 -Wall -Wextra -Werror` ile geçti. `tests/cpp/test_zmq_manager_idle.cpp` Unitree'de gerçek `ZMQManager` sınıfını donanımsız yerel ZMQ üzerinden çalıştırdı; stream açıldı, geçersiz `pose` paketleri gönderilirken zaman aşımıyla planner moduna döndü. Unitree'deki kaynakların SHA-256 değerleri yerel SONIC kaynağıyla eşleşti; patch'li `zmq_manager.hpp` ve tüm `g1_deploy_onnx_ref.cpp` kaynak dosyası robot mimarisinde `g++ -std=c++2a -fsyntax-only` ile geçti. Patch'li ikili Unitree'de `/tmp/vla-sonic-idle-buildroot/target/release/g1_deploy_onnx_ref` konumuna, ayrı kaynak kopyasından başarıyla derlendi. Kaynak ağacındaki DDS `.so` dosyaları Git LFS işaretçisi olduğundan derlemenin geçici kopyasında robotta zaten kurulu gerçek DDS kütüphaneleri kullanıldı. Çalışır tam kontrol döngüsü, simülasyon ve fiziksel ayakta-idle ölçümü henüz yapılmadı. Robotun kurulu kaynağına veya çalışan kontrol sürecine dokunulmadı.

Aynı gün yapılan sonraki doğrulamada patch'li tam ikili, `lo` DDS arayüzü, v1.1 model ve `zmq_manager` ile robot bilgisayarında açıldı; `g1_debug` çıkışını `16557` portuna bağladı ve gerçek robot LowState'i gelmediği için “LowState is not available, waiting for robot to be ready” noktasında kaldı. Süreli kapanış temizdi. Bu motor döngüsünde idle veya robot hareketi kanıtı değildir. `tests/test_robot_http_camera.py` ile mevcut kamera testleri toplam 8 geçti, 2 atlandı; HTTP istemcisi taze RGB JPEG'i kabul edip bayat/503 yanıtını reddetti. Robot üzerinde renk servisinin ilk canlı denemesinde `Frame didn't arrive within 2000`, sonra aygıt meşgul hatası alındı; servis durduruldu, kurulu bir servis bırakılmadı.
