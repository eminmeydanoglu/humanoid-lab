# SONIC ile Isaac G1

Bu yol Isaac Sim'deki G1'i, resmî SONIC kontrolcüsünün gördüğü **gerçek robot**
gibi sunar. Upstream decoder, model ve planner semantiği korunur. Pinli
`g1_deploy_onnx_ref` binary'sinin tek kaynak farkı, simülasyonu gerçek robot
ağından ayırmak için DDS domain'inin `42` yapılmasıdır; Isaac tarafı yalnız
robotun Unitree DDS topiklerini konuşur.

```
./dev.sh isaac-g1-sonic dex3          # 1. terminal: Isaac + robot
./dev.sh sonic-controller             # 2. terminal: resmî SONIC (planner + klavye)
```

Isaac komutu varsayılan olarak süresiz çalışır; pencere veya terminal
kapatılıncaya kadar açık kalır. Yalnız süreli bir koşu istendiğinde örneğin
`--duration 60` verilir. `--test` koşulları, açıkça süre verilmezse otomatik
olarak 12 saniyeyle sınırlıdır.

Sıra önemli değil: Isaac açılmadan SONIC'i başlatırsan SONIC "LowState is not
available" diyerek bekler ve Isaac açılınca devam eder.

`./dev.sh sonic-controller` Isaac'e ait tekil bir kontrolcü rolüdür. Komut ikinci
kez çalıştırılırsa önceki Isaac kontrolcüsünü kapatıp onun yerini alır. Bu işlem
yalnız launcher'ın `HUMANOID_LAB_CONTROLLER_TARGET=isaac` ile işaretlediği sürece
uygulanır; bağımsız MuJoCo/SONIC deployment süreçleri süreç adına bakılarak
öldürülmez.

## İki terminalde ne olur

**Isaac terminali.** Sahne, G1 ve head camera'yı kurar, fizik döngüsünü yönetir,
robot state'ini DDS'e yayınlar ve SONIC'in gönderdiği eklem komutlarını uygular.
Başlangıçta robot, resmî MuJoCo döngüsündeki gibi pelvisten bir yay bandıyla
asılıdır; bandın ne zaman bırakıldığı log'da görünür
(`isaac_g1_support_released`). Kontrolcü yokken, komut gecikmişken veya
kontrolcü kapandığında robot gerçekten pasiftir ve düşer.

**SONIC terminali.** Resmî dağıtımın kendi akışıdır: planner yüklenir, klavye o
terminalin TTY'sinde çalışır, `]` ile kontrol başlar, `T` referans hareketi
çalar, `N`/`P` hareket değiştirir, `R` başa alır, `O` acil durdurur. Isaac
tarafında hiçbir özel mod yoktur; bu terminal kapandığında robot pasife döner.

## Neye dikkat edilir

- Robot, politikayı başlatmadan önce serbest bırakılmaz. Bant `]` basılıp
  kontrolcü setpoint'leri hareket etmeye başlayınca bırakılır; operatörün MuJoCo
  penceresinde `9`'a basmasıyla aynı ana denk gelir. Bant en fazla
  ilk geçerli kontrolcü komutundan sonra `support.max_seconds` kadar kalır,
  sonra kendiliğinden bırakılır ve bu da log'a yazılır. SONIC yüklenirken
  kontrolcü komutu yoksa süre tüketilmez.
- Klavye kabul edildiğinde SONIC'in kendi durum geçişleri terminalde görünür;
  tuş → state → `BodyJointCommand` → Isaac hareketi zinciri bu şekilde izlenir.
- DDS simülasyona özeldir: domain `42` ve `lo` arayüzü kullanılır, gerçek robot
  ağına çıkılmaz.

## Ölçülen davranış

Aşağıdakiler bu depodaki kabul ölçütleridir ve Raider'daki koşularda
doğrulanmıştır:

| Durum | Beklenen | Nasıl ölçülür | Son ölçüm |
|---|---|---|---|
| Kontrolcü yok | Robot düşer | `--test passive-fall` | PASS, `robot_fell` + kamera akıyor |
| Test kontrolcüsü | Hedef eklemler fiziksel tepki verir | `--test controlled-hold` | PASS, 5 eklem izlendi, ilerleme 0.92 |
| Komut kesilince | TTL sonrası pasif | `--test controlled-hold` | PASS, `passive_after_command_stop` |
| SONIC geçerli | Robot ayakta kalır | `--test controller-hold` | PASS, bant bırakıldıktan sonra ~40 s kök yüksekliği 0.757–0.79 m |
| SONIC öldürüldü | TTL içinde pasif ve düşer | `--test controller-hold` + `SIGKILL` | PASS, son komuttan **51 tick (0.255 s)** sonra pasif, kök 0.09 m |
| Kontrol sırasında kamera | Akmaya devam eder | `--test controller-hold` | PASS, 1607 frame / 1402 değişen, 480×640×3 |
| Klavye → hareket | Ölçülebilir yer değiştirme | `T` ile referans hareket | Kök x/y ≈ 0.5 m yer değiştirdi, eklemler referansı izledi |

Koşu özeti ayrıca `root_z_trace`, `command_trace` (komut/ölçüm/tork/kök konumu)
ve kontrolcü istatistiklerini (komut sayısı, yaş, TTL, stale poll, reddedilen
komut) içerir.

Bozuk komut güvenliği: çözülemeyen, sonlu olmayan veya bildirilen eklem
sırasına uymayan bir komut döngüyü düşürmez; sayılır ve komut yokmuş gibi
davranılır (`rejected_commands`), robot pasife geçer.

Tork yolu tek ve açıktır: kontrolcünün verdiği tork, kendi limitleriyle
sınırlandıktan sonra doğrudan fizik motoruna yazılır. Aracın kendi actuator
modelleri kendi torklarını hesapladığı ve dışarıdan verilen torku yok saydığı
için bu yol bilerek onların dışındadır.

SONIC Dex3 profili ayrıca MuJoCo modelinin pasif eklem dinamiklerini PhysX'e
uygular: bütün 43 eklemde armature `0.01`, viskoz sürtünme `0.05`, Coulomb
sürtünmesi ise ekleme göre `0.1` veya `0.2 Nm` olur. IsaacLab'in varsayılan G1
actuator armature değerleri (`0.001`–`0.03`) bu profilde kullanılmaz.

## Engebeli arazi seçeneği

Aynı kontrol hattı, düz zemin yerine InstinctLab parkour dünyasında:

```
./dev.sh isaac-g1-sonic-rough dex3    # 1. terminal: Isaac + robot, engebeli arazi
./dev.sh sonic-controller             # 2. terminal: resmî SONIC (değişmez)
```

Zemin, `configs/profiles/isaac-g1-sonic-rough-dex3.json` içindeki `terrain`
bloğuyla seçilir. Arazi burada tarif edilmez; pinli InstinctLab checkout'undaki
`instinctlab.tasks.parkour.config.parkour_env_cfg` modülünden **olduğu gibi**
alınır (sub-terrain karışımı, malzemeler, kenar silindiri katmanı). Böylece
`./dev.sh instinct-parkour` ile aynı dünyadır; bu pakette bir kopyası tutulmaz.

- `terrain.preset`: şu an tek değer, `instinct_parkour_rough`.
- `terrain.max_init_terrain_level`: robotun başladığı zorluk bandı. `0` onu en
  kolay kareye sabitler (düz sayılabilecek bir başlangıç), eğitimdeki gibi
  rastgele daha zor bir karede başlamasını isterseniz yükseltin.

Koşuya özgü üç uyarlama: eğitim sahnesinin 5 m duvarları kapatılır, ızgara
4×10'a iner (izlenebilir koşu için), ve profilde dünya koordinatında verilen
şeyler — hata ayıklama kamerası, kayıt kamerası, destek bandının çıpası —
ortamın başlangıç noktasına göre kaydırılır. Başlangıçta
`{"event":"isaac_g1_terrain", ...}` satırı hangi dünyanın kurulduğunu yazar ve
aynı özet koşu çıktısındaki `terrain` alanında saklanır.

Durum: seçenek Raider'da koşuldu (headless, kontrolcüsüz serbest koşu, 20 s).
Engebeli arazi CPU fiziğinde ~370 Hz / RTF 1.8x, aynı sahne `cuda:0` ile
~51 Hz / RTF 0.25x. Düz zemin aynı koşulda ~424 Hz / RTF 2.1x; arazinin
maliyeti ~%13. Varsayılan `cpu` bu yüzden korunuyor.

## Model karşılaştırması (ölçülmüş)

Isaac tarafındaki gövde, pinli SONIC kaynağındaki Dex3 G1 USD'sinden gelir; o
USD'nin kaynağı `gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf`.
SONIC'in MuJoCo modeli ise `gear_sonic/data/robot_model/model_data/g1/
g1_29dof_with_hand.xml`. İkisi karşılaştırıldığında:

- **Eklem adları ve eksenleri birebir aynı.** 29 gövde joint'inin isim kümesi
  örtüşüyor; eksen vektörleri ve yönleri (işaretleri dahil) farklı değil.
- **Bacak kütleleri birebir aynı.** 44 ortak link'in kütlesi karşılaştırıldı;
  görünen farkların çoğu MuJoCo'nun fixed joint'li gövdeyi ebeveyne
  birleştirmesinden geliyor, ör. `left_wrist_yaw_link`: URDF 0.0846 + sabit
  `left_hand_palm_link` = 0.4574, MuJoCo'da tam olarak 0.4574.
- **Kaynak asset'te bir gerçek fark var: üst gövde.** MuJoCo `torso_link` kütlesi 9.598 kg;
  URDF'te sabit çocuklarıyla birleşince 7.817 kg (kendi 6.78 + head, d435,
  mid360, logo, imu). Aradaki ~1.78 kg fark toplam kütle farkıyla
  (36.165 vs 34.394) neredeyse birebir örtüşüyor. Yani politikanın gördüğü
  üst gövde, Isaac'te yaklaşık %23 daha hafif.

Aktif Dex3 SONIC profili bu farkı çalışma anında düzeltir. Kütleler pinli MJCF
dosyasından gövde adına göre okunur; MuJoCo'nun ebeveyne kaynakladığı dokuz
sabit çocuk PhysX'in pozitif kütle şartı nedeniyle `1 g` yapılır. Ölçülen toplam
kütle `34.3942 kg`'dan `36.1742 kg`'a çıkar ve aynı yöntemle hesaplanan referans
toplamına eşit olur.

43-DoF MuJoCo actuator dizisinde sol el, sol ve sağ kolun arasında yer alır.
Bu nedenle gövde limitleri dizinin ilk 29 elemanı değildir. Köprü gövdeyi
`[ilk 22] + [sağ kol 7]`, elleri kendi yedi elemanlık dilimlerinden kurar; aksi
halde sağ kol `0.7 Nm` parmak limitlerine kadar kırpılır.

Fizik motorları yine farklıdır: temas çözücüsü, çarpışma geometrisi ve gövde
atalet tensörleri henüz bit düzeyinde aynı değildir. Buradaki eşleme; eklem
adları/eksenleri, gövde kütleleri, armature ve pasif eklem sürtünmesini kapsar.

## Görünüm: beyaz robot düzeltmesi

Pinlenmiş Dex3 USD'si 49 görsel mesh'in 48'ini tek bir OmniPBR materyaline
bağlar; o materyalin albedo'su `(1,1,1)` ve metallic/roughness değeri hiç
yazılmamış. Bu yüzden hem dinamik simülatör hem GRAIL replay robotu düz beyaz
gösteriyordu; kaynak URDF'in "yedi link koyu (0.2), 42 link beyaz (0.7)"
ayrımı URDF → USD dönüşümünde kaybolmuştu.

Düzeltme `configs/assets/g1_29dof_with_hand_rev_1_0_appearance.usda`: pinlenmiş
USD'yi `prepend references` ile referans alıp **yalnız materyalleri** değiştiren
bir kaplama. Geometri, eklem, kütle, çarpışma ve fizik özelliklerine
dokunmaz; iki materyalin (gövde kabukları açık gri metalik, koyu parçalar
near-black polimer) renk/metalness/roughness değerlerini ve bu materyallere
bağlanan yirmi iki linki yazar. Dosya GRAIL deposundan birebir kopyalandı.

Dex3 asset'ini kullanan yedi profil bu kaplamaya işaret eder; kalan profiller
(`no_hands`, `inspire-ftp`) başka asset'ler kullanır ve değişmedi.

Doğrulama, gerçek WebRTC hattından (istemci akışı çözüyor, kare istemcinin
kendi kompozitöründen alınıyor) iki koşu karşılaştırılarak yapıldı:

| | Robottaki pikseller |
|---|---|
| Kaplamasız (pinlenmiş asset) | tek düz beyaz kütle; koyu piksel oranı ~%2 |
| Kaplamalı | koyu baş/visör, pelvis, kalça ve ayak bileği yatakları, Dex3 elleri; koyu piksel oranı ~%8 |

Kareler: `.generated/benchmarks/ui/sonic-appearance-before-after.png`
(beyaz = öncesi, iki tonlu = sonrası).

## Performans (bu makinede ölçülen)

`scripts/benchmark-sim.py` ile, tek Isaac süreci, WebRTC akışı açık, 10 s
ısınma + 20 s pencere, o penceredeki örneklerin medyanı. "Pacing öncesi/sonrası"
satırları aynı komutun deadline tabanlı pacing değişikliğinden önce ve sonra
ölçülmüş halidir; `kaçırılan` sütunu bu iki satır arasında karşılaştırılamaz,
çünkü sayaç anlam değiştirdi (aşağıya bakın).

| Koşu | Komut | Fizik | RTF | render | step | render_call |
|---|---|---|---|---|---|---|
| Düz, serbest | `--controller none` | 207 Hz | 1.03 | 26 FPS | 3.7 ms | 9.0 ms |
| Düz, serbest, DLSS-G kapalı | `--controller none --no-dlssg` | 213 Hz | 1.06 | 27 FPS | 3.7 ms | 8.0 ms |
| Düz, arayüzsüz | `--controller none --headless` | 419 Hz | 2.09 | — | 2.4 ms | — |
| Düz, pacing, öncesi | (varsayılan) | 160 Hz | 0.80 | 20 FPS | 4.5 ms | 9.7 ms |
| Düz, pacing, sonrası | (varsayılan) | 162 Hz | 0.81 | 20 FPS | 4.5 ms | 9.6 ms |
| Engebeli, serbest | `--controller none` | 195 Hz | 0.97 | 24 FPS | 4.0 ms | 8.8 ms |
| Engebeli, pacing, öncesi | (varsayılan) | 132 Hz | 0.66 | 17 FPS | 4.5 ms | 20.3 ms |
| Engebeli, pacing, sonrası | (varsayılan) | 139 Hz | 0.69 | 18 FPS | 4.5 ms | 17.3 ms |

`--perf_detail` adım başına dağılımı verir: düz akışlı koşuda ~3.7 ms'lik adımın
~3.2 ms'si PhysX, ~0.3 ms'si turun girdilerini hazırlamak, ~0.05 ms'si sahne
tampon tazelemesi; render çağrısı ~9 ms ve her 8 tick'te bir. **Arayüz, fizik
hızının yaklaşık yarısını alıyor**: aynı sahne arayüzsüz 2.09x RTF koşuyor.

Ölçülüp **kazanç sağlamayan** şeyler (bu yüzden anahtar yapılmadı): PhysX iş
parçacığı sayısı (4 > 16 > 24; 1 ve 2 fark etmedi), PhysX → USD geri yazma
sıklığı, asset'in contact sensörleri (bu depoda hiçbir şey okumuyor) ve render
sıklığı — render aralığını 16'ya çıkarmak çağrı başına maliyeti ~9 ms'den
~18-22 ms'ye çıkarıyor, yani saniye başına render maliyeti aynı kalıyor ve
ulaşılan hız değişmiyor (1.03 ve 0.98 RTF ölçüldü, varsayılanda 1.03).
Tekrarlanabilir tek render kazancı `--no-dlssg`: aynı pencerede 206.5 → 213.0 Hz
ve render çağrısı 9.0 → 8.0 ms.

Pacing artık mutlak deadline ile yürüyor: geciken bir tur bir sonraki turda
telafi ediliyor, her turda `sleep(dt - elapsed)` ile uyanma gecikmesi kalıcı
kayba dönüşmüyor. Eşleşen 20 s pencerelerde düz profil 159.7 Hz / 0.80 RTF'den
162.0 Hz / 0.81 RTF'ye çıktı (15 s pencereli ikinci bir koşu 0.82 verdi); paced koşular
varyansın baskın olduğu ölçümler, yani bu etkinin büyüklüğü, bir garanti değil.
`pacing_overruns` sayacı değişimden önce kendi işi bir periyodu aşan turları,
sonrasında ise mutlak takvimin gerisinde kalan turları sayıyor; bu yüzden iki
satırın "kaçırılan" değeri karşılaştırılmaz.

## Sınırlar

- Bu yol GR00T, CloudWalk veya görev başarısı içermez; yalnız gövde kontrol
  hattıdır.
- SONIC gövde kontrolcüsüdür. Dex3 el komutları DDS'ten gelir ve uygulanır;
  Inspire el varyantında el bilerek pasif bırakılır.
- Fiziksel robot davranışı kanıtı değildir.
- Ayakta durma ve klavye kontrolü doğrulandı. Dinamik bir referans hareket
  (zıplama içeren dans) çalınırken robot birkaç saniye sonra dengesini
  kaybediyor. Bunun Isaac–MuJoCo model farkından mı yoksa hareketin
  zorluğundan mı geldiği henüz bilinmiyor; resmî MuJoCo döngüsüyle aynı
  hareketin karşılaştırması yapılmadan bu konuda iddia edilmiyor.
