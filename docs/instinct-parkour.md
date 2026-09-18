# Instinct parkour oynatımı

`./dev.sh instinct-parkour`, Project-Instinct G1 parkour checkpoint'ini Isaac
Sim'de oynatır. Checkpoint yalnızca ONNX export'larıyla dağıtıldığı için upstream
`play.py` bu modeli doğrudan yükleyemez; `scripts/play-instinct-parkour.py` aynı
kontrol akışını depth encoder ve actor ONNX modelleriyle kurar.

```bash
./dev.sh instinct-parkour
./dev.sh instinct-parkour-drive  # yalnızca yerel Isaac X11 penceresi varsa
```

WebRTC koşusunda klavye olaylarını WebRTC istemcisi üzerinden gönderin;
`instinct-parkour-drive` yerel `DISPLAY` üzerinde bir Isaac penceresini
`xdotool` ile hedefler ve headless streaming server'a olay enjekte edemez.

Varsayılan üretim yolu CUDA fizik ve sensörler, 200 Hz fizik
(`dt=0.005`), `decimation=4` ile 50 Hz policy, 64×36 ve 50 Hz ray-cast depth,
WebRTC arayüzü ve duvar saatine pacing kullanır. Depth penceresini yalnızca
görsel olarak kapatmak için `--no_depth_window`; playback kırpmasını kapatmak
için `--no_tuning` kullanılabilir.

## Sonuç

Bu makinede görülen 24 çekirdek doygunluğu PhysX'in ihtiyacı değildi. İki küçük
ONNX Runtime oturumu varsayılan olarak makinedeki çekirdek sayısına göre ayrı
iş parçacığı havuzları kuruyor ve boşta beklerken spin atıyordu. Sonuç, süreçte
~2258% CPU kullanımı, hostta ~%98 CPU ve yüksek scheduler baskısıydı.

Playback artık her iki ONNX oturumu için açıkça şunları kullanır:

- `intra_op_num_threads=1`
- `inter_op_num_threads=1`
- `ORT_SEQUENTIAL`
- intra-op ve inter-op spinning kapalı

Bundan sonra süreç yaklaşık %108.8 CPU'ya, host yaklaşık %4.8 CPU'ya ve Linux
CPU pressure `some avg10` yaklaşık 60.74'ten 0.02'ye indi. Yani kalan darboğaz
CPU çekirdek kapasitesi değil; seri GPU fizik/sensör/render işi ve bu işlerin
CPU tarafındaki bekleme noktalarıdır.

## Önce/sonra

Aşağıdaki streamed sonuçlarda gerçek depth ve gerçek policy açıktır; dolayısıyla
policy davranışı açısından geçerli üretim ölçümleridir.

| Aşama | Streamed RTF | Yaklaşık policy loop | Ana değişiklik |
|---|---:|---:|---|
| İlk durum | 0.28 | 14/s | ONNX havuzları 24 çekirdeği spin ile dolduruyordu |
| ONNX havuz düzeltmesi | 0.33 | 16.5/s | CPU doygunluğu ve runqueue baskısı kalktı |
| Güvenli playback kırpması | 0.35 | 17.5/s | Eğitim/reward'a özel kullanılmayan işler çıkarıldı |
| Kit rate limiter düzeltmesi | **0.46** | **22.8/s** | Runner pacing yaparken ikinci kez bekleten Kit limiter kapatıldı |

Kit'in streamed experience dosyası ana döngü rate limiter'ını açık bırakıyordu.
Runner zaten mutlak deadline ile pacing yaptığı için bu ikinci bir beklemeydi.
`/app/runLoops/main/rateLimitEnabled=false` sonrasında render çağrısı yaklaşık
20.56 ms'den 7.94 ms'ye düştü. Normal `--realtime` davranışını runner korur;
`--no_realtime` yalnızca ulaşılabilir tavanı ölçer.

Son üretim doğrulamasında normal paced streamed koşu 0.45 RTF verdi. Ayrı bir
`--no-keyboard` koşusunda environment'ın eğitim aralığındaki hız komutu robotu
yaklaşık 2.4 m ilerletti (`vx` kısa süre 0.9–1.1 m/s); canlı depth ve policy ile
RTF yine 0.46 kaldı. Yani sonuç yalnızca duran robot durumuna ait değil.
Dağıtılan ONNX dosyalarının batch boyutu sabit 1 olduğu için playback artık
`--no-keyboard` modunda da varsayılan olarak bir environment açar ve açıkça
verilen `--num_envs != 1` değerini inference'a gelmeden reddeder.

## Süre nereye gidiyor?

Geçerli streamed, pacing'siz ayrıntılı koşunun policy loop başına duvar saati:

| Bölüm | ms/loop | Yorum |
|---|---:|---|
| Dört `sim.step` çağrısı | 21.59 | Dört adet 5 ms fizik alt adımı; ana kritik yol |
| Render | 7.94 | Kit limiter kapatıldıktan sonraki değer |
| Policy observation | 6.48 | Zorunlu depth history/crop/noise/gather işi dahil |
| Scene write | 2.88 | Action/state'in simülasyona yazılması |
| Scene update | 2.02 | Kamera ve contact sensor güncellemeleri dahil |
| Command | 1.17 | Komut yöneticisi |
| Termination | 0.53 | Düşme ve sınır kontrolleri |
| ONNX policy | 0.41 | Depth encoder + actor; darboğaz değil |
| Depth debug penceresi | 0.56 | Policy sensöründen ayrı görüntüleme maliyeti |
| **Toplam `env.step`** | **42.89** | **22.8 loop/s, 0.46 RTF** |

50 Hz policy için bir loop'un gerçek-zaman bütçesi 20 ms'dir. Dört fizik adımı
tek başına yaklaşık 21.6 ms sürdüğü için mevcut `dt=0.005`, `decimation=4` ve
fizik fidelity'siyle 1.0 RTF, render ve observation sıfır olsa bile mümkün
değildir. 1.0 RTF için fizik maliyetini azaltmak veya eğitim dinamiğini
değiştiren `dt`/decimation değişikliğini davranış testiyle kabul etmek gerekir.

## Depth maliyetini ayırma

`--diagnostic_no_depth`, depth sensörünü ilk frame'den sonra donduran yalnızca
profil amaçlı bir kontroldür. Bu koşuda robot davranışı geçerli değildir ve
üretim sonucu olarak kullanılamaz.

Headless karşılaştırmada:

- Gerçek depth ile observation yaklaşık 4.7–6.5 ms/loop.
- Depth dondurulunca observation yaklaşık 0.65 ms/loop'a indi.
- `sim.step` yaklaşık 20.8 ms'de kaldı.
- Tanısal RTF yaklaşık 0.74'e çıktı.

Bu, depth'in ana maliyetinin `sim.step` içinde değil; observation oluşturma,
history, noise, crop ve gather hattında ödendiğini gösterir. Sensör wrapper'ının
~0.12 ms görünmesi GPU işinin asenkron olmasından kaynaklanır; işin beklemesi
daha sonraki çağrıda tahsil edilebilir.

## Güvenli playback kırpması

Varsayılan playback, policy girdisini ve düşme recovery'sini koruyarak yalnızca
kullanılmayan eğitim işlerini kaldırır:

- reward manager ve critic/AMP observation grupları,
- AMP motion reference ve ona bağlı `dataset_exhausted` termination,
- yalnızca reward için kullanılan iki height scanner ve leg-volume sensor,
- `register_virtual_obstacles` startup event'i,
- debug işaretleri ve pahalı `mesh_boxes` gösterimi.

Head depth camera policy observation'ı olduğu için, contact sensor ise torso
contact ile düşme reset'ini beslediği için korunur.

## Doğru ölçüm yöntemi

```bash
./dev.sh instinct-parkour --no_realtime --step_detail --duration 6
scripts/benchmark-sim.py --case instinct-parkour-paced --case instinct-parkour-free --perf-detail
```

- Telemetriyi startup sırasında değil, ilk steady `[perf]` satırından sonra alın.
- Pencereli RTF'yi ardışık örneklerden `Δt_sim / Δt_wall` ile hesaplayın;
  playback simülasyon zamanını milisaniye hassasiyetinde yazar. Playback'in kendi
  `loops/s` alanı koşu başından beri kümülatiftir.
- Aynı anda tek Isaac süreci çalıştığını `docker top` ile doğrulayın.
- CPU için süreç CPU'su, `pidstat -t`, `%wait`, host CPU pressure ve runqueue;
  GPU için `nvidia-smi dmon/pmon` birlikte okunmalıdır.
- Python wall timer değerleri CUDA/PhysX çağrılarındaki senkronizasyon beklemesini
  içerir; kernel süresi değildir. Profil kodu ölçümü değiştirmemek için
  `torch.cuda.synchronize()` eklemez.
- `--diagnostic_no_depth` sonucu policy performansı olarak raporlanmamalıdır.

Host `perf`, `kernel.perf_event_paranoid=4` nedeniyle bu kullanıcı için kapalı;
ölçümler `pidstat`, CPU pressure, NVIDIA telemetrisi, PyTorch/CUPTI ve Kit'in
mevcut profiler altyapısıyla yapıldı.
