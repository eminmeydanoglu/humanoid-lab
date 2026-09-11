# humanoid-lab

Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + SONIC + GR00T N1.7 icin tek-container dev environment (tek NVIDIA GPU host).

## 1. Kurulum (bir kez)

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git && cd humanoid-lab
./setup.sh                    # host preflight + lock render + image build + start + smoke
```

Docker/NVIDIA toolkit kurulu degilse `setup.sh` lock'taki birebir versiyonlarla
kurulum komutlarini yazdirir; `--verify-digests` Isaac Sim digest'ini NGC'den cekip lock'a yazar:

```bash
./setup.sh --verify-digests
```

## 2. Gunluk kullanim

```bash
./dev.sh            # ana container shell (yoksa olusturur/baslatir)
./dev.sh isaac      # isaac-sonic env  : Python 3.11, Isaac Lab + SONIC
./dev.sh sonic-sim  # sonic-sim env    : Python 3.11, MuJoCo + G1
./dev.sh groot      # groot-n17 env    : Python 3.12, Isaac-GR00T N1.7
./dev.sh doctor     # host + container validation raporu
./dev.sh smoke      # container icinde normal smoke testler
./dev.sh groot-finetune-smoke pipeline  # istege bagli GR00T iki-adim egitim hatti smoke testi
./dev.sh rebuild    # image'i ayni lock ile yeniden build et
./dev.sh stop       # container'i durdur (veriler kalir)
```

## 3. Isaac G1 simulatoru (Kapi 1)

Yalnız Isaac Lab, G1 asset'i ve head camera kullanır; SONIC,
GR00T, CloudWalk, DDS veya başka bir controller yüklemez. Fizik döngüsünün tek
sahibi `SimulatorService`tir.

```bash
./dev.sh isaac-g1 dex3                     # akıcı ana GUI, doğrudan Play
./dev.sh isaac-g1 inspire-ftp
./dev.sh isaac-g1 no_hands --headless
./dev.sh isaac-g1 dex3 --head-camera-window # ikinci head-camera viewport'u
./dev.sh isaac-g1 dex3 --test passive-fall # controller'siz düşüş kabulü
./dev.sh isaac-g1-test-controller dex3 --headless # deterministik komut/TTL kabulü
```

Genel seçenekler `--headless`, `--head-camera-window`, `--duration SECONDS` ve
`--device {cpu,cuda}`'dır. Controller seçimi profile aittir; controller'lı bir
profil yalnız güvenli pasif fallback için `--controller none` ile kapatılabilir.
Normal çalışma `COMPLETED` ile biter; düşüş/hold kabulü yalnız ilgili
test komutunda uygulanır.

Tek robotlu bu sahnede fizik varsayılan olarak CPU'da, dört worker thread ile çalışır;
`--device cuda` açık bir geri dönüş seçeneğidir. Fizik zaman adımı 0,005 saniyedir
(200 Hz simülasyon zamanı); gerçek çalışma hızı terminalde ayrıca raporlanır. Ana RTX
viewport her sekiz fizik adımında güncellenir. Dış döngü bu render aralığının sahibidir;
Kit'in iç güncelleme adımı `1`dir. Aynı aralığın iki kat uygulanması render çağrısını
yaklaşık 40 ms bekletiyor ve RTF'yi yarıya düşürüyordu.

Normal GUI koşusu yalnız ana viewport'u ve `G1 Simulator` kontrol penceresini açar.
Bu viewer yolu RTX balanced görüntü, DL denoiser ve desteklenen GPU'da DLSS Frame
Generation kullanır. Terminaldeki FPS gerçek render çağrısıdır; üretilen sunum karelerini
saymaz. `--head-camera-window` 640×480 sensörü ve GPU-backed ikinci viewport'u açar;
ek render maliyeti beklenir. `Reset Robot` başlangıç pozunu geri yükler ve timeline'ı
paused bırakır. Koşuyu `Ctrl-C` veya pencereyi kapatarak durdur.

Simülatör özel MP4, trajectory, JSONL veya provenance kaydı üretmez. Passive-fall
kabulü kamera boyutunu ve değişen frame sayısını bounded sayaç/hash ile ölçer;
görüntü dizisini RAM'de veya diskte biriktirmez. Performans değerleri çalışırken
terminale saniyede bir yazılır.

GUI komutları çağıran terminalin canlı `DISPLAY` değerini kullanır. `.env` içindeki
display artık mevcut değilse ve hostta tek bir aktif X11 socket'i varsa bu display
otomatik seçilir; böylece masaüstünün yeniden girişten sonra `:0` ile `:1` arasında
değişmesi Isaac penceresini sessizce bozmaz.

Raider'da Isaac Sim 5.1 bazen `simulation_app.close()` çağrısından dönmez. CLI GUI'de
en fazla 2 saniye (headless koşuda 20 saniye) bekler; süre aşılırsa
`forced_exit=true` yazarak yalnız kendi sürecini sonlandırır. `dev.sh` ayrıca tekil
çalışma kilidi ve PID/process-group temizliği uygular; kesilen SSH/terminal istemcisi
arka planda Isaac süreci bırakamaz. Bu, graceful Kit shutdown kanıtı değildir.

### Isaac Lab demolari

G1 disindaki hazir demolar icin `isaac-demo` kullan; ham container kabugunda
`python` calistirmak gerekli Kit ortamini kurmaz:

```bash
./dev.sh isaac-demo quadrupeds.py
./dev.sh isaac-demo bipeds.py
```

Bu komut container'da varsayilan olarak `--headless` ekler. Uzak goruntu icin
once `./scripts/install-isaac-webrtc-client.sh` ile istemciyi kur, demoyu
`--livestream 2` ile baslat ve `./dev.sh webrtc-client` ile baglan. Dogrudan X11
penceresi (`--gui`) deneyseldir.

> **Raider notu (2026-09-04):** Isaac Sim 5.1'in resmi `runheadless.sh` streaming
> uygulamasi bu RTX 5090 Laptop sisteminde `librtx.scenedb.plugin.so` icinde
> cokuyor. Bu nedenle `isaac-stream` komutu, surucu/Isaac Sim uyumlulugu
> duzeltilene kadar korumali olarak basarisiz olur ve crash dongusu baslatmaz.

## 3b. SONIC ile Isaac G1

Isaac'teki G1'i resmî SONIC kontrolcüsüne bağlar. Upstream decoder, model ve
planner semantiği korunur; pinli deployment binary'sinin tek kaynak farkı,
simülasyon DDS'ini gerçek robot domain'inden ayıran domain `42` değişikliğidir.
Isaac yalnız robotun Unitree DDS topiklerini konuşur. Kontrolcü ayrı bir
terminalde, resmî inference'taki gibi planner ve klavye ile açılır.

```bash
./dev.sh isaac-g1-sonic dex3     # 1. terminal: Isaac + robot (ayrica inspire-ftp)
./dev.sh sonic-controller        # 2. terminal: resmî SONIC (planner + klavye)
```

SONIC terminalinde `]` kontrolü başlatır, `T` referans hareketi çalar,
`N`/`P` hareket değiştirir, `O` acil durdurur. Ayrıntı, kabul ölçütleri ve
ölçülen davranış: `docs/sonic-isaac.md`.

## 4. Modeller (bir kez)
```bash
./dev.sh hf-login       # HF token host cache'ine yazilir (image'e/.env'e girmaz)
./dev.sh fetch-models            # pinli model revision'lari + MODEL_PROVENANCE.json (sha256)
```

Not: GR00T N1.7 backbone'u `nvidia/Cosmos-Reason2-2B` gated — once HF'de lisansi kabul et.

GR00T veri formati, embodiment config'i, iki-adim `pipeline` / `pretrained` /
`eval` smoke komutlari ve 40 GiB VRAM siniri `scripts/groot-finetune-smoke.sh`
icinde ve git gecmisindeki fine-tuning rehberinde belgelidir. Bu GPU/data
maliyetli akis `setup.sh` ve normal `./dev.sh smoke` icinde otomatik calismaz.
Bu hattin SONIC/SONIC-Isaac entegrasyonu Kapi 3 kapsamindadir; Kapi 1 yalniz
simulatoru kapsar.

## 5. Dogrulama / tanalama

```bash
./doctor.sh                 # host'tan tam rapor (${HUMANOID_DATA_ROOT}/diagnostics/)
./scripts/verify-pins.sh    # lock pin'lerini upstream'e karsi dogrula
./scripts/ci-lint.sh        # bash/python/yaml statik kontroller
```

## 6. Container icinde environment secicileri

```bash
use-isaac-sonic | use-sonic-sim | use-groot | use-none
show-env            # aktif env + python
```

## Surum kilidi

Tum surumlerin tek kaynagi `versions.lock.yaml`'dir; `scripts/render-lock-env.py`
bunu `.generated/versions.env`'e render eder, Compose/Dockerfile/scriptler yalnizca
oradan okur. Dogrulanmamis pinler (`required: false`) setup'i bloklar. Kalici veri
koku `HUMANOID_DATA_ROOT`'tur; varsayilan `.env` degeri repo icindeki `data/`
dizinidir. Bu dizin Compose ile container'daki model, cache, tanilama ve cikti
dizinlerine bind-mount edilir.

Venv'ler de `${HUMANOID_DATA_ROOT}/venvs/` altinda kalicidir. Container baslarken
kilit ve pinli kaynak commit'i kontrol edilir; yalnizca farkli olan environment
`uv sync --frozen` ile guncellenir. Bu nedenle gunluk kod veya shell betigi
degisiklikleri venv'leri yeniden kurmaz; yeni bir Python bagimliligi ise lock'a
eklenmeden kalici hale gelemez.

Bir ortamin lock dosyasini degistirdikten sonra `./dev.sh sync` calistir. Bu,
imaji yeniden kurmadan yalniz degisen environment'i gunceller; indirilen
paketler `${HUMANOID_DATA_ROOT}/uv-cache/` icinde yeniden kullanilmak uzere kalir.

> **Kalici mount uyarisi:** `/opt/venvs` ile Isaac cache dizinleri yazilabilir
> host bind-mount'laridir. Container icinde bunlari elle degistirmek veya `pip`
> ile paket eklemek bu degisiklikleri kalicilastirir ve lock/fingerprint
> beklentisini bozabilir. Lock veya kaynak degisikliginde `./dev.sh sync`, imaj
> degisikliginde `./dev.sh rebuild` kullan; ardindan `./dev.sh doctor` ve
> `./dev.sh smoke` ile dogrula. Ayni fingerprint altindaki elle bozulmus bir
> ortam otomatik olarak onarilmaz.
