# humanoid-lab

Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + SONIC + GR00T N1.7 icin tek-container dev environment (tek NVIDIA GPU host).

## 1. Kurulum (bir kez)

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git && cd humanoid-lab
./setup.sh                    # host preflight + lock render + image build + start + smoke
```

Docker/NVIDIA toolkit kurulu degilse `setup.sh` lock'taki birebir versiyonlarla kurulum komutlarini yazdirir; `--verify-digests` Isaac Sim digest'ini NGC'den cekip lock'a yazar:

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

### İlk Isaac Lab dünyaları

`python` komutunu ham container kabuğunda çalıştırmak yerine `isaac-demo`
komutunu kullan. Bu komut Isaac Sim'in gerekli Kit ortamını ve
`isaac-sonic` venv'ini birlikte kurar:

```bash
./dev.sh isaac-demo quadrupeds.py
./dev.sh isaac-demo procedural_terrain.py
./dev.sh isaac-demo bipeds.py
```

Bu komutlar container'da varsayılan olarak `--headless` ekler. Isaac Sim
container'ı desteklenen biçimde headless Python uygulamaları çalıştırır; doğrudan
X11/masaüstü Kit penceresi açmak desteklenen bir yol değildir ve RTX renderer
çökmesine yol açabilir. Uzak görüntü için ayrı WebRTC istemcisi kurulup demo
`--livestream 1` veya `--livestream 2` ile başlatılmalıdır. Aynı Raider
masaüstünde doğrudan pencereyi denemek için açıkça `--gui` verilebilir; bu
deneysel X11 yoludur ve yalnızca GUI smoke testi geçerse kullanılmalıdır.

`h1_locomotion.py`, etkileşimli klavye girdisi ile RSL-RL pretrained H1 policy'si
gerektirir. Bu policy henüz bu repoda pinlenmiş veya `fetch-models` kapsamına
alınmış değildir; dolayısıyla aşağıdaki komut tek başına çalışır bir policy
indirme adımı değildir. Policy'yi kaynağı ve revizyonu doğrulanarak örneğin
upstream Isaac Lab demosunun beklediği konuma sağladıktan sonra streaming
istemcisi üzerinden çalıştır:

```bash
./dev.sh isaac-demo h1_locomotion.py --livestream 2
```

İstersen `--headless` argümanını açıkça da verebilirsin; iki kez eklenmez.

### WebRTC ile görsel Isaac Lab oturumu

Docker container'ında desteklenen görsel yol WebRTC'dir. İlk kez yalnızca hosta
resmî istemciyi kur:

```bash
./scripts/install-isaac-webrtc-client.sh
```

> **Raider notu (2026-09-04):** İstemci kuruludur ancak Isaac Sim 5.1'in
> resmî `runheadless.sh` streaming uygulaması bu RTX 5090 Laptop sisteminde
> `librtx.scenedb.plugin.so` içinde çöküyor. Bu nedenle `isaac-stream` komutu,
> sürücü/Isaac Sim uyumluluğu düzeltilene kadar korumalı olarak başarısız olur;
> yanlışlıkla crash döngüsü başlatmaz.

Uyumluluk düzeltildikten sonra bir demoyu streaming modunda başlatmak için bir
terminalde şunu çalıştır:

```bash
./dev.sh isaac-stream quadrupeds.py
```

Bu komut NVIDIA'nın `isaacsim.exp.full.streaming.kit` experience'ını ve
`--no-window` seçeneğini kullanır; normal GUI veya genel `--livestream` yolu
yerine container için tasarlanmış WebRTC çalışma biçimidir. Isaac Sim hazır
olana kadar bekle, sonra ikinci bir terminalde istemciyi aç:

```bash
./dev.sh webrtc-client
```

İstemci Raider üzerinde çalışıyorsa varsayılan `127.0.0.1` adresiyle bağlan.
Başka bir bilgisayardan bağlanırken Raider adresini kullan; TCP `49100` ve UDP
`47998` erişilebilir olmalıdır. Bir Isaac instance'a aynı anda yalnızca bir
istemci bağlanabilir.

## 3. Modeller (bir kez)

```bash
./dev.sh hf-login       # HF token host cache'ine yazilir (image'e/.env'e girmaz)
./dev.sh fetch-models            # pinli model revision'lari + MODEL_PROVENANCE.json (sha256)
```

Not: GR00T N1.7 backbone'u `nvidia/Cosmos-Reason2-2B` gated — once HF'de lisansi kabul et.

GR00T veri formati, embodiment config'i, iki-adim `pipeline` / `pretrained` / `eval` smoke komutlari ve 40 GiB VRAM siniri icin [GR00T N1.7 fine-tuning rehberine](docs/groot-n17-finetuning.md) bak. Bu GPU/data maliyetli akıs `setup.sh` ve normal `./dev.sh smoke` icinde otomatik calismaz.

## 4. CloudWalk GR00T + SONIC + Isaac

Bu yol yalnız Raider'da çalıştırılır. `scripts/run-cloudwalk-isaac.py` aynı dataset-kalibre scene builder'ını iki modda kullanır:

```text
interactive GUI: paused scene → Timeline Play → free-base PhysX
headless rollout: Isaac RGB/state → GR00T → SONIC C++ → Isaac G1 + Inspire FTP
```

G1, `g1_29dof_inspire_hand.usd` asset'i ile 29 gövde ve 24 Inspire el eklemi olarak kurulur. CloudWalk modunda root serbesttir ve gravity açıktır; upstream asset'in fixed-base manipülasyon varsayımları kullanılmaz. Masa, bottle, oda ve görsel child primleri `sim.reset()`ten önce author edilir. Physics başladıktan sonra USD topology'sini GUI'den değiştirme; bottle/table pose'unu komut satırından değiştirip oturumu yeniden başlat.

GUI'de sahneyi açmak için:

```bash
ssh raider
cd ~/code/humanoid-lab
docker exec -it humanoid-lab-dev bash -lc '
  source /opt/humanoid-lab/entrypoint.sh
  use-isaac-sonic
  exec /opt/venvs/isaac-sonic/bin/python \
    /workspace/humanoid-lab/scripts/run-cloudwalk-isaac.py \
    --interactive \
    --capture-path /workspace/humanoid-lab/outputs/cloudwalk-gui.jpg
'
```

Scene Timeline'da paused başlar. Play ile gravity ve serbest-root PhysX çalışır. `--bottle-position X Y Z`, `--table-position X Y Z` ve `--table-size X Y Z` yalnız başlangıç scene config'ini değiştirir. Yeni değer için uygulamayı kapatıp yeniden başlat.

Headless closed-loop koşusu dört ayrı süreç başlatır: upstream GR00T server, VLA worker, native SONIC decoder ve Isaac owner. Varsayılan başlatıcı:

```bash
./scripts/run-cloudwalk-closed-loop.sh
```

Koşu sırasında GR00T yüklenene kadar doğrulanmış SONIC standing reference gönderilir; böylece serbest taban cold-start sırasında kontrolsüz bırakılmaz. `outputs/` Git dışında runtime kanıt dizinidir. Kod yolu ve sözleşmeler: `tools/cloudwalk_scene.py`, `tools/cloudwalk_closed_loop.py`, `tools/sonic_isaac_inspire_adapter.py`, `scripts/cloudwalk-vla-worker.py`, `scripts/run-cloudwalk-isaac.py`.

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

## Sürüm kilidi

Tüm sürümlerin tek kaynağı `versions.lock.yaml`'dır; `scripts/render-lock-env.py`
bunu `.generated/versions.env`'e render eder, Compose/Dockerfile/scriptler yalnızca
oradan okur. Doğrulanmamış pinler (`required: false`) setup'ı bloklar. Kalıcı veri
kökü `HUMANOID_DATA_ROOT`'tur; varsayılan `.env` değeri repo içindeki `data/`
dizinidir. Bu dizin Compose ile container'daki model, cache, tanılama ve çıktı
dizinlerine bind-mount edilir.

Venv'ler de `${HUMANOID_DATA_ROOT}/venvs/` altında kalıcıdır. Container başlarken
kilit ve pinli kaynak commit'i kontrol edilir; yalnızca farklı olan environment
`uv sync --frozen` ile güncellenir. Bu nedenle günlük kod veya shell betiği
değişiklikleri venv'leri yeniden kurmaz; yeni bir Python bağımlılığı ise lock'a
eklenmeden kalıcı hâle gelemez.

Bir ortamın lock dosyasını değiştirdikten sonra `./dev.sh sync` çalıştır. Bu,
imajı yeniden kurmadan yalnız değişen environment'ı günceller; indirilen
paketler `${HUMANOID_DATA_ROOT}/uv-cache/` içinde yeniden kullanılmak üzere kalır.

> **Kalıcı mount uyarısı:** `/opt/venvs` ile Isaac cache dizinleri yazılabilir
> host bind-mount'larıdır. Container içinde bunları elle değiştirmek veya `pip`
> ile paket eklemek bu değişiklikleri kalıcılaştırır ve lock/fingerprint
> beklentisini bozabilir. Lock veya kaynak değişikliğinde `./dev.sh sync`, imaj
> değişikliğinde `./dev.sh rebuild` kullan; ardından `./dev.sh doctor` ve
> `./dev.sh smoke` ile doğrula. Aynı fingerprint altındaki elle bozulmuş bir
> ortam otomatik olarak onarılmaz.
