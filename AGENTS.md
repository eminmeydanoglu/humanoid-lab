# Çalışma ortamı

Burası **Raider**'dır: eğitim, simülasyon ve çalışma zamanı doğrulaması için
kullanılan tek makine. Docker, GPU, Isaac Sim/Isaac Lab, MuJoCo, SONIC ve GR00T
işleri burada çalıştırılır; başka bir host'a SSH gerekmez.

## Yerleşim

- Checkout: `/home/aksoy-msi/code/humanoid-lab-main`
- Container: `humanoid-lab-dev` (`humanoid-lab/dev:5.1.0`), checkout
  `/workspace/humanoid-lab` olarak bağlıdır.
- Kalıcı veri kökü: `HUMANOID_DATA_ROOT=/home/aksoy-msi/code/humanoid-lab-main/data`
  → container'da `/data`, `/cache`, `/outputs`. Modeller, venv'ler
  (`/opt/venvs`), Isaac cache'i, uv/HF cache'i ve koşu çıktıları buradadır.
- GPU: RTX 5090 Laptop 24 GB, sürücü 580.173.02. Kapı 1'de fizik CPU'da koşar,
  render GPU'dadır.
- Araç: Ubuntu 24.04, `/bin/bash`. Gömülü upstream kaynaklar `/opt/src/*`
  (isaaclab, sonic, isaac-groot) salt-okunur kabul edilir; değişiklik gerekiyorsa
  `patches/` altında ayrı patch olarak tutulur ve image katmanında uygulanır.

## Kurallar

- Tekrar indirme/derleme yapmadan önce kalıcı veri köküne bak. Model
  ağırlıkları, TensorRT engine'leri (`.trt`), venv'ler ve Isaac cache'i oradan
  gelir; image build'i ve engine üretimi tek seferliktir.
- Uzun süren işleri gereksiz tekrarlama: image katmanı ve veri kökü zaten
  önbellektir. Yeni bir bağımlılık veya kaynak değişikliği cache'i geçersiz
  kılıyorsa bunu açıkça söyle.
- Çalıştırılmadan geçti denmez. Bir davranış iddiası yalnız o koşunun çıktısıyla
  (komut, stdout, metrik, kanıt dosyası) desteklenir; doğrulanmayan kısım açıkça
  belirtilir.
- Uygulama çalıştırma komutları: `./setup.sh`, `./dev.sh`, `./doctor.sh`,
  `./scripts/smoke-test.sh`. GUI için `DISPLAY` çağıran terminalden gelir.
