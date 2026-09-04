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
./dev.sh smoke      # container icinde smoke testler
./dev.sh rebuild    # image'i ayni lock ile yeniden build et
./dev.sh stop       # container'i durdur (veriler kalir)
```

## 3. Modeller (bir kez)

```bash
./dev.sh hf-login       # HF token host cache'ine yazilir (image'e/.env'e girmaz)
./dev.sh fetch-models   # pinli revision'lar + MODEL_PROVENANCE.json (sha256)
```

Not: GR00T N1.7 backbone'u `nvidia/Cosmos-Reason2-2B` gated — once HF'de lisansi kabul et.

## 4. Dogrulama / tanalama

```bash
./doctor.sh                 # host'tan tam rapor (humanoid-lab-data/diagnostics/)
./scripts/verify-pins.sh    # lock pin'lerini upstream'e karsi dogrula
./scripts/ci-lint.sh        # bash/python/yaml statik kontroller
```

## 5. Container icinde environment secicileri

```bash
use-isaac-sonic | use-sonic-sim | use-groot | use-none
show-env            # aktif env + python
```

Global python alias yoktur; PATH yalnizca secilen env'e yonelir (prompt'ta gorunur).

## Surum kilidi

Tum versiyonlarin tek kaynagi `versions.lock.yaml`; `scripts/render-lock-env.py` bunu `.generated/versions.env`'e render eder, Compose/Dockerfile/scriptler yalnizca oradan okur. Dogrulanmamis pinler (`required: false`) setup'i blocklar. Kalici tum veri host bind-mount'larindadir (`~/humanoid-lab-data/`).
