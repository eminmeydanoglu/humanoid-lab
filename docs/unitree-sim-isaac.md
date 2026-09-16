# unitree_sim_isaaclab — Isaac Lab task suite

Bu yol, [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
task paketini (**ManagerBasedRLEnv** + `gym.register`) mevcut `humanoid-lab-dev`
container'ına entegre eder. Isaac Sim 5.1.0, IsaacLab 2.3.2 ve Unitree DDS
altyapısı (`unitree_sdk2py` + cyclonedds 0.10.2) isaac-sonic ile **paylaşılır**;
SONIC kontrolcüsünün kendisi bu task'lar tarafından kullanılmaz (paketin kendi
`dds/`, `layeredcontrol/`, `action_provider/` katmanı vardır).

## Mevcut ortamlara dokunulmaz

- Kaynak `/opt/src/unitree-sim` image katmanıdır (pin `versions.lock.yaml`).
- Ortam kendi venv'inde yaşar: `/opt/venvs/unitree-sim`. `isaac-sonic`,
  `sonic-sim`, `groot-n17`, `psi0` **değiştirilmez**.
- Yeni `locks/unitree-sim` girdisi `locks/isaac-sonic`'ten türetilir: torch 2.7.0,
  torchvision 0.22.0, numpy 1.26.4, scipy 1.15.3 ve isaaclab paketleri aynı
  pinlerdedir. `gear_sonic` çıkarılır; upstream `requirements.txt` paketleri
  (rerun-sdk, pyzmq, logging_mp, onnxruntime, onnx, pynput, aiortc, aiohttp)
  eklenir. teleimager editable ve `--no-deps` kurulur; kendi runtime
  bağımlılıkları (opencv-python, pyyaml) lock'ta açıkça belirtilir.

## Kurulum

```bash
./setup.sh --verify-digests   # NGC digest'i lock'a yazar (yeni image katmanı için)
./dev.sh rebuild              # /opt/src/unitree-sim katmanıyla image'ı yeniden kur
./dev.sh sync                 # unitree-sim venv'ini kurar + assets symlink'ini tazeler
./dev.sh fetch-unitree-sim-assets   # ~1 GiB+ HF dataset -> data root (ağ gerekir)
```

Asset'ler Hugging Face **dataset** reposundan (`unitreerobotics/unitree_sim_isaaclab_usds`,
`assets.zip`) gelir; image'a girmez. `$HUMANOID_DATA_ROOT/models/unitree-sim-assets/`
altına açılır, `ASSET_PROVENANCE.json` yazılır ve `/opt/src/unitree-sim/assets`
buraya symlink edilir (upstream `.gitignore` `assets/`'ı yoksayar, kaynak katmanı
bit düzeyinde aynı kalır).

## Çalıştırma

```bash
# GUI (X11 gerekir)
./dev.sh unitree-sim --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129 --device cuda

# GUI'siz (offscreen render)
./dev.sh unitree-sim --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129 --device cuda --headless
```

`sim_main.py`'nin kendi `--duration`'u yoktur; süreli koşu için dıştan
`timeout` kullanın. Task listesi için `sim_main.py` içindeki örnek komutlara
bakın (`Isaac-PickPlace/Stack/Move-*`, `--robot_type g129|h1_2`). Eşzamanlı
ikinci bir koşu `/tmp/humanoid-lab-unitree-sim.lock` ile engellenir.

Notlar:
- WebRTC streaming bu host'ta kapalıdır; `--headless` (offscreen) kullanın.
- `sim_main.py` kanal 1 DDS kullanır (gerçek robot protokolü). Mevcut SONIC
  izolasyonu domain 42/loopback'tir; ikisini aynı anda çalıştırırken DDS
  çakışmasını doğrulayın.

## Doğrulama

```bash
./dev.sh smoke        # unitree-sim import + AppLauncher yolu + assets kontrolü
./scripts/verify-pins.sh --strict
```

Ölçülen sonuçlar (Raider, RTX 5090):

| Kontrol | Sonuç |
|---|---|
| `./dev.sh smoke` | PASS=22 WARN=0 BLOCKED=0 FAIL=0 |
| `verify-pins.sh` | PASS (unitree commit + HF dataset revision doğrulandı) |
| Asset indirme | `assets.zip` 1245 MiB → 423 dosya / 1692 MiB |
| Headless koşu (`--headless --enable_cameras --device cuda`) | Sahne 3.1 s'de kuruldu, kameralar hazır, DDS 4 publisher (`lowstate`, `dex3`, `sim_state`, `rewards`), fizik döngüsü ~25 Hz / ~40 ms |

Koşu sırasında `sim_main.py` sürekli çalışır; süreli test için `timeout -s INT N python sim_main.py ...` kullanılır (SIGINT düzgün kapanış yapar). WebRTC sertifikaları `bootstrap-venvs.sh` tarafından `~/.config/xr_teleoperate/` altına bir kerelik üretilir; streaming bu host'ta kapalıdır, kameralar ZMQ ile yayınlanır.

## Bilinen sınırlar

- Yeni lock, isaac-sonic'e göre 13 paylaşılan pakette **patch** sürüm farkı taşır (ör. anyio 4.15.0→4.15.1, fsspec 2026.6.0→2026.7.0, hydra-core 1.3.2→1.3.7). Yalnız yeni `unitree-sim` ortamını etkiler; diğer ortamlar kendi lock'larıyla kuruludur.
- IsaacLab 2.3.2 ile upstream task API'si arasında sürüklenme çıkarsa düzeltme `patches/` altında ayrı patch olarak tutulmalıdır.
- **Worktree ve container:** container bir git worktree'yi mount ediyorsa, worktree'nin `.git` işaretçisi ortak depoya (mount dışına) gittiği için `sync_psi0` git meta verisini okuyamaz. `bootstrap-venvs.sh` bu durumda Psi0'ı uyarıyla atlar ve mevcut venv'i korur; Psi0'ı tazelemek için container'ı ana checkout'u mount ederek çalıştırın (feature branch ana dala alındıktan sonra normal durum budur).
- SONIC kaynağındaki `unitree_sdk2py/.../utils/lib/crc_*.so` git-lfs dosyasıdır; image katmanı bunları da çeker (SONIC `--disable-crc-check` kullanır, unitree DDS'i kullanmaz).

