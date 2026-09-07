# GR00T N1.7 fine-tuning smoke akışı

Bu depo GR00T N1.7 General Availability (GA) kaynak kodunu, modeli ve Python ortamını zaten sabit revizyonlarla sağlar. Bu rehber, tam eğitim yerine veri ve eğitim hattını güvenle doğrulayan **iki optimizer adımlı** isteğe bağlı smoke testini anlatır.

> Bu checkout'ta Docker, GPU, model indirme ve eğitim çalıştırılmaz. Bunları Raider'daki `/home/aksoy-msi/code/humanoid-lab` checkout'ında `ssh raider` ile yürütün.

## 1. Depodaki pinler ve bileşenler

| Bileşen | Bu depodaki kaynak |
|---|---|
| GR00T kaynak kodu | NVIDIA Isaac-GR00T `1a1837f20538b7d7e21f977a11a5aee14f99803c` |
| Fine-tuning entrypoint | `gr00t/experiment/launch_finetune.py` |
| Python environment | `groot-n17`: Python 3.12, PyTorch 2.9.0, CUDA 12.8, Flash Attention 2.8.3 |
| Ana model | `nvidia/GR00T-N1.7-3B` revizyon `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos backbone | `nvidia/Cosmos-Reason2-2B` revizyon `9ce19a195e423419c349abfc86fd07178b230561` |
| Örnek veri | Pinli Isaac-GR00T `demo_data/cube_to_bowl_5`, mounted hedef `/data/datasets/groot/cube_to_bowl_5` |
| Örnek embodiment | `NEW_EMBODIMENT`, `examples/SO100/so100_config.py` |

`versions.lock.yaml` bu değerlerin tek kaynağıdır. Yeni bir GR00T repository, ikinci virtual environment veya ayrı bir lock dosyası oluşturulmaz. Docker image dataset içermez. Resmî örnek veri hosttaki kalıcı mount altında `/data/datasets/groot/cube_to_bowl_5` konumunda tutulur; indirme komutu pinli revision, `meta/modality.json`, Parquet/MP4 varlığı ve Git LFS pointer durumunu doğrular.

Upstream lock x86_64 için DeepSpeed 0.17.6 içerir. Isaac Sim runtime image'ında CUDA toolkit (`nvcc`) bulunmadığından Accelerate, tek-GPU launcher DeepSpeed seçmemesine rağmen bu paketi import ederken durur. `bootstrap-venvs.sh`, lock sync'inden sonra yalnız bu kullanılmayan paketi kaldırır. Bu environment DeepSpeed dağıtık eğitim konfigürasyonlarını desteklemez; smoke komutu zaten tek GPU düz `python` yolunu kullanır.

Resmî temel kaynaklar:

- [N1.7 GA README](https://github.com/NVIDIA/Isaac-GR00T/blob/1a1837f20538b7d7e21f977a11a5aee14f99803c/README.md)
- [Fine-tuning launcher](https://github.com/NVIDIA/Isaac-GR00T/blob/1a1837f20538b7d7e21f977a11a5aee14f99803c/gr00t/experiment/launch_finetune.py)
- [Fine-tuning configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/1a1837f20538b7d7e21f977a11a5aee14f99803c/gr00t/configs/finetune_config.py)
- [Yeni embodiment rehberi](https://github.com/NVIDIA/Isaac-GR00T/blob/1a1837f20538b7d7e21f977a11a5aee14f99803c/getting_started/finetune_new_embodiment.md)
- [NVIDIA minimal fine-tune testi](https://github.com/NVIDIA/Isaac-GR00T/blob/main/tests/getting_started/test_finetune_new_embodiment_md.py)
- [Simülasyon verisiyle fine-tuning akışı](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/groot-fine-tuning-sim.html)

## 2. Kurulum, model erişimi ve önkoşullar

Image güncellendiğinde Raider'da önce rebuild ve environment sync çalıştırın. Ardından HF kimlik doğrulaması ve pinli model indirmesi yapılır:

```bash
ssh raider
cd /home/aksoy-msi/code/humanoid-lab
./dev.sh rebuild
./dev.sh sync
./dev.sh hf-login
./dev.sh fetch-models
./dev.sh fetch-groot-demo-data
./dev.sh doctor
./dev.sh smoke
```

`fetch-models` her başarılı model indirmesi için `/data/models/<model>/MODEL_PROVENANCE.json` yazar; dosya repo, revision ve bütün indirilen dosyaların SHA-256 bilgisini içerir. Fine-tuning smoke bu dosyada ana modelin repo ve revision değerlerini lock ile karşılaştırır; yalnız `config.json` bulunması yeterli değildir.

`fetch-groot-demo-data`, yalnız `cube_to_bowl_5` içeriğini geçici sparse checkout ile indirip `/data/datasets/groot/cube_to_bowl_5` altına taşır. Böylece image katmanında ikinci bir dataset kopyası bulunmaz. Dataset kökündeki `DATASET_PROVENANCE.json`, resmî repo ve pinli GA commit bilgisini kaydeder; smoke bunu da doğrular.

GR00T N1.7 ağırlıklarını indirmek açık olabilir; ancak Cosmos backbone erişimi Hugging Face lisans kabulü gerektirebilir. `GatedRepoError`, HTTP 401 veya 403 bir eğitim sonucu değildir: Cosmos model sayfasında erişimi kabul edin, `./dev.sh hf-login` çalıştırın ve yeniden deneyin.

NVIDIA gerçek fine-tuning için tek GPUda en az 40 GB VRAM belirtir. Raider'ın RTX 5090 Laptop GPU'su bu eşiğin altında beklenir. Bu nedenle weightsiz `pipeline` testi zorunlu ilk doğrulamadır; gerçek ağırlıklı test, donanım eşiği nedeniyle varsayılan olarak açıkça `BLOCKED` olur.

## 3. Eğitim entrypoint'i ve smoke parametreleri

Resmî entrypoint `train.py` değil, aşağıdaki dosyadır:

```text
gr00t/experiment/launch_finetune.py
```

Depodaki `scripts/groot-finetune-smoke.sh` bunu doğrudan çağırır. Tek GPU için `torchrun` veya DataParallel değil, `groot-n17` environment içinden düz `python` kullanılır.

Smoke için ortak eğitim parametreleri şöyledir:

```text
--base-model-path /data/models/groot_n17_base
--dataset-path /data/datasets/groot/cube_to_bowl_5
--embodiment-tag NEW_EMBODIMENT
--modality-config-path examples/SO100/so100_config.py
--num-gpus 1
--max-steps 2
--save-steps 2
--save-total-limit 1
--global-batch-size 2
--dataloader-num-workers 0
--shard-size 64
--num-shards-per-epoch 1
```

`pipeline` moda eklenen `--skip-weight-loading`, model ağırlıklarını indirme/yükleme yolunu kasıtlı olarak atlar. Böylece veri yükleme, modality kaydı, forward/backward, optimizer, checkpoint ve processor yazma hattı düşük maliyetle sınanır. Bu mod pretrained model uyumluluğunu kanıtlamaz; onun için `pretrained` mod kullanılır.

## 4. GR00T veri sözleşmesi ve embodiment yapılandırması

GR00T, [LeRobot v2](https://github.com/huggingface/lerobot) veri yerleşimini kullanır; buna ek olarak dataset kökünde şu dosya gerekir:

```text
meta/modality.json
```

Bu dosya state, action, image ve dil/embodiment alanlarının GR00T modality tanımını bağlar. İlk smoke için mounted dataset store içindeki resmî `/data/datasets/groot/cube_to_bowl_5` kullanılmalıdır; Parquet veya video dosyalarının Git LFS pointer olması geçerli veri değildir.

Her datasetin embodiment etiketi, dataset modality metadata'sı ve `--modality-config-path` birlikte uyumlu olmalıdır. Bu smoke'un sabit üçlüsü:

```text
NEW_EMBODIMENT
examples/SO100/so100_config.py
/data/datasets/groot/cube_to_bowl_5
```

Hazır embodiment ve modality yapılandırmaları kaynak checkout'ın `examples/` dizininde ve `EmbodimentTag` tanımında bulunur. Hazır bir yapılandırmayı kullanırken o yapılandırmanın desteklediği tag ve modality alanlarını değiştirmeyin. Yeni robot için resmî [custom embodiment rehberindeki](https://github.com/NVIDIA/Isaac-GR00T/blob/1a1837f20538b7d7e21f977a11a5aee14f99803c/getting_started/finetune_new_embodiment.md) gibi bir Python config oluşturun; bu dosya `NEW_EMBODIMENT` için modality config'i kaydetmeli ve veri alanlarıyla bire bir eşleşmelidir.

### Veri dönüştürme ve statistics

- **LeRobot v2:** `meta/modality.json` ekleyip doğru modality config ile doğrudan kullanın.
- **LeRobot v3:** Resmî `scripts/lerobot_conversion/convert_v3_to_v2.py` ile önce v2'ye dönüştürün, sonra GR00T modality metadata'sını doğrulayın.
- **HDF5, simülasyon veya özel demonstration formatı:** Önce LeRobot/GR00T yerleşimine dönüştürün; HDF5 dosyasını launcher'a doğrudan vermeyin. NVIDIA'nın [simülasyon akışı](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/groot-fine-tuning-sim.html) bu dönüşümün genel biçimini gösterir.
- **Action horizon veya `delta_indices` değişirse:** `gr00t/data/stats.py` ile statistics'i aynı modality config ve action tanımıyla yeniden üretin. Aksi halde normalizasyon veya index hatası beklenir.

## 5. Çalıştırma

Komut GPU/data maliyetli olduğu için `setup.sh` ve normal `./dev.sh smoke` tarafından çağrılmaz.

```bash
# Önce iki-adım, pretrained ağırlık yüklemeyen eğitim hattı doğrulaması.
./dev.sh groot-finetune-smoke pipeline

# 40 GiB veya üstü GPUda gerçek ağırlıklı iki-adım doğrulaması.
./dev.sh groot-finetune-smoke pretrained

# Eşik altı GPUda yalnızca açık best-effort tercihiyle denenir.
./dev.sh groot-finetune-smoke pretrained --allow-low-vram

# En yeni başarılı pipeline checkpoint-2 için kısa open-loop kontrolü.
./dev.sh groot-finetune-smoke eval
```

Betik varsayılan olarak çıktıyı şu biçimde yeni bir dizine yazar:

```text
/outputs/gr00t-n17-finetune-smoke/<pipeline|pretrained|eval>-<UTC timestamp>/
```

Aynı çıktı konumunu otomasyon için açıkça seçmek isterseniz hedef dizin önceden mevcut olmamalıdır:

```bash
./dev.sh groot-finetune-smoke pipeline \
  --output-dir /outputs/gr00t-n17-finetune-smoke/pipeline-manual-001
```

`eval`, en yeni `pipeline-*/checkpoint-2` dizinini bulur. Farklı bir checkpoint için:

```bash
./dev.sh groot-finetune-smoke eval \
  --checkpoint-dir /outputs/gr00t-n17-finetune-smoke/pipeline-<timestamp>/checkpoint-2
```

## 6. Çıktılar ve başarı ölçütü

Her çalıştırma `run.log`, `command.sh`, `source_revision`, `model_provenance.json`, `base_model_config.json`, GPU bilgisi ve VRAM ölçümünü saklar. Eğitimin başarılı sayılması için şunların tümü gerekir:

```text
checkpoint-2/
checkpoint-2/experiment_cfg/
checkpoint-2/processor/
checkpoint-2 içindeki model/config artifact'leri
```

Ek olarak launcher exit code'u `0` olmalı ve log iki adıma ulaştığını göstermelidir. İki adımda loss düşmesi veya robotun görevi öğrenmesi başarı ölçütü **değildir**.

`eval` resmi `gr00t/eval/open_loop_eval.py` aracını beş open-loop adımla çalıştırır. Finite MSE, finite MAE ve en az bir ground-truth/prediction plot artifact'i arar; değerler için kalite eşiği uygulanmaz.

## 7. Sık hatalar

| Belirti | Neden ve çözüm |
|---|---|
| `BLOCKED` ve 40 GiB VRAM mesajı | Pretrained mod için GPU bellek eşiği karşılanmıyor. Weightsiz `pipeline` çalıştırın; bilinçli best-effort deneme için `--allow-low-vram` ekleyin. |
| `GatedRepoError`, 401 veya 403 | Cosmos erişimini Hugging Face'te kabul edin ve `./dev.sh hf-login` çalıştırın. |
| `MODEL_PROVENANCE.json` veya revision hatası | `./dev.sh fetch-models` ile pinli modeli yeniden indirin; rastgele yerel checkpoint kullanmayın. |
| Eksik Parquet/MP4, LFS pointer veya dataset provenance hatası | `./dev.sh fetch-groot-demo-data --force` ile mounted dataset kopyasını yeniden oluşturun. |
| `No modality config registered` | Dataset tag, `meta/modality.json` ve modality config eşleşmiyor. İlk testte sabit SO100/`NEW_EMBODIMENT` üçlüsünü kullanın. |
| Statistics/index hatası | Action alanını veya `delta_indices` değerini değiştirdiniz; statistics'i yeniden üretin. |
| Batch/GPU bölünebilirlik hatası | Smoke için `--global-batch-size 2 --num-gpus 1` değerlerini koruyun. |
| `--action-horizon` eval'da reddediliyor | N1.7 evaluator `--execution-horizon` kullanır. |

## 8. N1.7 sürüm notu: README ve checkpoint config çelişkisi

N1.7 README, N1.6'ya göre varsayılan selected layer değerinin 16'dan 12'ye ve diffusion layer sayısının 32'den 16'ya indirildiğini yazar. Buna karşın pinli `nvidia/GR00T-N1.7-3B` checkpoint'inin `config.json` dosyasında `select_layer = 16` ve `diffusion_model_cfg.num_layers = 32` bildirildiği kamuya açık [NVIDIA issue #755](https://github.com/NVIDIA/Isaac-GR00T/issues/755)'te açıklanmıştır.

Bu yüzden smoke script README değerlerini elle zorlamaz: pinli checkpoint'in `config.json` dosyasını otorite kabul eder, o dosyayı ve model revision'ını çıktıya kaydeder. Pretrained ağırlıklarla eğitimde config'i manuel olarak 12/16'ya çevirmek checkpoint uyumsuzluğuna yol açabilir.
