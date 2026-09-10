# eski_kotu — arşivlenmiş eski implementasyon

Bu dizin, `en_iyi_plan.md` öncesi SONIC / GR00T / CloudWalk / F310 / ROMP
hattına ait dosyaları tek yerde toplar. Hiçbir dosya silinmedi: checkout'taki
izlenen dosyalar `git mv` ile taşındı, yalnız Raider'da bulunan ROMP dosyaları
da senkronizasyon öncesi stash'ten bu ağaca korunarak alındı.

## Neden burada

Plan §2.1 tek kanonik yol istiyor: yeni hat
`src/humanoid_lab/simulators/isaac/` altında yaşıyor ve buradaki hiçbir modülü
import etmiyor. Eski dosyalar aynı isim uzayında durduğu sürece
`run-sonic-isaac.py` ile `run-isaac-g1.py`, `tools/sonic_isaac_*` ile
`src/humanoid_lab/...` karışma riski taşıyordu.

## Dizin yapısı

Alt-ağacın iç düzeni olduğu gibi korundu (`tools/`, `scripts/`, `tests/`,
`configs/`, `patches/`). Bu sayede taşınan eski testlerin `parents[1] / "tools"`
ve `parents[1] / "scripts" / ...` göreli yolları arşiv içinde hâlâ çözülüyor.

```
archive/eski_kotu/
├── tools/      sonic_isaac_*, cloudwalk_*, isaac_g1_free_base, f310_*, sonic_isolated_lifecycle
├── scripts/    run-sonic-isaac, run-cloudwalk-*, sonic-isaac-*, build/run-sonic-isolated-*,
│               run-release-acceptance, verify-release-evidence, make-dataset-preview, f310 tunnel
├── tests/      test_sonic_isaac_*, test_cloudwalk_*, test_sonic_isolated_lifecycle,
│               test_dev_sh_sonic_isaac, test_release_evidence, test_f310_unitree_bridge,
│               native/*.cpp, fixtures/*.json
├── configs/    cloudwalk_scene.json, cloudwalk_wood.png,
│               sonic_isaac_deploy.json, sonic_v1_1_safe_standing_reference.json
├── patches/    sonic-f310-bridge-*.patch, sonic-deploy-sim-domain.patch
├── romp_teleop/ eski ROMP poz akışı, SONIC köprüsü, tanı araçları ve örnek NPZ'ler
├── bug_repor.md, cloud.md, groot-n17-finetuning.md
└── README.md
```

## Bilerek yerinde bırakılanlar

Bu dosyalar eski hatta ait olsa da paylaşılan altyapıya bağlı oldukları için
taşınmadı:

| Yol | Bağlı olduğu yer |
| --- | --- |
| `patches/sonic-sim-dds-isolation.patch` | `Dockerfile:103` imaja kopyalıyor |
| `scripts/smoke-test.sh` | `Dockerfile:104`, `setup.sh` |
| `scripts/fetch-groot-demo-data.sh` | `Dockerfile:104` |
| `scripts/groot-finetune-smoke.sh` | `Dockerfile:104` |
| `scripts/validate-groot-demo-fixture.sh` | `Dockerfile:59` |
| `containers/` | `compose.yaml` bind-mount, `CYCLONEDDS_URI`, `Dockerfile:58,102` |
| `locks/` | `Dockerfile:101`, `scripts/render-lock-env.py` |
| `scripts/render-lock-env.py` | `setup.sh`, `ci-lint.sh`, `verify-pins.sh`, `fetch-models.sh` |
| `scripts/{ci-lint,verify-pins,doctor-report}.{sh,py}` | repo düzeyi kalite kapıları |
| `scripts/fetch-models.sh`, `scripts/record-provenance.py` | model indirme + provenance |
| `scripts/install-isaac-webrtc-client.sh` | Isaac düzeyi görüntüleme, SONIC/GR00T'a özel değil |
| `tests/test_{fetch_groot_demo_data,groot_finetune_smoke,validate_groot_demo_fixture}.sh` | yukarıdaki imaja gömülü script'leri sınıyor |
| `README.md` | repo giriş dokümanı |

## Bilinen kırıklar

Taşıma sonrası bu arşiv içi testler artık çalışmaz, çünkü kökte kalan yollara
bakıyorlar:

- `tests/test_dev_sh_sonic_isaac.py` → `archive/eski_kotu/dev.sh` yok (dev.sh kökte).
- `tests/test_sonic_isaac_contract.py`, `tests/test_sonic_isaac_session.py`,
  `tests/test_sonic_isaac_session_cli.py` → `parents[1] / "containers" / "cyclonedds-sim.xml`.

`dev.sh` içindeki `sonic-sim`, `sonic-isaac`, `cloudwalk-sim`,
`cloudwalk-controller`, `release-acceptance` girdileri de arşivlenen script'leri
gösteriyor. `dev.sh`'in yeni hattı ilgilendiren tek girdisi `isaac-g1` ve o
bloğa dokunulmadı. Eski girdilerin temizliği plan §11.6 gereği ayrı bir adımdır.

## Kaldırma

Plan §11.6: yeni karşılık kanıtlandığında ilgili legacy yol kaldırılır. Bu dizin
o noktada tek commit ile silinebilir.
