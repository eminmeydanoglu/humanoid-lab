# GRAIL Replay — kısa uygulama planı

## Hedef

Tek komutla yerel GRAIL `pickup_table` kümesindeki herhangi bir geçerli motion’ı Isaac’te açmak:

```bash
./dev.sh grail-replay pickup_table__apple_0__000
```

Komut:

- doğru Dex3 G1 robotunu,
- motion’a ait object USD, texture ve scale’i,
- metadata’daki masa konumu/ölçüsünü,
- mevcut torso-mounted head camera’yı,
- gözle kontrol için dış kamerayı

tek Isaac oturumunda kurar ve kayıtlı robot + nesne yörüngesini **kinematik** oynatır. SONIC, DDS, controller, policy ve augmentation bu işin kapsamında değildir.

## Mevcut koddan kullanılacaklar

- GRAIL okuma/dönüşüm: `third_party/GRAIL/grail/visualization/prepare_vis_shard.py`
  - `convert_motion_lib_to_trajectories(...)`
  - joblib internal-key fallback
  - robot quaternion `xyzw → wxyz`
  - MuJoCo → Isaac body DOF sırası
  - `hand_dof_pos` eklenmesi
  - object pose/scale ve table pose
- Kinematik state yazma örneği: `third_party/GRAIL/grail/visualization/batch_render_replay.py`
  - `write_joint_state_to_sim`
  - `write_root_state_to_sim`
  - object USD xform güncellemesi
  - masa proxy’si ve bozuk material fallback’i
- Robot ve head camera tanımı: `configs/profiles/isaac-g1-dex3.json`
- Kamera sözleşmesi: `src/humanoid_lab/simulators/isaac/contracts.py::CameraSpec`
- Isaac başlatma/kapatma ve tek-process kilidi: mevcut `src/humanoid_lab/simulators/isaac/cli.py` + `dev.sh::run_isaac_g1`

`visualize_single.sh` doğrudan kullanılmayacak: yalnız dış kamera üretir, farklı conda varsayar ve kendi `AppLauncher` sürecini açar.

## Yapılacak değişiklikler

### 1. Replay servisi

Yeni dosya:

```text
src/humanoid_lab/simulators/isaac/replay.py
```

`ReplayService` şunları yapar:

1. Varsayılan veri kökünde sequence key’i çözer:
   `/data/datasets/grail/data/pickup_table`.
2. `robot`, `objects`, `meta` ve `object_usd` dosyalarının varlığını doğrular.
3. `convert_motion_lib_to_trajectories(..., motion_filter={key}, quat_convention="xyzw")` çağrısını geçici shard üzerinde çalıştırır ve tam bir trajectory üretildiğini doğrular. Bu helper yalnız `table_pos` taşıdığı için `table_size` ve `table_quat` gerekiyorsa aynı `meta/<key>.pkl` ayrıca okunur.
4. Mevcut `isaac-g1-dex3.json` profilinden robot asset’i ve head-camera calibration’ını yükler. Ayrı/kopya replay profili oluşturulmaz.
5. Tek `SimulationContext` kurar:
   - gravity kapalı,
   - actuator tracking yok,
   - bir source frame = bir render frame,
   - süre ve writer FPS doğrudan `trajectory["fps"]` değerinden gelir; `pickup_table` için 25 Hz.
6. Robot, object, table, head camera ve dış kamerayı aynı stage’de oluşturur.
7. Her frame’de robot root/joint state’i ile object/table transform’unu doğrudan yazar; `sim.forward()` + `sim.render()` sonrasında iki kamerayı okur.
8. GUI modunda dış viewport ve head-camera penceresini gösterir; headless modda iki MP4 üretir.

Replay mevcut dinamik `SimulatorService` içine eklenmez; onun physics/controller davranışı değişmez. Ortak kalan parçalar profile, contracts, Isaac launcher ve process lifecycle’dır.

### 2. Mevcut Isaac CLI’da replay seçimi

`src/humanoid_lab/simulators/isaac/cli.py` içine yalnız gerekli argümanlar eklenir:

```text
--replay SEQUENCE_KEY
--replay-data-root PATH       # opsiyonel
--replay-output-dir PATH      # opsiyonel
```

Aynı `AppLauncher` oluşturulduktan sonra:

- `--replay` varsa `ReplayService`,
- yoksa mevcut `SimulatorService`

çalışır. Böylece ikinci Kit/AppLauncher süreci ve ayrı cleanup yolu oluşmaz.

### 3. `grail-replay` komutu

`dev.sh` içine yeni dispatch eklenir:

```bash
./dev.sh isaac-g1 grail-replay <sequence-key> [--headless]
```

Bu yol mevcut `run_isaac_g1` fonksiyonunu ve `isaac-g1-dex3.json` profilini kullanır. Böylece aynı container environment, X11 kontrolü, Isaac process kilidi, pidfile ve teardown davranışı korunur.

## Kritik doğruluk kuralları

- Public release için quaternion convention açıkça `xyzw` seçilir; `auto` kullanılmaz.
- Robot articulation joint adları/sırası, trajectory’nin 29 body + 14 hand sırasıyla bir kez doğrulanır. Sıra eşleşmiyorsa positional write yapılmaz; isim tabanlı mapping kurulur.
- Mevcut profile `physics_dt=0.005` dinamik sim içindir. Replay frame süresi bundan alınmaz; motion’ın kendi `fps` alanından alınır.
- Head camera update period replay’de her render frame’i olacak şekilde ayarlanır; profile’ın dinamik render periyodu körlemesine kullanılmaz.
- Object USD yerinden referans edilir; texture relative path’lerinin bozulmaması için başka dizine kopyalanmaz.
- Object scale pozitif/finite olmalıdır.
- Masa için yalnız sabit `2.0 × 0.6 × 0.04` varsaymak yerine mevcutsa `meta.table_size` ve `table_quat` da korunur; mevcut `pickup_table` verisinde boyut sabit olsa bile loader genel kalır.
- Robot, object ve masa aynı origin shift’e tabi tutulur; relative geometry değişmez. Pickup-table için cosmetic scene yaw uygulanmaz.
- Eksik/bozuk dosya sessizce atlanmaz; komut açık hata ile non-zero çıkar.

## Çıktı

Varsayılan:

```text
/outputs/grail-replay/<sequence-key>/
├── external.mp4
├── head.mp4
└── manifest.json
```

`manifest.json` en az şunları içerir:

- sequence key ve kaynak yolları,
- frame sayısı ve FPS,
- trajectory DOF şekli,
- robot asset’i,
- object USD ve uygulanan scale,
- table pose/size,
- head/external camera çözünürlükleri,
- quaternion convention,
- sonuç ve hata bilgisi.

## Test ve kabul

### Otomatik testler

- Sequence-key çözümleme ve eksik dosya hataları.
- Joblib tek-inner-key fallback.
- `xyzw → wxyz`, DOF reorder ve 43-DOF doğrulaması.
- Object scale ve table metadata aktarımı.
- Aynı input için 250 trajectory frame’i → 250 video frame’i.
- Head-camera frame’lerinin klip boyunca değiştiğinin hash ile kontrolü.
- Replay modunda SONIC/controller import veya DDS başlangıcı olmadığının testi.

### Gerçek Isaac kabul koşusu

```bash
./dev.sh grail-replay pickup_table__apple_0__000 --headless
```

Geçmesi gerekenler:

- `250 frame @ 25 Hz`, `dof_pos=(250,43)`;
- external ve head MP4’lerin ikisi de 250 frame ve boş değil;
- robot düzgün pozda, sağ Dex3 el trajectory ile hareket ediyor;
- masa ve object doğru relative konumda;
- object USD/texture/scale doğru;
- object trajectory boyunca robot eliyle uyumlu hareket ediyor;
- head camera robotla beraber hareket ediyor ve değişen RGB üretiyor;
- süreç SONIC olmadan temiz kapanıyor.

Apple örneği geçtikten sonra en az bir texture’lı bottle/can ve bir texture’sız object ile aynı komut doğrulanır. Bu üç örnek geçince “herhangi bir geçerli `pickup_table` satırını açabilen replay yolu” tamamlanmış sayılır.
