🔎 humanoid-lab Audit Sentezi — Nihai Rapor

 Hedef: raider:/home/aksoy-msi/code/humanoid-lab · main @ 5b93f47 (22 kirli dosya, kullanıcıya ait WIP) · container humanoid-lab-dev (imaj
 humanoid-lab/dev:5.1.0) · 2026-09-04
 Kaynak kanıt: 8 dalga raporu (baseline, facts, static, isaac, docker, ros, mujoco, groot) — yinelemeler birleştirildi.

 ────────────────────────────────────────────────────────────────────────────────

 1) Genel Sonuç

 Karar: SİSTEM SAĞLIKLI — kullanılabilir durumda. Temel zincirlerin tamamı (kurulum→imaj→venv→env seçici→import; Isaac headless runtime; MuJoCo G1; GR00T
 metadata; kalıcılık/izin; güvenlik duruşu) uçtan uca doğrulandı. 2 gerçek FAIL var, ikisi de dar kapsamlı ve aksiyonlanabilir (ölü lint script'i; bozuk mesh
 kopyası). P1 düzeyinde 2 bulgu (DDS domain çakışması riski; smoke-test'in GR00T adımının koşulsuz FAIL etmesi) karar bekliyor.

 ┌─────────┬───────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Sayaç   │ Değer                 │ Not                                                                                                                    │
 ├─────────┼───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ PASS    │ 39 benzersiz          │ Dalga raporlarında 83 işaret; yinelenen kontroller (SSH ×6, GPU ×6, mount probe ×6, secret ×3, fingerprint ×2 …)       │
 │         │ doğrulama             │ tekilleştirildi                                                                                                        │
 ├─────────┼───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ FAIL    │ 2                     │ B-1 (ci-lint.sh committed bug), B-7 (g1-mujoco mesh stub)                                                              │
 ├─────────┼───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ RISK    │ 5 kanıtlı             │ B-2, B-12, B-16, B-17, B-18 — tümü kanıtla destekli; hiçbiri gerçekleşen olay değil, yapılandırma/izin kanıtlı risk    │
 ├─────────┼───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ NOT_RUN │ 9 kalem               │ Aşağıda; hiçbiri FAIL değil                                                                                            │
 └─────────┴───────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 Test edilenler (özet): host/container GPU+torch/CUDA, 22 bind-mount sözleşmesi, UID/GID/group zinciri, yazma/RO probe'ları (host 5/5, container 11/11 + 14
 Kit cache kökü), venv fingerprint 3/3, lock zinciri (versions.lock.yaml → deterministik render → imaj → container rev-parse → upstream verify-pins: 3 commit
 + 3 revision canlı doğrulandı), 3 venv'in import setleri, Isaac Sim 5.1.0 headless sim döngüsü (0 Error), G1 MuJoCo modeli + 500 adım rollout + EGL render,
 GR00T config/processor/shard-metadata (ağırlıksız) + flash_attn sm_120 forward, statik kalite (bash -n, py_compile, YAML, git diff --check, compose config),
 secret hijyeni, ROS/DDS aktif-olmama duruşu.

 Hariçler (NOT_RUN — kapsam/kural):
 1. ./dev.sh sync ve ./dev.sh rebuild — kullanıcı isteğiyle bilerek atlandı → NOT_RUN, FAIL değil
 2. doctor.sh / smoke-test.sh tam koşumu (yan etki kararı)
 3. GR00T ağırlık yükleme / inference / TRT engine build
 4. Model indirme (fetch-models, hf-login) — yasak; gated Cosmos lisans uyarısı by-design
 5. Fiziksel robot bağlantısı / ROS publish / ros2 discovery testleri — host-net + domain-0 riski nedeniyle yasak
 6. H1 runtime koşusu (yalnız statik inceleme)
 7. uv lock --check (kirli lock'ların runtime doğrulaması)
 8. X11 bağlantı testi
 9. docker compose config doğrulama koşumu (statik okuma yapıldı)

 İşaret uzlaştırması (birleştirme notları):
 - GROOT dalgası /data kökü yazma testini FAIL işaretledi; BASELINE ve DOCKER aynı durumu tasarım gereği beklenen red olarak işaretledi → gerçek FAIL
   sayılmaz (tasarım notu İ6).
 - verify-pins: FACTS NOT_RUN dedi, STATIC koşturup PASS aldı → birleştirildi: koşuldu, upstream doğrulandı.
 - Cosmos-Reason2-2B: BASELINE "hf-cache'te mevcut (4.6 GB)" vs GROOT "yerelde yok" → çelişki çözülemedi, sonraki testlere alındı.
 - Mount sayımı dalgalarda 19/22/23 olarak farklı; tutarlı olan: compose'daki 22 bind-mount container'da birebir (inspect toplam sayımı kimlik/dizine göre
   değişiyor).

 ────────────────────────────────────────────────────────────────────────────────

 2) Gerçek Bulgular

 ### B-1 · P0

 - Bileşen: scripts/ci-lint.sh
 - Başlık: find, ./data'yı prune etmediği için script set -e altında ilk adımda RC=1 ile ölüyor
 - Repro: timeout 300 ./scripts/ci-lint.sh
 - Beklenen: bash -n / shellcheck / python-yaml / determinizm / lock bölümleri koşsun, RC∈{0,2}
 - Gerçek: tek çıktı find: './data/isaac-cache/kit-data/documents/Kit/shared/screenshots': Permission denied; sıfır kontrol çalıştı, RC=1
 - Kanıt: mode-000 dizin (stat → mode 000, owner aksoy-msi); SCRIPTS=$(find …) ataması set -e altında script'i bitiriyor; HEAD'deki sürümde aynı satır →
   committed bug, kirli ağaçtan değil
 - Kök neden: prune listesinde yalnız ./.git ve ./.generated; ./data (venvs/uv-cache/isaac-cache) taranıyor; mode-000 dizin find'a RC=2 döndürüyor. Ayrıca
   80+ vendored script (tqdm/wandb/IsaacLab sdist) yanlışlıkla lint kapsamına giriyor
 - Minimal düzeltme: find'e -path ./data -prune ekle + || true ile bağla (veya sabit script listesi)
 - Gerekli karar: Tek satırlık düzeltmenin main'e alınması (davranış daraltıcı) — kullanıcı kararı

 ### B-2 · P1 (kanıtlı RISK)

 - Bileşen: compose.yaml (dev) + sonic-sim venv + gear_sonic mujoco_sim köprüsü
 - Başlık: Sim köprüsü gerçek robotla aynı DDS domain'inde: host network + DOMAIN_ID 0; tek komutla G1 kanallarına erişebilecek yapıda
 - Repro: docker exec humanoid-lab-dev /opt/venvs/sonic-sim/bin/python -m gear_sonic.scripts.run_sim_loop (başlatılmadı; risk statik + config kanıtlı)
 - Beklenen: Sim trafiği gerçek robot ağından izole (farklı domain / kısıtlı interface) veya başlatma öncesi açık gate
 - Gerçek: Container host network'te; ROS_DOMAIN_ID=0 config ile aynı; INTERFACE boşsa multicast tüm UP arayüzlere açılır; robot NIC (enp130s0) DOWN ama
   Wi-Fi UP
 - Kanıt: base_sim.py:564: ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"]); bridge: ChannelPublisher("rt/lowstate", …)
   (+odostate/secondary_imu/dex3/wireless); compose network_mode: host, ROS_DOMAIN_ID=0; WBC config'leri INTERFACE: "lo" sabit (o yol izole); hiçbir
   publisher başlatılmadı, pgrep temiz
 - Kök neden: Host network zorunluluğu (Isaac/robot erişimi) + varsayılan domain 0'ın değiştirilmemesi; sim köprüsünde izolasyon config'e bırakılmış, gate
   yok
 - Minimal düzeltme: Sim komutlarında CYCLONEDDS_URI ile interface'i lo'ya sabitle; gerçek-robot oturumlarında ROS_DOMAIN_ID'yi 0'dan farklıya zorla (ör.
   42); run_sim_loop'ta INTERFACE zorunlu parametre olsun
 - Gerekli karar: Sim test domain'i (ör. 42) + interface kilidi (lo) kabulü; robot NIC/IP belirlenince direct-dds-test.sh'a aynı değerlerin yazılması

 ### B-3 · P1

 - Bileşen: scripts/smoke-test.sh (satır 129 bloğu) / groot-n17
 - Başlık: GR00T metadata testi import gr00t.model eksikliği nedeniyle her zaman FAIL
 - Repro: smoke-test'in heredoc'u: AutoConfig.from_pretrained(ckpt, trust_remote_code=True) (gr00t importu olmadan)
 - Beklenen: PASS — registry ile config yüklenir
 - Gerçek: ValueError: … model type Gr00tN1d7 … Transformers does not recognize
 - Kanıt: import'suz FAIL; import gr00t.model eklenince OK (model_type=Gr00tN1d7, class=Gr00tN1d7Config); smoke-test.sh:130
 - Kök neden: Gr00tN1d7Config, transformers Auto mapping'ine yalnız gr00t.model import edilince kaydoluyor
 - Minimal düzeltme: Heredoc'a tek satır: import gr00t.model
 - Gerekli karar: Dosya değişikliği bu auditte yasaktı — kullanıcı onayı ile 1 satırlık yama

 ### B-4 · P2

 - Bileşen: README.md
 - Başlık: Kalıcılık yolu dokümanı gerçeği yansıtmıyor (~/humanoid-lab-data/ ↔ gerçek <repo>/data)
 - Repro: grep -n humanoid-lab-data README.md ↔ .env HUMANOID_DATA_ROOT
 - Beklenen: Doküman dünya gerçeğiyle aynı path
 - Gerçek: README.md:109,125,127,135 ~/humanoid-lab-data/...; .env + setup.sh HUMANOID_DATA_ROOT=$ROOT/data; mount source'ları .../humanoid-lab/data/*
 - Kanıt: yukarıdaki grep/mount kayıtları
 - Kök neden: setup.sh default'u değişmiş, README eski anlatımı taşıyor
 - Minimal düzeltme: README'de "kalıcı veri HUMANOID_DATA_ROOT (default: repo içi data/)" anlatımı; mutlak ~/humanoid-lab-data ifadesi kaldırılsın
 - Gerekli karar: Doküman düzeltmesi kullanıcıya (kirli tree onun)

 ### B-5 · P2

 - Bileşen: README + scripts/fetch-models.sh + versions.lock.yaml
 - Başlık: h1_locomotion için policy indirme adımının repoda karşılığı yok
 - Repro: grep -rn h1 scripts/ versions.lock.yaml
 - Beklenen: README adımı → pinli policy/indirme komutu
 - Gerçek: fetch-models yalnız lock'taki 3 modeli indirir (sonic/groot/cosmos); H1 policy pinli değil, komut yok; h1_locomotion.py etkileşimli RSL_RL
   OnPolicyRunner demosu
 - Kanıt: grep boş (yalnız demo scripti); H1'e özgü arg/env kısıtı dev.sh/compose/doctor'da tanımsız
 - Kök neden: Policy henüz lock'a/dokümana eklenmemiş
 - Minimal düzeltme: README'ye policy kaynağı (repo/revision) ekle veya "henüz sağlanmadı" notu düş
 - Gerekli karar: Pin ekleme kararı kullanıcıda

 ### B-6 · P2

 - Bileşen: dev.sh isaac-demo + docker compose exec lifecycle
 - Başlık: Host-side timeout kopunca container içi demo süreci yaşamaya devam ediyor
 - Repro: timeout -k 5 60 ./dev.sh isaac-demo quadrupeds.py --headless; ardından container'da ps -eo args | grep quadrupeds
 - Beklenen: Timeout sonrası exec süreci de sonlansın
 - Gerçek: Host exit 124 dönerken python …/quadrupeds.py --headless (PID 1758) ~95 s daha çalıştı
 - Kanıt: EXIT_CODE=124; süreç kaydı; test dalgası kendi sürecini kill etti → temiz
 - Kök neden: docker compose exec, client bağlantısı koptuğunda exec sürecini otomatik kill etmez
 - Minimal düzeltme: Otomasyonda timeout'u docker compose exec komutunu sarmalayacak konumlandır veya koşu-sonrası container-içi süreç doğrulama/temizlik
   adımı ekle
 - Gerekli karar: Uzun Isaac koşuları için standart süreç-lifecycle mekanizması (host wrapper mı, container-side timeout mu)

 ### B-7 · P2 (FAIL testinin kaynağı)

 - Bileşen: data/runtime/g1-mujoco (real olmayan varyant)
 - Başlık: Mesh STL'ler 130–131 B stub → XML yüklenemiyor (g1-mujoco-real sağlam)
 - Repro: MUJOCO_GL=egl /opt/venvs/sonic-sim/bin/python -c "import mujoco;
   mujoco.MjModel.from_xml_path('/workspace/humanoid-lab/data/runtime/g1-mujoco/scene_43dof.xml')"
 - Beklenen: Load OK (real varyantta olduğu gibi)
 - Gerçek: ValueError: decoder failed for mesh '.../left_hip_roll_link.STL' … stl_decoder: number of faces…
 - Kanıt: head_link.STL 131 B (g1-mujoco) vs 932.784 B (real); md5 farklı (7d8675… vs ea9a08…); mesh tarihleri 21 Ağu vs 3 Eyl
 - Kök neden: Eski/kırpılmış kopya (muhtemelen yarıda kalmış transfer/LFS)
 - Minimal düzeltme: g1-mujoco/meshes/ içeriğini g1-mujoco-real/meshes/ (veya gear_sonic sdist referansı) ile senkronla — bu oturumda sync yasağı nedeniyle
   yapılmadı
 - Gerekli karar: Senkronu kim/ne zaman yapacak; bu dizin aktif mi terk mi

 ### B-8 · P3

 - Bileşen: Repo kökü (hijyen)
 - Başlık: scripts/'e taşınan scriptlerin kök kopyaları silinmemiş; smoke-test.sh ayrışmış bayat kod
 - Repro: diff smoke-test.sh scripts/smoke-test.sh; diff install-isaac-webrtc-client.sh scripts/install-isaac-webrtc-client.sh
 - Beklenen: Tek doğruluk kaynağı
 - Gerçek: Kök smoke-test.sh eski (transformers tabanlı GR00T kontrolü; AppLauncher/rsl_rl bölümü yok) ve Dockerfile yalnız scripts/smoke-test.sh'i imaja
   kopyalıyor → kök kopya ölü/yanıltıcı; webrtc kopyası birebir identical (zararsız)
 - Kanıt: diff çıktıları; COPY scripts/smoke-test.sh /opt/humanoid-lab/smoke-test.sh
 - Kök neden: Taşıma sırasında kök temizlenmemiş
 - Minimal düzeltme: Kök smoke-test.sh (ve opsiyonel webrtc kopyası) silinsin
 - Gerekli karar: Kullanıcı (kirli tree; bu auditte silinmedi)

 ### B-9 · P3

 - Bileşen: doctor.sh (CONTAINER bloğu)
 - Başlık: Dead-container durumunda hem warn (skip) hem err basılıp rc=2 — çıktı kafa karıştırıcı
 - Repro: Container durdurulup ./dev.sh doctor (koşulmadı; kod okundu)
 - Beklenen: Tek net sinyal (skip = yalnız warn)
 - Gerçek: "mount contract not auditable" hata tonuyla rc=2; davranış bilinçli görünüyor
 - Kanıt: doctor.sh CONTAINER bloğu
 - Kök neden: Skip/hata durumları ayrıştırılmamış
 - Minimal düzeltme: Dead-container durumunu yalnız warn yap
 - Gerekli karar: Dokunma/düzeltme kararı kullanıcıda

 ### B-10 · P3

 - Bileşen: Isaac Sim Kit extension search path
 - Başlık: Repo alt dizinleri extension olarak taranıp 8 uyarı üretiliyor (işlevsel etki yok)
 - Repro: Herhangi isaac-demo koşusunun ilk saniyeleri
 - Beklenen: Sessiz veya daraltılmış search path
 - Gerçek: .git, .generated, containers, docs, data, locks, ros, scripts için ×8 "extension.toml doesn't exist" uyarısı; 0 Error
 - Kanıt: kit_20260904_123210.log [Warning] [omni.ext.plugin]
 - Kök neden: Repo kökü Kit extension search path'inde
 - Minimal düzeltme: Search path daralt veya uyarıyı bilinçli gürültü olarak belgele
 - Gerekli karar: İstenen davranış mı, yapılandırma düzeltmesi mi

 ### B-11 · P3

 - Bileşen: Host (gpu.foundation)
 - Başlık: Performans ile ilgili host uyarıları — simülasyonu engellemediler
 - Repro: Kit log'u
 - Beklenen: —
 - Gerçek: CPU performance profile is set to powersave; IOMMU is enabled; PCIe link width current (8) vs maximum (16)
 - Kanıt: Kit log uyarıları; demo koşusu PASS
 - Kök neden: Host güç/BIOS yapılandırması
 - Minimal düzeltme: İlerideki performans dalında değerlendir
 - Gerekli karar: CPU governor / host BIOS ayarına dokunulacak mı

 ### B-12 · P3 (kanıtlı RISK)

 - Bileşen: data/hf-cache
 - Başlık: 6 root:root kalıntı (xet cache + marker'lar) → kullanıcı UID'si xet alt ağacına yazamaz
 - Repro: find data/hf-cache -uid 0
 - Beklenen: Tüm mount içerikleri 1000:1000
 - Gerçek: hf-cache/xet/ (755 root), 3 × xet log, .agent_harnesses.json, .check_for_update_done, .check_for_skill_update_done → root:root (3 Eyl)
 - Kanıt: drwxr-xr-x 4 0 0 hf-cache/xet vb. ls çıktıları
 - Kök neden: 3 Eyl'de root EUID'li bir süreç yazmış (erken/root container veya sudo'lu host oturumu)
 - Minimal düzeltme: sudo chown -R 1000:1000 data/hf-cache/xet data/hf-cache/.agent_harnesses.json data/hf-cache/.check_for_* — veya içerik atılabilir kabul
   edilip silinerek yeniden oluşsun
 - Gerekli karar: Chown mu, sil-yeniden-oluşsun mu (root-owned olduğu için sudo gerekli; dokunulmadı)

 ### B-13 · P3

 - Bileşen: compose.yaml volume varsayılanları
 - Başlık: ${HUMANOID_*_ROOT:-/dev/null} fallback'i boşken geçersiz bind üretir (ENOTDIR)
 - Repro: .env olmadan compose config (statik okuma; koşulmadı)
 - Beklenen: Eksik zorunlu değişken compose'ta net reddedilmeli
 - Gerçek: /dev/null/datasets gibi "not a directory" hatası üretecek bind; mevcut .env ile etkisiz
 - Kanıt: ${HUMANOID_SOURCE_ROOT:-/dev/null}:/workspace/humanoid-lab, ${HUMANOID_DATA_ROOT:-/dev/null}/datasets:/data/datasets
 - Kök neden: Hata yakalanması :? yerine sessizce runtime'a bırakılmış
 - Minimal düzeltme: ${HUMANOID_DATA_ROOT:?set HUMANOID_DATA_ROOT in .env} söz dizimine geç (ISAAC arg değişkenlerindeki desen zaten mevcut)
 - Gerekli karar: Yok

 ### B-14 · P3

 - Bileşen: doctor.sh ROS/DDS bölümü
 - Başlık: Domain çakışması / multicast riski için kontrol veya uyarı yok (yalnız bilgi echo)
 - Repro: ./dev.sh doctor → doctor.sh:147-148
 - Beklenen: Host-net + domain 0 + publish-capable venv kombinasyonu en azından WARN
 - Gerçek: Yalnız ROS_DOMAIN_ID/RMW echo; risk skorlaması yok
 - Kanıt: doctor.sh:147,148,219
 - Kök neden: Kapsam env-report düzeyinde tutulmuş (ros/README Stage A tasarım aşamasında)
 - Minimal düzeltme: Tek satırlık kontrol + "WARN: sim bridge real-robot domain'iyle çakışabilir" notu
 - Gerekli karar: Uyarı fail-gate mi info mu

 ### B-15 · P3

 - Bileşen: MuJoCo EGL / PyOpenGL shutdown
 - Başlık: EGLError <exception str() failed> destructor gürültüsü (stderr); exit kodu etkilenmez
 - Repro: Herhangi EGL render sonrası interpreter kapanışı
 - Beklenen: Temiz çıkış
 - Gerçek: Exception ignored in: Renderer.__del__ / GLContext.__del__ → EGLError, exit=0
 - Kanıt: mujoco/egl/__init__.py:131
 - Kök neden: Bilinen PyOpenGL/MuJoCo kozmetik sorunu (bağlam serbest bırakılırken hata bayrağı sorgusu)
 - Minimal düzeltme: Script'lerde explicit r.close() veya log süzgeci
 - Gerekli karar: Yok

 ### B-16 · P3 (kanıtlı RISK)

 - Bileşen: Host tooling
 - Başlık: shellcheck kurulu değil → ci-lint'in shellcheck bölümü sessizce atlanıyor
 - Repro: command -v shellcheck (raider host)
 - Beklenen: Bölüm koşsun
 - Gerçek: == shellcheck missing (skipping…) yolu
 - Kanıt: ci-lint çıktısı
 - Kök neden: Host'ta paket yok
 - Minimal düzeltme: apt install shellcheck (ci-lint zaten kendi mesajını basıyor)
 - Gerekli karar: Kurulum kullanıcıya ait

 ### B-17 · P3 (kanıtlı RISK)

 - Bileşen: data/checkpoints
 - Başlık: Checkpoint yazma hedefi root-owned ve boş → ileride fetch/checkpoint yazımı izin hatası verecek
 - Repro: Container'da ls -ld /data/checkpoints + yazma probe
 - Beklenen: ubuntu sahipli, yazılabilir
 - Gerçek: drwxr-xr-x root root, boş; (kök /data ve /cache root — korunum tasarımı, ayrıca İ6)
 - Kanıt: ls + probe kayıtları; /data/models/... ubuntu sahipli (fetch hedefi OK)
 - Kök neden: Dizin root tarafından oluşturulmuş; yalnız alt mount içerikleri ubuntu sahipli
 - Minimal düzeltme: İhtiyaç anında host'ta chown 1000:1000 data/checkpoints; kök izinlerine dokunma
 - Gerekli karar: Host sahibi müdahalesi (yetki bu auditte yok)

 ### B-18 · P3 (kanıtlı RISK)

 - Bileşen: /opt/venvs + /cache/isaac mount'ları
 - Başlık: Kalıcı venv/cache container içinden yazılabilir ve guard yok → tek yanlış komutla 32 GB venv + cache ezme potansiyeli
 - Repro: Container'da 14 mount kökünde geçici dosya yaz-sil (hepsi OK — kalıcılık için gereklilik)
 - Beklenen: Kalıcılık ile koruma dengesi bilinçli seçilmiş olmalı
 - Gerçek: Yazılabilir (gerekli); ne dokümantasyonda ne doctor'da ezme riskine dair uyarı yok; bu auditte dokunulmadı
 - Kanıt: ISAAC dalgası 14/14 yazma testi + fingerprint mekanizmasının varlığı (yanlış yazım fingerprint'i bozar ama içerik ezilirse geri gelmez)
 - Kök neden: Kalıcılık-tasarımının doğal sonucu; risk kabulü belgelenmemiş
 - Minimal düzeltme: README/doctor'a tek satır uyarı; opsiyonel per-venv yedek/rebuild stratejisi notu
 - Gerekli karar: Risk kabul mü, koruma (uyarı + belge) mi

 ### Bilgi düzeyi (aksiyon gerektirmez)

 ┌────┬────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────┐
 │ #  │ Bileşen                    │ Başlık / Öz                                                                                    │ Kök neden             │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ1 │ fetch-models.sh            │ Ölü eval: grep '^MODELS_' her zaman boş (render-lock-env MODELS_ üretmiyor); zararsız, kafa    │ Satır artıkları       │
 │    │                            │ karıştırıcı                                                                                    │                       │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ2 │ groot_n17_base checkpoint  │ experiment_cfg/config.yaml python/object:groot.vla… tag'i çözülemiyor (paket adı gr00t); JSON  │ NVIDIA eski iç paket  │
 │    │                            │ config'ler yeterli, model yükleme etkilenmiyor                                                 │ şeması                │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ3 │ sonic-sim venv             │ onnxruntime/tensorrt yok; ONNX/TRT dosyaları /data/runtime/sonic-deploy-models/ altında mevcut │ Bilinçli minimal venv │
 │    │                            │ → inference runtime deploy tarafında                                                           │                       │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ4 │ isaac-stream               │ Bilinçli kalıcı exit-1 (librtx.scenedb çökmesi, RTX 5090 Laptop + driver 595.84); README       │ NVIDIA sürüm/Isaac    │
 │    │                            │ Raider notuyla uyumlu; alternatif yol (isaac-demo --headless + webrtc AppImage) kurulu         │ uyumu                 │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ5 │ compose                    │ Healthcheck tanımsız; şu an container kaynaklı dinleyen port yok (ss temiz)                    │ Minimal tasarım       │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ6 │ Mount korunumu (tasarım    │ /data+/cache kökleri root, /data/runtime :ro, /isaac-sim 750 root:isaac-sim → yazma reddi      │ HUMANOID_DATA_ROOT    │
 │    │ doğrulaması)               │ beklenen davranış; alt mount'lar yazılabilir                                                   │ korunumu              │
 ├────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────┤
 │ İ7 │ DEVELOPER_UID/GID          │ 1000:1000 zinciri host↔container↔.env tutarlı; farklı GID'li hosta taşınırsa setup.sh          │ —                     │
 │    │                            │ append_env ile çözer                                                                           │                       │
 └────┴────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

 3) Ek Bölümler

 ### 3a) Başarılı doğrulanan yollar (uçtan uca)

 1. Kurulum/pin zinciri: versions.lock.yaml (schema 1) → deterministik .generated/versions.env (2× byte-aynı) → compose args → imaj 5.1.0 (digest-pinli base)
    → container rev-parse == lock → upstream verify-pins: 3 repo commit + 3 model revision canlı → venv fingerprint 3/3 güncel (uv sync runtime'da
    tetiklenmez)
 2. Isaac Sim headless runtime: use-isaac-sonic → ISAAC_PATH/CARB_APP_PATH/EXP_PATH zinciri → Vulkan + RTX 5090 → isaaclab.python.headless.kit experience
    yükleme → sim döngüsü (quadrupeds, cuda:0) → Kit log 0 Error, izin/shader/GPU hatası yok → 14 cache/log/config mount yazılabilir → orphan temizliği
 3. MuJoCo G1: use-sonic-sim (izolasyon CLEAN) → gerçek scene_43dof.xml (nq=50, sdist referansıyla byte-aynı) → mj_step → 500 adım pasif rollout (stabil
    çökme, çarpışma sağlam) → 64×64 NVIDIA EGL offscreen render
 4. GR00T N1.7 (ağırlıksız): use-groot (PYTHONPATH/LD_LIBRARY_PATH sızıntısı yok) → torch 2.9.0+cu128, sm_120 → flash_attn 2.8.3 bf16 forward →
    config/processor parse → safe_open: 2 shard / 1031 tensör / BF16 → TRT 10.15 import → checkpoint rev == lock == MODEL_PROVENANCE → inference entrypoint
    importları
 5. Kalıcılık/izin: 22 bind-mount compose↔container birebir; /data/runtime gerçekten RO; host 5/5 + container 11/11 yaz-sil; temp kalıntı 0; root-owned obje
    yalnız hf-cache (B-12)
 6. Güvenlik duruşu: .env'de credential yok (HF token host cache'te, "image'e/.env'e girmaz" iddiası kodla uyumlu); isaac-stream crash-döngü kilidi;
    foxy/direct-dds-test.sh bilinçli stub (exit 2); publish-capable kod yalnız /opt/src/sonic köprüsünde ve hiç başlatılmadı
 7. Statik kalite: bash -n ×18, py_compile, YAML, git diff --check, compose config -q, lint öncesi/sonrası tree-hash değişmedi

 ### 3b) Eksik asset / model / checkpoint yolları

 - Cosmos-Reason2-2B backbone (GR00T full-load önkoşulu): hf-cache'te models--nvidia--Cosmos-Reason2-2B (4.6 GB) var, ama GROOT dalgası model kökünde
   çözümleyemedi → full model load doğrulanamadı (çelişkili işaret; sonraki test)
 - H1 locomotion policy: pinli değil, indirme komutu yok (B-5)
 - data/runtime/g1-mujoco: mesh stub → onarım/senkron gerekli (B-7)
 - /data/checkpoints, /data/datasets, /data/rosbags: boş; checkpoints root-owned (B-17)
 - sonic-sim venv: onnxruntime/tensorrt yok (İ3) — inference runtime eksik

 ### 3c) Dokümantasyon–gerçek farkları

 1. README ~/humanoid-lab-data/ ↔ gerçek HUMANOID_DATA_ROOT=<repo>/data (B-4)
 2. README h1 policy indirme adımı ↔ repoda karşılık yok (B-5)
 3. Doğrulanan uyumlar: README'deki 13 komuttan 12'si birebir mevcut; isaac-stream Raider notu gerçek davranışla (kalıcı exit-1) uyumlu; "HF token
    image'e/.env'e girmaz" iddiası kodla uyumlu

 ### 3d) Sonraki en değerli 5 test

 1. ci-lint'i canlandır: B-1 tek satırlık prune fix + shellcheck kurulumu + tam koşum → RC=0 beklentisi (kalite kapısı ölü durumdan çıkıyor)
 2. smoke-test'i geçer kıl: B-3 tek satırlık import gr00t.model yaması + tam ./dev.sh smoke koşumu (GPU + model load dahil)
 3. GR00T full-load dalgası: Cosmos-Reason2-2B'nin hf-cache kopyasından çözümlenmesi + ağırlık yükleme + tek inference (izin/gated lisans kararı sonrası)
 4. DDS izolasyon kanıtı: CYCLONEDDS_URI/lo sabitleme + test domain'i ile run_sim_loop; gerçek ağa sızmama doğrulaması (robot yok, pasif gözlem)
 5. g1-mujoco mesh onarımı: g1-mujoco-real kaynaklı mesh senkronu + non-real varyant load/mj_step doğrulaması

 ────────────────────────────────────────────────────────────────────────────────

 ⚠️ Kural uyumu: Hiçbir dosya değiştirilmedi (tree-hash doğrulandı); sync/rebuild kullanıcı isteğiyle atlandı (NOT_RUN); tüm yazma sınamaları benzersiz
 geçici dosyalarla yapıldı ve silindi (kalıntı 0); secret değer hiçbir rapora yazılmadı; riskler yalnız kanıtla listelendi, safe sınırla yapılmayan hiçbir
 işlem FAIL sayılmadı.

 ↳ Full result: /home/emin/.pi/workflows/projects/humanoid-lab-ad06ec89f3cd/runs/humanoid-lab-runtime-audit-mtmxgvqg-8lh243.json
