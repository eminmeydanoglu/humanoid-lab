# BlockStacking değerlendirme hataları — düzeltmeler ve ölçümler

Kapsam: GR00T ve psi0 BlockStacking değerlendirmesinde bulunan hatalar (exp01–exp18).
Kaynaklar: `data/outputs/blockstacking-debug/followup-diagnosis/REPORT.md`, aynı dizindeki `summary.json` ve `contract-evidence.md`, `data/outputs/blockstacking-debug/experiments/*/REPORT.md`, exp15–18 raporları.

Sınıflar:
- **A** — değerlendirme zincirindeki gerçek hatalar (ölçümleri etkiledi)
- **B** — deney sırasında bulunan kendi ölçüm/alet hatalarımız (ürün nedeni değiller)
- **C** — hata değil, eksik bilgi

---

## A. Değerlendirme zincirindeki hatalar

### A1. Aksiyonlar yanlış zaman tabanıyla ilerletiliyordu (iki model)

**Görünen:** Modellerin ürettiği hareket planı, simülasyon zamanına göre olması gerekenden hızlı tüketiliyordu.

**Mekanizma:** Satır ilerletme duvar saatine bağlıydı. Simülasyon gerçek zamandan yavaş aktığı için (RTF 0.670–0.673) saniyedeki uygulanan satır sayısı nominalin üstüne çıkıyordu.

**Ölçüm:**
- GR00T: 73.6–73.9 satır/sim-s (eğitim kopyası 50 Hz). Etkin adım aralığı 0.01353–0.01359 sim-s → 40 adımlık horizon **0.541–0.544 sim-s**'de tükeniyordu (tasarım: 0.8 s). Kaynak: `experiments/12-runtime-contract-probes/groot-action-sim-cadence.json`
- psi0: duvar-saati koşusunda uygulanan hız **43.35 aksiyon/sim-s** (nominal 30 Hz), 1.311 aksiyon / 30.24 sim-s. Kaynak: `experiments/14-policy-clock-ab`

**Düzeltme:** Simülasyon saatinden beslenen policy clock.
- Yeni: `src/humanoid_lab/psi0_bridge/sim_clock.py`, `policy_clock.py`, `groot_clock_runtime.py`
- GR00T: `scripts/run-groot-vla.py` wrapper'ı satır zamanlamasını ve gecikme telafisini bu saatten hesaplıyor
- psi0: `session.py` 30 satır/sim-s; `serve-psi0-sonic.py` içinde `SimulationClockRtcController`; saat sıfırlanır/duraklarsa koşu fail-closed kapanıyor
- Bayraklar: `--policy-clock wall|simulation`, `--policy-clock-file`, `--policy-clock-timeout-s` (varsayılan `wall`)

**Doğrulama:**
- exp15 GR00T: 1.498/1.498 satır uygulandı; sim saniyesi 0–28 için tam **50 satır/sim-s**, 29. saniyede 48; 80 chunk kuruldu
- exp18 psi0: 873/887 aksiyon, **28.87/29.31 /sim-s**, 30 Hz tamlık %96.23/%97.71

### A2. GR00T sol el sıra sözleşmesi yanlış uygulanıyordu

**Görünen:** Sol el komutları kayıtlı veriyle uyuşmuyordu; başparmak etkilenmeden işaret ve orta parmak karışıyordu.

**Mekanizma:** Üç yerde üç farklı sıra var:

| Nerede | Sol el sırası |
|---|---|
| Canlı simülasyon / komut arayüzü | başparmak, orta, işaret |
| GR00T eğitim gözlemi (`state.left_hand`) | işaret, orta, başparmak |
| GR00T eğitim aksiyon etiketi (`teleop.left_hand_joints`) | başparmak, işaret, orta |

Eski varsayılan (compatibility) aksiyon tarafında hiç çeviri yapmıyordu (`left_hand_action_to_live` doğrudan dönüyor), gözlem tarafında ise ham yazma vardı (`result[5:7] = source[3:5]`). Yani model ne yaptığını yanlış parmağa söylüyordu ve elin durumunu da yanlış biliyordu.

**Ölçüm:** Demo replay karşılaştırmasında GR00T sol el komutları ile kayıtlı SONIC komutları arasında ortalama **0.0875 rad**, en fazla **1.687 rad** fark. Gerçek kayıtta işaret ve orta parmak birbirinden ortalama 0.064, en fazla **1.44 rad** ayrışıyor (yani karışma küçük bir kayma değil).

**Düzeltme:** `src/humanoid_lab/psi0_bridge/groot_hand_contract.py`
- Gözlem: RobotModel'den **önce** `(0,1,2,5,6,3,4)` permütasyonu (RobotModel "baş,işaret,orta" alıp "işaret,orta,baş" çıkarıyor)
- Aksiyon: `concat_action`'dan **sonra** aynı permütasyon
- Bayrak: `--groot-left-hand-contract model-independent` (varsayılan `compatibility` korundu)

**Doğrulama:** Gerçek pinlenmiş SONIC `RobotModel` + gerçek episode-0 frame-504 değerleriyle entegrasyon testi; 17 test geçti. Not: ilk önerilen gözlem haritası yanlış sınırda uygulanınca çift permütasyon oluşuyordu; bağımsız inceleme yakalayıp düzeltti.

### A3. psi0 RTC uyuşmazlığı (eğitimde yok, değerlendirmede var)

**Görünen:** Token dizisi demonstrasyonlardan ~4.3× yavaş değişiyordu; "checkpoint kötü strateji öğrenmiş" şüphesi test edilemiyordu.

**Mekanizma:** Eğitim: `model.rtc=false`, 30 adımlık chunk, 30 adımlık uygulama ufku. Değerlendirme: launcher koşulsuz `--rtc` geçiyor; sunucu gizli minimum 15 tick sonra yeniden planlıyor ve `guidance_alpha=0.9` ile süreklilik harmanı uyguluyor. Ayrıca `enable_rtc` ilan edilmesine rağmen controller seçimine uygulanmıyordu.

**Düzeltme:** `scripts/serve-psi0-sonic.py` (upstream `assert cfg.rtc` kaldırıldı; gerçek `config.rtc` ile `Server` kuruluyor; `/info` gerçek `rtc_enabled` bildiriyor), `src/humanoid_lab/psi0_bridge/open_loop.py`, bayraklar `--psi0-rtc-off` ve `--psi0-action-exec-horizon 1..30`. Guided RTC varsayılan kaldı.

**Ölçüm:** exp18, aynı checkpoint-40000, gravity-on, sim clock — guided vs RTC-off:

| Metrik | Guided | RTC-off |
|---|---:|---:|
| Uygulanan aksiyon | 873 | 887 |
| Uygulama hızı (/sim-s) | 28.87 | 29.31 |
| Kol takip MAE (rad) | 0.0458 | 0.0586 |
| Palm-z takip MAE (m) | 0.0467 | 0.0617 |
| Token geçiş L1 | 0.2295 | 0.2573 |
| Aynı token oranı | %4.36 | %13.43 |
| En uzun sabit tutma | 4 satır | 5 satır |
| Küp hareketi / kaldırma | 0 / 0 | 0 / 0 |

n=1. Mod, yeniden planlama ve son-satır tutma semantiğini de değiştirdiği için izole "guidance" karşılaştırması değildir.

### A4. Yerçekimi telafisi yoktu; kol hedefin altında kalıyordu

**Görünen:** Komut edilen el yüksekliği ile gerçekleşen arasında 21–28 cm fark; el her koşuda hedefin **altında**.

**Mekanizma:** Kontrolcü `tau_ff ≡ 0` ile sadece oransal çalışıyor (`kp = 14.25 N·m/rad`, motor armatüründen türetilmiş; f=10 Hz, ζ=2). Denge: `kp·e = τ_yerçekimi`. Yük arttıkça sarkma artar — asılı kol ~1 N·m (~5 cm), yukarıda uzatılmış kol omuzda 5–7, belde 12–14 N·m (0.21–0.28 m).

**Kanıt:** Kayıtlı P-terimi ile URDF yerçekimi momenti oranı 1.00–1.21; tork özdeşliği 1.1e-5 N·m; bel-pitch sarkması tek başına palm farkının %20–28'i. Kaynak: `experiments/03-tracking-validation/REPORT.md`

**Düzeltme:** Opt-in `--gravity-feedforward`; kaynak `root_physx_view.get_gravity_compensation_forces()`; `SimulatorService.gravity_feedforward`, telemetri `body_gravity_feedforward_torque`, `dev.sh` iletimi. Varsayılan kapalı.

**Ölçüm:**
- Deterministik tutuşlar: palm-z MAE **0.4269 → 0.0114 m (−%97.3)** ve **0.2691 → 0.0490 m (−%81.8)**; sıfır-yerçekimi negatif kontrolü **0.0183 → 0.1430 m** (terim yükün olmadığı yerde bozuyor → gerçekten yerçekimine özgü). Kaynak: `experiments/04-gravity-comp-ab/REPORT.md`
- exp13 demo replay: palm MAE 0.0875/0.1092 → **0.0170/0.0140 m**; gövde MAE 0.1153 → **0.0533 rad**
- exp17 GR00T: 15.221 tick boyunca 29 ekleme uygulandı, ortalama |τ| 1.16 Nm, max 15.58 Nm; aktif gövde MAE **0.1321 → 0.0586 rad**
- Uyarı: plant değiştiği için policy hedefleri kaydı (sol/sağ medyan −0.073 / −0.162 m) ve görev yine başarısız oldu; yerçekimi telafisinin doğru yeri eğitim simulatorü, değerlendirmeye sonradan eklemek plant uyuşmazlığı yaratıyor.

### A5. Reset sonrası başlangıç teslimi (initial pose) yarış koşulu

**Görünen:** Bazı koşular referans hareketle, bazıları önceki koşunun son token'ıyla başlıyordu.

**Mekanizma:** İlk poz komutu "gönder ve unut" biçiminde; subscriber hazır olmadan yayınlanıyor ve uygulandığı doğrulanmıyordu. Ayrıca `reset_queued` bilgisi "reset uygulandı" onayı sayılıyordu.

**Ölçüm:** Canonical hücrede 3 settle'ın hiçbirinde initial-pose mesajı gitmemiş; A1'de 12 eklemde 4.15 rad başlangıç farkı. Reliable handshake sonrası 750/751/751 mesaj 49.3 Hz ve aynı token; başlangıç yayılımı **%96 azaldı** (kol 510 → 20.4 mrad; palm 282.9 → 19.2 mm); ilk saniyedeki target farkı 7.6–12.9 cm düştü fakat etki birkaç saniyede kayboldu.

**Durum:** Tam deterministik handshake uygulanmadı; `reset_queued` uygulanmış-reset onayı değildir, warmstart/hold diagnostikleri bu eksiği kapatmıyor. Kaynak: `final-diagnosis/REPORT.md`, `experiments/08-groot-initial-pose-handshake`

### A6. Alt gövde/bel kanalları normalizasyonda sıfırlanıyor (bilgi kaybı)

**Mekanizma:** Eğitim verisinde bacak/bel 15 kanalı sabit; bu yüzden q01 ≈ q99 ve percentile normalizer bu boyutları 0'a eşliyor. Değerlendirmede gelen hareketli değerler de 0'a normalize oluyor.

**Ölçüm:** GR00T checkpoint-40000 processor'ı ile: eğitim sabitleri → 15 sıfır; gerçek runtime değerleri → 15 sıfır; `outputs_identical=true`. psi0: 15 kanalın min/max aralığı sıfır; normalizer sıfır-aralıklı boyutları 0'a eşliyor. Kaynak: `experiments/12-runtime-contract-probes/groot-normalization-probe.json`

**Not:** İlk şüphe "bu yüzden sayılar şişiyor (amplification)" idi; her iki model için de çürütüldü. Kalan kusur anlamsal: iki model de kullanılabilir alt gövde bilgisi almıyor. Düzeltme yok; eğitim/normalizasyon sözleşmesi konusu olarak sınırlama diye kayıtlı.

---

## B. Deney sırasında bulunan kendi ölçüm/alet hatalarımız

Bunlar özgün başarısızlığın nedeni değil; deneylerin geçerliliğini korumak için bulunup düzeltildi.

| # | Hata | Ölçüm / belirti | Düzeltme |
|---|---|---|---|
| B1 | exp14 sim-clock hücresinde eksik aksiyon uygulaması | Dönen 77 hedefin yalnız 23'ü uygulandı → hücre geçersiz | `session.py`: her uyanışta tek gözlem, dönen her aksiyon bir kez yayınlanır, eski sınırlar atlanır; `serve-psi0-sonic.py`: controller sim-zamanı sınırlarında. Test: 0.67× hız, 0.23/0.37/0.29 s gecikmeler |
| B2 | exp15 ilk corrected denemede yanlış backend | GR00T istendi, `fine-tuned` (psi0) servis edildi → koşu reddedildi | Açık `--model groot --model-id groot`, startup provenance logu, capture metadata'da backend + checkpoint hash'leri; rerun1/2 gerçek `groot` + `checkpoint-40000` |
| B3 | Warmstart reset-öncesi baseline taşınmaması | `no fresh post-reset Isaac simulation clock sample`; 0/1509 kare | Güvenilir reset-öncesi örnek `model_controller.py`'de saklanıp `arm()`'a veriliyor; sonrasında 1509/1509 kare uygulandı |
| B4 | GR00T wrapper'da sahte reset hatası | `PolicyClockError: inference crossed an Isaac simulation reset` | Tamamlanan inference için taze saat örneği alınıyor |
| B5 | Launcher-only bayrağın upstream parser'a sızması | `--left-hand-contract` Tyro parse hatası | Bayrak wrapper içinde tüketiliyor |
| B6 | Positive-control şema okuması | Telemetri `kind` şeması ve Parquet artefaktı yanlış okunuyordu | Gerçek Parquet okunuyor; 1.509 indeksin tamamının uygulanması zorunlu; öğrenilmiş policy trafiği yalnız replay başladıktan sonra denetleniyor |
| B7 | Offline psi0 RTC anchor testinde geçmiş kirlenmesi | Süreksiz anchor'lar arasında önceki chunk taşındı → sonuç geçersiz, geri çekildi (`11-offline-demo-fit/psi0-val.json` saklandı) | Yerine bağımsız `Psi0Model.predict_action` (8 difüzyon adımı, sabit seed, row-k hizalaması): held-out token MAE h0 0.02236 → h16–29 0.02564 (mean taban 0.08474/0.08491); eller 0.03020 → 0.04207 (mean 0.21220/0.22758) |
| B8 | psi0 gözlem penceresinde yanlış satır | 21 satırlık pencerede `t+10` satırı dönüştürülmüş `t` ile karşılaştırıldı (normalize fark 0.02503; satırlar arası ham fark 0.04111) | Merkez satır (index 10) kullanıldı → maks fark 0.0, birebir eşitlik; jitter/noise doğrulamada kapalı (`no_aug=True`), inference girdisinde değişiklik gerekmedi |
| B9 | Provenance iddiası | "Aktif konteynerler sibling checkout bind ediyor" iddiası eksik/yanıltıcıydı | Gerçek: `humanoid-lab-dev` → mevcut checkout (dev=66313, inode=1966242; altı dosyanın SHA-256'sı birebir). Sibling konteynerler var ama exp15/17/18 onları kullanmadı. Dar sınırlama: exp15 manifesti kaynak SHA kaydetmemiş |
| B10 | Test koşusu ortamı | İlk toplu pytest host'ta: 261 passed, 32 skipped, 5 environment-dependent failed (pyarrow, torch, psi modülü yok) | Eksikler doğru konteynerlerde kapatıldı: 4 + 1 + 11 = **16 passed**. Tek koşuda 277'lik tam suite çalıştırılmadı |

---

## C. Hata değil, eksik bilgi

- **Eğitim rig'i kalibrasyonu yok:** kamera intrinsics/distortion, seçilen göz, torso/pelvis ekstrinsi, masa-robot dönüşümü ve episode küp/bant pozları repoda yok. Simülatör profili yeniden üretilebilir ama eğitim rig'i olduğu kanıtlanmış değil → metrik yeniden kurulum ve kesin kök neden ayrımı bloke. Kaynak: `followup-diagnosis/contract-evidence.md`
- **Görüntü etkisi kanıtlı, kaynağı belirsiz:** exp16'da iyi desteklenen dört örnekte görüntü değişimi token MAE 0.0516 (stokastik taban 0.0347); 12/12 eşleşmiş etki p95 üstü. Kamera, nesne yerleşimi, görünüş ve poz confounded.
- **Başarı kanıtı yok:** Hiçbir geçerli canlı koşuda başarılı kavrama/kaldırma elde edilmedi (tüm hücrelerde küp lift/displacement = 0). Bu yüzden "checkpoint mı, transfer geometrisi mi, kalan plant/lifecycle mı" sorusu açık.
