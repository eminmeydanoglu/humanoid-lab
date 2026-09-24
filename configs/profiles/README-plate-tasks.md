# PickApple ve PickGum değerlendirme sahneleri

`./dev.sh psi0-isaac-eval --task PickApple ...` veya `--task PickGum ...` seçimi hem Isaac sahnesini hem de politika komutunu değiştirir. `--task` verilmezse mevcut BlockStacking sahnesi ve komutu kullanılır. Mevcut checkpoint, sunucu ve kayıt bayrakları aynı biçimde verilir. Karşılaştırmalı çalışmalarda `--scene-profile` ile sahne profili ayrıca seçilebilir; böyle bir seçimde görevin komutuyla profilin eşleştiğini operatör kontrol etmelidir.

| Görev | Eğitim verisindeki tam komut | Kaynak görüntü | Sahne profili |
| --- | --- | --- | --- |
| PickApple | `Put the apple into the plate.` | [Kaynak](references/pickapple-source.jpg) · [Eski Isaac kamerası](references/pickapple-isaac.jpg) | [PickApple profili](isaac-g1-sonic-pickapple-dex3.json) |
| PickGum | `Put the gum into the plate.` | [Kaynak](references/pickgum-source.jpg) · [Eski Isaac kamerası](references/pickgum-isaac.jpg) | [PickGum profili](isaac-g1-sonic-pickgum-dex3.json) |

Komutlar `data/datasets/psi0-unitree-dex3-sonic-v1-mini/train/meta/tasks.jsonl` içindeki kayıtlarla birebir aynı. Görüntüler sırasıyla yerel `unitree-sonic-latent-replay-20260917/G1_Dex3_PickApple_Dataset/episode_000012/source.mp4` ve `G1_Dex3_PickGum_Dataset/episode_000034/source.mp4` videolarının yaklaşık 2. saniyesinden alınmıştır.

Her iki görev profili `base_profile` ile BlockStacking değerlendirme profilindeki robotu, kontrolcüyü, başlangıç pozunu, fizik adımını ve baş kamerasını devralır. BlockStacking'in yeşil üçüncü küp varyantı da aynı yolla asıl BlockStacking profilini devralır. Baş kamerası robotun `torso_link` parçasına bağlıdır; konumu ve yönü üç görev için tektir ve yalnız asıl profilde tanımlıdır. Görev profilleri yalnız sahneyi değiştirir. Elma yaklaşık 6,7 cm çaplı kırmızımsı bir küre, sakız yaklaşık 6,1 × 3,8 × 1,5 cm koyu mavi bir kutu, tabak ise 18 cm çaplı pembe veya turkuaz bir disk olarak modellenmiştir. Kaynak görüntüdeki el pozları, gerçek meyve dokusu, sakız ambalajı ve tabağın içbükey biçimi burada modellenmemiştir.

Aşağıdaki eski görsel kanıtlar, görev profillerinde ayrı kamera yönü varken çekildi; güncel ortak kamera görüşünü doğrulamaz. Yalnız sahne nesneleri ve sıfırlama için kanıt olarak kullanılabilirler. Geçici sabit tabanlı koşular `data/outputs/eval-plate-scene-smoke/{pickapple,pickgum}-fixed-visual/` altında `head-before.jpg`, `head-after.jpg`, `endpoint-reset.json`, `metrics.json` ve `isaac.log` dosyalarını üretti. Her iki canlı koşuda baş kamera JPEG'i döndü, Reset bölüm kimliğini 0'dan 1'e yükseltti ve `scene_probe` nesne merkezini başlangıç konumunda ölçtü. Üretim profilindeki kontrolcülü model çıkarımı bu çalışmada sınanmadı.

Her iki eski serbest tabanlı profil ayrıca SONIC DDS sağlayıcısı etkin ve model çıkarımı kapalı olarak 12 saniyelik kısa koşudan geçti. Kayıtlar `data/outputs/eval-plate-scene-smoke/{pickapple,pickgum}-production/` altında `isaac.log`, `metrics.json` ve `head.jpg` dosyalarıdır. İkisinde de `result=COMPLETED`, sağlayıcı `sonic_dds`, 219 baş kamera karesi ve canlı nesne/tabak `scene_probe` ölçümü vardır. Bu kayıtlar kamera düzeltmesinden önceki profillere aittir. Harici SONIC komutu olmadan robotun eklemleri pasif kaldığı için bu koşunun baş kamerası sonunda zemini gösterir; etkin kontrol altında görev başarısının kanıtı değildir.

Sahne sıfırlaması dinamik nesneyi başlangıç konumuna geri yazar. Canlı `scene_probe` çıktısı masa, tabak üst yüzeyi ve nesne merkezini raporlar. Görev başarısı yalnız nesneye bakılarak varsayılmamalıdır; mevcut arayüz otomatik başarı ölçütü içermez.
