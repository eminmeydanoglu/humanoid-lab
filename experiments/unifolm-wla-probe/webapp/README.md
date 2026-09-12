# UnifoLM ER oynatıcı (web arayüzü)

Tek sayfalık yerel arayüz: solda G1 kafa kamerasından alınmış hazır kareler, sağda
model + prompt, altta cevap. `Point modu`'nda modelin cevabındaki koordinatlar karenin
üzerine çizilir.

## Çalıştırma

```bash
experiments/unifolm-wla-probe/webapp/serve.sh          # port 8321, oturumdan bağımsız
PORT=9000 experiments/unifolm-wla-probe/webapp/serve.sh
```

Betik sunucuyu `setsid nohup` ile arka planda başlatır, PID'i `/tmp/er_web_<port>.pid`
dosyasına yazar, log `/tmp/er_web.log`. Elle çalıştırmak istersen:

```bash
cd experiments/unifolm-wla-probe/webapp
/home/aksoy-msi/code/humanoid-lab-main/data/venvs/unifolm-wla/bin/python app.py \
  --port 8321 --warm \
  --frames-dir /home/aksoy-msi/code/humanoid-lab-main/data/outputs/unifolm-wla-probe/frames
```

- Model dizini verilmezse sırayla `--model-root`, `HUMANOID_DATA_ROOT`, depo içi
  `data/` ve `~/code/*/data/models/unifolm-wla-1.0` denenir.
- `--warm` varsayılan modeli sunucu açılışında yükler; yoksa ilk istek ~15 sn bekler.
- Kareler `--frames-dir` altındaki PNG'lerdir; `manifest.jsonl` varsa meyve/komut
  bilgisi kare adına eşlenir.

Adresler:

| nereden | adres |
|---|---|
| bu makineden | http://localhost:8321/ |
| tailnet IP | http://100.126.18.76:8321/ |
| MagicDNS (emin-1 dâhil) | http://aksoy-msi-raider-16-max-hx-b2wj.tail2e36f3.ts.net:8321/ |

Durdurmak için: `kill $(cat /tmp/er_web_8321.pid)`.

## Prompt şeridi

Her koşuda, modele giden prompt fotoğrafın üst kısmına büyük yazıyla basılır (koyu
yarı saydam şerit). Şeritte **sadece doğal dildeki prompt** görünür; sohbet
şablonunun `<|im_start|>user`, `<|vision_start|>`, `<|im_end|>` gibi token'ları
şeride girmez. Şerit fotoğraf ile işaretçi katmanı arasında durur, böylece kırmızı
işaretçiler her zaman şeridin üstünde kalır.

Yazı boyutu metin uzunluğuna ve ekran genişliğine göre değişir (masaüstünde 19 px,
mobilde 14 px; uzun promptlarda küçülür), şerit karenin en fazla %72'sini kaplar.
Prompt sığmazsa alta bir solma efekti eklenir; tam metin her zaman görselin altındaki
`Modele giden tam dize (şablon token'larıyla)` alanında durur. Ölçülen kaplama:
masaüstünde %15, mobilde %24 (yıldız meyvesi karesi, `bbox` şablonu).

## Dosyalar

| dosya | ne yapar |
|---|---|
| `app.py` | stdlib HTTP sunucusu, iş kuyruğu, `/api/*` uçları, point ayrıştırma |
| `engine.py` | checkpoint yükleme + greedy üretim; aynı anda tek model bellekte |
| `points.py` | cevaptaki koordinatları bulup piksele çevirir (ölçek kipleri burada) |
| `static/index.html`, `static/style.css`, `static/app.js` | arayüz |

Yeni paket kurulmadı: sunucu Python stdlib, GPU işi mevcut `unifolm-wla` venv'inde.
İstekler tek işçi thread'de sıraya girer; bu yüzden tarayıcıyı yenilemek koşuyu bozmaz.

## Koordinat ölçeği (kalibrasyon)

Modelin verdiği sayılar karenin piksel ızgarasında **değil**, 0-1000 normalize
ızgarasında. Kanıt: 4 karede modelin `bbox` cevabı 0-1000 kabul edilip piksele
çevrildiğinde, bağımsız bir görsel tahminle (64 piksellik ızgara üzerinden okunan
nesne merkezi) şu kadar örtüşüyor:

| kare | modelin ham cevabı | 0-1000 → piksel | bağımsız tahmin | sapma |
|---|---|---|---|---|
| `apple_ep000000_f0` | `[(255, 415)]` | (163, 199) | (165, 198) | 2 px |
| `pear_ep000074_f0` | `[(816, 512)]` | (522, 246) | (535, 242) | 13 px |
| `grapes_ep000000_f0` | `[(791, 434)]` | (506, 208) | (497, 210) | 9 px |
| `starfruit_ep000111_f0` | `[(311, 836)]` | (199, 401) | (195, 400) | 4 px |

Bu yüzden `Ölçek kipi` varsayılanı `0-1000 normalize`. Ham sayılar arayüzde her
zaman ayrıca gösterilir, çünkü `auto`/`raw` kipleri de denenebilir olmalı.

Bilinmesi gerekenler:

- `Grasp point` şablonu sayı döndürüyor ama dikeyde kayıyor: elma karesinde model
  (161, 226) diyor, nesne merkezi (165, 198). Yatay iyi, dikey ~28 px aşağıda.
- `Point to the {target} in the image.` ifadesi **koordinat döndürmüyor**, model
  "Counting the apple shows a total of 1." cevabını veriyor. Arayüzdeki dört şablon
  koordinat döndüren ifadelerdir.
- ER-Flow `bbox` sorusuna JSON döndürüyor:
  `[{"bbox_2d": [812, 641, 996, 783], "mask": "..."}]`.
  `bbox_2d` dört sayı olduğu için kutu olarak çizilir; `mask` alanı VQ-VAE/RVQ
  çözücüsü yayınlanmadığı için çözülemez, sadece metin olarak görünür.

## Uçlar

| uç | ne yapar |
|---|---|
| `GET /api/config` | kareler, modeller, şablonlar, varsayılanlar |
| `GET /api/status` | bellekteki model, GPU ayırması, kuyruk |
| `POST /api/generate` | `{frame_id, model, mode, prompt\|template+target, scale, max_new_tokens, assistant_prefix}` → `{job_id}` |
| `GET /api/job/<id>` | `queued/running/done/error` + sonuç |
| `POST /api/warm` | modeli önceden yükle |
