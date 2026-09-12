# ER-1 point/bbox koordinatlarının ölçeği (kalibrasyon)

Arayüzü yazarken çıkan soru: ER-1'in verdiği sayılar hangi ızgarada? `[(816, 512)]`
cevabı 640x480 bir karede piksel olamaz (512 > 480, 816 > 640). Üç varsayım test
edildi: ham piksel, 0-1000 normalize, sayı/2.

Yöntem: 4 karede aynı `bbox` sorusu soruldu, cevaplar üç kipe göre piksele çevrildi
ve bağımsız bir görsel tahminle karşılaştırıldı. Bağımsız tahmin, karelerin üzerine
64 piksellik koordinat ızgarası çizilip bir görsel modele okutulmasıyla alındı; bu
tahminin kendisi de yaklaşık (±10 px), yani aşağıdaki sapmalar yöntemin hassasiyet
sınırında.

Soru: `Locate the {fruit} in the image and output its bounding box.` (model: ER-1,
greedy, 48-64 token).

| kare | modelin ham cevabı | ham piksel kabul edilirse | 0-1000 kabul edilirse | sayı/2 kabul edilirse | bağımsız tahmin | 0-1000 sapması |
|---|---|---|---|---|---|---|
| `apple_ep000000_f0` | `[(255, 415)]` | (255, 415) | **(163, 199)** | (128, 208) | (165, 198) | 2 px |
| `pear_ep000074_f0` | `[(816, 512)]` | kare dışı | **(522, 246)** | (408, 256) | (535, 242) | 13 px |
| `grapes_ep000000_f0` | `[(791, 434)]` | kare dışı | **(506, 208)** | (396, 217) | (497, 210) | 9 px |
| `starfruit_ep000111_f0` | `[(311, 836)]` | kare dışı | **(199, 401)** | (156, 418) | (195, 400) | 4 px |

Sonuç: koordinatlar 0-1000 normalize ızgarada (Qwen-VL konvansiyonu). Dört karenin
üçünde ham piksel yorumu zaten kare dışına düşüyor; dördüncüsünde (elma) tesadüfen
kare içinde kalıyor ve yanlış yere işaret koyuyor. Bu yüzden arayüz varsayılanı
`norm1000`; `auto` kipi "kareye sığmıyorsa normalize et" kuralıyla elma karesinde
yanlış karar veriyor, o yüzden varsayılan değil.

## Şablon karşılaştırması

Aynı kare ve hedef için beş ifade denendi (elma ve yıldız meyvesi kareleri):

| ifade | elma | yıldız meyvesi | koordinat döndü mü |
|---|---|---|---|
| `Point to the {t} in the image.` | `Counting the apple shows a total of 1.` | `Counting the starfruit shows a total of 1.` | hayır (0/2) |
| `... Answer with the pixel coordinate as [(x, y)].` | `[(255, 418)]` | `[(311, 833)]` | evet |
| `Give the center point of the {t} in the image as [(x, y)].` | `[(258, 419)]` | `[(314, 834)]` | evet |
| `Where is the {t}? Answer with the pixel coordinate.` | `[(257, 423)]` | `[(316, 836)]` | evet |
| `Point to the {t} in the image and output its coordinate.` | `[(265, 418)]` | `[(315, 836)]` | evet |

`Give the center point ... as [(x, y)]` arayüzün varsayılan şablonu: elma karesinde
piksele çevrilince (165, 201), bağımsız tahmin (165, 198).

## Grasp point kayıyor

`Where should the robot grasp the {t}? Give the grasp point.` sayı döndürüyor ama
dikeyde tutarsız: elma karesinde (161, 226) — nesne merkezi (165, 198), sapma
(-4, +28). Diğer üç karede sapma (+8, -38), (+9, +6), (+8, +16). Yatay eksen iyi,
dikey eksen 30-40 piksele kadar kayabiliyor.

## ER-Flow bbox cevabı

ER-Flow aynı soruya JSON döndürüyor:

```
[{"bbox_2d": [812, 641, 996, 783], "mask": "..."
```

`bbox_2d` dört sayı olduğu için arayüz kutu olarak çiziyor: 0-1000 kabul edilince
piksel (520, 308)-(637, 376). Armut karesinde bağımsız tahmin merkezi (535, 242) ve
yaklaşık 62x92 boyut, yani kutu dikeyde belirgin biçimde aşağıda ve olduğundan büyük.
`mask` alanı VQ-VAE/RVQ çözücüsü yayınlanmadığı için çözülemiyor, metin olarak kalıyor.

Bu ölçümler `webapp/app.py` üzerinden, arayüzün kullandığı API'nin aynısıyla alındı.
