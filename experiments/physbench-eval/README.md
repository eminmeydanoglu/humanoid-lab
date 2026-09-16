# PhysBench değerlendirmesi — UnifoLM-ER-1 & Cosmos-Reason2-2B

İki robotik VLM'nin [PhysBench](https://physbench.github.io/) test seti (10.002 öge) üzerindeki
performansını ölçen çalışmanın kayıtları. Ana çıktı: **`report.html`** (tarayıcıda açın).

## Modeller

| Model | Taban | Rol |
|---|---|---|
| `unifolm-wla-1.0/UnifoLM-ER-1` | Qwen3-VL-4B | Unitree'nin gömülü algılama/grounding modeli (robotik ince ayar) |
| `cosmos_reason_backbone` (nvidia/Cosmos-Reason2-2B) | Qwen3-VL-2B-Instruct | NVIDIA'nın fiziksel-AI akıl yürütme modeli; GR00T N1.7'nin VLM omurgası |

## Protokol (resmî veri ve skorlamayla)

- **Veri:** resmî `USC-PSI-Lab/PhysBench` deposundan `test.json` + `image.zip` + `video.zip`;
  GT için repodaki `eval/physbench/test_answer.json` (skorlama `print_test_score.py` ile).
- **Medya:** soru metnindeki `<video>`/`<image>` yer tutucuları, resmî koddaki gibi sırayla
  `file_name` listesine eşlendi. Videolar **tüm klip boyunca eşit aralıklı 8 kareye** indirildi
  (resmî varsayılan `test_frame=8`), tüm medya uzun kenar **≤640 px** JPEG'e normalize edildi.
- **Prompt:** videolu ögelerde resmî "kareler dizisi" açıklaması + soru + resmî bitiş talimatı
  (`Answer with the option's letter from the given choices directly...`). İki model birebir aynı
  promptu gördü; düşünme/zincir-akıl-yürütme istemi verilmedi (Cosmos'un `<think>` formatı kapalı).
- **Üretim:** greedy (`do_sample=False`), en fazla 32 yeni token.
- **Skorlama iki biçimde:** (1) *harf* — modelin çıktısından çıkarılan A-D harfi;
  (2) *resmî string* — repodaki `answer.startswith(gt)` kuralı ham çıktıya uygulanır.
  Temiz 4-seçenekli GT'si olan 9.978 öge skorlandı (24 bozuk etiket dışlandı).
- **Donanım:** tek RTX 5090 (aksoy), bf16, SDPA, modeller sırayla koşuldu.
- **Hız düzeltmesi:** Qwen3-VL yama-gömme katmanı (`Conv3d(3,H,2,16,16)`, kernel=stride) matematiksel
  olarak eşdeğer `F.linear` ile değiştirildi (≈170× hızlanma, çıktı farkı 6,5e-4). Bu, sonuçları
  değiştirmez; yalnızca koşu süresini kısaltır.

## Yeniden üretim

```bash
# 0) veri (yerel): 7,43 GB indirme + arşiv açma
python fetch_parallel.py            # /home/aksoy-msi/code/humanoid-lab-main/data/physbench

# 1) kare önbelleği: items.jsonl + media/ (uzun kenar ≤640)
python build_cache.py --workers 14

# 2) aksoy'a aktarım (tek tar akışı) ve koşular
python /tmp/physbench_mon/transfer_cache.py
ssh aksoy 'cd ~/physbench && bash pilot_val.sh'      # 200 örneklik val pilotu
ssh aksoy 'cd ~/physbench && bash full_test.sh'      # 10.002 ögelik tam test

# 3) skorlama + rapor (yerel)
bash finalize.sh
```

## Dosya haritası

- `report.html` — sonuç raporu (kendi kendine yeten HTML).
- `scores_er1.json`, `scores_cosmos.json` — resmî scorer çıktıları (kategori/yetenek/alt sınıf + mod kırılımı).
- `pair.json` — iki modelin anlaşma/ortak doğruluk analizi.
- `summary_er1_test.json`, `summary_cosmos_test.json` — koşu özetleri (süre, token, VRAM).
- `cache_stats.json` — önbellek istatistikleri (öge/medya sayısı, video uzunluk dağılımı).
- Harness ve scriptler: `humanoid-lab-main/data/physbench/` (`run_eval.py`, `build_cache.py`,
  `score.py`, `make_report.py`, `pair_analysis.py`).

## Sonuçlar

Ayrıntılı tablolar ve grafikler `report.html` içinde. Özet:

| | ER-1 (4B) | Cosmos-Reason2-2B |
|---|---|---|
| Genel doğruluk (harf) | **%47,96** (4.785/9.978) | **%44,25** (4.415/9.978) |
| Resmî string skoru | %47,96 | %44,25 |
| Geçersiz çıktı | %0,00 (0) | %0,00 (0) |
| Örnek başına süre (izole pilot / tam koşu) | 187 ms / 382 ms | 128 ms / 346 ms |
| İşlem döngüsü süresi | 63,7 dk | 57,7 dk |
| VRAM zirvesi | 9,81 GB | 5,60 GB |

Kategori (ER-1 / Cosmos / GPT-4o referansı): property 52,8 / 50,3 / 56,9 ·
relationships 63,0 / 49,0 / 64,8 · scene 31,8 / 36,6 / 30,15 · dynamics 45,4 / 41,5 / 46,99.

Ayrıntılı tablo, grafikler ve 12 maddelik bulgu listesi `report.html` içinde.

## Bilinen sınırlar

- Kare örnekleme tüm klibi tarar; resmî açık-kaynak yardımcı fonksiyon videonun *ilk* 8 karesini alır
  (uzun kliplerde bu, klip başının ~0,3 saniyesine denk gelir). GPT-4o yolu ise klip boyunca sabit
  adımla 8 kare alır. Bizim seçim prompt metninin tarif ettiği ve GPT-4o yolunun amaçladığı protokole denktir.
- Çözünürlük ≤640 px; GPT-4o protokolü 512 px kullanır (yani bu bütçe API modellerinden cömerttir).
- Tek koşu, greedy, tek seed; baz modeller (Qwen3-VL-4B/2B-Instruct) koşulmadı.
- PhysBench videoları ögeler arasında paylaşılır (6.017 videolu öge ↔ 3.273 benzersiz klip),
  yani videolu ögeler istatistiksel olarak bağımsız değildir.
