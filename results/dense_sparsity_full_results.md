# Kart 5 — ASAMA 2 (--full): noron-bazli MLP predictor sonuclari

**Tarih:** 2026-07-04
**Veri:** scripts/sparsity_data_v2 (Qwen2.5-3B, 560 prompt, 6476 token, L6/18/30, d_ffn=11008)
**Kurulum:** 2 katmanli MLP (2048 -> 1024 ReLU -> 11008), BCE + Adam, 12 epoch,
prompt-seviyesi split (sizinti onlemi, seed=0). Sandbox CPU'da NumPy replikasyonu ile
kosuldu (`scripts/np_full_runner.py`); torch versiyonu `analyze_sparsity_v2.py --full`
icinde ayni protokolle mevcut, GPU dogrulamasi icin:

    python scripts/analyze_sparsity_v2.py --data scripts/sparsity_data_v2 --full --k-frac 0.20

## Sonuclar

recall>=0.90 icin acilmasi gereken noron orani (= butce) ve net FFN kazanci
(predictor maliyeti %19.8 FFN dustukten sonra):

| Katman | k    | recall@topk | butce(r=0.9) | net kazanc |
|--------|------|-------------|--------------|------------|
| L6     | 0.20 | 0.613       | %48          | **%32.0**  |
| L6     | 0.30 | 0.674       | %54          | %26.7      |
| L30    | 0.20 | 0.564       | %69          | %11.0      |
| L30    | 0.30 | 0.598       | %75          | %5.7       |
| L18    | 0.20 | 0.449       | %78          | %2.5       |
| L18    | 0.30 | 0.527       | %79          | %1.3       |

Ham raporlar: `scripts/sparsity_data_v2/sparsity_report_full_k{20,30}.json`
(recall@butce egrileri, per-neuron recall dagilimi dahil).

## Bulgular

1. **MLP lineer probu GECMIYOR.** k=0.30'da MLP recall@topk 0.674/0.527/0.598
   (L6/18/30) vs onceki lineer 0.676/0.549/0.607. Nonlineerlik bir sey katmadi ->
   sinyal buyuk olcude LINEER. Predictor secimi artik kapasite degil maliyet sorunu:
   dusuk-rank lineer (2048 -> r -> 11008) mantikli aday.
2. **recall 0.9 hedefi PAHALI.** En iyi katmanda (L6) bile noronlarin ~%48'ini acmak
   gerekiyor; PowerInfer'in "%10-20 ile isi kurtar" rejiminden uzak. L18'de butce %78
   -> skip pratikte anlamsiz.
3. **Katman sirasi yine sabit: L6 > L30 > L18** (3. kez ayni desen — acik soru duruyor).
4. **Net kazanc tavani ~%32 (sadece L6, k=0.20).** Tum katmanlara uygulansa ortalama
   cok daha dusuk; attention + diger maliyetler dahil degil.
5. Per-neuron recall medyani dusuk (L6 k=0.20'de 0.47, p10=0.12): bircok noron
   neredeyse hic dogru tahmin edilemiyor.

## Karar onerisi

Kapi **YARI ACIK, egim asagi**: sinyal gercek ama recall-0.9 rejiminde kazanc yalnizca
erken katmanlarda ve mutevazi. Devam icin en ucuz iki test:
- **Veri olcegi:** 6476 token / ~112 test prompt az olabilir; 5-10x veri ile L6
  butcesi dusuyor mu? (lineer predictor yeterli oldugundan bu deney ucuz)
- **Kalite testi (asil olcut):** recall 0.9 maskeyle skip edilen gercek uretimde
  perplexity/kalite kaybi ne? Recall vekil metrik; belki 0.8 bile yetiyor.

Bu ikisi de negatifse Kart 1 gibi temiz STOP yazilabilir.

---

# Kart 6 — Oracle skip: butce vs perplexity -> ERKEN PATLAMA

**Tarih:** 2026-07-04 | `scripts/oracle_quality.py`, 100 prompt (training_v1.jsonl),
Qwen2.5-3B-Instruct, 36 katmanin TAMAMINDA gercek-aktivasyon top-B maskesi (predictor yok).

| Butce B | ppl    | delta   |
|---------|--------|---------|
| baseline| 12.502 | —       |
| 0.50    | 12.796 | +2.4%   |
| 0.40    | 13.405 | +7.2%   |
| 0.30    | 15.083 | +20.7%  |
| 0.20    | 20.599 | +64.8%  |

Rapor: `scripts/oracle_quality_report.json`

## Bulgular

1. **Ust sinir artik belli.** Oracle = herhangi bir predictor'un ulasabilecegi EN IYI
   durum. Kalite ancak B~0.50'de dayaniyor (+2.4%); B=0.40 bile +7.2% -> script'in
   kendi esigi ("B=0.40 ~ baseline") saglanmadi. SiLU yamaci (ReLU'suz Qwen) yari
   yolda birakiyor — literaturle uyumlu (PowerInfer/DejaVu kazanclari ReLU'lu ya da
   ReLUfied modellerde).
2. **Kart 5 ile birlesik tablo:** B=0.50'de predictor recall'u L6 0.91 / L30 0.81 /
   L18 0.73 (k=0.20 egrisi) -> gercek predictor maskesiyle ppl kaybi +2.4%'un
   UZERINDE olur. Hesap tarafinda tavan: %50 skip - predictor maliyeti
   (MLP %19.8, dusuk-rank lineer ~%10) = FFN'de net ~%30-40, toplam FLOP'ta kabaca
   ~%25-30 — karsiliginda >+2.4% ppl. Zayif takas.
3. Sinyalin lineer oldugu bulgusu (Kart 5) burada tali kaldi: predictor ne kadar
   iyilesse de oracle tavanini asamaz.

## Karar

**Vanilla-Qwen dense skip hatti: STOP'a cok yakin.** Sinyal var (Kart 3-5) ama
oracle tavani ekonomik degil. Kapanmadan once TEK ucuz test:

- **ReLUfied hazir model spike'i (train GEREKTIRMEZ):** oracle_quality.py'de modeli
  degistir (or. ProSparse-LLaMA / TurboSparse ailesi, hazir yayimlanmis ReLUfied
  checkpoint'ler) ve ayni B-egrisini cikar. Literatur dogruysa B=0.10-0.15'te
  delta ~0 gorulmeli -> o zaman tum predictor hatti O modellerde degerli.
- Spike de negatifse: Kart 1 gibi temiz STOP, "dense skip yalnizca ReLU'lu
  modellerde anlamli" notuyla kapat.

---

# Kart 7 — OPT-1.3b (dogustan ReLU) oracle: PLATO VAR -> POZITIF

**Tarih:** 2026-07-04 | `oracle_quality.py --model facebook/opt-1.3b`, ayni 100 prompt,
24 katman, hook fc2 girdisi (= relu(fc1(x))). Rapor: `scripts/oracle_quality_opt13b.json`

**Dogal sifir orani: %96.1** (a==0; Qwen/SiLU'da ~%0)

| Butce B | ppl    | delta  |   | Qwen2.5-3B (Kart 6) |
|---------|--------|--------|---|---------------------|
| baseline| 17.899 | —      |   | —                   |
| 0.50    | 17.899 | -0.0%  |   | +2.4%               |
| 0.30    | 17.899 | -0.0%  |   | +20.7%              |
| 0.20    | 17.896 | -0.0%  |   | +64.8%              |
| 0.15    | 17.868 | -0.2%  |   | —                   |
| 0.10    | 17.924 | +0.1%  |   | —                   |
| 0.05    | 18.096 | +1.1%  |   | —                   |

## Bulgular

1. **Hipotez dogrulandi: sorun fikirde degil, aktivasyonda.** ReLU modelde token
   basina noronlarin sadece ~%4'u canli; B >= dogal-canli-oran oldukca top-B maske
   fiilen kayipsiz. B=0.05'te (canli oranin ALTINA inince) bile sadece +1.1%.
2. DejaVu/PowerInfer literaturuyle birebir uyum (ana deney modelleri zaten OPT'ydi).
3. **Ekonomi tersine dondu:** Qwen'de tavan %50 skip + >2.4% ppl idi; ReLU modelde
   %85-90 skip ~0 ppl. Predictor hatti (Kart 3-5'teki tum altyapi) ReLU modellerde
   deger kazaniyor — ustelik gorev kolaylasir: %20-30 "sicak" degil, ~%4 "canli"
   noron tahmini (DejaVu bunu dusuk-rank predictor'la cozuyor).

## Karar

- **Vanilla-SiLU Qwen dense skip: temiz STOP** (Kart 6). "Dense skip yalnizca
  ReLU('fied) modellerde anlamli" — artik olculmus, iki satirlik tablo konusuyor.
- **Hat ReLU modellere pivotluyor.** Siradaki iki adim (ucuzdan pahaliya):
  a) collect/analyze v2'yi OPT-1.3b'ye uyarla (hook fc1 cikisi): x -> canli-maske
     predictor LIFT'i ReLU modelde olc. Beklenti: cok daha kolay gorev, recall >>.
  b) 7B ReLUfied dogrulama (Bamboo-7B / ReluLLaMA-7B, --limit 50 --max-tokens 128):
     "modern model + ReLUfication tasiyor mu" — Qwen'i ReLUfy etme kararinin vekili.

## Kaydedilen kapilar

- domain -> noron/expert haritasi (Stage A, spike) — kapali
- dense'i etiketle "bolme" — kapali
- **vanilla-SiLU dense skip — KAPALI (Kart 6+7: oracle tavani ekonomik degil,
  ReLU kontrolu platoyu dogruladi)**
