\# Activation Sparsity Predictability — Arastirma Notlari

Son guncelleme: 2026-06-27

Sahip: Bulut (Kler3) | Ortam: RTX 5060 8GB / 32GB RAM, CPU-torch venv, Colab H100 (egitim icin)



\## Amac

FFN noron seyrekligini hesaplanmadan ONCE tahmin edip atlamak -> hesap+bellek

kazanci (PowerInfer/DejaVu mekanizmasi). Buyuk hedef: enerji/hesap verimliligi,

"MoE'nin ilerisi", 20W paradoksu. Bellek-sigdirma DEGIL, az-calistirma hatti.



\## ANA BULGU (2 haftalik zincirin ozeti)

Skip kazancinin on kosulu SERT seyreklik (ReLU-tarzi gercek sifirlar).

\- Domain sinyali seyrekligi BILMIYOR (Kart 1) — token aktivasyonu BILIYOR (Kart 3)

\- Ama SiLU'da sinyal olsa da kuyruk yumusak -> skip kaliteyi tasimiyor (Kart 5-6)

\- ReLU modelde ayni protokol: %96 dogal sifir, %90 skip bedava (Kart 7)

Sonuc: vanilla-SiLU hatti STOP (mimari sinir). ReLU hatti ACIK.

GUNCEL (Kart 13): ReLU hattinda DEGER KANITI alindi — OPT-1.3b, r512 predictor

(300k token), B=0.30-0.40: FFN'de net %44-54 azalma, +1.1-2.9% ppl. Kritik

kaldirac VERI OLCEGI cikti (6k'daki tum negatifler aclik artefaktiydi).



\## Karsilastirma tablosu (ayni protokol, oracle skip, butce -> ppl delta)

| Butce | Qwen2.5-3B (SiLU) | OPT-1.3b (ReLU) |

|---|---|---|

| dogal sifir orani | \~0 | %96.1 |

| B=0.50 | +2.4% | -0.0% |

| B=0.30 | +20.7% | -0.0% |

| B=0.20 | +64.8% | -0.0% |

| B=0.15 | — | -0.2% |

| B=0.10 | — | +0.1% |

| B=0.05 | — | +1.1% |

SiLU: puruzsuz yamac, her dilim acitiyor. ReLU: plato + gec kirilma.



\## Kartlar



\### Kart 1 — Stage A: dense domain-static sparsity -> STOP (+0.037)

Qwen2.5-3B, 560 prompt, domain-conditioned top-k mask.

Coverage lift +0.037 (MoE tarafi +0.186'ydi). Global top-%20 zaten %90

kapsiyor -> tavan yok. Blok/katman/k-sweep her levered'da negatif.

Dosya: dense\_sparsity\_stageA\_results.md, benchmark\_results/dense\_sparsity\_1781676512.npz



\### Kart 2 — MoE routing spike: L20 CV=2.77, domain deseni degil -> paused

Qwen3.6-35B-A3B. Routing dengesiz ama insan-okunur domain'e haritalanmiyor.



\### Kart 3 — x -> sicak maske tahmini: SINYAL VAR (dogrulanmis)

Qwen2.5-3B, L6/18/30, k=%20 magnitude, lineer probe.

LIFT = +0.176 / +0.124 / +0.164 (L6/L18/L30).

Prompt-seviyesi held-out split ile AYNI (v1 token-split: +0.177/+0.125/+0.162)

\-> sizinti degil, gercek genelleme. Recall 0.47-0.61.

Stage A karsilastirmasi: domain +0.037 vs token-aktivasyon \~+0.15 (4-5x).



\### Kart 4 — k-sweep (0.10/0.20/0.30): LIFT k'dan BAGIMSIZ

L6 \~+0.17, L18 \~+0.125, L30 \~+0.155 her k'da (L18: 0.125/0.124/0.126).

Katman sirasi sabit: L6 > L30 > L18 (3 olcumde ayni). Recall mutlak

seviyesi k ile artiyor (k=0.30'da L6 0.676). k secimi sinyali etkilemiyor

\-> muhendislik kriteriyle secilebilir.



\### Kart 5 — --full (noron-bazli MLP + recall@butce + maliyet): kapi yari acik

(Cowork, NumPy reimpl; torch sandbox'a kurulamadi — GPU torch dogrulamasi

HALA ACIK is.) 6 kosu: L6/18/30 x k=0.20/0.30.

\- MLP lineer probu GECEMEDI -> sinyal lineer + doymus; predictor ucuz kalabilir

\- recall 0.9 icin en iyi katmanda (L6) noronlarin %48'i acilmali

\- net kazanc tavani \~%32, sadece L6; L18'de fiilen 0

Dosya: dense\_sparsity\_full\_results.md, sparsity\_data\_v2/sparsity\_report\_full\_k{20,30}.json



\### Kart 6 — Oracle skip, 36 katman, butce vs ppl (Qwen, 100 prompt, CPU fp32)

baseline ppl 12.50. B=0.50: +2.4% | 0.40: +7.2% | 0.30: +20.7% | 0.20: +64.8%.

Egri plato YAPMIYOR -> yumusak kuyruk (SiLU) imzasi. Kart 5 ile birlesik

bilanco: gercekci kazanc \~%30 FFN, ppl +4-6% tahmini -> zayif.

Mekanizma hipotezi: SiLU gercek sifir uretmiyor (Stage A'da count@1e-2

metriginin doymasi ayni sebep). Dosya: oracle\_quality\_report.json



\### Kart 7 — ReLU kontrolu (OPT-1.3b, ayni protokol): PLATO VAR

Dogal sifir %96.1 (ort. canli \~%3.9). B=0.20'ye kadar delta tam 0;

B=0.10 +0.1%; B=0.05 sadece +1.1% (Qwen B=0.50'de +2.4 idi).

SiLU-vs-ReLU hipotezi iki modelli kontrollu karsilastirmayla DOGRULANDI.

Ekonomi tersine dondu: Qwen "%50 skip + >2.4% ppl" vs ReLU "%85-90 skip + \~0 ppl".

Kart 3-5 predictor altyapisi copte degil — yanlis modeldeymis.

Dosya: oracle\_quality\_opt13b.json



\### Kart 8 — OPT canli-maske predictor (x -> a>0): SINYAL RELU'DA DA VAR (on sonuc)

Ayni protokol, --mask-mode live, L4/12/20, 8060 token (560 prompt).

Dogal canli oran: L4 %0.7 / L12 %4.5 / L20 %6.8 (Kart 7 ortalamasi %3.9 ile tutarli).

Lineer probe LIFT: L4 +0.075 (zayif) / L12 +0.125 / L20 **+0.243** (tum olcumlerin en buyugu).

KATMAN SIRASI TERS DONDU: Qwen L6>L30>L18, OPT L20>L12>L4. Erken katman OPT'de

neredeyse statik (base 0.478, canli %0.7) — sinyal gec katmanlarda.

Duzeltme kosusu (--arch linear --ffn-mats 2): MLP yine lineeri gecemedi (4. kez;

artik kural). Lineer egriyle butce(r=0.9): L4 %18 / L12 %29 / L20 %34

(MLP'nin %47-50'si kotumsermis). LIFT: +0.117 / +0.189 / +0.312.

w-recall teshisi: L20 0.685>0.604 (kacirilanlar minik -> ppl bedeli dusuk olmali),

L4 0.475<0.595 (kacirilanlar BUYUK -> L4 gorundugunden kotu).

SVD rank sweep (egitilmis W kesildi, retrain yok): SINYAL DUSUK BOYUTLU.

rank=64'te maliyet %50 -> %1.9, butce neredeyse bozulmuyor:

| Katman | butce(full) | butce(r64) | net(r64) |

|---|---|---|---|

| L12 | %29 | %33 | +64.7% |

| L20 | %34 | %37 | +61.0% |

| L4  | %18 | %41 (r256: %36) | +56.6% |

Tatli nokta rank 64-128 (L4 rank'e duyarli, 256 ister ama mutlak maliyeti kucuk).

VERDICT: ReLU modelde dusuk-rank lineer predictor ile FFN'de ~%55-65 net hesap

kazanci mumkun (recall-0.9 vekiliyle). MLP 4/4 lineeri gecemedi -> kural.

Kart 9 (uctan uca ppl) onaylandi — vekil metrik tartismasi orada biter.

Dosya: sparsity\_data\_opt/sparsity\_report\_live.json, sparsity\_report\_full\_live\*.json



\### Kart 9 — predictor-in-the-loop ppl: TEK KATMAN UCUZ, 24 KATMAN PATLIYOR

predictor\_quality.py: 24 katmanda rank-128 (full-linear + SVD kesme), rastgele

prompt-split, held-out 112 prompt, UC EGRI ayni eval setinde. Bulgular:

\- v1 tuzagi: SIRALI split = istemeden domain-holdout (baseline 42.8 vs 26.4).

&#x20; Yan bulgu: domain kaymasinda statik de predictor de cokuyor (+275%) ->

&#x20; predictor egitimi karisik-domain olmak zorunda.

\- Mekanik saglam: pred B=0.95 -> +0.4%; oracle B>=0.25 bit-ayni baseline

&#x20; (canli <= %8.2, ayni hook yolu).

\- TEK katman ucuz: L12 pred +0.9% (B=0.30; statik +1.7% -> predictor katkisi

&#x20; dusuk butcede gercek), L3 <= +0.7%.

\- 24 katman birlikte: statik +403..+1401%, pred +467..+2029% (B=0.40..0.25).

&#x20; Bagimsiz birikim tahmini (1.009^24 ~ +24%) KAT KAT asiliyor ->

&#x20; SUPERLINEER birikme: temiz x'te egitilen predictor, onceki katmanlarin

&#x20; maskesiyle KAYAN x'te bozuluyor; hata çig gibi buyuyor.

\- Kok neden: recall@butce egrisi duzlesiyor (butce 0.60'ta bile 0.972);

&#x20; 0.99'a butceyle ULASILMIYOR — kuyruktaki ~%3 canli kutle bu predictor +

&#x20; 6k token veriyle x'ten tahmin edilemiyor.

VERDICT: per-layer recall ~0.90-0.95, in-loop'ta YETMIYOR (DejaVu'nun ~0.99

rejiminin sebebi buymus — bunu kendi elimizle olctuk). Kapi kapanmadi ama

esik yukseldi: cikislar Siradaki'de. Odul hala masada: in-loop cozulurse

B=0.30 + rank-128 = FFN'de ~%66 net.

Dosya: predictor\_quality.py, predictor\_quality\_report.json, predictor\_weights\_r128.pt



\### Kart 10 — Progressive (hata-farkindalikli) egitim: YARIYA indirdi, YETMEDI

train\_budget=0.30, 24 katman-sirali toplama (\~65 dk CPU). Ayni eval seti:

| B | static v2->prog | pred v2->prog |

|---|---|---|

| 0.40 | +403% -> +208% | +467% -> +247% |

| 0.30 | +907% -> +394% | +1264% -> +536% |

| 0.25 | +1401% -> +568% | +2029% -> +895% |

1\. Dagilim kaymasi GERCEK ama IKINCIL (\~2x iyilesme; felaket bandi surdu).

&#x20;  Dominant sorun: kuyruk tahmin edilemezligi (6k token, rank-128, recall

&#x20;  egrisi 0.60 butcede 0.972'de duzlesiyor).

2\. SURPRIZ: in-loop'ta STATIK > PREDICTOR (0.30'da +394 vs +536; offline'da

&#x20;  tam tersi). Yorum: statigin hatasi SABIT -> ag tutarli bir budanmis-alt-ag

&#x20;  goruyor; predictor'in hatasi token-basina DEGISKEN -> gurultu birikiyor.

&#x20;  "Error consistency" per-layer recall kadar onemli — literatur karsilastirmasi

&#x20;  icin not: PowerInfer'in statik-agir hibrit tasarimi bununla tutarli.

VERDICT: post-hoc predictor, BU veri olceginde, 24-katman tam skip'i tasimiyor.

Kalan iki cikis: kismi-katman (dakikalar, ckpt hazir) ve veri olcegi (Colab).

Dosya: predictor\_quality\_prog.json, predictor\_weights\_r128\_prog30.pt



\### Kart 11 — Kismi-katman skip: IKI BLOK DA TASIMIYOR (6k token'da)

Ayni ckpt (--load-preds), ayni eval seti, teshis modu (--only-layers).

ERKEN BLOK (L1-8, canli %0.6-1.5), B=0.15/0.10:

&#x20; static +86/+214%, pred +615/+1372%. "Olu katman = guvenli katman" YANLIS:

&#x20; erken katmanin az sayidaki canlisi token-basina KAOTIK ve BUYUK

&#x20; (L4 w-recall 0.475 < duz recall bunu onceden haber vermisti).

&#x20; Tasarim dersi: katman butcesi canli oranla degil TAHMIN-EDILEBILIRLIKLE

&#x20; olceklenmeli.

ORTA BLOK (L12-17), B=0.40/0.30:

&#x20; static +55/+108%, pred +16/+34%. Iki bulgu:

&#x20; 1) Kisa blokta PREDICTOR statigi 3-4x geciyor — 24 katmanda tersiydi.

&#x20;    Doz-yanit iliskisi: az katman -> adaptivite kazanir; cok katman ->

&#x20;    degisken hata gurultusu birikir, tutarlilik kazanir. Error-consistency

&#x20;    bulgusu (Kart 10) saglamlasti.

&#x20; 2) Mutlak seviye yine kotu: 6 katman + B=0.40 = FFN'in \~%14'u icin +%16 ppl.

VERDICT: kismi-katman da KAPALI (bu veri olceginde). Tum kurtarma yollari

olculdu: tam-24 (Kart 9), progressive (Kart 10), erken blok, orta blok.

Tek kalan kaldirac: VERI OLCEGI. O da recall'u \~0.99'a tasimazsa OPT

post-hoc predictor hatti ilkesel STOP.

Dosya: predictor\_quality\_early.json, predictor\_quality\_mid.json



\### Kart 12 — Veri olcekleme egrisi (Colab H100, wikitext): EGRI CANLI, PLATO YOK

scaling\_probe\_colab.py: 350k token, sabit 30k test (belge-split), ic ice

6k/30k/100k/300k. frac@0.99 (full linear):

| N | L4 | L12 | L20 |

|---|---|---|---|

| 6k | 0.96 | 0.95 | 0.96 |

| 30k | 0.86 | 0.79 | 0.81 |

| 100k | 0.52 | 0.43 | 0.40 |

| 300k | **0.29** | **0.18** | **0.21** |

rec@0.40: 0.75 -> 0.92 -> 0.98 -> **0.994-0.999**. Her 3.3x veri egriyi belirgin

dusuruyor, doyma isareti yok.

1\. KART 8-11'IN KOTUMSER TABLOSU VERI ACLIGI ARTEFAKTIYMIS. Kuyruk x'ten

&#x20;  ogreniliyor; 6k token'la ogrenilemiyormus. "Kuyruk kaotik" degil "kuyruk ac".

2\. Yeni darbogaz: r128 buyuk veride KAPASITEYE takiliyor (frac@0.99 300k'da

&#x20;  full 0.18-0.29 vs r128 0.37-0.49) -> rank 256-512 (OPT'de r512 maliyet %15.6).

3\. 6k saglamasi Kart 8 ile tutarli (frac@0.90 \~0.7 wikitext'te; synth'te %18-34

&#x20;  idi — synth daha dar dagilim, beklenen yonde fark).

VERDICT: post-hoc predictor hatti YASIYOR. Son test: 300k-egitimli 24-katman

predictor ile in-loop ppl (Kart 13) — birikme 0.999 recall'da hala sorun mu?

Dosya: scaling\_probe\_colab.py, scaling\_probe\_results.json



\### Kart 13 — DEGER KANITI: 300k-egitimli r512 predictor, in-loop ppl (Colab)

24 katman, wikitext held-out 200 belge, baseline ppl 24.41. Ayni protokol,

tek degisken veri olcegi (6k -> 300k) + rank (128 -> 512):

| B | static | pred | pred@6k (Kart 9, ref) | net FFN kazanc\* |

|---|---|---|---|---|

| 0.40 | +1151% | **+1.1%** | +467% | %44 |

| 0.30 | +3213% | **+2.9%** | +1264% | %54 |

| 0.25 | +5629% | **+4.6%** | +2029% | %59 |

| 0.20 | +9095% | **+8.3%** | — | %64 |

\*net = 1 - B - 0.156 (r512 maliyeti, OPT 2-matmul FFN'e gore)

1\. Kart 9-11'in TUM negatifleri veri acligiymis. rec@0.40\~0.999'da birikme

&#x20;  yonetilebilir cikti — "error consistency" kaygisi yuksek recall'da eriyor.

2\. Statik tamamen oldu (+1151% vs +1.1%): buyuk-veri rejiminde predictor'in

&#x20;  katkisi uc mertebe. Stage A'dan beri suren "girdi sinyali gercek mi"

&#x20;  sorusunun kesin cevabi: GERCEK ve ppl duzeyinde belirleyici.

3\. SONUC CUMLESI (olculmus): OPT-1.3b'de dusuk-rank lineer predictor

&#x20;  (r512, %15.6 FFN maliyet) + B=0.30-0.40 butce = FFN hesabinda net %44-54

&#x20;  azalma, +1.1-2.9% ppl bedeliyle. PowerInfer/DejaVu mekanizmasi kendi

&#x20;  olcum zincirimizle yeniden uretildi ve butun basarisizlik kipleri haritalandi.

Dosya: colab\_train\_all\_predictors.py, predictor\_quality\_wt300k.json,

predictor\_weights\_wt300k\_r512.pt (Colab'dan INDIR — runtime silinir!)



\## Kapali kapilar (tekrar acma)

\- domain -> noron/expert haritasi (Kart 1 + Kart 2; iki mimaride de yok)

\- dense'i etiketle "bolme" (sanal konum != fiziksel ayrisma)

\- SiLU modelde skip-tabanli kazanc (Kart 5+6+7: sinir mimaride, muhendislikte degil)

\- predictor'i buyutme/inceltme SiLU'da (Kart 5: MLP lineer probu gecemedi, sinyal doymus)



\## Acik sorular

\- L6 > L30 > L18 tahmin-edilebilirlik sirasi (3 olcumde ayni) — neden?

\- OPT'de sira TERS (L20 > L12 > L4) — SiLU/ReLU farki mi, mimari mi?

&#x20; (OPT erken katmanlar neredeyse statik: canli %0.7, base recall 0.478)

\- Kart 5 NumPy sonuclarinin GPU-torch dogrulamasi (yarim saat, dusuk oncelik)

\- Esit-olmayan katman butcesi oracle egrisini iyilestirir mi (dusuk oncelik, SiLU kapandigi icin buyuk olasilikla gereksiz)



\## Siradaki

DEGER KANITI ALINDI (Kart 13). Arastirma fazi kapandi; secenekler:

1\. MUHENDISLIK FAZI: maske != hizlanma. Gercek kazanc icin skip'i fiilen

&#x20;  HESAPLAMAMAK gerek (gather/sparse-matmul; llm-manager CPU/GGUF baglaminda

&#x20;  row-skip fc1/fc2). Ilk olcum: naive PyTorch gather ile duvar-saati.

&#x20;  Ayrica rank/butce Pareto taramasi + katman-basi butce (erken katmanlar

&#x20;  Kart 11'e gore hassas — kontrol et).

2\. GENELLEME: ayni zincir baska ReLU/ReLUfied modelde (Bamboo-7B/ReluLLaMA)

&#x20;  — "OPT'ye ozgu mu" sorusunu kapatir. Colab'da \~1 saat.

3\. YOL AYRIMI (artik gerekceli): Qwen'i ReLUfy etmek yatirim olarak

&#x20;  savunulabilir hale geldi — kazanc zinciri ReLU'da olculdu. ProSparse

&#x20;  tarifi, yaz projesi olcegi; karar sana.

4\. YAZIM: 13 kart + 2 model + tum basarisizlik kipleri haritali —

&#x20;  blog/rapor olarak yazmaya deger bir hikaye oldu.

&#x20;  a) hazir ReLU/ReLUfied modelle hatti sonuna goturmek

&#x20;     (7B dogrulama RAM sinirinda, --limit 50 --max-tokens 128 ile sona birak)

&#x20;  b) kendi modelini ReLUfy etmek (ProSparse tarifi; DIKKAT: milyarlarca

&#x20;     token continued pretraining ister, LoRA olceginde degil — kapsami

&#x20;     bilerek gir, yaz projesi olcegi)



\## Dosyalar

scripts/collect\_sparsity\_v2.py, analyze\_sparsity\_v2.py, oracle\_quality.py, predictor\_quality.py

sparsity\_data\_v2/ (Qwen, 560 prompt, L6/18/30, prompt-id'li)

oracle\_quality\_report.json (Qwen), oracle\_quality\_opt13b.json (OPT)

dense\_sparsity\_stageA\_results.md, dense\_sparsity\_full\_results.md



\## Yontem notlari (degismez)

\- Her olcum: soru / kurulum / headline sayi / verdict / ne ogrendik

\- Pozitif sonucu KIRMAYA calis (split kontrolu, k-sweep) — kirilmazsa kart yaz

\- Vekil metrik uyarisi: recall vekildir, asil olcut uctan uca kalite (ppl)

\- Ayni protokol, farkli model = kontrollu karsilastirma; tek seferde tek degisken

\- SIRALI split yasak: dosya domain-sirali olabilir -> rastgele prompt-split (seed'li)

\- Per-layer metrik COMPOSE ETMEZ: in-loop olcum sart (Kart 9 dersi)

