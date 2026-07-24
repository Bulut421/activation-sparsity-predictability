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

GUNCEL (Kart 19): sinyal OPT'ye ozgu degil — ReluLLaMA-7B'de de var (2. model).

Dogal sifir spektrumu tezi dogruluyor: dogustan(%96) > ReLUfied(%67) > SiLU(~0).

OPT olcekleme 1M'e uzatildi: frac@0.99 hala dusuyor (%4-12), plato yok —

r256+ gerekiyor (r128 son-mili ifade edemiyor). Kart 18'in dusuk-B olumu

kismen aclikmis olabilir (Kart 20 adayi: 1M + r256+ ile dusuk-B dirilir mi).



\## Karsilastirma tablosu (ayni protokol, oracle skip, butce -> ppl delta)

| Butce | Qwen2.5-3B (SiLU) | ReluLLaMA-7B (ReLUfied) | OPT-1.3b (dogustan ReLU) |

|---|---|---|---|

| dogal sifir orani | \~0 | %67 | %96.1 |

| B=0.50 | +2.4% | +0.0% | -0.0% |

| B=0.30 | +20.7% | +0.1% | -0.0% |

| B=0.20 | +64.8% | +0.8% | -0.0% |

| B=0.15 | — | +2.0% | -0.2% |

| B=0.10 | — | +6.6% | +0.1% |

| B=0.05 | — | +22.1% | +1.1% |

SiLU: puruzsuz yamac. Dogustan-ReLU: sert plato, gec kirilma. ReLUfied: ARADA —

plato var ama daha kisa/yumusak (B=0.20'ye kadar bedava, sonra hizli kalkis).

SPEKTRUM = TEZIN DOGRULANISI: dogal sifir orani "seyreklik ne kadar egitime

gomulu" ile monoton — dogustan(%96) > sonradan-ReLUfied(%67) > hic(SiLU ~0).

Sonradan-secicilik, gomulu seciciligin olculebilir sekilde YUMUSAK taklidi.



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



\### Kart 14 — Kademe 0 mikro-benchmark: FLOP->saniye cevrimi GERCEK; yerlesim bedava

bench\_stage0.py, OPT FFN boyutlari (2048->8192->2048), 4 varyant, CPU.

T=1 (decode, 8 thread), full=2.38ms:

| B | gather (anlik dilim) | prebuilt (onceden dilim) | block (bitisik tavan) |

|---|---|---|---|

| 0.05 | 1.00x | **55.3x** | 43.2x |

| 0.20 | 0.41x | 6.8x | 6.8x |

| 0.30 | 0.33x | **3.9x** | 3.9x |

| 0.40 | 0.26x | 2.7x | 2.8x |

T=64 (prefill): naive gather BILE kazaniyor (1.1-5.0x); prebuilt 2.5-22x.

th=1: ayni desen (prebuilt 3.4x @ 0.30).

1\. **prebuilt ~ block HER NOKTADA** -> dagitik-ama-onceden-dilimlenmis =

&#x20;  bitisik. Bellek yerlesimi BEDAVA; PowerInfer-tarzi sicak/soguk yerlesim

&#x20;  muhendisligi bu donanimda GEREKSIZ. Tum bedel ANLIK dilimleme kopyasi.

2\. T=1 prebuilt superlineer (3.9x > ideal 3.3x @ B=0.30; B=0.05'te 55x):

&#x20;  matris-vektor bellek-bound, dilim kuculdukce cache rejimi degisiyor.

3\. Olu yol: token-basina maske + yeniden dilimleme (decode'da 0.2-0.6x).

SONUC: decode kazanc yolu iki secenege indi —

&#x20; a) MASKE KALICILIGI: maskeyi N token'da bir guncelle + prebuilt dilim

&#x20;    (ara cozum; N ve recall bedeli OLCULEBILIR: eldeki npz'lerden ardisik

&#x20;    token canli-kume ortusmesi -> Kart 15, model kosusu gerektirmez)

&#x20; b) kopyasiz fused gather-matmul kernel (buyuk is, tavani block verdi)

Dosya: bench\_stage0.py, bench\_stage0\*.json



\### Kart 15 — Maske kaliciligi: ZAYIF -> decode yolu kernel'e daraldi

Eldeki OPT npz'lerinden (synth, L4/12/20), model kosusu yok.

\- Ciplak kalicilik: recall(live(t+1) | live(t)) = 0.394 (L12); gap=16'da 0.224.

&#x20; Canli kumeler token-basina SERT degisiyor.

\- Union(live, N token): N=16 -> butce %28, sonraki-token recall 0.771;

&#x20; N=32 -> %41, 0.865. Hedef rejim ~0.999'un mertebe altinda.

\- Melez (canli(t) + statik-top30): L12 0.827 / L20 0.718 / L4 0.951 (j=1).

&#x20; Yine yetmez (L4 istisna gibi ama Kart 11: erken katman kacaklari pahali).

\- METOD NOTU: ReLU'da "top-B by |a|" esigi 0'a dusup TUM tensoru secebilir

&#x20; (sifir baglari) — bu metrik ReLU'da tanimsiz, olcumde bug olarak yasandi.

SONUC: "maskeyi N token sabit tut" ara cozumu OLDU (yuksek-recall rejiminde).

Kalan tek acik: egitilmis r512 predictor'un top-%30 maskesinin bayatlama

egrisi (bolluk payi yavas cozulure belki N=2-4 kurtarir; Colab ~10 dk,

beklenti dusuk). Esas yol artik net: KOPYASIZ gather-matmul kernel —

Kart 14 tavani (block ~ prebuilt ~ 3.9x @ B=0.30) kernel'in ulasabilecegi

hedefi zaten olctu; llm-manager CPU baglaminda AVX/C++ row-skip.

Dosya: mask\_persistence.py (koda dokuldu, 0c),
sparsity\_data\_opt/mask\_persistence\_report.json



\### Kart 16 — Prefill chunk-union entegrasyonu: NEGATIF (union sismesi)

prefill\_sparse\_proto.py: gercek OPT prefill, C token'lik parca + katman-basi

UNION maske, gather fc1/fc2 (compute fiilen atlaniyor). 30 wikitext belgesi, B=0.30:

| C | dense tok/s | sparse tok/s | speedup | union butce | ppl delta |

|---|---|---|---|---|---|

| 8 | 47.5 | 19.5 | 0.41x | %62.4 | +1.5% |

| 32 | 84.1 | 47.2 | 0.56x | %77.0 | +0.9% |

| 128 | 87.6 | 57.0 | 0.65x | %82.6 | +0.8% |

1\. **Union sismesi oldurucu:** 8 token'in top-%30 maskeleri bile %62'ye

&#x20;  birlesiyor (Kart 15 churn'unun dogrudan sonucu). Dilimlenmis FFN ~%62-83

&#x20;  FLOP + dilimleme kopyasi + predictor overhead'i -> her C'de kayip.

2\. Kalite beklendigi gibi: union noron ekler -> ppl +0.8-1.5%

&#x20;  (Kart 13 per-token +2.9'dan iyi). Sorun hizda degil kalitede olsaydi

&#x20;  cozum vardi; tersi.

3\. METOD DERSI: Kart 14'un T=64 kazanci "64 token AYNI maskeyi paylasir"

&#x20;  varsayimindaydi. Mikro-benchmark varsayimlari gercek is yukuyle

&#x20;  eslesmeli — paylasilamayan maske, paylasilmis-maske benchmark'ini bosa dusurur.

SONUC: prefill compute-skip (agirlik dilimleme yoluyla) KAPALI. Prefill zaten

compute-bound ve hizli; kazanc aranacak yer DECODE. Orada da Kart 14'un

"prebuilt ~ block" bulgusu umut veriyor: dagitik satirlar bir kez okunursa

bitisikten farksiz -> KOPYASIZ gather-matvec kernel'i (satirlari dogrudan

okuyan, materialize etmeyen) block'un ~3.9x'ine ulasabilir. Kart 17 adayi.

Dosya: prefill\_sparse\_proto.py, prefill\_proto\_report.json



\### Kart 17 — Kopyasiz gather-matvec kernel (numba, decode): YOL ACIK

bench\_kernel\_kart17.py, T=1, maske her cagrida farkli, dogruluk 1e-6 OK.

torch-full 2.293ms | numba-full 3.273ms (taban 0.70x) | predictor r512 0.260ms

| B | numba-gather | +pred (GERCEK decode) | tavan (prebuilt) |

|---|---|---|---|

| 0.05 | **12.85x** | 5.22x | 13.11x |

| 0.20 | 3.14x | 2.32x | 3.77x |

| 0.30 | 2.30x | **1.82x** | 3.12x |

| 0.40 | 1.54x | 1.31x | 2.33x |

1\. **Kernel fikri kayipsiz:** numba-ici kazanc (full/gather) B=0.30'da 3.29x

&#x20;  = FLOP oraninin kendisi. Tavana mesafe TAMAMEN numba-MKL taban farki

&#x20;  -> C++/AVX cogunu kapatir. B=0.05'te tavanla bas basa.

2\. Ilk OLCULMUS gercek decode kazanci (predictor dahil): 1.82x @ B=0.30

&#x20;  (+2.9% ppl, Kart 13), 2.32x @ B=0.20 (+8.3% ppl). Uctan uca kaba tahmin

&#x20;  (FFN ~2/3 decode FLOP): ~1.4x token gecikmesi @ B=0.30.

3\. Predictor 0.26ms = kernelin %26'si (B=0.30) — ucuzlatma kalemi belli:

&#x20;  r256 (Kart 12 kapasite Pareto'suna bak) ya da predictor'i da numba'ya alma.

4\. B=0.05 CAZIBESI: 5.22x — ama o butcede ppl OLCULMEDI (Kart 13 en dusuk

&#x20;  0.20'ydi, +8.3%). "Ultra-muhafazakar mod" ancak kalite olcumunden sonra.

VERDICT: Kart 14 tavani prototiple dogrulandi; kernel yolu ACIK.

Muhendislik hedefi netlesti: llm-manager C++/AVX row-skip + predictor

entegrasyonu, calisma noktasi B=0.20-0.30 bandinda kalite/hiz Pareto'su.

Dosya: bench\_kernel\_kart17.py, bench\_kernel\_kart17.json



\### Kart 18 — Dusuk-butce ppl egrisi (Faz 0a): ultra-muhafazakar mod OLU

predictor\_quality.py, ayni 200 wikitext belge, r512/300k ckpt, uc egri:

| B | oracle | static | pred | (kernel hizi, Kart 17) |

|---|---|---|---|---|

| 0.15 | +0.0% | +13382% | +16.4% | ~x |

| 0.10 | +0.0% | +25709% | +39.0% | ~x |

| 0.05 | +1.4% | +36595% | **+186.2%** | 5.22x |

1\. Oracle tavani saglam: B=0.10'a kadar tam 0, B=0.05'te +1.4%

&#x20;  (Kart 7 synth'te +1.1 idi — wikitext teyidi, ayni eval setinde).

2\. **Fark = saf tahmin hatasi ve bolluk payiyla olceklenir:** B=0.30'da

&#x20;  butce/canli ~6x -> hata affedilir (+2.9%); B=0.05'te ~1.1x -> her hata

&#x20;  vurur (+186%). Dusuk-B rejimi predictor kalitesine EN duyarli bolge.

3\. Statik bu bantta tamamen anlamsiz (+13k-37k%).

VERDICT: 5.22x cazibesi bu predictor kalitesinde gomuldu; calisma noktasi

B=0.20-0.30 bandinda kaliyor (1.82-2.32x @ +2.9-8.3%). NOT: ilkesel degil

koşullu olum — dusuk-B tavani acik (oracle ~0), fark tahmin hatasi, ve

Kart 12 egrisi hala canliydi -> 0b'deki 1M-token noktasi bu banda tekrar

baktirabilir (beklenti mutevazi).

Dosya: predictor\_quality\_lowB.json



\### Kart 19 — Genelleme (ReluLLaMA-7B) + OPT 1M-token olcekleme (Faz 0b)

Iki bolum, tek oturum (Colab A100).

BOLUM A — ReluLLaMA-7B (ReLUfied, gated FFN, L5/16/26, 23.5k token):

\- Dogal sifir %67 (OPT %96, Qwen ~0 arasinda — ARADA; spektrum tabloda).

\- Oracle: B=0.30'a kadar +0.1%, B=0.20 +0.8%, B=0.15 +2.0%, B=0.10 +6.6%,

&#x20; B=0.05 +22.1%. Plato VAR ama OPT'den KISA (OPT B=0.10'a kadar bedavaydi).

\- Sinyal: LIFT +0.162/+0.182/+0.256 (L5/16/26). Katman sirasi L26>L16>L5

&#x20; (gec>erken) — OPT deseni (L20>L12>L4), Qwen'in TERSI. ReLU-ailesi imzasi.

\- Full linear: recall 0.66-0.76, **w-recall 0.78-0.87** (buyuk noronlari

&#x20; yakaliyor, kucukleri kaciriyor — iyi desen). SVD dusuk-rank yine calisti:

&#x20; r256 net kazanc +40-45% (L16/L5), +33% (L26). frac@0.9 %48-54 ama bu

&#x20; 23.5k-token rejimi (OPT 6k gibi kotumser; veriyle duser).

\- SONUC: sinyal OPT'ye OZGU DEGIL. Ikinci ReLU-ailesi model, ayni zincir,

&#x20; ayni nitel sonuc. Iddia 1-model'den 2-model'e cikti (yazim icin sarttı).

\- GATED-RELU tuzagi calisti: A!=0 duzeltmesi olmasa canli oran (up<0 negatif-

&#x20; canlilar) yanlis olurdu. (a=relu(gate)*up; sifirlar gate<0'dan.)

BOLUM B — OPT-1.3b, olcekleme egrisine 1M nokta eklendi (scaling_probe):

| N | L4 frac@.99 | L12 | L20 | rec@0.40 |

|---|---|---|---|---|

| 300k | 0.30 | 0.19 | 0.21 | 0.994-0.999 |

| **1M** | **0.04** | **0.09** | **0.12** | **1.000** |

\- Egri HALA DUSUYOR, plato yok. 1M'de recall-0.99 icin sadece %4-12 noron

&#x20; (full-rank). rec@0.40 tam 1.000.

\- r128 KAPASITE DUVARI keskinlesti: frac@0.90 mukemmel (%2-7) ama frac@0.99

&#x20; kotu (%43-94; L4'te 300k'da 0.45 -> 1M'de 0.94!). Son mil (0.90->0.99)

&#x20; RANK istiyor; sinyal orada (full 0.04) ama r128 ifade edemiyor.

&#x20; -> 1M rejiminde r256-512 SART (Kart 12 kapasite bulgusunun kesin hali).

\- IMA: Kart 18'in dusuk-B olumu (300k'da) KISMEN aclik artefaktiymis. 1M +

&#x20; yeterli rank ile full frac@0.99=%4-12 -> dusuk-B rejimi DIRILEBILIR

&#x20; (gerek: 1M-egitimli r256+ predictor + in-loop ppl dogrulama). Kart 20 adayi.

Dosya: oracle\_relullama.json, sparsity\_data\_rll/, scaling\_probe\_results.json (1M)



\### Kart 21 — Faz 1: sifirdan dense ReLU bebek-model (TinyStories) EGITILDI

train\_baby.py, ~17.5M param (d=384, 8 kat, 6 head, FFN=fc1->ReLU->fc2),

466M token train, 20k step, RTX 5060 (~53 dk/seed). Iki seed.

\- Kalite: val ppl 4.45 (s0) / 4.47 (s1); uretimler tam tutarli TinyStories

&#x20; ("...she saw a big, scary monster with big teeth and red eyes. Lily was

&#x20; scared, but..."). Kontrol modeli A hazir (2 kopya).

\- **GURULTU CUBUGU (Faz 3 icin kritik): val 1.492 vs 1.496 -> Δ%0.27.**

&#x20; Cok dar. Faz 3'te "B ~ A" testi: B, A'nin ~%0.5'i icindeyse "esit" denir.

\- **DOGAL SEYREKLIK ~%89 (katman %85-95), SIFIR regularizasyonla.** Vanilla

&#x20; LM egitimi, hicbir seyreklik baskisi yok — yine de ReLU FFN kendiliginden

&#x20; %89 sifir uretiyor. "Dogustan secicilik" sifirdan-egitimde, minik olcekte,

&#x20; iki seed'de TEKRARLANABILIR sekilde dogrulandi. Kontrol modeli A ZATEN seyrek.

\- Katman deseni: L0 %85 (sinir/embedding dibi dususu), L1 tepe %95, sonra

&#x20; cikisa dogru monoton dusus L7 %85. OPT'ye KABACA benzer (erken-orta en

&#x20; seyrek, cikis en az) — ReLU-ailesi imzasi. (OPT: erken %99 -> gec %93.)

FAZ 3 ICIN IKI SOMUT SONUC:

1. A zaten ~%89 seyrek -> A uzerindeki ORACLE tavani (zorunlu on-adim) OPT

&#x20;  gibi plato yapmali (skip ~B=0.11'e = dogal canli orana kadar bedava).

2. Dogustan-goz B ANLAMLI olmasi icin butcesini dogal-canli ALTINA (~%11)

&#x20;  hedeflemeli — yoksa A'nin zaten yaptigi seyrekligi tekrar etmis olur.

Dosya: train\_baby.py, prepare\_tinystories.py, tinystories\_data/baby\_s{0,1}.pt,

baby\_report\_s{0,1}.json



\### Kart 22 — Faz 2: ProSparse-mini (seyreklik-cezali fine-tune) TUTTU

train\_prosparse.py: A'yi (baby\_s0) FFN aktivasyonuna progresif L1 cezasiyla

fine-tune. lambda ORCUYLE kalibre edildi: |a|≈0.031 olcup lambda=10 sectik

(ceza ~0.1 = CE'nin ~%7'si; lambda=0.05/0.5 cok zayifti, sifir kipirdamadi).

6000 step, RTX 5060.

\- **Sifir %89.4 -> %95.6 (+6.2 puan), ppl 4.557 -> 4.561 (+%0.1, GURULTU ICINDE).**

&#x20; OPT'nin %96'sina yaklasti — minik from-scratch modelde neredeyse bedava.

\- Egitim dinamigi ILGINC: ppl once yukseldi (step 2500'de 4.87, +%7) sonra

&#x20; INDI (step 6000'de 4.56). Model kisitlamayi absorbe etti — yeterli step'le

&#x20; kaliteyi geri kazandi. |a| 5x kuculdu (0.031->0.006) ve sifir yukseldi

&#x20; (sadece kuculme degil, gercek sparsification).

\- Uretim tutarli ("Anna was only three years old, but she was brave and strong.

&#x20; She had a special gift..."). Model = Faz 3'un 3. kolu (baby\_prosparse.pt).

TEZ RAFINMANI (onemli): "post-hoc secicilik kaliteyi kırar" cercevem BU

OLCEKTE fazla kabaymis — post-hoc SEYREKLIK itmesi ppl'de ~bedava cikti

(yeterli step'le). Demek ki dogustan-vs-post-hoc farki asil ppl'de DEGIL,

SECIMIN tahmin-edilebilirliginde olacak: dogustan-goz secim yapar (tahmin

edilecek oracle yok), post-hoc goz oracle'i taklit etmek zorunda (recall

0.999 yuku). Faz 3 (ii) tam bunu olcecek — ProSparse daha seyrek (%95.6)

oldugu icin predictor'i belki DAHA kolay, ama yuk hala taklit yuku.

Dosya: train\_prosparse.py, baby\_prosparse.pt, prosparse\_report\_prosparse.json



\### Kart 23 — Faz 3 on-adim: bebek-model oracle tavani (A + ProSparse)

oracle\_baby.py (oracle\_quality'nin BabyGPT hali). Iki tez tahmini de ONAYLANDI:

\- A (dogal sifir %89.3, canli %10.7): plato ~B=0.15'e kadar (+0.7-1.3%),

&#x20; kenar ~canli oranda; alti sert kalkis: B=0.11 +3.3%, 0.08 +6.4%, 0.05 +19.5%.

&#x20; -> Dogal-seyrek ReLU bebek-modelde OPT-tarzi plato (Kart 7 deseni bebekte).

\- ProSparse (sifir %95.6, canli %4.4): plato COK daha asagi — B=0.11 -0.1%,

&#x20; 0.08 +2.1% (A'da +6.4%'tu!), 0.05 +7.5%. Seyrek model ~3x agresif skip'e

&#x20; izin veriyor -> ProSparse'in deploy degeri OLCULDU.

KARAR: B modeli hedef butcesi ~%12.5 aktif (G=16 blok, k=2) — A'nin dogal

canlisinin (%10.7) hemen alti, post-hoc'un acimaya basladigi bolge. Basari:

B burada A'nin baseline'ina yakin + A'ya takilan predictor'i geciyorsa tez tutuyor.

Dosya: oracle\_baby.py, oracle\_baby\_baby\_s0.json, oracle\_baby\_baby\_prosparse.json



\### Kart 24 — Faz 3: dogustan-goz B (blok-router) EGITILDI, temiz ama oracle'i gecmedi

train\_born.py, BornGPT ~17.6M, G=16/k=2 (=%12.5 aktif), sifirdan 20k step,

A ile ayni veri/butce. Switch load-balancing (alpha=0.01).

\- **Router COKMEDI:** entropi 0.999-1.000 (uniform), max-blok %7 (ideal %6.25).

&#x20; Load-balancing tek seferde tuttu, alpha ayari gerekmedi. Kart 2 patolojisi yok.

\- Egitim: A'yi bastan sona %2-4 farkla golgeledi; uretim tutarli TinyStories.

\- Final val ppl **4.751** (%12.5 aktif). Karsilastirma:

&#x20; - A baseline (dense %100 FFN): ~4.48 -> B +%6

&#x20; - A oracle @0.125 (~4.57): B +%4 USTUNDE -> **oracle'i GECMEDI**

\- Yorum: dense agirliklar + kusursuz secim, sifirdan-routing'i ~%4 yeniyor

&#x20; (B kapasiteyi routing kisitiyla sinirladi). Bu bir kismi negatif — ama

&#x20; oracle ULASILAMAZ; asil test gerçek predictor.

VERDICT BEKLEMEDE: tez "dogustan-goz taklit yukunu kaldirir" -> B vs A'nin

GERCEK predictor'i (@0.125) karsilastirilmali (faz3\_compare.py). Bütçe≈canli

oran (%10.7) oldugunda A-predictor pahali olmali (Kart 9-11); B onu geciyorsa

tez tutar. Ayrica dogustan-goz DUSUK butcede (k=1, %6.25) daha cok parlamali —

orada A-oracle bile bozuluyor (@0.05 +19.5%), gercek predictor felaket olur.

Dosya: train\_born.py, born\_G16k2.pt, born\_report\_G16k2.json

\### Kart 25 — Faz 3 KARAR: DOGUSTAN-GOZ KAZANDI (taklit yuku olculdu)

faz3\_compare.py, butce %12.5, AYNI 120 val batch (cross-script gurultusu yok):

| Kol | ppl | Δ dense |

|---|---|---|

| A-baseline (dense %100 FFN) | 4.481 | — |

| A-oracle (ULASILAMAZ tavan) | 4.573 | +2.1% |

| A-static (girdiye bakmaz) | 1338 | +29763% |

| A-predictor (full-rank, 500k tok, EN IYI SANS) | 5.916 | +32.0% |

| **born-B (dogustan-goz, blok-router)** | **4.712** | **+5.2%** |

KARAR: born 4.712 vs A-predictor(en-iyi) 5.916 -> **BORN %20 ONDE**. born

oracle'a (4.573, +2.1%) neredeyse degiyor; en iyi predictor 6x uzakta.

SAGLAMLIK NOTU: ilk kosu (rank-128, 200k) A-predictor'i 14.55 (+225%) vermisti

— o EKSIK EGITIMDI. Full-rank + 500k ile en iyi sansini verdik: yine +32%.

Yani birikme felaketi (+225%) veri/kapasiteyle hafifliyor ama TAKLIT YUKU

kalmiyor: en iyi predictor bile oracle'in 30 PUAN uzerinde.

TAKLIT YUKU (olculmus) = predictor(+32%) - oracle(+2.1%) = ~30 puan ppl.

Dogustan-goz oracle'in sadece ~3 puan ustunde -> yukun ~%90'ini KALDIRIYOR.

MEKANIZMA: butce≈canli oran (%10.7) oldugunda post-hoc recall-paysiz TAHMIN

eder, kusurlu recall 8 katmanda birikir (Kart 9). Dogustan-goz TAHMIN ETMEZ

KARAR verir — kacirilacak "gercek aktivasyon" yok, agirliklar secime gore

orulmus. A-static +29763%: Stage A "global-core tuzagi"nin ppl teyidi.

Bu, projenin ana tezinin OLCULMUS cekirdegi: dogustan secici, post-hoc'un

acidigi butcede (1/8 FFN) oracle-yakini kalite tutuyor.

CAVEAT (durust): tek butce (%12.5), minik model, TinyStories, tek seed.

Genelleme: k=1 agresif (born daha cok parlamali) + buyuk model + 2. seed sirada.

Dosya: faz3\_compare.py, faz3\_compare\_b125.json



\### Kart 26 — Faz 3 guclendirme: agresif butce (k=1, %6.25) -> TREND

train\_born.py --k 1 (=%6.25 aktif, A canli %10.7'nin ALTINDA), sifirdan 20k.

Router k=1'de bile COKMEDI (entropi 1.000, max-blok %6.6 ≈ ideal %6.25).

faz3\_compare @%6.25 (full-rank predictor, 500k, ayni batch):

| Kol | ppl | Δ dense |

|---|---|---|

| A-baseline | 4.481 | — |

| A-oracle (ulasilamaz) | 5.035 | +12.4% |

| A-predictor (EN IYI SANS) | 11.572 | +158% |

| **born-B k1** | **5.482** | **+22.4%** |

BUTCELER-ARASI TREND (asil sonuc — nokta degil TREND):

| Butce | oracle | predictor | born | born vs pred | taklit yuku (pred-oracle) |

|---|---|---|---|---|---|

| %12.5 | +2.1% | +32% | +5.2% | **-20%** | ~30 puan |

| %6.25 | +12.4% | +158% | +22.4% | **-53%** | ~146 puan |

Butce daraldikca post-hoc predictor COKUYOR (+32->+158%), dogustan-goz zarifce

bozuluyor (+5.2->+22.4%). born'un oracle'a uzakligi her iki butcede de kucuk

(~3 / ~10 puan) -> **taklit yukunun ~%90'ini KALDIRIYOR, tutarli sekilde.**

Post-hoc'un yuku ise butceyle PATLIYOR (30->146 puan). Bu, tezin tek noktada

degil butce-eksemi boyunca TREND olarak dogrulanmasi.

Dosya: born\_G16k1.pt, born\_report\_G16k1.json, faz3\_compare\_b062.json



\### Kart 27 — born gurultu cubugu (tekrarlanabilirlik)

born G16k2 iki seed: seed0 val ppl 4.751, seed1 4.723 -> Δ%0.6. Kil payi;

dogustan-goz sonucu TEKRARLANABILIR, sansli init degil. Router iki seed'de de

kusursuz (entropi 0.999-1.000). Faz 3 kazanclari (-%20 @0.125, -%53 @0.0625)

bu gurultunun mertebelerce ustunde -> etki gercek.

(Not: seed1 kosusu born\_G16k2.pt'yi uzerine yazdi — dosya adi seed icermiyor;

iki seed ~ayni oldugu icin materyal etki yok, referans ckpt artik seed1.)

Dosya: born\_report\_G16k2.json (seed1 son hali)

\### Kart 28 — Router ablasyonu: KAZANCIN KAYNAGI CO-ADAPTASYON (routing degil)

Gerekce: entropi 1.0 router'in ISE YARADIGINI kanitlamaz. Kaynak (a) gövde mi

(b) ogrenilmis secim mi — ayirdik.

28a (eval-time swap, born\_G16k2, ayni batch): learned 4.72 -> random **1338**

(+28238%), fixed **1220** (+25743%). Govde router'a co-adapte -> swap COKUYOR.

Kosullu secim cesitliligi (Jaccard) **0.909** tum katmanlar -> routing girdiye

GERCEKTEN kosullu (sahte entropi degil). Ama co-adaptasyon confound'u -> 28b.

28b (SIFIRDAN egitim, tek fark router; ayni veri/butce/step):

| kol | ppl | ne ekler |

|---|---|---|

| dense A (%100 FFN) | 4.481 | referans |

| oracle @12.5% (ulasilamaz) | 4.573 | — |

| born LEARNED | 4.723 | — |

| born RANDOM (donuk router) | 4.827 | ogrenme: +0.10 (+%2) |

| born FIXED (girdi-bagimsiz) | 5.265 | girdi-kosullama: +0.44 |

| post-hoc predictor (en iyi) | 5.916 | co-adaptasyon: +0.65 |

AYRISMA (onem sirasi): CO-ADAPTASYON (post-hoc->fixed, +0.65) > GIRDI-KOSULLAMA

(fixed->random, +0.44) >> ROUTER-OGRENME (random->learned, +0.10).

1. **Ogrenilmis router EN AZ katki (+%2).** Tutarli girdi-bagimli HERHANGI bir

&#x20;  bolumleme (donuk-random dahil) yetiyor — govde etrafinda oruluyor.

2. **Fixed born bile post-hoc'u geciyor** (5.27 < 5.92). Born'un ustunlugu

&#x20;  routing zekasi DEGIL, gövdeyi seyrek-bilerek egitmek (co-adaptasyon).

3. Girdi-kosullama gercek katki (fixed 5.27 -> random 4.83): bolumleme girdiye

&#x20;  bagli olmali, ama OGRENILMIS olmasi sart degil.

TEZ RAFINMANI (zayiflatmaz, "neden"i duzeltir): born post-hoc'u yeniyor cunku

TAKLIT degil CO-ADAPTASYON. "Ogrenilmis secim" hikayenin yildizi degil.

CAVEAT (kritik): tiny olcek (k=2/G=16, 17.5M). Olcekte cok blok + zor gorevde

ogrenilmis routing cok daha onemli olabilir — "ogrenme onemsiz" KUCUK-OLCEK

ARTEFAKTI olabilir (6k-token negatifleri gibi). Faz 4c olcek testi bunu yoklar.

Dosya: router\_ablation.py, router\_ablation\_born\_G16k2.json,

born\_G16k2\_s0\_random.pt, born\_G16k2\_s0\_fixed.pt (+ report'lari)

\### Kart 29 — Faz 4a: granularite supurmesi -> ESIK + PLATO

Sabit %12.5 butce, degisen blok (train\_born --G/--k). Router hepsinde saglam

(entropi 1.000, max-blok ideale yakin, cokus yok).

| kol | ppl | secim uzayi C(G,k) | k |

|---|---|---|---|

| G8k1 (192-noron blok) | 5.430 | 8 | 1 (harman yok) |

| G16k2 (96-noron) | 4.723 | 120 | 2 |

| G32k4 (48-noron) | 4.723 | 35960 | 4 |

EGRI: kaba (G8) +%15 kotu; G16'da doyuyor; G32 HIC eklemiyor (4.723=4.723).

Yani granularite bir ESIGE kadar onemli (cok kaba = kotu), otesi PLATO.

CONFOUND (kayitli): G8k1 ayni zamanda TEK k=1 kolu (tek blok, gate-harman yok).

12.5% butcede k=1 -> zorunlu G=8. Yani "kaba blok" ile "k=1 harman yok"

ayrilamiyor (ayni butcede). Temiz olan: G16=G32 platosu (ikisi de k>=2, zengin,

birebir esit). Pratik sonuc: k>=2 kullan, G16 yeter, G32 gereksiz.

KART 28+29 BIRLESIK TABLO (born-eye neye ihtiyac duyar):

1. CO-ADAPTASYON (govdeyi seyrek-bilerek egit) — en buyuk (Kart 28).

2. GIRDI-KOSULLU bolumleme (fixed degil) — Kart 28.

3. YETERINCE ZENGIN/ince partisyon, k>=2 (cok kaba degil) — Kart 29; G16 yeter.

IHTIYAC DUYMADIGI: ogrenilmis (random yeter, K28) routing; sonsuz ince blok (K29).

FALSIFIKASYON COZULDU: "random~learned VE granularite duz -> ProSparse'a coker"

kosulunun IKINCI yarisi YANLIS cikti (granularite DUZ DEGIL, esik var). Yani

born ProSparse'a COKMUYOR: "co-adaptive + girdi-kosullu + blok-yapili yeterince

ince seyreklik" ProSparse'tan (tek sabit desen) DAHA ZENGIN bir sey. Router'in

akilli olmasi sart degil, ama partisyon uzayi girdi-kosullu ve yeterince ince olmali.

CAVEAT: tiny olcek; esik/plato yeri (G16) olcege gore kayabilir. 4c yoklar.

Dosya: born\_G8k1\_s0.pt, born\_G32k4\_s0.pt (+ report\_G{8k1,32k4}\_s0.json)



\### Kart 30 — Faz 4b: born decode kernel'i STOK matmul ile realize (implementasyon kritik)

bench\_born\_kernel.py: OPT-olcek FFN (d=2048 f=8192), T=1 decode, %12.5 butce.

Full vs born(cat) vs born\_loop(slice-view) vs posthoc(scatter) vs ceiling(prebuilt).

Sayilar (speedup, G16k2):

| kip | th=8 | th=1 |

|---|---|---|

| full | 1.00x | 1.00x |

| born (cat=fancy-index gather) | 2.12x | 1.31x |

| born\_loop (bitisik slice-VIEW) | **3.91x** | **7.41x** |

| posthoc (dagitik gather) | 2.17x | 1.22x |

| ceiling (prebuilt bitisik) | 6.73x | 8.75x |

KRITIK BULGU (implementasyon): W1[idx] fancy-index BITISIK olsa bile TAM gather-

kopya yapar -> born(cat) ≈ posthoc (avantaj COP). Ama born'un bloklari bitisik

oldugu icin W1[bas:bit] SIFIR-KOPYA slice-view kullanilabilir (born\_loop) ->

posthoc'un YAPAMADIGI sey (dagitik satir slice-view olmaz).

SONUC: born\_loop (slice-view) ≈ ceiling (7.41/8.75, th=1) VE posthoc'u ~6x geciyor

(7.41 vs 1.22). Born decode STOK matmul ile realize -> OZEL KERNEL GEREKSIZ.

Post-hoc'un K17 numba-0.70x ucurumu born'da YOK (bitisik = native matmul).

th=1 gercek decode rejimi (minik matvec overhead-bound; 8-thread kucukte zarar).

GRANULARITE-KERNEL: kaba blok kernel'i hafif seviyor (G8k1 9.66x, G16k2 7.41x?,

G32k4 6.32x @th8 — az/buyuk bitisik parca daha iyi). Kart 29 KALITEYI ince

istiyordu (G8 kotu) -> G16 iki tarafi dengeleyen TATLI NOKTA.

CAVEAT: micro-benchmark, tek FFN, CPU, T=1, stok-torch. (i) Gercek uctan-uca

decode tok/s (tum katman + attention + KV) = 4b-adim2, YAPILMADI. (ii) Python-

loop overhead'li born\_loop bir ALT SINIR; fuse'lu kernel ceiling'e daha yakin.

(iii) Prefill'de her token farkli blok -> MoE-dispatch (Kart 16), decode temiz.

Dosya: bench\_born\_kernel.py, (cikti kartta)



\### Kart 31 — Faz 4c: OLCEK (2x, 36M) -> makas DARALDI (durust, kismi negatif)

d=512 L=10 h=8 (~36M, 2x), TinyStories, ayni recete, Colab A100 (~20 dk/kosu).

Saturasyon YOK: A val ppl 4.14 (>3.5). Dogal seyreklik %89->**%92** (ölçekle artti).

| olcum | 17.5M | 36M |

|---|---|---|

| A-oracle @12.5% | +2.1% | +1.1% |

| A-predictor (en iyi) | +32% | +17.5% |

| born | +5.2% | +5.4% |

| **born vs post-hoc** | **-20%** | **-10.3%** |

| taklit yuku (pred-oracle) | ~30 puan | ~16 puan |

| learned vs random | +2.2% | +1.9% |

1. **Kart 28 OLCEKTE TUTUYOR:** born random 4.469 vs learned 4.387 = +1.9%

&#x20;  (≈ tiny +2.2%). "Ogrenilmis routing onemsiz" kucuk-olcek artefakti DEGIL (2x'e).

2. **Taklit yuku YARIYA indi** (30->16 puan) -> born goreli avantaji yariya

&#x20;  (-20%->-10%). Buyuk model daha seyrek (%92) -> post-hoc'un isi kolay. Kart 12'nin

&#x20;  (VERI olcegi post-hoc'u kolaylastirdi) MODEL-olcegi hali: hem veri hem model

&#x20;  olcegi post-hoc'u guclendiriyor. Born avantaji KISITLI rejimde en buyuk.

3. Born hala kazaniyor (-10.3%) ve born maliyeti sabit (+5.2->+5.4%). Born saglam.

AGRESIF REJIM TESTI (confound cozumu): born k=1 @36M (%6.25 < %8 canli),

faz3_compare @6.25%: A-oracle +6.4%, post-hoc **+77.6%**, born **+21.8%** ->

born -31.4%. Router k=1+olcekte de cokmedi (entropi 1.000).

TAM 2x2 TABLO (butce x olcek):

| butce | olcek | oracle | post-hoc | born | born-vs-pred | taklit yuku |

|---|---|---|---|---|---|---|

| %12.5 | 17.5M | +2.1% | +32% | +5.2% | -20% | ~30p |

| %12.5 | 36M | +1.1% | +17.5% | +5.4% | -10.3% | ~16p |

| %6.25 | 17.5M | +12.4% | +158% | +22.4% | -53% | ~146p |

| %6.25 | 36M | +6.4% | +77.6% | +21.8% | -31.4% | ~71p |

IKI EKSEN AYRISTI (temiz):

A. BUTCE ekseni (TEZ TUTUYOR): born avantaji agresif butcede cok daha buyuk

&#x20;  (36M'de -31% @6.25 vs -10% @12.5). Her olcekte. NARROWING butce-artefakti DEGIL.

B. OLCEK ekseni (yeni, durust): her butcede 2x olcek taklit yukunu ~YARIYA

&#x20;  indiriyor (30->16, 146->71); born maliyeti SABIT (+22->+22, +5->+5); post-hoc

&#x20;  yaklasiyor. Buyuk model daha seyrek (%89->%92) -> post-hoc'un isi kolay (Kart

&#x20;  12'nin veri-olceginin model-olcegi karsiligi).

SONUC: born HER butce+olcekte kazaniyor, agresif butcede net ustun; ama margin

olcekle eriyor. ~yariya-inme surerse cok buyukte post-hoc agresif butcede bile

yetisebilir — 2 noktadan EKSTRAPOLASYON, kesin degil. Tez: born-eye kisitli

rejimde (agresif butce, kucuk/orta model) en degerli; olcek imitasyon yukunu

kucultup makasi daraltiyor. "Her yerde daha iyi" DEGIL, "kisitli rejimi acan sey".

Dosya: faz3_compare_b125.json (36M), born_report_G16k2_s0{,_random}.json (36M),

baby_report_s0.json (36M, dogal sifir %92)



\### Kart 32 — Faz 4d-adim1: DEGISKEN-k (adaptif genislik) -> FAYDA YOK + korelasyon TERS

Soru: tek born-goz butcesini token-basi ADAPTIF dagitirsa (kolay token az blok / zor cok, ortalama %12.5 SABIT) sabit-k born'u (4.723) geciyor mu? Sabit top-k'da "zor token daha cok butce" TANIMSIZDI (her token tam k blok); 4d-adim1 bunu test eder. Mekanizma: bagimsiz sigmoid blok-kapilari, top-k YOK, aktif-sayi emergent (Kart 28'in 'ogrenilmis degisken-k' onerisi).

V1 (SOFT kapi, train_born_vark.py) -> SOFT-CHEAT, GERCEKLESTIRILEMEZ:

Yuzeyde POZITIF gorundu: val ppl 4.618, mean_g 0.125, corr(sayi,kayip) +0.166, zor/kolay 1.12x. AMA kapi aktivasyonu OLCEKLIYOR (sifirlamiyor) -> soft modda 16 blok da hesaplaniyor, tasarruf YOK. Falsifikasyon (eval_bornvar_hard.py, g>0.5 hard-esik): ppl 4.6 -> 551, compute %6. Kalite tamamen KESIRLI kapilardan geliyormus (ikili %78 = kapilarin %22'si tasiyici ortak kutle). Kok neden: train/deploy uyusmazligi (soft egit, hard deploy). DERS: "butce tuttu + corr>0" YETMEZ; deploy-esitligi (hard==soft) olcumu SART. Bu, 4d'nin Kart 9 ani (vekil-metrik tuzagi, in-loop/gercek olcum sart).

V2 (STE hard-forward, train_born_vark_ste.py) -> uyusmazlik YAPISAL kaldirildi:

g = g_hard + (g_soft - g_soft.detach()): forward 0/1 GERCEK secim, backward sigmoid. Forward artik HARD -> kesirli-kapi hilesi imkansiz (0.125'lik kapi hard'da 0 olur), rapor edilen val ppl DOGRUDAN deploy sayisi.

| kol | HARD ppl | Δ dense (4.481) |

|---|---|---|

| sabit-k born (G16k2) | 4.723 | +5.4% |

| degisken-k STE | 4.869 | +8.7%  (sabit-k'ya +3.1% KOTU) |

mean_g 0.123 (~%12.3, hedefe oturdu) + ikili %94 (v1'in %78'inden cok iyi = kararli hard-secim). Adaptasyon VAR: hard-sayi 1.96 (std ~1.0, buyuk token-arasi varyans; model sabit-2 yapmiyor, gercekten dagitiyor).

KORELASYON TERS DONDU: corr(sayi,kayip) = -0.111; kolay token 2.13 blok, zor token 1.81 blok (zor/kolay 0.85x). Model zor token'a AZ, kolay token'a COK blok veriyor — HIPOTEZIN TAM TERSI.

VERDICT (durust negatif, iki katmanli):

1. Adaptif-k sabit-k'yi GECMIYOR (+3.1%). Bu olcek/butcede degisken genislik sabit-k'ya bir sey katmiyor -> FIXED-k YETER.

2. ASIL BULGU: CE-zorlugu compute-tahsis EKSENI DEGIL. Yuksek-kayip token cogu zaman INDIRGENEMEZ surpriz (yeni isim, cumle basi) -> fazladan FFN yardim etmez; model blogu MARJINAL DEGERI olan yerde (tahmin-edilebilir yapida) harciyor. "Zor token daha cok butce hak eder" sezgisi bu olcekte YANLIS cikti.

ROADMAP ETKISI: GOZ HARITASI'nin gozler-arasi BUTCE TAKASI vizyonu ("bu token derin in ama seyrek hesapla") tam bu "zorluk -> butce" varsayimina dayaniyordu -> bu olcekte DESTEKLENMEDI. 4d-full (iki-goz agirlik+derinlik) gerekcesi ZAYIFLADI; yatirimdan once yeniden dusun. Kart 28 ritmi tekrar: ogrenilmis karar-katmani (routing / adaptif-butce) minik olcekte beklenenden AZ deger katiyor.

CAVEAT: (i) +3.1%'in bir kismi STE optimizasyon yanliligi olabilir (straight-through, top-k'dan lossier); AMA ters korelasyon STE-overhead'den BAGIMSIZ -> saglam bulgu (fayda-yoklugundan daha guclu iddia). (ii) tiny olcek (17.5M, %12.5); Kart 28/31 gibi olcekle kayabilir, ama ters korelasyonu duzeltir mi belirsiz. (iii) "zorluk" = token-CE; baska vekil (gate-margin, entropi, gradyan-normu) farkli sonuc verebilir (dusuk oncelik, acik soru).

Dosya: train_born_vark.py (v1 soft, negatif kayit), eval_bornvar_hard.py (falsifikasyon), train_born_vark_ste.py (v2 STE), bornvar_ste_report_G16t0.125_s0.json, bornvar_report_G16t0.125_s0.json (v1 soft-cheat kaydi)



\### Kart 33 — Faz 4b-adim2: UCTAN UCA decode tok/s -> born kerneli e2e KALIYOR (Amdahl-sinirli)

Soru: Kart 30'un izole FFN kazanci (~7x @th1) TUM modelde (tum katman + attention + KV-cache + head) ne kadar kaliyor? born SADECE FFN'i hizlandirir -> uctan uca kazanc FFN'in decode-PAYIYLA sinirli (Amdahl). bench_decode_e2e.py: B=1 tek-akis, KV-cache'li GERCEK decode (T=1/adim), 4 kol (none / dense / born-sliceview / posthoc-scatter+predictor), CPU th=1, 4 olcek. slice-view dogrulugu (verify): max|Δ|=0, birebir.

| olcek | ~par | FFN payi | FFN-only born | e2e born | e2e posthoc |

|---|---|---|---|---|---|

| baby d384/L8/f1536 | 17M | 48.8% | 1.67x | 1.24x | 0.75x |

| mid d768/L12/f3072 | 91M | 54.8% | 3.46x | 1.64x | 0.91x |

| large d1536/L24/f6144 | 693M | 61.0% | 5.51x | 2.00x | 1.08x |

| opt d2048/L24/f8192 | 1226M | 63.0% | 6.22x | 2.12x | 1.23x |

(teorik FFN FLOP orani her olcekte 8x = 1/butce.)

1. born kerneli e2e KALIYOR ve OLCEKLE ACILIYOR: e2e born 1.24 -> 2.12x. FFN-only 1.67 -> 6.22x, Kart 30 izole tavanina (~7x @th1) YAKLASIYOR — minik matvec'te Python-loop/overhead FFN'i boguyordu (baby 1.67x); buyudukce ~8x FLOP orani gorunur oluyor (opt 6.22x ~ Kart30 7.41x, tam-model baglami kucuk farki aciklar). Izole Kart 30 kernel olcumu TAM-MODELDE dogrulandi.

2. AMA Amdahl TAVANI belirleyici: FFN decode-zamaninin %49-63'u (olcekle artiyor); geri kalan (attn + head + norm + embed) born'a DOKUNULMAZ. Sonsuz-hizli FFN bile opt'ta e2e <= 1/(1-0.63) = 2.7x. born 2.12x -> mevcut basligin ~%80'ini aliyor. "8x FLOP" reklami e2e'de 2.1x'e iner -> FLOP orani != duvar-saati orani; DURUST decode sayisi budur.

3. POSTHOC E2E COKUYOR: 0.75 -> 1.23x (baby'de dense'ten YAVAS!). Scatter-gather (dagitik d_ff satir kopyasi) + predictor overhead kazanci yiyor; posthoc e2e ~ izole posthoc (Kart30 1.22x) cunku FFN'i neredeyse hic hizlandirmiyor. born HER olcekte posthoc'u e2e ~1.7-1.85x geciyor. Kart 30'un "born bitisik / posthoc dagitik" farki TUM-MODELDE, duvar-saatinde teyitli.

4. Projenin CEKIRDEK muhendislik sorusu ("az-calistirma gercek decode hizina cevriliyor mu") UCTAN UCA cevaplandi: EVET, ~2.1x gercek decode (1B-sekil, CPU, th1) — ve born'un BITISIK yapisi SART; post-hoc'un dagitik gather'i e2e neredeyse hicbir sey vermiyor (1.2x).

VERDICT: 4b POZITIF ve KAPANDI. born decode kerneli stok-matmul ile realize (Kart 30, izole ~7x) + uctan uca ~2x hizlanma (bu kart). "20W paradoksu / az-calistirma" north-star'ina ilk duvar-saati e2e sayisi. NOT: bildirilen Amdahl tahmini == D/B OZDESLIGI (bagimsiz dogrulama degil, ayristirmanin kendisi); asil sonuc TREND + posthoc karsilastirmasi.

CAVEAT: (i) th=1 (temiz kucuk-matvec / decode-latency rejimi); cok-cekirdek AYRI soru (Kart 30: th=8 born'a daha cok zarar veriyordu). (ii) FFN payi VOCAB (head) + BAGLAM UZUNLUGUNA bagli: burada vocab=8192, ctx ~64-128. Gercek 50k vocab / uzun baglam -> attn+head payi artar -> FFN payi ve e2e-born DUSER; uzun baglamda born (FFN gozu) tek basina yetmez, KV gozu (goz #2) gerekir -> GOZ HARITASI'na dogal kopru. (iii) rastgele agirlik (zamanlama deger-bagimsiz). (iv) CPU hedef (edge/llm-manager); GPU minik-batch launch-bound.

Dosya: bench_decode_e2e.py, bench_decode_e2e.json



\## Kapali kapilar (tekrar acma)

\- domain -> noron/expert haritasi (Kart 1 + Kart 2; iki mimaride de yok)

\- dense'i etiketle "bolme" (sanal konum != fiziksel ayrisma)

\- SiLU modelde skip-tabanli kazanc (Kart 5+6+7: sinir mimaride, muhendislikte degil)

\- maske kaliciligi / "N token'da bir guncelle" (Kart 15: churn cok sert)

\- prefill compute-skip, agirlik dilimleme yoluyla (Kart 16: union %62-83'e sisiyor)

\- predictor'i buyutme/inceltme SiLU'da (Kart 5: MLP lineer probu gecemedi, sinyal doymus)



\## Acik sorular

\- L6 > L30 > L18 tahmin-edilebilirlik sirasi (3 olcumde ayni) — neden?

\- OPT'de sira TERS (L20 > L12 > L4) — SiLU/ReLU farki mi, mimari mi?

&#x20; (OPT erken katmanlar neredeyse statik: canli %0.7, base recall 0.478)

\- Kart 5 NumPy sonuclarinin GPU-torch dogrulamasi (yarim saat, dusuk oncelik)

\- Esit-olmayan katman butcesi oracle egrisini iyilestirir mi (dusuk oncelik, SiLU kapandigi icin buyuk olasilikla gereksiz)

\- Birlesik secicilik: tum "gozler" (noron-skip / KV-cache / early-exit / router)

&#x20; tek secicilik ilkesiyle BIRLIKTE ogrenilebilir mi? (eklenti goz != dogustan goz;

&#x20; Kart 9-10 birikme problemi bunun kucuk golgesiydi)

&#x20; - Teknik zemin var: tum gozler ayni x'i (residual stream) okuyor; tek

&#x20;   dusuk-rank "bakis govdesi" + karar kafalari mumkun. Asil yenilik

&#x20;   GOZLER-ARASI BUTCE TAKASI: "bu token sig gec ama genis bak; su token

&#x20;   derin in ama seyrek hesapla" — bugun hicbir sistemde ogrenilmis degil.

&#x20;   Kart 9-10 uyarisi: gozler etkilesir, birikme katlanir -> ancak BIRLIKTE

&#x20;   egitilirse yapilabilir.

&#x20; - TAHMINCI-GOZ vs KARAR-GOZ ayrimi (16 kartin ozeti): eklenti goz bir

&#x20;   tahmincidir — modelden habersiz bir "gercek kume" vardir, onu bilmek

&#x20;   zorundadir; tum acilar (recall 0.999, birikme, churn) bu taklit yukunden.

&#x20;   Maske egitim dongusune girerse goz karar vericiye donusur: taklit

&#x20;   edilecek oracle kalmaz, sectigi sey hesabin kendisidir (MoE router'inin

&#x20;   "recall"u diye kavram olmamasi bundan). ReLUfication yarim cozum

&#x20;   (dunyayi goze okunur yapar, goz disarida kalir); tam cozum skip'i

&#x20;   egitime koyup "predictor'la dogmus" model — MoE'nin noron-granularitesi,

&#x20;   "MoE'nin ilerisi"nin muhtemelen en durust formulasyonu. (Dogustan gozun

&#x20;   da bakimi var: load-balancing, router collapse — Kart 2. "Yeterli" =

&#x20;   tahmin yukunden muaf, bedava degil.)



\## Siradaki

DEGER KANITI ALINDI (Kart 13). Arastirma fazi kapandi; secenekler:

KART 17 TAMAM — kernel yolu dogrulandi. 17 kartlik arastirma zinciri kapandi.

PLAN OTURUMU bekliyor. Masadaki kalemler:

&#x20; a) llm-manager C++/AVX row-skip + predictor entegrasyonu (ana muhendislik)

&#x20; b) predictor ucuzlatma: r256 Pareto + predictor'i kernel'e alma

&#x20; c) B=0.05-0.15 kalite olcumu (5.2x cazibesi icin ppl egrisi eksik)

&#x20; d) uctan uca gercek-model decode olcumu (tokens/s, tum katmanlar)

&#x20; e) genelleme: Bamboo-7B/ReluLLaMA ile ayni zincir (~1 saat Colab)

&#x20; f) yazim: blog/rapor (17 kart, 2 model, tum basarisizlik kipleri haritali)

&#x20; g) uzun vade: ReLUfication / skip'i egitime koyma ("dogustan goz",

&#x20;    Acik sorular'daki birlesik secicilik notuyla birlikte)

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

