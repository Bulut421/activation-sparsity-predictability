# ROADMAP — "Dogustan Goz" Fazi ve Sonrasi
Son guncelleme: 2026-07-24
Sahip: Bulut (Kler3) | Bu belge SPARSITY_NOTES.md'nin devami — o belge 17 kartlik
ARASTIRMA fazinin kaydi (kapandi), bu belge SIRADAKI fazlarin haritasi.
Cowork/Claude: yeni oturuma bu dosya + SPARSITY_NOTES.md ile baslanir.

## Durum ozeti (tek paragraf)
17 kartlik zincir kapandi. Kanitlanan: (1) FFN canli-noron kumesi x'ten
dusuk-rank lineer predictor ile tahmin edilebilir — ama SADECE ReLU-ailesi
modellerde ve yeterli veriyle (300k token; 6k'daki tum negatifler aclik
artefaktiydi). (2) Deger: OPT-1.3b, B=0.30-0.40 -> FFN net %44-54 azalma,
+1.1-2.9% ppl (Kart 13). (3) Duvar-saati: kopyasiz gather-matvec ile gercek
decode kazanci 1.82x @ B=0.30, predictor dahil (Kart 17). Kapanan yan yollar:
maske kaliciligi (churn sert, Kart 15), prefill agirlik-dilimleme (union
sismesi, Kart 16). ANA TEZ (Acik sorular'da): eklenti goz TAHMINCIDIR ve tum
acilar (recall 0.999, birikme, churn) taklit yukunden gelir; maske egitim
dongusune girerse goz KARAR VERICIYE donusur, taklit edilecek oracle kalmaz
(MoE router'inda "recall" kavraminin olmamasi). Nihai hedef bu tezin testi.

## Buyuk resim: 4 faz + 1 paralel hat
FAZ 0 (acik kapanislari topla + yazim)  ->  1-2 hafta, hafif
FAZ 1 (pretrain kasi + kontrol modeli)  ->  yeni beceri, dusuk risk
FAZ 2 (ProSparse-mini: yarim cozum)     ->  tanidik zemin (fine-tune), yeni kavram
FAZ 3 (dogustan goz vs eklenti goz)     ->  ana deney, tezin testi
KERNEL HATTI (paralel/park)             ->  llm-manager C++/AVX, bagimsiz ilerler

Ilke: her faz bir oncekinin CIKTISINI kullanir; hicbir adim buyuk bahis degil.
Faz 3'e varildiginda 3 tasarim sorusundan 2'si zaten olculmus olacak.

---

## FAZ 0 — Acik kapanislar + yazim (once bu)
Amac: yeni faza temiz masayla girmek. Hepsi ucuz, bekledikce deger kaybediyor.

0a. [TAMAM -> Kart 18] B=0.05-0.15 ppl egrisi
    Sonuc: ultra-muhafazakar mod olu (+186% @ 0.05); oracle tavani saglam
    (+1.4% @ 0.05); fark saf tahmin hatasi, bolluk payiyla olcekleniyor.
    Calisma noktasi B=0.20-0.30'da kaldi. Dusuk-B: kosullu olum, 1M-token
    noktasi (0b) tekrar baktirabilir.
    --- orijinal plan ---
    Kart 17'nin 5.22x @ B=0.05 cazibesi kalitesiz olculmedi (Kart 13 en dusuk
    0.20'ydi, +8.3%). predictor_quality.py mevcut haliyle yapar:
    --budgets 0.15,0.10,0.05 --modes oracle,static,pred
    --load-preds predictor_weights_wt300k_r512.pt
    (oracle sutunu Kart 7'yle kopru olur; --modes ACIK yazilsin.)
    Gereksinim: wikitext_eval.jsonl — yoksa regen_wikitext_eval.py ile
    deterministik yeniden uretilir (seed=0, birebir ayni belgeler).
    Cikti: Kart 18 (ultra-muhafazakar mod yasar mi olur mu).

0b. [TAMAM -> Kart 19] Genelleme + 1M olcekleme
    ReluLLaMA-7B: sinyal var (LIFT +0.16-0.26), OPT'ye ozgu DEGIL -> iddia
    2-model. Dogal sifir %67 = dogustan(%96)/SiLU(~0) arasi; spektrum tezi
    dogruluyor (ReLUfied = yumusak taklit). OPT 1M: frac@0.99 %4-12'ye dustu
    (plato yok) ama r128 son-mili tutmuyor -> r256+ sart. Kart 18 dusuk-B
    olumu kismen aclik olabilir -> KART 20 ADAYI: 1M + r256 in-loop dusuk-B.
    --- orijinal plan ---
    Ayni zincir ikinci ReLU modelde (Bamboo-7B ya da ReluLLaMA-7B; RAM icin
    --limit 50 --max-tokens 128). Soru: "OPT'ye ozgu mu?" — YAZIMDAN ONCE
    cevaplanmali, iddiayi tek-model'den iki-model'e cikarir.
    !! GATED-RELU TUZAGI: ReluLLaMA'da a = relu(gate) (.) up — sifirlar
    gate'ten gelir ama sifir-olmayan girdiler NEGATIF olabilir. Canli maske
    (A > 0) DEGIL (A != 0) olmali; yoksa negatif-canli noronlar olu sayilir
    ve Kart 19 yanlis etiketle olculur. (analyze_sparsity_v2.py duzeltildi;
    OPT'de davranis ayni — relu ciktisi zaten >= 0.) Bamboo secilirse de
    ayni kontrol: FFN'in son projeksiyon girdisinin isaret dagilimina BAK.
    Ayni oturumda 1M token noktasi da alinabilir (olcekleme doyma sorusu).
    Bellek notu: 3 katman x 1M token ~ 25GB+ CPU RAM (X fp16 + M bool) —
    H100 runtime'da olur ama bilerek girilsin; gerekirse 2 katmana dus.
    Cikti: Kart 19.

0c. GitHub guncelleme
    Repo Kart ~15'te kaldi. Eklenecek: Kart 16-17 (+ 0a/0b ciktilari),
    bench_kernel_kart17.py, prefill_sparse_proto.py, guncel SPARSITY_NOTES,
    bu ROADMAP. Kart 15'in analiz scripti sandbox'ta kalmisti — artik
    scripts/mask_persistence.py olarak yazildi, repoya girsin
    (tekrarlanabilirlik ilkesi: "sayilar kartta ama kod kayip" olmaz).
    README'ye 3-egri grafigi (K13 verisinden, 20 satir matplotlib).

0d. YAZIM + paylasim (Faz 1'den once, sonra degil)
    Gerekce: hikaye SU AN butun ve kapanmis; yeni faz baslayinca yazma
    enerjisi olur. Paylasim Faz 3'ten once geri bildirim / potansiyel
    isbirlikci getirir ("grubum olsun" hedefi). Icerik hazir: 17 kart,
    2 model, tum basarisizlik kipleri haritali, tahminci-goz/karar-goz tezi
    sonuc bolumu. Kanallar: blog (README'den turetilir) -> r/LocalLLaMA,
    EleutherAI Discord, HF. Ingilizce.

---

## FAZ 1 — Pretrain kasi + kontrol modeli (dense TinyStories)
[TAMAM -> Kart 21] ~17.5M ReLU GPT, 2 seed, val ppl 4.45/4.47, tutarli uretim.
  Gurultu cubugu Δ%0.27 (Faz 3 "esit" esigi ~%0.5). Yan bulgu BUYUK: dogal
  seyreklik ~%89 (regularizasyonsuz!) -> "dogustan secicilik" sifirdan-egitimde
  dogrulandi; kontrol A ZATEN seyrek. Sonuclar: baby_s{0,1}.pt, baby_report*.
  Faz 3 rafinman: (1) oracle-on-A plato yapmali (~B=0.11'e kadar bedava);
  (2) dogustan-goz B butcesi dogal-canli ALTINA (~%11) hedeflemeli, yoksa
  A'nin zaten yaptigini tekrar eder. --- orijinal plan asagida ---

Amac: (1) sifirdan egitim becerisi — bugune kadar hep fine-tune yapildi;
veri akisi, LR schedule, kayip egrisi okuma, checkpoint disiplini yeni kas.
(2) Faz 3'un KONTROL MODELI (dense ikiz A) burada uretilmis olur. Iki kus.

Kurulum onerisi:
- Olcek: ~10-30M parametre (TinyStories literaturu bu olcekte anlamli dil
  ogrenildigini kanitladi — Eldan & Li 2023). 4-8 katman, d_model 256-512.
- AKTIVASYON: ReLU (bilerek — Faz 2-3 zemini; SiLU kapisi zaten kapali).
- Veri: TinyStories (HF'de hazir), tokenizer kucuk (4-8k vocab yeterli).
- Donanim: Colab H100 birkac saat; RTX 5060 lokalde de dener (yavash ama olur).
- Basari olcutu: uretilen hikayeler tutarli mi (TinyStories degerlendirme
  gelenegi) + val loss egrisi saglikli mi. Mukemmellik ARANMAZ — bu bir
  kontrol modeli + ogrenme turu.
- IKI SEED: A modeli iki farkli seed'le egitilir (bu olcekte ucuz). Gerekce:
  Faz 3'un ana iddiasi "B ~ A ppl" — A'nin kosudan-kosuya ppl varyansi
  bilinmeden "yakin"in anlami yok. Seed-farki = gurultu cubugu; B-A farki
  bu cubugun icindeyse "esit" denebilir (tek-degisken ilkesinin istatistik ayagi).
- Yan olcum (bedava, merakli): bu ReLU bebek-modelde dogal sifir orani kac?
  Katman deseni OPT'ye benziyor mu? (Acik soru: katman-sirasi gizemi.)

Riskler: pretrain'in "ilk seferde calismamasi" normaldir; LR/init/stabilite
turlari beklenir. Bu ogrenmenin kendisi Faz 3 icin sart.

---

## FAZ 2 — ProSparse-mini (yarim cozum, ara basamak)
[TAMAM -> Kart 22] L1=10 (|a|≈0.031 olcup kalibre), 6000 step. Sifir
  %89.4 -> %95.6 (+6.2p), ppl +%0.1 (GURULTU ICINDE — ~bedava). Model
  baby_prosparse.pt = Faz 3'un 3. kolu. TEZ RAFINMANI: post-hoc SEYREKLIK
  itmesi bu olcekte ppl'de ~bedava -> dogustan-vs-post-hoc farki ppl'de
  DEGIL, SECIMIN tahmin-edilebilirliginde aranmali (Faz 3 ii). --- plan ---
Amac: turevlenebilirlik sorusunun YARISINI ucuza ogrenmek + Faz 3'e ucuncu
karsilastirma kolu uretmek. Tam "dogustan goz"e atlamadan, seyrekligi
SONRADAN-AMA-EGITIMLE guclendirme (ReLUfication'in minyaturu).

Kurulum:
- Faz 1'in dense modelini al, seyreklik regularizasyonuyla fine-tune et
  (ProSparse tarifi kucukte: kaydirilmis ReLU + L1-tarzi aktivasyon cezasi,
  kademeli artan katsayi).
- Olc: dogal sifir orani nereden nereye cikti? ppl bedeli? Ayni predictor
  zinciri (collect/analyze) bu modelde ne veriyor?
- Bu, senin fine-tune kaslarinla (Unsloth/LoRA donemi) tanidik zemin —
  yeni olan sadece kayip fonksiyonu.

Cikti: "yarim cozum" veri noktasi. Faz 3 tablosunda uc kol olacak:
dense+eklenti goz  vs  ProSparse-mini+eklenti goz  vs  dogustan goz.

---

## FAZ 3 — ANA DENEY: dogustan goz vs eklenti goz
[KARAR ALINDI -> Kart 23-25] TEZ TUTTU. Blok-router B (G16k2, %12.5 aktif)
  sifirdan egitildi, router COKMEDI (entropi 0.999). faz3_compare @%12.5,
  ayni batch: A-predictor EN IYI SANS (full-rank, 500k tok) +32% ppl; born-B
  +5.2% (oracle +2.1%'e yakin). Born, en iyi predictor'i %20 yeniyor.
  TAKLIT YUKU olculdu = predictor(+32) - oracle(+2.1) = ~30 puan; born yukun
  ~%90'ini kaldiriyor. (Ilk rank-128/200k kosu +225%'ti = eksik egitim;
  full-rank +32% saglam alt sinir.) Router cokmedi (entropi 0.999).
  GUCLENDIRME (Kart 26): k=1 (%6.25) -> born vs predictor farki %20'den %53'e
  ACILDI (predictor +32->+158%, born +5.2->+22.4%). Butce-ekseni boyunca TREND:
  born taklit yukunun ~%90'ini tutarli kaldiriyor; post-hoc'un yuku butceyle
  patliyor (30->146 puan). Tez tek nokta degil TREND olarak tuttu. Router k=1'de
  de saglam. Kalan (istege bagli): buyuk model + 2. born seed. --- plan asagida ---
Tezin testi: maske egitim dongusune girerse taklit yuku kalkar mi?

ON-ADIM (zorunlu, Kart 6 dersinin kurumsallasmasi): B'yi egitmeden ONCE
A uzerinde ORACLE butce->ppl egrisi cikarilir (oracle_quality.py, kucuk
uyarlama). B'nin hedef butcesi bu egriden secilir — tavani olcmeden hedef
koymak yasak. (Faz 1 yan olcumu dogal seyrekligi verir; dusukse Faz 3
butcesi ona gore ayarlanir — bebek modelde %95 beklenmeyebilir.)

Tasarim (geriye dogru kurgulandi):
- Model B ("dogustan"): Faz 1 ile AYNI mimari/veri/butce, tek fark FFN'de
  ogrenilmis secici — her token noronlarin sadece B%'si aktif, secici modelle
  BIRLIKTE egitiliyor.
- Karsilastirmalar:
  (i)   B vs A: ayni AKTIF-FLOP'ta ppl (adil karsilastirma aktif hesap
        uzerinden, toplam parametre uzerinden degil — MoE literatur standardi)
  (ii)  TAKLIT YUKU TESTI — operasyonel tanim (dis-predictor recall'u
        KULLANILMAZ: B'nin secicisi x'in deterministik fonksiyonu, dis
        predictor onu kopyalayip recall 1.0 tutturabilir, test bosa duser).
        Bunun yerine PERTURBASYON DUYARLILIGI: B'nin sectigi kumeden rastgele
        %p dusur vs A+predictor maskesinden %p dusur -> iki ppl-duyarlilik
        egrisi. Tez dogruysa: A-kolu %p'ye karsi sert kirilir (kacirilan
        oracle-uyesi pahali), B-kolu yumusak bozulur (kacirilacak oracle yok,
        model kendi secimine gore orulmus). Fark = taklit yukunun OLCUSU.
        Ek: birikme/churn B'de var mi (ayni in-loop protokol).
  (iii) Kart 2 patolojisi: secici cokuyor mu (hep ayni %B)? Olcum: secim
        entropisi / token-basina kume cesitliligi. Load-balancing analogu
        gerekirse ekle — "dogustan gozun bakimi var, bedava degil".

UC ACIK TASARIM SORUSU (Faz 3 oncesi cevaplanacak; 2'si Faz 1-2'den gelir):
1. Turevlenebilirlik: top-k secim turevlenemez. Adaylar: straight-through,
   softmax gating (MoE yolu), aktivasyon-cezali kendiliginden seyreklik
   (ProSparse yolu — Faz 2 bunu olcer). Kucuk olcekte hangisi stabil?
2. Granularite: noron mu blok mu? Kart 17 kernel gercegi blok lehine
   (egitimde stabil + donanimda hizli). Blok secilirse kernel hattiyla
   ayni noktaya baglanir. Faz 1 yan olcumu buna veri verir.
3. Cokus onleme: metrik bastan tanimli (yukarida iii).

Basari/basarisizlik tanimlari BASTAN:
- POZITIF: B, A'ya yakin ppl'de (aktif-FLOP esit) + gozu tahmin yukunden muaf
  (ii'de fark acik) -> tez dogrulandi, "MoE'nin ilerisi"nin ilk kaniti.
- NEGATIF de degerli: B egitilemiyorsa/cokuyorsa hangi kipte cokuyor —
  o da kart olur (Kart 9-11 gelenegi: basarisizlik kipleri haritalanir).

---

## KART 28 — Router teshisi + ABLASYON  [TAMAM -> SPARSITY_NOTES Kart 28]
SONUC: ayrisma = co-adaptasyon (post-hoc 5.92 -> fixed-born 5.27, +0.65) >
girdi-kosullama (fixed 5.27 -> random 4.83, +0.44) >> router-ogrenme
(random 4.83 -> learned 4.72, +0.10). Ogrenilmis router EN AZ katki (+%2);
fixed-born BILE post-hoc'u geciyor. Born'un ustunlugu CO-ADAPTASYON, routing
zekasi degil. Eval-swap: random/fixed co-adapte govdede cokuyor (+28000%);
kosullu cesitlilik 0.909 (routing gercekten girdiye bagli). CAVEAT: tiny olcek,
"ogrenme onemsiz" kucuk-olcek artefakti olabilir -> 4c yoklar. --- ozgun plan ---
Gerekce (Bulut): entropi 1.0 COKUS olmadigini kanitlar, router'in ISE YARADIGINI
kanitlamaz — rastgele router da entropi 1.0 verir. Kazanc (a) maskeye gore orulmus
GOVDEDEN mi, (b) ogrenilmis SECIMDEN mi? Henuz ayrilmadi. Bu, Faz 3 iddiasindaki
gercek acik: kapatmadan Faz 4'e genisleme kor.

ABLASYON — IKI SEVIYE (Claude eki, onemli ayrim):
  UCUZ (eval-time router swap, ayni ckpt, ayni batch, faz3_compare tarzi):
    1. ogrenilmis (mevcut)  2. rastgele-per-token (ayni k)  3. sabit-blok (girdi-bagimsiz)
    DIKKAT confound: govde THIS router'a co-adapte oldugu icin random/fixed'de
    cokebilir -> bu "kirilganlik" testi, "ogrenme gerekli mi"nin TAM cevabi DEGIL.
  DECISIVE (~1 saat egitim): SIFIRDAN rastgele-router modeli egit (router donuk-
    random, geri kalan ayni) -> final ppl born'a yakin mi?
      yakin   -> kazanc "egitilmis seyrek govde", routing tali -> tez ProSparse'a
                 cerceveleniyor (DURUST soyle; falsifikasyona bak)
      cok kotu -> ogrenilmis routing hakkini veriyor (tez guclenir)
    (Ayni mantik: sifirdan sabit-blok modeli = "girdi-kosullama gerekli mi".)

TESHIS (ucuz, loglardan/ckpt'den):
  - kosullu secim cesitliligi (marjinal entropi DEGIL): token basina secilen
    blok kumeleri ne kadar farkli? (dusukse routing girdiye az bakiyor)
  - katman deseni: erken/orta/gec routing davranisi
  - gate margin <-> token loss korelasyonu (zor token'da guven degisiyor mu)
  - entropi egitim egrisi (loglarda var, grafige dok)
NOT (Bulut): "zor token'da butce artiyor mu" sabit top-k ile TANIMSIZ. Bu soru
Faz 4 fikri doguruyor: OGRENILMIS DEGISKEN-k (kolay az blok / zor cok) —
butce-takasinin en kucuk hali, born'un dogal uzantisi + GOZ HARITASI takasina kopru.
Cikti: Kart 28.  (Numaralandirma: ablasyon=28; Faz 4 ciktilarini 29+'a kaydirdim —
Bulut taslaginda 4a=28'di, cakismayi cozmek icin.)

---

## FAZ 4 — born-eye'i genisletme  [Bulut, Claude ekleriyle]
Giris kosulu: KART 28 (ablasyon) ONCE. Kazancin kaynagi soylenmeden genisleme kor.

4a. [TAMAM -> NOTES_TR Kart 29] Granularite: ESIK+PLATO. G8k1 5.43 (kaba, +%15
    kotu) / G16k2 4.72 / G32k4 4.72 (birebir). Granularite bir esige kadar onemli,
    G16'da doyuyor, otesi bos. Confound: G8k1 ayni zamanda tek k=1 kolu. K28+K29
    birlesik: born-eye = co-adaptasyon + girdi-kosullu + yeterince-ince(k>=2)
    partisyon; ogrenilmis-router VE sonsuz-ince gerekmez. FALSIFIKASYON COZULDU:
    granularite DUZ DEGIL -> born ProSparse'a COKMEDI (daha zengin bir sey).
    --- ozgun plan asagida ---
    GRANULARITE SUPURMESI (~yarim gun). Sabit butce %12.5, degisen blok boyutu:
      python train_born.py --G 8  --k 1
      python train_born.py --G 32 --k 4
    (G16k2 var: born_report_G16k2.json). Soru: born ince-taneli MoE'den ne
    kazaniyor? granularite-ppl egrisi. Izle: router entropi + max-blok (kaba
    granularitede cokus riski yuksek — az blok = az cesitlilik).
    [Claude ek] confound: G degisince router-param + blok-ifade gucu birlikte
    degisir; egriyi "granularite" diye okurken akilda tut. Istege bagli uc nokta:
    G=96 (per-neuron limitine yakin) egriyi uzatir.
    KARAR: egri DUZ -> granularite onemsiz, born'un katkisi CERCEVE (mekanizma
    degil; yazida acikca soyle). Egri EGIMLI -> olculmus katki.
    Cikti: Kart 29.

4b. [TAMAM -> Kart 30 + 33] KERNEL: born decode STOK matmul ile
    realize. born_loop (bitisik slice-VIEW, kopyasiz) th=1'de 7.41x (ceiling 8.75x),
    posthoc'u ~6x geciyor (1.22x). KRITIK: W1[idx] fancy-index gather-kopya yapar
    (born_cat ≈ posthoc, avantaj cop) -> per-blok SLICE-VIEW sart. Ozel kernel
    GEREKSIZ (post-hoc'un numba-0.70x ucurumu born'da yok, bitisik=native).
    Granularite-kernel kabayi sever (G8 9.66x > G32 6.32x) ama K29 kaliteyi ince
    ister -> G16 tatli nokta. ADIM2 TAMAM (Kart 33): e2e decode ~2.1x @1B-sekil (CPU th1); FFN payi %49->63 olcekle artiyor, posthoc e2e sadece 1.2x -> born BITISIKLIGI sart. --- ozgun plan asagida ---
    KERNEL EVLILIGI (ana is). born bloklari BITISIK -> gather yok, duz dilim.
    [Claude ek — KRITIK] "gather yok" YALNIZCA DECODE'da gecerli (T=1: tek token,
    k bitisik blok, duz dilim, hizli). PREFILL'de (T>1) her token FARKLI blok
    secer -> standart MoE-dispatch problemi (Kart 16 union sismesinin ta kendisi).
    Yani born decode'da temiz, prefill'de "MoE dispatch verimi" (cozulmus ama tali).
    4b'yi DECODE tokens/sec uzerine kur.
    Adim 1: mikro-benchmark (bench_stage0 tarzi, born blok yapisi, gercek d_ffn —
            17.6M model overhead'e bogulur, kernel'i izole olc).
    Adim 2: BornGPT decode forward'ini gercek dilimli yap, tokens/sec.
    ASIL SAYI: teorik FLOP orani vs olculen duvar-saati orani (decode). Post-hoc
    hattinda ucurum vardi (K17 numba 0.70x MKL gerisi); born'da CAKISMASI beklenir
    (mimari kernel icin doguldu). Cakisiyorsa -> projenin ilk sorusuna (az calistirma/
    enerji) UCTAN UCA cevap. Ucurum -> nerede kayboldugu yeni kart.
    Cikti: Kart 30.

4c. [TAMAM -> NOTES_TR Kart 31] OLCEK 2x (36M), 2x2 (butce x olcek) tam. IKI EKSEN:
    (A) BUTCE: born agresif butcede cok ustun (36M -31%@6.25 vs -10%@12.5), her
    olcekte -> tez tutuyor, narrowing butce-artefakti DEGIL. (B) OLCEK: 2x, taklit
    yukunu ~yariya indiriyor her butcede (30->16, 146->71p); born maliyeti sabit;
    post-hoc yaklasiyor (buyuk model %92 seyrek). Born HER kutucukta kazaniyor ama
    margin olcekle eriyor. Kart 28 (random~learned) olcekte TUTUYOR. Tez: born
    kisitli rejimde en degerli. (Ekstrapolasyon 2 noktadan; kesin degil.) --- plan ---
    OLCEK (100-300M, Colab) — SADECE 4b pozitifse. Soru: born avantaji olcekle
    buyuyor mu kuculuyor mu?
    [Claude ek] IKI YARISAN etki, sonuca PESIN baglanma:
      (i)  buyuk model daha cok dogal seyrek -> post-hoc'un isi KOLAY (makas kapanir)
      (ii) daha cok katman -> post-hoc hatasi daha cok BIRIKIR (makas acilir)
    Hangisi baskin, olcum soyleyecek.

4d. [ADIM1 TAMAM -> Kart 32] DEGISKEN-k (adaptif genislik) kopru testi: born
    butcesini token-basi ADAPTIF dagittik (STE hard-forward, tam turevlenebilir).
    NEGATIF: sabit-k'yi gecmedi (HARD ppl 4.869 vs 4.723, +3.1%) VE korelasyon TERS
    (zor token AZ blok: corr -0.111, zor/kolay 0.85x) -> CE-zorlugu compute-tahsis
    EKSENI DEGIL. V1 soft-kapi soft-cheat'e cokmustu (hard-esik 4.6->551); STE ile
    gercek olctuk. ETKI: asagidaki GOZ HARITASI butce-takasinin temeli olan
    'zorluk->butce' varsayimi bu olcekte DESTEKLENMEDI -> 4d-full gerekcesi zayifladi.
    --- ozgun plan ---
    BIRLESIK GOZ (agirlik + derinlik, ortak butce) — en son, yuksek risk.
    GOZ HARITASI'ndaki 1+4 cifti. K9-10 uyarisi: gozler birlikte egitilmezse cig.

FALSIFIKASYON [GUNCEL — Kart 28 sonucu]: "random~ogrenilmis" KOSULU TETIKLENDI
(random 4.83 ~ learned 4.72). AMA born YINE DE post-hoc'u geciyor (fixed 5.27 <
5.92) -> born ProSparse'a COKMEDI: cunku co-adaptasyon (ne routing ne salt
seyreklik) asil mekanizma. Yani "ogrenilmis routing yildiz" iddiasi DUSTU, ama
"co-adapte seyrek govde post-hoc'u yener" iddiasi AYAKTA. Kalan falsifikasyon:
4a granularite de duz cikarsa -> MoE-routing cercevesi bu olcekte dekoratif,
born ~ "co-trained sabit seyrek alt-ag". O zaman durust cerceve: born'un degeri
BLOK-ROUTING degil CO-ADAPTIVE-SEYREK-EGITIM (ProSparse'in blok-yapili, girdi-
kosullu hali). Olcek (4c) bunu tersine cevirebilir (buyukte routing onem kazanir).

CERCEVE (Bulut pazarlama + Claude bilim): born dense'ten HER butcede kotu. Iddia
"born daha iyi" DEGIL: "born, agresif butce rejimini (%6-12) ERISILEBILIR kiliyor —
post-hoc orada cokuyor." PARETO CEPHESI dili: born, kalite-hesap Pareto'sunu
post-hoc'un giremedigi bolgeye uzatiyor. Genis butcede fark kucuk; GIZLEME.

DISIPLIN (Bulut): ckpt adlarina seed koy (born_G16k2_s1.pt); K27'de uzerine
yazildi, bir dahaki karsilastirmayi kirletir. (train_born.py'de tek satir — yapildi.)

---

## GOZ HARITASI (uzun vade referansi — Faz 3 tek gozun testi, tam vizyon bu)
Bes "goz", hepsi ayni sinyali (residual stream / ondan tureyen) okur:
1. Agirliklara bakan  : noron/blok-skip (bu projenin gozu; Faz 3 dogustan hali)
2. Baglama bakan      : KV-cache eviction (H2O/attention-decay; eski "C fikri")
3. Expert'lere bakan  : MoE router (tek dogustan ornek — Kart 2 patolojisiyle)
4. Derinlige bakan    : early-exit / dinamik derinlik
5. Diske bakan        : retrieval/RAG (evren=disk metaforunun kapisi)
Birlesik vizyon: bes goz + tek dusuk-rank "bakis govdesi" + gozler-arasi
BUTCE TAKASI ("sig gec ama genis bak / derin in ama seyrek hesapla") —
hicbir sistemde ogrenilmis degil. Kart 9-10 uyarisi: gozler etkilesir,
ancak BIRLIKTE egitilirse. Faz 3 pozitifse siradaki aday: goz 1 + goz 4
birlikte (ikisi de FFN/derinlik tarafinda, en az etkilesen cift olabilir —
en kucuk guvenli "iki gozu birlikte egit" adimi). KV/retrieval sonraya.

---

## KERNEL HATTI (paralel / park — enerji tercihine gore)
Bagimsiz muhendislik: llm-manager C++/AVX row-skip + predictor entegrasyonu.
- Hedef Kart 17'den belli: numba tabani MKL'e ~0.70x geride; C++/AVX cogunu
  kapatir -> gercekci hedef ~2.5-3x FFN @ B=0.30 (tavan 3.12x olculdu).
- Kalemler: (a) C++ prototip vs numba A/B; (b) predictor ucuzlatma
  (r256 Pareto — Kart 12 kapasite egrisine bak — ya da predictor'i kernel'e
  alma, su an maliyetin %26'si); (c) uctan uca gercek-model decode tok/s
  (tum katmanlar, gercek predictor, tokens/s — Kart 17 tek-FFN olcumuydu).
- Faz 3 blok-granularite secerse bu hat DOGRUDAN oraya hizmet eder.
- BONSAI-27B ZEMINI (prism-ml/Bonsai-27B-gguf, dogrulandi 2026-07, Apache-2.0):
  Qwen3.6-27B'nin 1-bit {-1,+1} (Q1_0_g128, 1.125 bpw, ~3.9GB, FP16'nin %89.5'i)
  ve ternary {-1,0,+1} (1.71 bpw, %94.6) temsili. AÇIK llama.cpp fork'u var
  (github.com/PrismML-Eng/llama.cpp, CUDA+Metal, 1-bit hybrid-attn kernel) +
  MLX fork. Kernel fazina "sifirdan AVX" yerine HAZIR ZEMIN: Q1_0 matvec'e
  row-skip eklemek. NEDEN ILGINC (3 kesisim, hepsi OLCULMEDI):
  * Bandwidth carpimi: skip kazanci FLOP'tan degil OKUNMAYAN satirdan (Kart 14/17).
    1-bit'te bytes/step zaten dusuk -> edge/CPU'da (llm-manager zemini) row-skip
    ustune deger. AMA sayfanin kendi H100-batch1 notu: orada binary~ternary
    (104.8 vs 98 tok/s) cunku kernel-launch/sync LATENCY tavani, bandwidth degil.
    Yani carpim EDGE/CPU hikayesi (bizim hedef), datacenter-batch1 degil.
  * Ternary sifirlari = EGITILMIS agirlik-seyrekligi (sadece ternary variantta;
    1-bit binary'de sifir yok). "Sert temsil > sonradan budama" tezimizin
    agirlik-uzayindaki kuzeni. Acik soru: ternary sifirlari satir-bazinda
    kumeleniyor mu? Kumeleniyorsa aktivasyon-skip ile AYNI kernelde birlesir.
  * Enerji: 0.275 mWh/token (M5 Pro) — datacenter'dan ~mertebe verimli;
    "20W paradoksu" motivasyonumuzun endustriyel teyidi.
  DIKKAT (duzeltme): backbone DENSE SwiGLU (MoE DEGIL — Kart 2'nin Qwen3-30B-A3B'si
  ayri). SwiGLU tam da aktivasyon-skip'in ISLEMEDIGI aile (Kart 5-7). Yani bu
  BELLEK ekseni; bizim compute-skip'imiz Bonsai'ye DOGRUDAN uygulanmaz —
  1-bit'lestirme aktivasyon dagilimini degistirmis OLABILIR ama olcmeden hukum yok.
  Kernel altyapisi degerli; aktivasyon-skip evliligi ayri bir olcum sorusu.
- Karar: Faz 0-1 sirasinda dokunma; Faz 2'de canin isterse yan ugras;
  Faz 3 granularite karari netlesince ciddi gir.

---

## ARAC ENVANTERI (yeniden kullanim haritasi)
Mevcut zincir Faz 0-3'un cogunu hazir karsiliyor:

| Arac | Ne yapar | Nerede yeniden kullanilir |
|---|---|---|
| collect_sparsity_v2.py | hook ile (x, a) cifti toplar, prompt-id'li | Faz 1 yan olcum, Faz 2 olcum, Faz 3 (ii) — model-agnostik, OPT/Qwen algilar; TinyStories modeli icin kucuk uyarlama |
| analyze_sparsity_v2.py | LIFT + recall@butce + SVD rank + --full | Faz 2-3'te ayni protokol — "ayni metrik, farkli model" ilkesi |
| oracle_quality.py | butce->ppl, oracle/mimari-algilamali | Faz 1-2-3 kalite olcumu; bebek-model icin hook yolu ayni |
| predictor_quality.py | in-loop 3-egri ppl (oracle/statik/pred) | Faz 0a hemen; Faz 3 (ii) ana karsilastirma araci |
| bench_stage0.py / bench_kernel_kart17.py | duvar-saati mikro/kernel | Kernel hatti; Faz 3 blok karari icin boyut degistirip tekrar |
| scaling_probe_colab.py / colab_train_all_predictors.py | veri-olcek egrisi / 24-katman predictor egitimi | Faz 0b genelleme; Faz 3 (ii) eklenti-goz kolu |
| prefill_sparse_proto.py | chunk-union gather prefill (Kart 16 negatifi) | referans/negatif-kanit; Faz 3'te union davranisi tekrar bakilirsa |
| regen_wikitext_eval.py | wikitext_eval.jsonl deterministik yeniden uretim | Faz 0a; ckpt/jsonl kaybolursa her yerde |
| mask_persistence.py | Kart 15 analizi (churn/union/melez) — sandbox'tan koda dokuldu | Faz 0c repo; Faz 3 churn olcumu B modelinde tekrar |
| SPARSITY_NOTES.md kart formati | soru/kurulum/sayi/verdict/ders | TUM fazlar — numaralandirma 18'den devam |

Eksik araclar (yenisi yazilacak):
- Faz 1: pretrain dongusu (model tanimi + egitim scripti) — en buyuk yeni parca
- Faz 2: seyreklik-cezali kayip eklentisi (kucuk)
- Faz 3: ogrenilmis-secici FFN modulu + birlikte-egitim (ana yeni parca);
  secim-entropisi olcumu (kucuk)

---

## Karar noktalari (simdiden isaretli)
- K18/K19 (Faz 0a/0b) sonuclari yazinin son halini etkiler — o yuzden yazim
  0a/0b'DEN SONRA, Faz 1'den ONCE.
- Faz 2 -> 3 gecisi: ProSparse-mini "aktivasyon-cezali seyreklik stabil
  egitiliyor" derse Faz 3'te o yol; demezse straight-through/gating denenir.
- Granularite karari Faz 1 yan olcumu + kernel gercekleriyle Faz 3 basinda.
- Her faz sonunda: kartlari yaz, GitHub'i guncelle, SONRA ilerle.

## Yontem notlari (SPARSITY_NOTES'tan devralinir, degismez)
- soru / kurulum / headline sayi / verdict / ders
- pozitifi KIRMAYA calis; kirilmazsa kart
- vekil metrik uyarisi; in-loop sart; sirali split yasak; tek degisken ilkesi
- (yeni, Kart 16 dersi) mikro-benchmark varsayimi gercek is yukuyle eslesmeli
