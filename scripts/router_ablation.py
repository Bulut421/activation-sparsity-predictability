"""
router_ablation.py  -  Kart 28: eval-time router swap + teshis (UCUZ, egitim yok)
==================================================================================
Gerekce: entropi 1.0 COKUS olmadigini kanitlar, router'in ISE YARADIGINI
kanitlamaz — rastgele router da entropi 1.0 verir. Bu script iki soruyu ucuza
yoklar (ayni ckpt, ayni val batch, egitim yok):

  1. EVAL-TIME SWAP: ogrenilmis born'un router SECIMINI degistir:
       learned (mevcut) | random (rastgele k blok) | fixed (sabit ilk-k blok)
     UYARI (confound): govde THIS router'a co-adapte -> random/fixed'de cokmesi
     "kirilganlik" gosterir, "ogrenme gerekli"nin TAM cevabi DEGIL. Tam cevap
     icin sifirdan random-router egit (train_born.py --router random; Kart 28b).
     Ama random~learned CIKARSA (cokmezse) bu zaten carpici: routing govde icin
     onemsiz demektir, ucuza ogreniriz.

  2. KOSULLU SECIM CESITLILIGI (marjinal entropi DEGIL): token basina secilen
     blok kumeleri ne kadar FARKLI? (mean pairwise Jaccard mesafesi)
       ~0 -> tum token'lar ayni bloklari seciyor (routing girdiye bakmiyor,
             marjinal entropi yuksek olsa bile — "sahte cesitlilik")
       yuksek -> routing girdiye gercekten kosullu
     + katman deseni (erken/orta/gec) + marjinal entropi/max-blok.

Kosu: python router_ablation.py --born tinystories_data/born_G16k2.pt
"""

import argparse, math, os, json
import numpy as np
import torch
from train_born import BornGPT, get_batch


@torch.no_grad()
def ppl_on(model, batches):
    ls = []
    for x, y in batches:
        _, loss = model(x, y)
        ls.append(loss.item())
    return math.exp(float(np.mean(ls)))


def set_override(model, mode):
    for b in model.blocks:
        b.ffn.override = mode                 # None | "random" | "fixed"


@torch.no_grad()
def selection_diversity(model, x, sample=256):
    """Her katmanda: token-basi secili blok kumeleri ne kadar farkli?
    mean pairwise Jaccard mesafesi (ayni-k k-hot vektorler). 0=hepsi ayni."""
    set_override(model, None)
    model(x)                                  # last_mask'leri doldur
    out = {}
    for i, b in enumerate(model.blocks):
        m = b.ffn.last_mask.reshape(-1, b.ffn.G).float()   # [N, G] k-hot
        N = m.shape[0]
        idx = torch.randperm(N, device=m.device)[:min(sample, N)]
        s = m[idx]                            # [M, G]
        k = model.k
        inter = s @ s.T                       # [M, M]  kesisim buyuklugu
        union = 2 * k - inter
        dist = 1.0 - inter / (union + 1e-9)
        M = s.shape[0]
        off = ~torch.eye(M, dtype=torch.bool, device=m.device)
        out[i] = round(float(dist[off].mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--born", default="tinystories_data/born_G16k2.pt")
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--eval-batches", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    ck = torch.load(args.born, map_location=dev)
    c = ck["config"]
    model = BornGPT(c["vocab"], c["d"], c["L"], c["h"], c["blk"],
                    c["d_ff"], c["G"], c["k"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"[i] {args.born}  G={c['G']} k={c['k']} aktif={c['k']/c['G']:.1%}  cihaz={dev}")

    # sabit val batch'leri (tum modlar ayni gorur)
    g = torch.Generator(device="cpu").manual_seed(1234)
    blk = c["blk"]
    batches = []
    for _ in range(args.eval_batches):
        ix = torch.randint(len(val) - blk - 1, (args.batch,), generator=g)
        xb = torch.stack([torch.from_numpy(val[i:i+blk].astype(np.int64)) for i in ix])
        yb = torch.stack([torch.from_numpy(val[i+1:i+1+blk].astype(np.int64)) for i in ix])
        batches.append((xb.to(dev), yb.to(dev)))

    # --- 1) EVAL-TIME SWAP ---
    report = {"born": args.born, "active": c["k"]/c["G"]}
    print("\n== eval-time router swap (ayni ckpt, ayni batch) ==")
    for mode in ("learned", "random", "fixed"):
        set_override(model, None if mode == "learned" else mode)
        p = ppl_on(model, batches)
        report[f"ppl_{mode}"] = round(p, 4)
        base = report.get("ppl_learned", p)
        print(f"[{mode:>8}]  ppl {p:8.3f}   delta vs learned {(p/base-1)*100:+7.1f}%")
    set_override(model, None)

    # --- 2) KOSULLU SECIM CESITLILIGI + marjinal ---
    xb, _ = batches[0]
    div = selection_diversity(model, xb)
    ent, mx = model.router_stats()
    report["selection_diversity_by_layer"] = div
    report["marginal_entropy"] = round(ent, 4)
    report["max_block"] = round(mx, 4)
    print("\n== kosullu secim cesitliligi (Jaccard mesafesi; 0=hepsi ayni) ==")
    print("   katman:", div)
    print(f"   ortalama {np.mean(list(div.values())):.3f}  |  "
          f"marjinal entropi {ent:.3f} (1=uniform)  max-blok {mx:.1%}")

    out = os.path.join(args.data, f"router_ablation_{os.path.basename(args.born).replace('.pt','')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {out}")
    print("Okuma:")
    print("  random ~ learned (cokmez)  -> routing govde icin onemsiz; tez ProSparse'a")
    print("     cerceveleniyor. random COKER -> govde routing'e bagli (ama co-adapte")
    print("     confound; kesin cevap sifirdan --router random egitimi = Kart 28b).")
    print("  cesitlilik ~0 -> routing girdiye bakmiyor (sahte entropi).")
    print("  cesitlilik yuksek + random coker -> ogrenilmis kosullu secim gercek.")


if __name__ == "__main__":
    main()
