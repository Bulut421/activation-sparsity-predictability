"""
oracle_baby.py  -  Faz 3 ON-ADIM: BabyGPT icin oracle butce->ppl (zorunlu)
===========================================================================
Kart 6 dersinin kurumsallasmasi: dogustan-goz B'nin butcesini SECMEDEN once
tavani olc. oracle_quality.py'nin BabyGPT hali (HF degil, kendi modelimiz).

Her katmanda FFN aktivasyonu a=relu(fc1(x)) top-B magnitude maskelenir (oracle),
val ppl olculur. ReLU modelde beklenti (Kart 7): B > dogal-canli-oran oldukca
BEDAVA (sadece zaten-sifirlar kesilir), B < canli-oran'da ppl kalkar.

Iki tez tahmini:
  - A (%89 sifir, ~%11 canli): plato ~B=0.11'e kadar -> OPT gibi.
  - ProSparse (%95.6 sifir, ~%4.4 canli): plato daha asagi (~B=0.045) ->
    daha seyrek model daha agresif skip'e izin verir (deploy degeri).

Kosu:  python oracle_baby.py --ckpt tinystories_data/baby_s0.pt
       python oracle_baby.py --ckpt tinystories_data/baby_prosparse.pt
"""

import argparse, json, math, os
import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from train_baby import BabyGPT, get_batch

BUDGET = {"frac": None}
STATS = {"zero": 0.0, "n": 0}


def oracle_hook(module, inp):
    a = inp[0]                                   # = relu(fc1(x)) >= 0
    if BUDGET["frac"] is None:                   # baseline: dogal sifir olc
        STATS["zero"] += (a == 0).float().mean().item(); STATS["n"] += 1
        return None
    k = max(1, int(a.shape[-1] * BUDGET["frac"]))
    thr = a.abs().topk(k, dim=-1).values[..., -1:]
    return (a * (a.abs() >= thr),)


@torch.no_grad()
def masked_ppl(model, data, blk, bs, dev, iters=80):
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, blk, bs, dev)
        _, loss = model(x, y)
        losses.append(loss.item())
    return math.exp(float(np.mean(losses)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tinystories_data/baby_s0.pt")
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--budgets", default="0.20,0.15,0.11,0.08,0.05,0.03")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    ck = torch.load(args.ckpt, map_location=dev)
    c = ck["config"]
    model = BabyGPT(c["vocab"], c["d"], c["L"], c["h"], c["blk"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    hooks = [b.fc2.register_forward_pre_hook(oracle_hook) for b in model.blocks]
    print(f"[i] {args.ckpt}  cihaz={dev}  blk={c['blk']}")

    report = {"ckpt": args.ckpt}
    budgets = [None] + [float(b) for b in args.budgets.split(",")]
    for b in budgets:
        BUDGET["frac"] = b
        ppl = masked_ppl(model, val, c["blk"], args.batch, dev)
        name = "baseline" if b is None else f"{b:.2f}"
        report[name] = round(ppl, 4)
        base = report.get("baseline", ppl)
        print(f"[B={name:>8}]  ppl={ppl:8.3f}   delta={(ppl/base-1)*100:+6.1f}%")
        if b is None and STATS["n"]:
            zf = STATS["zero"] / STATS["n"]
            report["natural_zero"] = round(zf, 4)
            print(f"[i] dogal sifir={zf:.1%}  canli~{1-zf:.1%}  "
                  f"(plato bu butcenin ustunde bedava olmali)")
    for h in hooks:
        h.remove()
    out = os.path.join(args.data,
                       f"oracle_baby_{os.path.basename(args.ckpt).replace('.pt','')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {out}")
    print("Okuma: platonun bittigi B = B modeli icin ANLAMLI en dusuk butce.")
    print("B'yi bu platonun ALTINA hedefle ki dogustan-goz A'dan fazlasini yapsin.")


if __name__ == "__main__":
    main()
