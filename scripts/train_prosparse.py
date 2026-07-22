"""
train_prosparse.py  -  Faz 2: ProSparse-mini (seyreklik-cezali fine-tune)
==========================================================================
Faz 1'in dense modelini (A) al, FFN aktivasyonuna PROGRESIF L1 cezasi
ekleyerek fine-tune et -> dogal sifir orani %89'dan YUKARI it, ppl bedelini
olc. ProSparse tarifinin (Song 2024) minyaturu; "yarim cozum" veri noktasi:
seyreklik SONRADAN-AMA-EGITIMLE guclendirilir (dogustan-goz'e ara basamak).

Mekanizma (mevcut hook altyapisi):
  fc2 pre-hook a = relu(fc1(x)) alir:
    (1) opsiyonel FATReLU esigi:  a <- a * (a > threshold)   [kaydirilmis ReLU]
    (2) L1 cezasi icin |a| biriktirir
  kayip = CE + lambda(step) * mean_layer(mean|a|)
  lambda 0'dan lambda_max'a rampalanir (progresif — ani seyreklik kaliteyi kirar).

Olcum: fine-tune ONCE/SONRA dogal sifir (esik dahil efektif) + val ppl.
Karar: sifir NE KADAR yukseldi, ppl NE KADAR arttı? (Faz 3 tablosunda kol.)

Gereksinim: train_baby.py (ayni klasor) + baby_s0.pt + tinystories_data/.
Kosu: python train_prosparse.py --init tinystories_data/baby_s0.pt --l1 0.05
"""

import argparse, math, os, time, json
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from train_baby import BabyGPT, get_batch, eval_loss


STATE = {"collect": False, "thr": 0.0, "l1_store": []}


def fc2_pre_hook(module, inp):
    a = inp[0]                                   # = relu(fc1(x))  >= 0
    if STATE["thr"] > 0.0:
        a = a * (a > STATE["thr"])               # FATReLU (kaydirilmis ReLU)
    if STATE["collect"]:
        STATE["l1_store"].append(a.abs().mean())
    return (a,)


@torch.no_grad()
def measure_zeros(model, data, blk, bs, dev, thr, iters=20):
    """Efektif sifir orani: relu(fc1) <= thr olan noron% (thr=0 -> Kart 21 ile ayni)."""
    stats = {i: [] for i in range(len(model.blocks))}
    hs = []

    def mk(i):
        def h(m, inp, out):                      # out = fc1(x); relu(out)<=thr <=> out<=thr
            stats[i].append((out <= thr).float().mean().item())
        return h
    for i, b in enumerate(model.blocks):
        hs.append(b.fc1.register_forward_hook(mk(i)))
    was = STATE["collect"]; STATE["collect"] = False
    for _ in range(iters):
        x, _ = get_batch(data, blk, bs, dev)
        model(x)
    STATE["collect"] = was
    for h in hs:
        h.remove()
    per = {i: round(float(np.mean(v)), 4) for i, v in stats.items()}
    return per, round(float(np.mean(list(per.values()))), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="tinystories_data/baby_s0.pt")
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--l1", type=float, default=0.05, help="lambda_max (ANAHTAR — sweep)")
    ap.add_argument("--ramp", type=int, default=1500, help="lambda 0->max ramp adimi")
    ap.add_argument("--threshold", type=float, default=0.0, help="FATReLU esigi (0=kapali)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-4, help="fine-tune (pretrain'den dusuk)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="prosparse")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    STATE["thr"] = args.threshold

    tok = ByteLevelBPETokenizer(os.path.join(args.data, "vocab.json"),
                                os.path.join(args.data, "merges.txt"))
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    ck = torch.load(args.init, map_location=dev)
    c = ck["config"]
    model = BabyGPT(c["vocab"], c["d"], c["L"], c["h"], c["blk"]).to(dev)
    model.load_state_dict(ck["model"])
    print(f"[i] A yuklendi: {args.init}  cihaz={dev}  l1_max={args.l1} "
          f"ramp={args.ramp} thr={args.threshold} steps={args.steps}")

    # --- ONCE olcum (esik uygulanmadan A'nin dogal hali; sonra esikli efektif) ---
    z0_per, z0 = measure_zeros(model, val, args.block, args.batch, dev, 0.0)
    STATE["collect"] = False
    ppl0 = math.exp(eval_loss(model, val, args.block, args.batch, dev))
    print(f"[ONCE] dogal sifir(thr=0)={z0:.1%}  val ppl={ppl0:.3f}  katman={z0_per}")

    hooks = [b.fc2.register_forward_pre_hook(fc2_pre_hook) for b in model.blocks]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def lam(step):
        return args.l1 * min(1.0, step / max(1, args.ramp))

    def lr_at(step):
        w = 200
        if step < w:
            return args.lr * step / w
        r = (step - w) / max(1, args.steps - w)
        return 0.1 * args.lr + 0.5 * 0.9 * args.lr * (1 + math.cos(math.pi * r))

    hist = []
    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if step % args.eval_every == 0:
            STATE["collect"] = False
            vl = eval_loss(model, val, args.block, args.batch, dev)
            _, zc = measure_zeros(model, val, args.block, args.batch, dev, args.threshold)
            # ham |a| buyuklugu (L1 terim degeri) -> lambda kalibrasyonu icin
            STATE["l1_store"].clear(); STATE["collect"] = True
            with torch.no_grad():
                xb, _ = get_batch(val, args.block, args.batch, dev)
                model(xb)
            STATE["collect"] = False
            amag = float(torch.stack(STATE["l1_store"]).mean()) if STATE["l1_store"] else 0.0
            hist.append({"step": step, "val": round(vl, 4),
                         "ppl": round(math.exp(vl), 3), "zeros": zc,
                         "lam": round(lam(step), 4), "a_l1": round(amag, 4)})
            print(f"[{step:5d}] val ppl {math.exp(vl):.3f}  efektif sifir {zc:.1%}  "
                  f"lam {lam(step):.3f}  |a|={amag:.4f}  ceza={lam(step)*amag:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            torch.save({"model": model.state_dict(), "config": c,
                        "tag": args.tag, "threshold": args.threshold, "step": step},
                       os.path.join(args.data, f"baby_{args.tag}.pt"))

        if step == args.steps:
            break

        x, y = get_batch(train, args.block, args.batch, dev)
        STATE["l1_store"].clear(); STATE["collect"] = True
        _, ce = model(x, y)
        STATE["collect"] = False
        l1 = torch.stack(STATE["l1_store"]).mean() if STATE["l1_store"] else 0.0 * ce
        loss = ce + lam(step) * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for h in hooks:
        h.remove()

    # --- SONRA olcum (esikli efektif sifir + esiksiz ham, karsilastirma) ---
    zr_per, zr = measure_zeros(model, val, args.block, args.batch, dev, args.threshold)
    z_raw_per, z_raw = measure_zeros(model, val, args.block, args.batch, dev, 0.0)
    STATE["collect"] = False
    ppl1 = math.exp(eval_loss(model, val, args.block, args.batch, dev))
    print(f"\n[SONRA] efektif sifir={zr:.1%} (ham relu {z_raw:.1%})  val ppl={ppl1:.3f}")
    print(f"[DELTA] sifir {z0:.1%} -> {zr:.1%} (+{zr-z0:.1%})  "
          f"ppl {ppl0:.3f} -> {ppl1:.3f} ({(ppl1/ppl0-1)*100:+.1f}%)")

    rep = os.path.join(args.data, f"prosparse_report_{args.tag}.json")
    with open(rep, "w") as f:
        json.dump({"init": args.init, "l1_max": args.l1, "ramp": args.ramp,
                   "threshold": args.threshold, "steps": args.steps,
                   "zeros_before": z0, "zeros_after_effective": zr,
                   "zeros_after_raw_relu": z_raw, "zeros_after_by_layer": zr_per,
                   "ppl_before": round(ppl0, 3), "ppl_after": round(ppl1, 3),
                   "ppl_delta_pct": round((ppl1/ppl0-1)*100, 2),
                   "history": hist}, f, indent=2)
    ex = tok.encode("Once upon a time").ids
    out = model.generate(torch.tensor([ex], device=dev), 50)[0].tolist()
    print("   ornek: " + tok.decode(out).replace("\n", " ")[:180])
    print(f"[ok] ckpt -> baby_{args.tag}.pt\n[ok] rapor -> {rep}")
    print("Okuma: sifir belirgin arttiysa (ppl kabul edilebilir) -> ProSparse tutuyor,")
    print("Faz 3 tablosunda 3. kol. Kaliteyi kirdiysa --l1 dusur / --ramp uzat.")


if __name__ == "__main__":
    main()
