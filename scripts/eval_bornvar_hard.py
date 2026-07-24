"""
eval_bornvar_hard.py  -  4d-adim1 FALSIFIKASYON: soft vs HARD gate ppl.
=======================================================================
Degisken-k born SOFT sigmoid kapiyla egitildi. Kapi aktivasyonu OLCEKLER,
sifirlamaz -> soft modda 16 blogun HEPSI hesaplanir (compute tasarrufu YOK,
%100 FFN, sadece gate ile carpim). Rapor edilen ppl 4.618 bu YOGUN sayidir.

Gercek %12.5 butce ancak kapilari ESIKLERSEK (g>0.5 -> 1/0) gerceklesir.
O zaman: kac blok gercekten acik? ppl ne olur? -> sabit-k born 4.723 @12.5%
ile ADIL kiyas ancak bu HARD sayiyla yapilir.

Kosu (Colab, ckpt + data zaten orada):
  python eval_bornvar_hard.py --ckpt tinystories_data/bornvar_G16t0.125_s0.pt
"""
import argparse, numpy as np, torch
import torch.nn.functional as F
from train_born_vark import BornVarGPT, VarRoutedFFN, eval_ppl
from train_baby import get_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tinystories_data/bornvar_G16t0.125_s0.pt")
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--baseline", type=float, default=4.723, help="sabit-k born @12.5%")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=dev)
    c = ck["config"]
    m = BornVarGPT(c["vocab"], c["d"], c["L"], c["h"], c["blk"], c["d_ff"],
                   c["G"], c["target"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    val = np.memmap(f"{args.data}/val.bin", dtype=np.uint16, mode="r")

    # 1. SOFT (egitimdeki gibi, surekli kapi) — saglama, ~4.618 gelmeli
    soft = eval_ppl(m, val, c["blk"], args.batch, dev, iters=args.iters)

    # 2. HARD (kapilari esikle = gercek blok-skip; deploy sayisi)
    thr = args.thresh
    orig = VarRoutedFFN.forward

    def hard_forward(self, x):
        g = (torch.sigmoid(self.router(x)) > thr).float()
        self.last_count = g.sum(-1).detach()          # [B,T] gercek aktif blok
        ng = g.repeat_interleave(self.bs, dim=-1)
        return self.fc2(F.relu(self.fc1(x)) * ng)

    VarRoutedFFN.forward = hard_forward
    hard = eval_ppl(m, val, c["blk"], args.batch, dev, iters=args.iters)

    # 3. hard butce: kac blok gercekten aciliyor
    cnts = []
    with torch.no_grad():
        for _ in range(40):
            x, y = get_batch(val, c["blk"], args.batch, dev); m(x, y)
            cnts.append(torch.stack([b.ffn.last_count for b in m.blocks]).mean().item())
    VarRoutedFFN.forward = orig
    blocks = float(np.mean(cnts)); pct = blocks / c["G"] * 100.0

    print(f"\n[FALSIFIKASYON] esik={thr}  ckpt={args.ckpt}")
    print(f"  soft ppl (yogun, %100 FFN, olcekli): {soft:.3f}   (egitim ~4.618)")
    print(f"  HARD ppl (gercek blok-skip)        : {hard:.3f}   <- DEPLOY sayisi")
    print(f"  hard butce                         : {blocks:.2f}/{c['G']} blok = {pct:.1f}% compute")
    print(f"  sabit-k born hedef                 : {args.baseline:.3f} @ 12.5%")
    if hard <= args.baseline:
        print(f"  -> ADIL KAZANC: hard {hard:.3f} <= {args.baseline:.3f}"
              f"  (ustelik {pct:.1f}% compute'ta) -> iki-goz 4d'ye yesil isik")
    else:
        print(f"  -> soft-cheat: hard {hard:.3f} > {args.baseline:.3f}"
              f"  -> kazanc surekli kapidan geldi; koprude dur / --gamma arttir")


if __name__ == "__main__":
    main()
