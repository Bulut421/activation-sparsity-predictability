"""
faz3_compare.py  -  Faz 3 KARAR: dogustan-goz vs eklenti-goz, TEK eval tablosu
================================================================================
Tezin karari. AYNI val batch'leri, AYNI butce (born'un k/G'si) uzerinde:
  A-baseline  : dense, %100 FFN (ust referans)
  A-oracle    : top-B |a| (ULASILAMAZ tavan — gercek aktivasyonu bilir)
  A-static    : en sik canli B noron (girdiye bakmaz — Stage A tuzagi)
  A-predictor : x'ten dusuk-rank lineer predictor, IN-LOOP (gercek eklenti-goz)
  born-B      : blok-router B modeli (dogustan-goz)

Tez: born-B, A-PREDICTOR'i geciyorsa dogustan-goz kazandi (oracle'i degil —
o ulasilamaz; asil kiyas gercek predictor, cunku deploy onu kullanir).
Butce≈canli oranda A-predictor pahali olmali (tahmin hatasi, Kart 9-11);
born secim yapar, tahmin etmez -> avantaj orada.

Kosu: python faz3_compare.py --A tinystories_data/baby_s0.pt \
        --born tinystories_data/born_G16k2.pt
"""

import argparse, math, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from train_baby import BabyGPT, get_batch
from train_born import BornGPT

MODE = {"m": "baseline", "B": None}
XSTASH = {}
PRED = {}          # L -> (W [d_ff,d_model], b)
STATIC = {}        # L -> top-k index tensor (butceye gore doldurulur)


def fc1_pre(L):
    def h(mod, inp):
        XSTASH[L] = inp[0]           # fc1 girdisi = predictor girdisi x
    return h


def fc2_pre(L):
    def h(mod, inp):
        a = inp[0]                   # relu(fc1(x)) >= 0
        if MODE["m"] == "baseline" or MODE["B"] is None:
            return None
        k = max(1, int(a.shape[-1] * MODE["B"]))
        if MODE["m"] == "oracle":
            thr = a.abs().topk(k, dim=-1).values[..., -1:]
            return (a * (a.abs() >= thr),)
        if MODE["m"] == "static":
            key = (L, k)
            if key not in STATIC:
                idx = STATIC["freq"][L].topk(k).indices
                v = torch.zeros(a.shape[-1], dtype=torch.bool, device=a.device)
                v[idx] = True
                STATIC[key] = v
            return (a * STATIC[key],)
        if MODE["m"] == "predictor":
            W, b = PRED[L]
            logits = XSTASH[L] @ W.T + b
            thr = logits.topk(k, dim=-1).values[..., -1:]
            return (a * (logits >= thr),)
        return None
    return h


@torch.no_grad()
def collect(model, data, blk, bs, dev, n_tokens):
    """A'dan (x, canli-maske) topla: x=fc1 girdisi, mask=(fc1>0).
    Bellek-dostu: onceden ayrilmis diziye dilim dilim yazar (liste+cat yok,
    peak ~yari). 500k x 8 kat ~ 9GB (X fp16 + M bool)."""
    L = len(model.blocks)
    d_in = model.blocks[0].fc1.in_features
    d_ff = model.blocks[0].fc1.out_features
    X = {i: torch.empty(n_tokens, d_in, dtype=torch.float16) for i in range(L)}
    M = {i: torch.empty(n_tokens, d_ff, dtype=torch.bool) for i in range(L)}
    pos = {i: 0 for i in range(L)}
    hs = []

    def mkx(i):
        def h(m, inp):
            v = inp[0].reshape(-1, inp[0].shape[-1]).to(torch.float16).cpu()
            p = pos[i]; n = min(v.shape[0], n_tokens - p)
            if n > 0: X[i][p:p+n] = v[:n]
        return h

    def mkm(i):
        def h(m, inp, out):
            v = (out.reshape(-1, out.shape[-1]) > 0).cpu()
            p = pos[i]; n = min(v.shape[0], n_tokens - p)
            if n > 0:
                M[i][p:p+n] = v[:n]; pos[i] = p + n
        return h
    for i, b in enumerate(model.blocks):
        hs.append(b.fc1.register_forward_pre_hook(mkx(i)))
        hs.append(b.fc1.register_forward_hook(mkm(i)))
    while pos[0] < n_tokens:
        x, _ = get_batch(data, blk, bs, dev)
        model(x)
    for h in hs:
        h.remove()
    return X, M


def train_pred(X, M, dev, rank, epochs=6, bs=8192, lr=1e-3):
    """dusuk-rank lineer predictor (x->canli-maske), SVD ile rank-r'ye kes."""
    d_in, d_out = X.shape[1], M.shape[1]
    net = nn.Linear(d_in, d_out).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    N = len(X)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            bidx = perm[i:i+bs]
            xb = X[bidx].float().to(dev); yb = M[bidx].float().to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    with torch.no_grad():
        W = net.weight.detach().float()
        if rank and rank < min(W.shape):
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            W = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        return W.contiguous(), net.bias.detach().float()


@torch.no_grad()
def ppl_on(model, batches):
    ls = []
    for x, y in batches:
        _, loss = model(x, y)
        ls.append(loss.item())
    return math.exp(float(np.mean(ls)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", default="tinystories_data/baby_s0.pt")
    ap.add_argument("--born", default="tinystories_data/born_G16k2.pt")
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--collect-tokens", type=int, default=200000)
    ap.add_argument("--rank", type=int, default=128, help="predictor rank (0=full)")
    ap.add_argument("--eval-batches", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--budget", type=float, default=None, help="None=born'un k/G'si")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")

    ckA = torch.load(args.A, map_location=dev); cA = ckA["config"]
    A = BabyGPT(cA["vocab"], cA["d"], cA["L"], cA["h"], cA["blk"]).to(dev).eval()
    A.load_state_dict(ckA["model"])
    ckB = torch.load(args.born, map_location=dev); cB = ckB["config"]
    B = BornGPT(cB["vocab"], cB["d"], cB["L"], cB["h"], cB["blk"],
                cB["d_ff"], cB["G"], cB["k"]).to(dev).eval()
    B.load_state_dict(ckB["model"])
    budget = args.budget if args.budget else cB["k"] / cB["G"]
    print(f"[i] A={args.A}  born={args.born}  butce={budget:.1%} (born k/G)  rank={args.rank}")

    # sabit val batch'leri (TUM modlar ayni gorur)
    g = torch.Generator(device="cpu").manual_seed(1234)
    blk, bs = cA["blk"], args.batch
    batches = []
    for _ in range(args.eval_batches):
        ix = torch.randint(len(val) - blk - 1, (bs,), generator=g)
        x = torch.stack([torch.from_numpy(val[i:i+blk].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(val[i+1:i+1+blk].astype(np.int64)) for i in ix])
        batches.append((x.to(dev), y.to(dev)))

    # A: topla + predictor egit + static frekans
    t0 = time.time()
    print(f"[i] A'dan {args.collect_tokens:,} token toplaniyor...")
    X, M = collect(A, train, blk, bs, dev, args.collect_tokens)
    for L in range(cA["L"]):
        STATIC.setdefault("freq", {})[L] = M[L].float().mean(0).to(dev)
        W, b = train_pred(X[L], M[L], dev, args.rank)
        PRED[L] = (W.to(dev), b.to(dev))
        print(f"  L{L}: predictor egitildi (canli {M[L].float().mean():.1%})  "
              f"({time.time()-t0:.0f}s)")
    del X, M

    # A modellerine hook tak
    hs = []
    for L, blk_m in enumerate(A.blocks):
        hs.append(blk_m.fc1.register_forward_pre_hook(fc1_pre(L)))
        hs.append(blk_m.fc2.register_forward_pre_hook(fc2_pre(L)))

    report = {"budget": round(budget, 4), "rank": args.rank}
    for m in ("baseline", "oracle", "static", "predictor"):
        MODE["m"] = m; MODE["B"] = None if m == "baseline" else budget
        p = ppl_on(A, batches)
        report[f"A_{m}"] = round(p, 4)
        base = report.get("A_baseline", p)
        print(f"[A {m:>9}]  ppl {p:7.3f}   delta {(p/base-1)*100:+6.1f}%")
    for h in hs:
        h.remove()

    # born-B (ayni batch'ler)
    pB = ppl_on(B, batches)
    report["born"] = round(pB, 4)
    base = report["A_baseline"]
    print(f"[born G{cB['G']}k{cB['k']}]  ppl {pB:7.3f}   delta {(pB/base-1)*100:+6.1f}%  "
          f"(dogustan-goz, {budget:.1%} aktif)")

    # karar
    dp = report["A_predictor"]
    print(f"\n[KARAR] born {pB:.3f} vs A-predictor {dp:.3f}  "
          f"-> {'BORN KAZANDI' if pB < dp else 'A-predictor onde'} "
          f"({(pB/dp-1)*100:+.1f}%)")
    print(f"        (referans: A-oracle {report['A_oracle']:.3f} ulasilamaz tavan; "
          f"A-baseline {base:.3f} dense)")
    out = os.path.join(args.data, f"faz3_compare_b{int(budget*1000):03d}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] rapor -> {out}")


if __name__ == "__main__":
    main()
