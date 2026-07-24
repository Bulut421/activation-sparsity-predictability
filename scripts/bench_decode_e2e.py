"""
bench_decode_e2e.py  -  Faz 4b-adim2: UCTAN UCA gercek decode tok/s
====================================================================
Kart 30 tek-FFN mikro-benchmark'ti (born_loop slice-view 7.41x @th1). Soru simdi:
o kernel kazanci TUM MODELDE (tum katman + attention + KV-cache + head) ne kadar
kaliyor? Amdahl: born SADECE FFN'i hizlandirir; attention/head/norm ayni. Uctan
uca kazanc = FFN'in decode-zamanindaki PAYIYLA sinirli.

Olctugumuz (B=1 tek-akis decode, KV-cache'li, T=1/adim):
  - tok/s: none (FFN atla) / dense / born (slice-view) / posthoc (scatter+predictor)
  - FFN payi   = (t_dense - t_none) / t_dense
  - e2e hizlanma(born) = t_dense / t_born
  - FFN-only hizlanma  = (t_dense - t_none) / (t_born - t_none)   [Kart 30 ile kopru]
  - Amdahl tahmini vs olculen e2e (tutarlilik)
  - teorik FFN FLOP orani = 1/butce (=8x @ %12.5)
Olcek supurmesi (--sweep): baby(17.6M) -> opt(1.3B-sekil). Kucukte overhead-bound
(FFN payi dusuk), buyukte compute-bound (FFN payi -> kernel tavanina yaklasir).

Not: agirliklar RASTGELE — duvar-saati deger-bagimsizdir (bench_born_kernel ilkesi).
Mimari/boyut gercek; born blok-secimi T=1'de her token 2 BITISIK 96-dilim = ayni
zamanlama (hangi blok fark etmez). "Gercek-model" = tam model, tek-FFN degil.

Kosu:  python bench_decode_e2e.py --threads 1            # tek config (baby)
       python bench_decode_e2e.py --threads 1 --sweep    # baby->mid->large->opt
       python bench_decode_e2e.py --verify               # slice-view dogrulugu
CPU onerilir (edge/llm-manager hedefi + FFN compute-bound rejim). GPU'da minik
model launch-bound (--device cuda ile bakabilirsin, ama hedef rejim degil).
"""

import argparse, time, statistics
import torch
import torch.nn.functional as F


def _g(*shape, device, dtype):
    return torch.randn(*shape, device=device, dtype=dtype) * 0.02


class Layer:
    def __init__(self, d, h, d_ff, G, r, device, dtype):
        self.d, self.h, self.dh = d, h, d // h
        self.d_ff, self.G, self.bs = d_ff, G, d_ff // G
        z = lambda n: torch.zeros(n, device=device, dtype=dtype)
        self.ln1w = torch.ones(d, device=device, dtype=dtype); self.ln1b = z(d)
        self.ln2w = torch.ones(d, device=device, dtype=dtype); self.ln2b = z(d)
        self.Wqkv = _g(3 * d, d, device=device, dtype=dtype); self.bqkv = z(3 * d)
        self.Wo = _g(d, d, device=device, dtype=dtype); self.bo = z(d)
        self.W1 = _g(d_ff, d, device=device, dtype=dtype); self.b1 = z(d_ff)
        self.W2T = _g(d_ff, d, device=device, dtype=dtype)   # [d_ff,d]: W2T[r] = BITISIK view
        self.Wr = _g(G, d, device=device, dtype=dtype); self.br = z(G)          # router (born)
        self.Pr1 = _g(r, d, device=device, dtype=dtype)                          # predictor (posthoc)
        self.Pr2 = _g(d_ff, r, device=device, dtype=dtype)
        self.K = self.V = None

    def reset(self):
        self.K = self.V = None

    def attn_step(self, x):                                  # x: [1,1,d]
        qkv = F.linear(x, self.Wqkv, self.bqkv)              # [1,1,3d]
        q, k, v = qkv.split(self.d, dim=2)
        q = q.view(1, 1, self.h, self.dh).transpose(1, 2)    # [1,h,1,dh]
        k = k.view(1, 1, self.h, self.dh).transpose(1, 2)
        v = v.view(1, 1, self.h, self.dh).transpose(1, 2)
        self.K = k if self.K is None else torch.cat([self.K, k], dim=2)
        self.V = v if self.V is None else torch.cat([self.V, v], dim=2)
        y = F.scaled_dot_product_attention(q, self.K, self.V)  # tek query, tum key'lere bakar
        y = y.transpose(1, 2).reshape(1, 1, self.d)
        return F.linear(y, self.Wo, self.bo)

    def ffn(self, x1, mode, k, budget):                     # x1: [d] -> [d]
        if mode == "none":
            return torch.zeros_like(x1)
        if mode == "dense":
            a = torch.relu(self.W1 @ x1 + self.b1)           # [d_ff]
            return a @ self.W2T                              # [d]
        if mode == "born":
            logits = self.Wr @ x1 + self.br                  # [G]
            topv, topi = logits.topk(k)
            gate = topv.softmax(0)
            bs = self.bs
            y = torch.zeros(self.d, device=x1.device, dtype=x1.dtype)
            for j in range(k):                               # k BITISIK slice-view (kopyasiz)
                g = int(topi[j]); r = slice(g * bs, (g + 1) * bs)
                a = torch.relu(self.W1[r] @ x1 + self.b1[r]) * gate[j]
                y = y + a @ self.W2T[r]
            return y
        if mode == "posthoc":
            s = self.Pr2 @ torch.relu(self.Pr1 @ x1)         # dusuk-rank predictor -> [d_ff]
            Bn = max(1, int(round(budget * self.d_ff)))
            idx = s.topk(Bn).indices                         # DAGITIK satirlar
            a = torch.relu(self.W1[idx] @ x1 + self.b1[idx]) # gather-kopya (scatter)
            return a @ self.W2T[idx]
        raise ValueError(mode)

    def step(self, x, mode, k, budget):                     # x: [1,1,d]
        x = x + self.attn_step(F.layer_norm(x, (self.d,), self.ln1w, self.ln1b))
        h = F.layer_norm(x, (self.d,), self.ln2w, self.ln2b).view(self.d)
        return x + self.ffn(h, mode, k, budget).view(1, 1, self.d)


class Model:
    def __init__(self, d, L, h, d_ff, G, r, vocab, blk, device, dtype):
        self.d, self.vocab, self.blk = d, vocab, blk
        self.layers = [Layer(d, h, d_ff, G, r, device, dtype) for _ in range(L)]
        self.Wemb = _g(vocab, d, device=device, dtype=dtype)
        self.Wpe = _g(blk, d, device=device, dtype=dtype)
        self.lnfw = torch.ones(d, device=device, dtype=dtype)
        self.lnfb = torch.zeros(d, device=device, dtype=dtype)
        self.nparam = (sum(p.numel() for lyr in self.layers
                           for p in (lyr.Wqkv, lyr.Wo, lyr.W1, lyr.W2T, lyr.Wr))
                       + self.Wemb.numel())

    def reset(self):
        for lyr in self.layers:
            lyr.reset()

    def step(self, tok, pos, mode, k, budget):              # -> next token id (argmax)
        x = (self.Wemb[tok] + self.Wpe[pos]).view(1, 1, self.d)
        for lyr in self.layers:
            x = lyr.step(x, mode, k, budget)
        x = F.layer_norm(x, (self.d,), self.lnfw, self.lnfb).view(self.d)
        logits = self.Wemb @ x                              # [vocab]  (tied head)
        return int(logits.argmax())


@torch.no_grad()
def run_arm(model, mode, k, budget, prompt, decode, warmup, trials, sync):
    prefill = len(prompt)
    per_tok = []
    for _ in range(trials):
        model.reset()
        for p in range(prefill):                            # prefill (KV-cache doldur, timing disi)
            model.step(prompt[p], p, mode, k, budget)
        tok, pos = prompt[-1], prefill
        for _ in range(warmup):
            tok = model.step(tok, pos, mode, k, budget); pos += 1
        sync()
        t0 = time.perf_counter()
        for _ in range(decode):
            tok = model.step(tok, pos, mode, k, budget); pos += 1
        sync()
        per_tok.append((time.perf_counter() - t0) / decode)
    return statistics.median(per_tok)                       # sn/token


def verify_born(d=384, d_ff=1536, G=16, k=2):
    """slice-view born == full-mask born (gate'li top-k) mi? indeks/dilim dogrulugu."""
    torch.manual_seed(0)
    lyr = Layer(d, 6, d_ff, G, 512, "cpu", torch.float32)
    x1 = torch.randn(d)
    # slice-view (uretim yolu)
    y_slice = lyr.ffn(x1, "born", k, k / G)
    # referans: TUM bloklari hesapla, top-k'yi gate'le, geri kalani 0
    logits = lyr.Wr @ x1 + lyr.br
    topv, topi = logits.topk(k); gate = topv.softmax(0)
    a_full = torch.relu(lyr.W1 @ x1 + lyr.b1)
    mask = torch.zeros(d_ff); bs = d_ff // G
    gvec = torch.zeros(d_ff)
    for j in range(k):
        g = int(topi[j]); mask[g*bs:(g+1)*bs] = 1.0; gvec[g*bs:(g+1)*bs] = gate[j]
    y_ref = (a_full * mask * gvec) @ lyr.W2T
    err = (y_slice - y_ref).abs().max().item()
    print(f"[verify] born slice-view vs full-mask  max|Δ| = {err:.2e}  "
          f"-> {'OK' if err < 1e-4 else 'HATA!'}")


PRESETS = [   # (isim, d, L, h, d_ff)
    ("baby",  384,  8,  6, 1536),
    ("mid",   768, 12, 12, 3072),
    ("large", 1536, 24, 16, 6144),
    ("opt",   2048, 24, 32, 8192),
]


def bench_config(name, d, L, h, d_ff, args, dev, dtype, sync):
    G, k = args.G, args.k
    budget = k / G
    blk = args.prefill + args.decode + args.warmup + 4
    model = Model(d, L, h, d_ff, G, min(args.r, d_ff), args.vocab, blk, dev, dtype)
    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab, (args.prefill,)).tolist()
    a = {m: run_arm(model, m, k, budget, prompt, args.decode, args.warmup, args.trials, sync)
         for m in ("none", "dense", "born", "posthoc")}
    tps = {m: 1.0 / a[m] for m in a}
    ffn_share = (a["dense"] - a["none"]) / a["dense"]
    e2e_born = a["dense"] / a["born"]
    e2e_post = a["dense"] / a["posthoc"]
    ffn_only = ((a["dense"] - a["none"]) / (a["born"] - a["none"])
                if a["born"] > a["none"] else float("nan"))
    amdahl = 1.0 / ((1 - ffn_share) + ffn_share / ffn_only) if ffn_only == ffn_only else float("nan")
    print(f"\n=== {name:5s}  d={d} L={L} h={h} d_ff={d_ff}  ~{model.nparam/1e6:.0f}M  "
          f"butce {budget:.1%} (G{G}k{k})  threads={torch.get_num_threads()} ===")
    print(f"  tok/s   none {tps['none']:7.1f} | dense {tps['dense']:7.1f} | "
          f"born {tps['born']:7.1f} | posthoc {tps['posthoc']:7.1f}")
    print(f"  FFN payi (dense)      : {ffn_share:5.1%}")
    print(f"  e2e hizlanma born     : {e2e_born:5.2f}x   (posthoc {e2e_post:4.2f}x)")
    print(f"  FFN-only hizlanma born: {ffn_only:5.2f}x   (Kart30 ~7x @th1 kopru)")
    print(f"  Amdahl tahmini        : {amdahl:5.2f}x   (olculen e2e {e2e_born:.2f}x ile tutarli mi)")
    print(f"  teorik FFN FLOP       : {1/budget:5.2f}x")
    return {"name": name, "d": d, "L": L, "d_ff": d_ff, "params_M": round(model.nparam/1e6, 1),
            "budget": budget, "sec_per_tok": a, "tok_s": tps, "ffn_share": ffn_share,
            "e2e_born": e2e_born, "e2e_posthoc": e2e_post, "ffn_only_born": ffn_only,
            "amdahl_pred": amdahl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--r", type=int, default=512, help="posthoc predictor rank")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep", action="store_true", help="baby->mid->large->opt")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="bench_decode_e2e.json")
    args = ap.parse_args()

    if args.verify:
        verify_born(); return
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dev = args.device or "cpu"
    dtype = torch.float32
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

    configs = PRESETS if args.sweep else [("baby", args.d_model, args.n_layer,
                                           args.n_head, args.d_ff)]
    print(f"[i] device={dev} dtype=fp32  prefill={args.prefill} decode={args.decode} "
          f"trials={args.trials}  (agirlik rastgele; duvar-saati deger-bagimsiz)")
    rows = [bench_config(n, d, L, h, f, args, dev, dtype, sync) for (n, d, L, h, f) in configs]

    import json
    with open(args.out, "w") as fp:
        json.dump({"device": dev, "threads": torch.get_num_threads(),
                   "prefill": args.prefill, "decode": args.decode, "rows": rows}, fp, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("OKUMA: born ~ dense'ten HIZLI olmali (FFN payi kadar). FFN payi olcekle "
          "BUYUMELI (buyuk modelde FFN baskin) -> e2e born hizlanmasi kernel tavanina yaklasir.")
    print("posthoc < born beklenir (scatter-gather + predictor overhead; Kart 30 ~6x fark).")


if __name__ == "__main__":
    main()
