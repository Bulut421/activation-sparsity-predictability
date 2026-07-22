"""
bench_stage0.py  -  Kademe 0: FLOP kazanci duvar-saatine ceviriliyor mu?
=========================================================================
Model YOK. Sadece OPT-1.3b FFN boyutlarinda iki matmul (2048->8192->2048).

Uc varyant, ayni is:
  (a) full   : y = W2 @ relu(W1 @ x)              [referans]
  (b) gather : canli satirlar DAGINIK (gercek durum)
               W1[idx] @ x -> relu -> W2[:, idx] @ a
  (c) block  : ayni sayida satir ama BITISIK (ideal yerlesim tavani)
               W1[:k] @ x  -> relu -> W2[:, :k] @ a

Okuma:
  (b) > (a) suresi  -> naive gather kaybettiriyor (BEKLENEN negatif)
  (c) << (a)        -> ideal yerlesimde kazanc var; (b)-(c) farki =
                       muhendislik isinin buyuklugu
  (c) ~ (a)         -> bu donanimda hat kapali (memory-bound tavan)

Kart 14 adayi. CPU'da ~2-3 dk.
"""

import argparse, time, json
import torch

def bench(fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0   # ms/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--d-ffn", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=1,
                    help="ayni anda islenen token (decode=1, prefill=daha buyuk)")
    ap.add_argument("--budgets", default="0.05,0.20,0.30,0.40")
    ap.add_argument("--threads", type=int, default=0, help="0 = torch varsayilani")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default="bench_stage0.json")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    d, f, T = args.d_model, args.d_ffn, args.batch_tokens
    W1 = torch.randn(f, d)          # fc1: [d_ffn, d_model]
    W2 = torch.randn(d, f)          # fc2: [d_model, d_ffn]
    x  = torch.randn(T, d)

    # (a) FULL
    def full():
        a = torch.relu(x @ W1.T)     # [T, f]
        return a @ W2.T              # [T, d]
    t_full = bench(full, iters=args.iters)

    print(f"[i] d_model={d} d_ffn={f} tokens={T} threads={torch.get_num_threads()}")
    print(f"[a] FULL           : {t_full:8.3f} ms   (referans)")

    report = {"config": vars(args), "full_ms": t_full, "budgets": {}}

    for b in [float(s) for s in args.budgets.split(",")]:
        k = max(1, int(f * b))

        # canli indeksler: DAGINIK (rastgele, sirali degil - gercek durum)
        idx = torch.randperm(f)[:k]
        # onceden dilimlenmis agirliklar (statik yerlesim varsayimi YOK:
        # her cagrida gather maliyeti dahil olsun diye icerde dilimliyoruz)
        def gather():
            W1s = W1[idx]                    # gather: daginik satir kopyala
            a = torch.relu(x @ W1s.T)        # [T, k]
            W2s = W2[:, idx]                 # gather: daginik sutun kopyala
            return a @ W2s.T
        t_g = bench(gather, iters=args.iters)

        # bitisik blok: ayni k, ardisik bellek (ideal yerlesim TAVANI)
        W1b = W1[:k].contiguous()
        W2b = W2[:, :k].contiguous()
        def block():
            a = torch.relu(x @ W1b.T)
            return a @ W2b.T
        t_b = bench(block, iters=args.iters)

        # gather'in "yerlesim onceden yapilmis" hali de ilginc:
        # indeksler sabitse dilimleme BIR KEZ yapilir -> block'a esdeger mi?
        W1g = W1[idx].contiguous()
        W2g = W2[:, idx].contiguous()
        def gather_prebuilt():
            a = torch.relu(x @ W1g.T)
            return a @ W2g.T
        t_gp = bench(gather_prebuilt, iters=args.iters)

        sp_g  = t_full / t_g
        sp_b  = t_full / t_b
        sp_gp = t_full / t_gp
        print(f"[B={b:.2f} k={k:5d}]  gather: {t_g:8.3f} ms ({sp_g:4.2f}x)   "
              f"prebuilt: {t_gp:8.3f} ms ({sp_gp:4.2f}x)   "
              f"block: {t_b:8.3f} ms ({sp_b:4.2f}x)")
        report["budgets"][f"{b:.2f}"] = {
            "k": k, "gather_ms": t_g, "gather_prebuilt_ms": t_gp,
            "block_ms": t_b, "speedup_gather": sp_g,
            "speedup_prebuilt": sp_gp, "speedup_block": sp_b,
        }

    with open(args.out, "w") as fjson:
        json.dump(report, fjson, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("Okuma: speedup <1 = kayip. gather vs prebuilt farki = ANLIK dilimleme")
    print("bedeli (token-basina degisen maske). prebuilt vs block farki = bellek")
    print("yerlesiminin bedeli. block = bu donanimin teorik tavani.")


if __name__ == "__main__":
    main()