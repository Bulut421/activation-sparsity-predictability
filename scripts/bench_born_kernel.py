"""
bench_born_kernel.py  -  Kart 30 (Faz 4b): born decode kernel'i realize edilebilir mi?
========================================================================================
Soru: born-eye'in bloklari BITISIK. Decode'da (T=1) secili k blogu okumak =
BITISIK gather (k adet ardisik dilim), post-hoc'un DAGITIK satir gather'i degil.
Kart 14 dersi: dagitik per-call gather olduruyordu; bitisik/prebuilt hizliydi.
Born mimarisi tam bu bitisik durumda doguyor -> kernel "bedava" olabilir mi?

Model YOK; gercek OPT-olcek FFN boyutlari (17.6M bebek overhead'e bogar).
T=1 (decode), butce %12.5. Varyantlar:
  full         : relu(W1@x)@W2            (dense referans)
  born         : router + k BITISIK blok per-call gather (cat) + matmul  [born decode]
  born_loop    : ayni ama blok-blok VIEW (kopyasiz, Python-loop overhead'li)
  posthoc      : k*bs DAGITIK satir per-call gather + matmul  (Kart 17 gather kipi)
  ceiling      : bitisik dilim ONCEDEN kopyalanmis (per-call gather yok) = tavan

Okuma:
  born ~ ceiling  &&  born >> posthoc  -> born'un bitisik yapisi decode kernel'ini
     STOK matmul ile realize ediyor (numba/C++ ozel kernel gereksiz). 4b POZITIF.
  born ~ posthoc  -> bitisiklik yardim etmiyor, ozel kernel lazim.

Kosu: python bench_born_kernel.py        (torch, CPU; --threads 1 de dene)
"""

import argparse, time
import torch


def bench(fn, warmup=20, iters=500):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0   # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--d-ffn", type=int, default=8192)
    ap.add_argument("--G", type=int, default=16, help="blok sayisi")
    ap.add_argument("--k", type=int, default=2, help="aktif blok (aktif%=k/G)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    d, f, G, k = args.d_model, args.d_ffn, args.G, args.k
    bs = f // G                       # blok boyutu (noron)
    budget = k / G
    x = torch.randn(d)
    W1 = torch.randn(f, d); b1 = torch.randn(f)
    W2T = torch.randn(f, d)           # W2 transpoze: W2T[rows] BITISIK view
    Wr = torch.randn(G, d)            # router

    # (a) FULL
    def full():
        a = torch.relu(W1 @ x + b1)   # [f]
        return W2T.T @ a              # [d]
    t_full = bench(full, iters=args.iters)

    # ortak: router + top-k secim (born'da her cagride)
    def route():
        logits = Wr @ x
        topv, topi = logits.topk(k)
        return (topv.softmax(0)), topi

    # (b) BORN: k bitisik blogu cat ile topla (kucuk kopya), tek matmul
    def born():
        gate, topi = route()
        rows = torch.cat([torch.arange(int(g)*bs, (int(g)+1)*bs) for g in topi])
        gvec = gate.repeat_interleave(bs)                 # [k*bs]
        a = torch.relu(W1[rows] @ x + b1[rows]) * gvec    # [k*bs]
        return W2T[rows].T @ a
    t_born = bench(born, iters=args.iters)

    # (c) BORN_LOOP: blok-blok VIEW (kopyasiz), Python-loop
    def born_loop():
        gate, topi = route()
        y = torch.zeros(d)
        for j in range(k):
            g = int(topi[j]); r = slice(g*bs, (g+1)*bs)
            a = torch.relu(W1[r] @ x + b1[r]) * gate[j]   # W1[r], W2T[r] = view
            y = y + W2T[r].T @ a
        return y
    t_loop = bench(born_loop, iters=args.iters)

    # (d) POSTHOC: k*bs DAGITIK satir per-call gather (Kart 17 gather kipi)
    kk = k * bs
    def posthoc():
        idx = torch.randperm(f)[:kk]                      # dagitik
        a = torch.relu(W1[idx] @ x + b1[idx])
        return W2T[idx].T @ a
    t_post = bench(posthoc, iters=args.iters)

    # (e) CEILING: bitisik dilim ONCEDEN kopyalanmis (per-call gather yok)
    rows0 = torch.cat([torch.arange(j*bs, (j+1)*bs) for j in range(k)])
    W1c = W1[rows0].contiguous(); b1c = b1[rows0].contiguous()
    W2c = W2T[rows0].contiguous()
    def ceiling():
        a = torch.relu(W1c @ x + b1c)
        return W2c.T @ a
    t_ceil = bench(ceiling, iters=args.iters)

    print(f"[i] d={d} f={f} G={G} k={k} -> aktif {budget:.1%} (blok={bs}) "
          f"threads={torch.get_num_threads()}")
    print(f"[a] full          : {t_full:7.4f} ms   (1.00x)")
    print(f"[b] born (cat)    : {t_born:7.4f} ms   ({t_full/t_born:5.2f}x)  <- decode kernel")
    print(f"[c] born_loop     : {t_loop:7.4f} ms   ({t_full/t_loop:5.2f}x)  (kopyasiz view)")
    print(f"[d] posthoc(scat) : {t_post:7.4f} ms   ({t_full/t_post:5.2f}x)  (dagitik gather)")
    print(f"[e] ceiling       : {t_ceil:7.4f} ms   ({t_full/t_ceil:5.2f}x)  (prebuilt tavan)")
    print(f"[+] teorik FFN    : {1/budget:5.2f}x   (1/butce, router haric)")
    print("\nOkuma: born ~ ceiling && born >> posthoc -> bitisik yapi decode'u STOK")
    print("matmul ile realize ediyor (ozel kernel gereksiz), 4b POZITIF.")
    print("born ~ posthoc -> bitisiklik yetmiyor, ozel kernel lazim.")


if __name__ == "__main__":
    main()
