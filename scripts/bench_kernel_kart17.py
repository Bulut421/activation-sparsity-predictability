"""
bench_kernel_kart17.py  -  Kart 17: kopyasiz gather-matvec kernel (DECODE)
===========================================================================
Kart 14: prebuilt ~ block -> dagitik satirlari dogrudan okumak bitisikten
farksiz. Oyleyse satirlari MATERIALIZE ETMEDEN okuyan bir kernel, block'un
~3.9x tavanina yaklasabilir. Kart 16 prefill'i kapatti; bu decode'un testi.

Varyantlar (T=1, OPT FFN boyutlari, fp32):
  (a) torch full        : referans (MKL/oneDNN)
  (b) torch prebuilt    : dilim onceden alinmis — Kart 14 tavani (kopya HARIC)
  (c) numba gather      : ASIL ADAY — W1[idx] satirlari + W2T[idx] satirlari
                          dogrudan okunur, kopya YOK, maske HER CAGRIDA farkli
  (d) numba full        : kontrol — numba'nin kendi tam FFN'i
                          (c/d = numba-ici kazanc; MKL-numba verim farkini ayirir)
  (+) predictor matvec  : r512 predictor'un T=1 bedeli (butceye dahil edilecek)

Okuma:
  c, a'ya karsi >= ~2.5x @ B=0.30 -> kernel yolu ACIK, AVX/C++ tasimaya deger
  c ~ a                           -> numba yetmiyor; C++/AVX prototipe gec
  c/d orani yuksek ama c/a dusukse -> numba tabani yavas, verdict yine C++

Kosu:  pip install numba
       python bench_kernel_kart17.py
"""

import argparse, json, time
import numpy as np
import torch

try:
    from numba import njit, prange, set_num_threads, get_num_threads
except ImportError:
    raise SystemExit("pip install numba")


@njit(parallel=True, fastmath=True, cache=True)
def nb_gather_ffn(x, W1, b1, W2T, b2, idx, a, y, nblk):
    k = idx.shape[0]
    d = x.shape[0]
    # fc1: secili satirlar, dogrudan oku (kopya yok)
    for j in prange(k):
        row = W1[idx[j]]
        s = 0.0
        for t in range(d):
            s += row[t] * x[t]
        s += b1[idx[j]]
        a[j] = s if s > 0.0 else 0.0
    # fc2: y = sum_j a[j] * W2T[idx[j]]  (W2T satirlari contiguous)
    blk = d // nblk
    for bi in prange(nblk):
        st = bi * blk
        en = d if bi == nblk - 1 else st + blk
        for t in range(st, en):
            y[t] = b2[t]
        for j in range(k):
            aj = a[j]
            if aj == 0.0:
                continue
            row = W2T[idx[j]]
            for t in range(st, en):
                y[t] += aj * row[t]
    return y


def bench(fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--d-ffn", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--budgets", default="0.05,0.20,0.30,0.40")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--nblk", type=int, default=16, help="fc2 d-blok sayisi")
    ap.add_argument("--out", default="bench_kernel_kart17.json")
    args = ap.parse_args()

    d, f, r = args.d_model, args.d_ffn, args.rank
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((f, d), dtype=np.float32)
    b1 = rng.standard_normal(f).astype(np.float32)
    W2T = rng.standard_normal((f, d), dtype=np.float32)   # W2.T contiguous
    b2 = rng.standard_normal(d).astype(np.float32)
    x = rng.standard_normal(d).astype(np.float32)

    tW1, tb1 = torch.from_numpy(W1), torch.from_numpy(b1)
    tW2T, tb2 = torch.from_numpy(W2T), torch.from_numpy(b2)
    tx = torch.from_numpy(x)

    # (a) torch full
    def torch_full():
        a_ = torch.relu(tx @ tW1.T + tb1)
        return a_ @ tW2T + tb2
    t_full = bench(torch_full, iters=args.iters)

    # (+) predictor bedeli (r512, T=1)
    V = torch.from_numpy(rng.standard_normal((r, d), dtype=np.float32))
    U = torch.from_numpy(rng.standard_normal((f, r), dtype=np.float32))
    def pred():
        return (tx @ V.T) @ U.T
    t_pred = bench(pred, iters=args.iters)

    # (d) numba full (kontrol)
    a_buf = np.empty(f, np.float32)
    y_buf = np.empty(d, np.float32)
    idx_full = np.arange(f, dtype=np.int64)
    nb_gather_ffn(x, W1, b1, W2T, b2, idx_full, a_buf, y_buf, args.nblk)  # JIT
    t_nbfull = bench(lambda: nb_gather_ffn(x, W1, b1, W2T, b2, idx_full,
                                           a_buf, y_buf, args.nblk),
                     iters=max(50, args.iters // 4))

    print(f"[i] d={d} f={f} numba threads={get_num_threads()}")
    print(f"[a] torch full : {t_full:7.3f} ms   [d] numba full : {t_nbfull:7.3f} ms"
          f"   (taban orani {t_full/t_nbfull:.2f}x)")
    print(f"[+] predictor r{r} matvec: {t_pred:7.3f} ms")

    # dogruluk kontrolu (bir kez, B=0.30)
    k0 = int(f * 0.30)
    idx0 = np.sort(rng.permutation(f)[:k0]).astype(np.int64)
    a0 = np.empty(k0, np.float32); y0 = np.empty(d, np.float32)
    nb_gather_ffn(x, W1, b1, W2T, b2, idx0, a0, y0, args.nblk)
    a_ref = np.maximum(x @ W1[idx0].T + b1[idx0], 0)
    y_ref = a_ref @ W2T[idx0] + b2
    err = np.abs(y0 - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
    print(f"[i] dogruluk: max rel err = {err:.2e}  {'OK' if err < 1e-4 else 'FAIL!'}")

    report = {"torch_full_ms": t_full, "numba_full_ms": t_nbfull,
              "predictor_ms": t_pred, "budgets": {}}
    # maske her cagrida degissin (decode gercegi): 32 farkli idx dondur
    for b in [float(s) for s in args.budgets.split(",")]:
        k = max(1, int(f * b))
        idx_pool = [np.sort(rng.permutation(f)[:k]).astype(np.int64)
                    for _ in range(32)]
        a_ = np.empty(k, np.float32); y_ = np.empty(d, np.float32)
        cnt = {"i": 0}
        def gather():
            cnt["i"] = (cnt["i"] + 1) % 32
            return nb_gather_ffn(x, W1, b1, W2T, b2, idx_pool[cnt["i"]],
                                 a_, y_, args.nblk)
        t_g = bench(gather, iters=args.iters)

        tW1s = tW1[idx_pool[0]].contiguous()
        tb1s = tb1[idx_pool[0]]
        tW2s = tW2T[idx_pool[0]].contiguous()
        def prebuilt():
            aa = torch.relu(tx @ tW1s.T + tb1s)
            return aa @ tW2s + tb2
        t_pb = bench(prebuilt, iters=args.iters)

        tot = t_g + t_pred        # kernel + predictor = gercek decode butcesi
        print(f"[B={b:.2f} k={k:5d}]  numba-gather: {t_g:7.3f} ms "
              f"({t_full/t_g:4.2f}x)   +pred: {tot:7.3f} ms ({t_full/tot:4.2f}x)   "
              f"torch-prebuilt(tavan): {t_pb:7.3f} ms ({t_full/t_pb:4.2f}x)")
        report["budgets"][f"{b:.2f}"] = {
            "k": k, "numba_gather_ms": t_g, "with_pred_ms": tot,
            "prebuilt_ms": t_pb,
            "speedup_vs_torch_full": round(t_full / t_g, 3),
            "speedup_with_pred": round(t_full / tot, 3),
            "ceiling_prebuilt": round(t_full / t_pb, 3)}

    with open(args.out, "w") as fj:
        json.dump(report, fj, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("Okuma: '+pred' sutunu gercek decode kazanci (predictor dahil).")
    print("numba-gather prebuilt tavanina yaklasiyorsa kernel yolu ACIK ->")
    print("llm-manager C++/AVX row-skip'e tasimaya deger. Yaklasamiyorsa")
    print("numba tabanina bak ([d] satiri): taban yavassa suc numba'da, C++ dene.")


if __name__ == "__main__":
    main()
