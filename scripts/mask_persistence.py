"""
mask_persistence.py  -  Kart 15: maske kaliciligi analizi (kod kaydi)
======================================================================
Kart 15'in sayilari sandbox oturumunda uretilmisti; tekrarlanabilirlik
ilkesi geregi koda dokuldu. Model kosusu YOK — collect_sparsity_v2
ciktisindaki npz'lerden calisir.

Uc olcum (katman basina):
  1. Ciplak kalicilik : recall( live(t+j) | live(t) )         — gap sweep
  2. Union penceresi  : N ardisik token'in canli-union butcesi +
                        bir SONRAKI token'i kapsama orani
  3. Melez            : live(t) UNION statik-topB  ->  live(t+j) recall

Kart 15 verdicti: OPT'de churn sert (gap=1'de ~0.39), union/melez hicbir
varyant yuksek-recall (~0.999) rejimine yaklasmiyor -> "maskeyi N token
sabit tut" kapali. (Kart 16'da prefill union sismesi ayni kokten.)

NOT: canli maske (A != 0) — gated-ReLU tuzagi (bkz. analyze_sparsity_v2).

Kosu: python mask_persistence.py --data sparsity_data_opt --budget 0.30
"""

import argparse, glob, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sparsity_data_opt")
    ap.add_argument("--budget", type=float, default=0.30, help="melez statik dolgu butcesi")
    ap.add_argument("--gaps", default="1,2,4,8,16")
    ap.add_argument("--windows", default="4,8,16,32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gaps = [int(g) for g in args.gaps.split(",")]
    wins = [int(w) for w in args.windows.split(",")]
    report = {}

    for path in sorted(glob.glob(os.path.join(args.data, "layer_*.npz"))):
        L = os.path.basename(path).replace("layer_", "").replace(".npz", "")
        d = np.load(path)
        M = (d["A"].astype(np.float32) != 0)      # canli maske (isaret-guvenli)
        P = d["P"]
        f = M.shape[1]
        k = int(f * args.budget)
        rep = {"live_rate": round(float(M.mean()), 4)}
        prompts = [np.where(P == p)[0] for p in np.unique(P)]

        # 1) ciplak kalicilik
        rep["stale"] = {}
        for j in gaps:
            rs = []
            for idx in prompts:
                if len(idx) <= j:
                    continue
                a, b = M[idx[:-j]], M[idx[j:]]
                rs.append(((a & b).sum(1) / (b.sum(1) + 1e-9)).mean())
            rep["stale"][j] = round(float(np.mean(rs)), 4)

        # 2) union penceresi
        rep["union"] = {}
        for N in wins:
            bud, rec = [], []
            for idx in prompts:
                if len(idx) <= N:
                    continue
                for s in range(0, len(idx) - N, N):
                    u = M[idx[s:s + N]].any(0)
                    bud.append(u.mean())
                    nxt = M[idx[s + N]]
                    rec.append((u & nxt).sum() / (nxt.sum() + 1e-9))
            rep["union"][N] = {"budget": round(float(np.mean(bud)), 4),
                               "next_recall": round(float(np.mean(rec)), 4)}

        # 3) melez: live(t) + statik-topB dolgu
        freq_order = np.argsort(-M.mean(0))
        st = np.zeros(f, bool); st[freq_order[:k]] = True
        rep["hybrid"] = {}
        for j in gaps:
            rs = []
            for idx in prompts:
                if len(idx) <= j:
                    continue
                stale = M[idx[:-j]] | st
                nxt = M[idx[j:]]
                rs.append(((stale & nxt).sum(1) / (nxt.sum(1) + 1e-9)).mean())
            rep["hybrid"][j] = round(float(np.mean(rs)), 4)

        report[L] = rep
        print(f"[L{L}] canli={rep['live_rate']:.1%}  "
              f"stale(j=1)={rep['stale'][gaps[0]]:.3f}  "
              f"union(N={wins[-1]})={rep['union'][wins[-1]]['budget']:.1%}"
              f"/{rep['union'][wins[-1]]['next_recall']:.3f}  "
              f"melez(j=1)={rep['hybrid'][gaps[0]]:.3f}")

    out = args.out or os.path.join(args.data, "mask_persistence_report.json")
    with open(out, "w") as fj:
        json.dump(report, fj, indent=2)
    print(f"\n[ok] rapor -> {out}")
    print("Okuma: hedef rejim ~0.999. Hicbir sutun yaklasmiyorsa 'maskeyi sabit")
    print("tut' kapali (Kart 15 verdicti). B modelinde (Faz 3) churn'u BUNUNLA olc.")


if __name__ == "__main__":
    main()
