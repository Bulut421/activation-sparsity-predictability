"""
analyze_sparsity.py  -  Sinyal var mi?  +  Predictor
=====================================================
Tek soru: "x (FFN girdisi) -> a'nin hangi noronlari sicak" tahmin edilebilir mi?

Iki asama (UCUZDAN PAHALIYA - onceki konusmadaki oneri):

  ASAMA 1  "sinyal var mi" (ucuz, once bunu yap):
     - a'dan sicak maskesi cikar (top-k magnitude, Stage A metrigi)
     - x -> mask icin TEK katmanli lineer predictor egit
     - recall ve precision olc
     - BASELINE: "her zaman global en sik sicak noronlari tahmin et" (statik)
     - predictor baseline'i ASIYOR mu?  ->  sinyal x'te VAR demektir

  ASAMA 2  "tam cozunurluk" (--full ile, sadece sinyal cikarsa):
     - noron basina ayri tahmin (MLP), per-neuron recall

KARAR SAYISI (Stage A'daki "coverage lift" gibi):
  predictor_recall - baseline_recall = LIFT
  LIFT buyukse  ->  x sonraki seyrekligi biliyor  ->  PowerInfer tutuyor  ->  KAPI ACIK
  LIFT ~0 ise   ->  seyreklik var ama x'ten ucuza tahmin edilemiyor  ->  temiz HAYIR
"""

import argparse, glob, os, json
import numpy as np
import torch
import torch.nn as nn


def hot_mask(A, k_frac):
    """a -> sicak noron maskesi. magnitude metrigi (Stage A ile ayni mantik)."""
    mag = np.abs(A)                                  # [N, d_ffn]
    k = int(A.shape[1] * k_frac)
    # her token icin top-k noron = sicak
    idx = np.argpartition(-mag, k, axis=1)[:, :k]
    M = np.zeros_like(mag, dtype=np.float32)
    np.put_along_axis(M, idx, 1.0, axis=1)
    return M                                         # [N, d_ffn] 0/1


def recall_precision(pred, true):
    """pred,true: 0/1 [N, d_ffn].  token basina ortala."""
    tp = (pred * true).sum(1)
    p = tp / (pred.sum(1) + 1e-9)
    r = tp / (true.sum(1) + 1e-9)
    return r.mean(), p.mean()


def global_baseline(M_train, M_test, k):
    """STATIK baseline: egitimde en sik sicak olan k noronu HER ZAMAN tahmin et.
       (domain yok, girdi yok - PowerInfer'in yenmesi gereken sey)"""
    freq = M_train.mean(0)                           # [d_ffn] her noronun sicak olma sikligi
    top = np.argpartition(-freq, k)[:k]
    pred = np.zeros_like(M_test)
    pred[:, top] = 1.0
    return recall_precision(pred, M_test)


def linear_probe(X_tr, M_tr, X_te, M_te, k, device, epochs=8):
    """ASAMA 1: x -> mask, tek katmanli lineer.  Ucuz sinyal testi."""
    d_in, d_out = X_tr.shape[1], M_tr.shape[1]
    net = nn.Linear(d_in, d_out).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Mtr = torch.tensor(M_tr, dtype=torch.float32, device=device)
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(net(Xtr[b]), Mtr[b])
            loss.backward(); opt.step()
    # tahmin: token basina top-k logit = sicak
    with torch.no_grad():
        Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
        logits = net(Xte).cpu().numpy()
    idx = np.argpartition(-logits, k, axis=1)[:, :k]
    pred = np.zeros_like(M_te)
    np.put_along_axis(pred, idx, 1.0, axis=1)
    return recall_precision(pred, M_te)


def full_predictor(X_tr, M_tr, X_te, M_te, k, device, hidden=1024, epochs=12,
                   target_recall=0.90, arch="mlp", ffn_mats=3, A_te=None):
    """ASAMA 2 (--full): x -> mask, 2 katmanli MLP (DejaVu/PowerInfer tarzi).

    Uc soruya cevap verir:
      1. MLP lineer probu asiyor mu?  (recall_topk)
      2. recall ~0.9 icin KAC noron acmak gerek?  (recall@butce egrisi)
         -> skip karari icin asil sayi: frac_for_target
      3. predictor UCUZ mu?  (predictor MAC / FFN MAC orani)
    """
    d_in, d_out = X_tr.shape[1], M_tr.shape[1]
    if arch == "linear":
        net = nn.Linear(d_in, d_out).to(device)
    else:
        net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                            nn.Linear(hidden, d_out)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Mtr = torch.tensor(M_tr, dtype=torch.float32, device=device)
    bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(net(Xtr[b]), Mtr[b])
            loss.backward(); opt.step()
    with torch.no_grad():
        Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
        logits = net(Xte).cpu().numpy()

    # (1) top-k tahmin - ASAMA 1 ile ayni protokol, karsilastirilabilir
    idx = np.argpartition(-logits, k, axis=1)[:, :k]
    pred = np.zeros_like(M_te)
    np.put_along_axis(pred, idx, 1.0, axis=1)
    r_topk, p_topk = recall_precision(pred, M_te)

    # buyukluk-agirlikli recall: kacirilan noronlar buyuk mu kucuk mu?
    # (kucuk |a| kacirmak ppl'e ucuz - duz recall bunu goremez)
    w_topk = None
    if A_te is not None:
        w = np.abs(A_te) * M_te
        w_topk = float((w * pred).sum() / (w.sum() + 1e-9))

    # (2) recall@butce: logit sirasinda ilk m noronu acarsak recall?
    order = np.argsort(-logits, axis=1)                 # [N, d_ffn]
    hits = np.take_along_axis(M_te, order, axis=1)      # sirali isabet 0/1
    cum = np.cumsum(hits, axis=1) / (M_te.sum(1, keepdims=True) + 1e-9)
    curve = cum.mean(0)                                 # recall@m
    budget_recall = {f"{f:.2f}": round(float(curve[int(d_out*f)-1]), 4)
                     for f in (0.20, 0.30, 0.40, 0.50, 0.60)}
    m_needed = int(np.searchsorted(curve, target_recall) + 1)
    frac_needed = min(m_needed / d_out, 1.0)

    # per-neuron recall (testte >=20 kez sicak olanlar)
    hot_count = M_te.sum(0)
    sel = hot_count >= 20
    per_n = (pred * M_te).sum(0)[sel] / hot_count[sel]

    # (3) maliyet: predictor MAC vs FFN MAC (Qwen: 3 matmul, OPT: 2 matmul)
    pred_mac = d_in * d_out if arch == "linear" else d_in * hidden + hidden * d_out
    ffn_mac = ffn_mats * d_in * d_out

    # (4) SVD rank taramasi (sadece linear): egitilmis W'yu kes, yeniden egitme yok.
    #     Soru: kac rank'te butce bozulmadan predictor ucuzlar?
    svd_sweep = None
    if arch == "linear":
        Wm = net.weight.detach().cpu().numpy()          # [d_out, d_in]
        bm = net.bias.detach().cpu().numpy()
        U, S, Vt = np.linalg.svd(Wm, full_matrices=False)
        svd_sweep = {}
        for r in (64, 128, 256, 512):
            Wr = (U[:, :r] * S[:r]) @ Vt[:r]
            lg = X_te @ Wr.T + bm
            o = np.argsort(-lg, axis=1)
            h = np.take_along_axis(M_te, o, axis=1)
            cv = (np.cumsum(h, 1) / (M_te.sum(1, keepdims=True) + 1e-9)).mean(0)
            fr = min((int(np.searchsorted(cv, target_recall)) + 1) / d_out, 1.0)
            pm = d_in * r + r * d_out
            svd_sweep[r] = {"frac_for_target": round(float(fr), 4),
                            "predictor_mac_ratio": round(pm / ffn_mac, 4),
                            "net_ffn_saving": round(1.0 - fr - pm / ffn_mac, 4)}

    return {
        "svd_rank_sweep": svd_sweep,
        "arch": arch, "hidden": hidden, "epochs": epochs, "ffn_mats": ffn_mats,
        "recall_topk": round(float(r_topk), 4),
        "precision_topk": round(float(p_topk), 4),
        "weighted_recall_topk": round(w_topk, 4) if w_topk is not None else None,
        "budget_recall": budget_recall,
        "target_recall": target_recall,
        "frac_for_target": round(float(frac_needed), 4),
        "per_neuron_recall_median": round(float(np.median(per_n)), 4),
        "per_neuron_recall_p10": round(float(np.percentile(per_n, 10)), 4),
        "predictor_mac_ratio": round(pred_mac / ffn_mac, 4),
        # net kazanc tahmini: skip ile FFN'in frac_for_target'i + predictor calisiyor
        "net_ffn_saving": round(1.0 - frac_needed - pred_mac / ffn_mac, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sparsity_data", help="collect_sparsity.py ciktisi klasoru")
    ap.add_argument("--k-frac", type=float, default=0.20, help="sicak kabul edilen noron orani")
    ap.add_argument("--split", type=float, default=0.8, help="train orani (held-out icin)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true", help="ASAMA 2: noron-bazli MLP predictor")
    ap.add_argument("--hidden", type=int, default=1024, help="--full MLP gizli boyut")
    ap.add_argument("--arch", choices=["mlp", "linear"], default="mlp",
                    help="--full predictor mimarisi (linear: MLP'nin gecemedigi durumda)")
    ap.add_argument("--ffn-mats", type=int, default=3,
                    help="FFN matmul sayisi: Qwen/LLaMA=3 (gate/up/down), OPT=2 (fc1/fc2)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--mask-mode", choices=["topk", "live"], default="topk",
                    help="live: M = (A > 0), ReLU modeller icin. Butce = ort. canli sayisi."
                         " topk sifirlar arasinda keyfi secim yapar -> ReLU'da etiket gurultusu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report = {}
    for path in sorted(glob.glob(os.path.join(args.data, "layer_*.npz"))):
        L = os.path.basename(path).replace("layer_", "").replace(".npz", "")
        d = np.load(path)
        X, A = d["X"].astype(np.float32), d["A"].astype(np.float32)
        N, d_ffn = A.shape
        if args.mask_mode == "live":
            # (A != 0): gated-ReLU modellerde (ReluLLaMA: a = relu(gate)*up)
            # sifir-olmayan girdiler NEGATIF olabilir; (A > 0) onlari olu sayardi.
            # OPT'de (relu ciktisi >= 0) davranis ayni.
            M_all = (A != 0).astype(np.float32)         # gercek canli maske
            k = max(1, int(round(M_all.sum(1).mean()))) # butce = ort. canli sayisi
            print(f"  [live] ort. canli oran: {k/d_ffn:.1%} ({k}/{d_ffn} noron)")
        else:
            M_all = None
            k = int(d_ffn * args.k_frac)

        # HELD-OUT ayrimi - EZBER mi GENELLEME mi (Stage A disiplini)
        # SIZINTI ONLEMI: ayni prompt'un token'lari ya HEP train ya HEP test.
        # (token-seviyesi split, komsu token benzerligi yuzunden LIFT'i sisirir)
        if "P" in d.files:
            P = d["P"]
            uniq = np.unique(P)
            np.random.shuffle(uniq)
            n_tr_p = int(len(uniq) * args.split)
            tr_prompts = set(uniq[:n_tr_p].tolist())
            mask_tr = np.isin(P, list(tr_prompts))
            tr = np.where(mask_tr)[0]
            te = np.where(~mask_tr)[0]
            print(f"  [split] prompt-seviyesi: {len(tr_prompts)} train / "
                  f"{len(uniq)-len(tr_prompts)} test prompt")
        else:
            print("  [uyari] P yok -> token-seviyesi split (sizinti riski, "
                  "collect'i v2 ile tekrar calistir)")
            n_tr = int(N * args.split)
            perm = np.random.permutation(N)
            tr, te = perm[:n_tr], perm[n_tr:]

        M = M_all if args.mask_mode == "live" else hot_mask(A, args.k_frac)
        M_tr, M_te = M[tr], M[te]
        X_tr, X_te = X[tr], X[te]

        # baseline (statik global) her iki asamada da lazim
        r_base, p_base = global_baseline(M_tr, M_te, k)

        if args.full:
            # ASAMA 2: noron-bazli MLP predictor + butce/maliyet analizi
            res = full_predictor(X_tr, M_tr, X_te, M_te, k, args.device,
                                 hidden=args.hidden, epochs=args.epochs,
                                 target_recall=args.target_recall,
                                 arch=args.arch, ffn_mats=args.ffn_mats,
                                 A_te=A[te])
            lift = res["recall_topk"] - float(r_base)
            report[L] = {
                "N_tokens": int(N), "d_ffn": int(d_ffn), "k": int(k),
                "k_frac": args.k_frac,
                "baseline_recall": round(float(r_base), 4),
                "LIFT_mlp": round(lift, 4),
                **res,
            }
            wtxt = (f"  w-recall={res['weighted_recall_topk']:.3f}"
                    if res.get("weighted_recall_topk") is not None else "")
            print(f"[L{L}] {args.arch} recall@topk={res['recall_topk']:.3f}{wtxt} "
                  f"(base {r_base:.3f}, LIFT {lift:+.3f})  "
                  f"recall>={args.target_recall:.2f} icin butce="
                  f"{res['frac_for_target']:.0%}  "
                  f"predictor maliyeti={res['predictor_mac_ratio']:.1%} FFN  "
                  f"net kazanc~{res['net_ffn_saving']:.1%}")
            if res.get("svd_rank_sweep"):
                for r, s in res["svd_rank_sweep"].items():
                    print(f"        rank={r:>4}: butce={s['frac_for_target']:.0%}  "
                          f"maliyet={s['predictor_mac_ratio']:.1%}  "
                          f"net={s['net_ffn_saving']:+.1%}")
        else:
            r_pred, p_pred = linear_probe(X_tr, M_tr, X_te, M_te, k, args.device)
            lift = r_pred - r_base
            report[L] = {
                "N_tokens": int(N), "d_ffn": int(d_ffn), "k": int(k),
                "baseline_recall": round(float(r_base), 4),
                "predictor_recall": round(float(r_pred), 4),
                "LIFT": round(float(lift), 4),
                "predictor_precision": round(float(p_pred), 4),
            }
            verdict = "SINYAL VAR (kapi acik)" if lift > 0.10 else \
                      "zayif" if lift > 0.03 else "SINYAL YOK (hayir)"
            print(f"[L{L}] N={N}  baseline_recall={r_base:.3f}  "
                  f"predictor_recall={r_pred:.3f}  LIFT={lift:+.3f}  -> {verdict}")

    # ozet
    mask_tag = "live" if args.mask_mode == "live" else f"k{int(args.k_frac*100):02d}"
    arch_tag = "_linear" if (args.full and args.arch == "linear") else ""
    tag = f"_full_{mask_tag}{arch_tag}" if args.full else \
          ("_live" if args.mask_mode == "live" else "")
    out = os.path.join(args.data, f"sparsity_report{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {out}")
    if args.full:
        print("\nKARAR:  frac_for_target dusuk (<~%50) ve net_ffn_saving > 0  ->  kapi ACIK, "
              "predictor'u inference'a tasimayi dusun")
        print("        frac_for_target ~1.0  ->  recall hedefi bu veriyle ulasIlamaz, "
              "daha buyuk predictor / daha cok veri dene")
    else:
        print("\nKARAR:  LIFT > 0.10  ->  PowerInfer tutuyor, tam-cozunurluk predictor'a gec (--full)")
        print("        LIFT ~ 0     ->  temiz HAYIR (Stage A gibi), neden tahmin edilemedigini incele")


if __name__ == "__main__":
    main()