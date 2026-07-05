"""
predictor_quality.py  -  Kart 9: predictor-in-the-loop, butce vs perplexity
============================================================================
ASIL OLCUT. Oracle degil, GERCEK predictor maskesiyle uctan uca ppl.

3 faz (tek komut):
  A) TOPLA : train prompt'lari gecerken her katmanda (x, canli-maske) topla
  B) EGIT  : katman basina TAM lineer predictor (BCE) -> SVD ile rank-r'ye kes
             (analyze_sparsity_v2 sweep'inde dogrulanan yol)
  C) OLC   : HELD-OUT prompt'larda butce -> ppl, UC EGRI:
             - oracle  (oracle_quality_opt13b.json'dan, referans)
             - STATIK  (train'den global en-sik-canli top-m, sabit, girdiye bakmaz)
             - predictor (dusuk-rank lineer, girdiye bakar)
             Predictor'in katkisi = statikle arasindaki fark (Stage A dersi:
             global-core tuzagina ppl seviyesinde tekrar dusme).

Sizinti disiplini: RASTGELE prompt-seviyesi split (seed'li). Sirali split
KULLANMA — dosya domain'e gore siraliysa istemeden domain-holdout olcersin
(v1 kosusunda eval baseline 42.8 vs beklenen ~18: bu tuzagin ta kendisi).
Oracle da AYNI eval setinde olculur -> uc egri ayni zeminde.

Simdilik OPT-mimarisi (fc1/fc2). SiLU hatti kapali (Kart 6-7).
"""

import argparse, json, math, time, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path, tok, skip, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            o = json.loads(line)
            if "prompt" in o:
                out.append(o["prompt"])
            elif "text" in o:
                out.append(o["text"])
            elif "messages" in o:
                out.append(tok.apply_chat_template(o["messages"], tokenize=False))
            if len(out) >= limit:
                break
    return out


def train_lowrank(X, M, rank, device, epochs=8, bs=4096, lr=1e-3, seed=0):
    """Tam lineer egit -> SVD ile rank-r'ye kes. Doner: (V [r,d_in], U [d_out,r], b)."""
    torch.manual_seed(seed)
    d_in, d_out = X.shape[1], M.shape[1]
    net = nn.Linear(d_in, d_out).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Mt = torch.tensor(M, dtype=torch.float32, device=device)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), Mt[b])
            loss.backward(); opt.step()
    with torch.no_grad():
        W = net.weight.detach()                      # [d_out, d_in]
        U_, S_, Vt_ = torch.linalg.svd(W, full_matrices=False)
        U = (U_[:, :rank] * S_[:rank])               # [d_out, r]
        V = Vt_[:rank]                               # [r, d_in]
        b = net.bias.detach()
    return V.contiguous(), U.contiguous(), b


@torch.no_grad()
def perplexity(model, tok, texts, device, max_tokens):
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n
        total_tok += n
    return math.exp(total_nll / total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--train-limit", type=int, default=448)
    ap.add_argument("--eval-limit", type=int, default=112)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--budgets", default="0.40,0.35,0.30,0.25",
                    help="tutulan noron orani (analyze butce onerisi: ~0.30-0.35)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="predictor_quality_report.json")
    ap.add_argument("--load-preds", default=None,
                    help="onceki kosunun .pt dosyasi -> Faz A+B atlanir (dakikalar)")
    ap.add_argument("--seed", type=int, default=0, help="rastgele prompt-split seed")
    ap.add_argument("--modes", default="oracle,static,pred",
                    help="teshis: olculecek egriler (virgullu)")
    ap.add_argument("--only-layers", default="all",
                    help="teshis: maske sadece bu katmanlara (or. '12' ya da '12,20')")
    ap.add_argument("--progressive", action="store_true",
                    help="Kart 10: hata-farkindalikli egitim — L_k egitilirken "
                         "L_0..L_{k-1} maskeleri TAKILI (24 toplama gecisi, yavas)")
    ap.add_argument("--train-budget", type=float, default=0.30,
                    help="--progressive'de egitim sirasinda takilan maske butcesi "
                         "(predictor bu butceye ozgu olur)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if args.device == "cpu" else torch.float16,
        device_map=args.device).eval()
    assert hasattr(model.model, "decoder"), "simdilik sadece OPT (fc1/fc2)"
    layers = model.model.decoder.layers
    nL = len(layers)
    print(f"[i] {nL} katman (OPT)")

    allp = load_prompts(args.prompts, tok, 0, args.train_limit + args.eval_limit)
    idx = np.random.default_rng(args.seed).permutation(len(allp))
    train_prompts = [allp[i] for i in idx[:args.train_limit]]
    eval_prompts = [allp[i] for i in idx[args.train_limit:]]
    print(f"[i] train={len(train_prompts)} / eval={len(eval_prompts)} prompt "
          f"(RASTGELE split, seed={args.seed}, kesisim yok)")

    t0 = time.time()
    ckpt_path = (f"predictor_weights_r{args.rank}_prog{int(args.train_budget*100)}.pt"
                 if args.progressive else f"predictor_weights_r{args.rank}.pt")
    if args.load_preds:
        ck = torch.load(args.load_preds)
        preds, static_order = ck["preds"], ck["static_order"]
        if ck.get("seed") != args.seed:
            print(f"[UYARI] ckpt seed={ck.get('seed')} != --seed {args.seed}: "
                  f"egitim/eval kesisebilir -> sizinti riski!")
        print(f"[i] predictor'lar yuklendi: {args.load_preds} (Faz A+B atlandi)")
    else:
        fn = collect_and_train_progressive if args.progressive else collect_and_train
        preds, static_order = fn(model, layers, tok, train_prompts, args, nL, t0)
        torch.save({"preds": preds, "static_order": static_order,
                    "seed": args.seed, "progressive": args.progressive,
                    "train_budget": args.train_budget if args.progressive else None},
                   ckpt_path)
        print(f"[ok] predictor agirliklari -> {ckpt_path} (tekrar icin --load-preds)")

    run_eval(model, layers, tok, eval_prompts, preds, static_order, args, nL)


def collect_and_train(model, layers, tok, train_prompts, args, nL, t0):
    # ---------------- FAZ A: TOPLA ----------------
    cap_x = {L: [] for L in range(nL)}
    cap_m = {L: [] for L in range(nL)}

    def x_hook(L):
        def h(module, inp):
            cap_x[L].append(inp[0].detach().to(torch.float16).cpu())
        return h

    def m_hook(L):
        def h(module, inp):
            cap_m[L].append((inp[0] > 0).cpu())      # canli maske, bool
        return h

    handles = []
    for L in range(nL):
        handles.append(layers[L].fc1.register_forward_pre_hook(x_hook(L)))
        handles.append(layers[L].fc2.register_forward_pre_hook(m_hook(L)))

    with torch.no_grad():
        for i, p in enumerate(train_prompts):
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=args.max_tokens).input_ids.to(args.device)
            model(ids)
            if (i + 1) % 100 == 0:
                print(f"  topla {i+1}/{len(train_prompts)} ({time.time()-t0:.0f}s)")
    for h in handles:
        h.remove()

    # ---------------- FAZ B: EGIT ----------------
    preds, static_order = {}, {}
    for L in range(nL):
        X = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap_x[L]]).float().numpy()
        M = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap_m[L]]).float().numpy()
        cap_x[L] = cap_m[L] = None                   # bellegi birak
        # statik baseline icin: train'de en sik canli noronlarin sirasi
        static_order[L] = torch.tensor(np.argsort(-M.mean(0)).copy())
        V, U, b = train_lowrank(X, M, args.rank, args.device, epochs=args.epochs)
        live = M.mean()
        preds[L] = (V, U, b)
        print(f"  egit L{L:>2}: N={len(X)} canli={live:.1%} rank={args.rank} "
              f"({time.time()-t0:.0f}s)")
        del X, M
    return preds, static_order


def collect_and_train_progressive(model, layers, tok, train_prompts, args, nL, t0):
    """Kart 10: hata-farkindalikli egitim. L_k icin (x, canli) L_0..L_{k-1}
    predictor maskeleri (butce=train_budget) TAKILI forward'dan toplanir ->
    egitim dagilimi = cikarim dagilimi. 24 toplama gecisi: yavas ama durust."""
    preds, static_order = {}, {}
    tb = args.train_budget
    state = {"cur": -1}
    xstash, xcap, mcap = {}, [], []

    def fc1_hook(L):
        def h(module, inp):
            if L < state["cur"]:
                xstash[L] = inp[0]
            elif L == state["cur"]:
                xcap.append(inp[0].detach().to(torch.float16).cpu())
        return h

    def fc2_hook(L):
        def h(module, inp):
            if L < state["cur"]:
                a = inp[0]
                V, U, b = preds[L]
                logits = (xstash[L] @ V.T) @ U.T + b
                m = max(1, int(a.shape[-1] * tb))
                thr = logits.topk(m, dim=-1).values[..., -1:]
                return (a * (logits >= thr),)
            if L == state["cur"]:
                mcap.append((inp[0] > 0).cpu())
            return None
        return h

    handles = []
    for L in range(nL):
        handles.append(layers[L].fc1.register_forward_pre_hook(fc1_hook(L)))
        handles.append(layers[L].fc2.register_forward_pre_hook(fc2_hook(L)))

    print(f"[i] PROGRESSIVE egitim: train_budget={tb} (predictor bu butceye ozgu)")
    for L in range(nL):
        state["cur"] = L
        xcap.clear(); mcap.clear()
        with torch.no_grad():
            for p in train_prompts:
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=args.max_tokens).input_ids.to(args.device)
                model(ids)
        X = torch.cat([t.reshape(-1, t.shape[-1]) for t in xcap]).float().numpy()
        M = torch.cat([t.reshape(-1, t.shape[-1]) for t in mcap]).float().numpy()
        static_order[L] = torch.tensor(np.argsort(-M.mean(0)).copy())
        V, U, b = train_lowrank(X, M, args.rank, args.device, epochs=args.epochs)
        preds[L] = (V, U, b)
        print(f"  prog L{L:>2}: N={len(X)} canli={M.mean():.1%} "
              f"({time.time()-t0:.0f}s)")
        del X, M
    for h in handles:
        h.remove()
    return preds, static_order


def run_eval(model, layers, tok, eval_prompts, preds, static_order, args, nL):
    # ---------------- FAZ C: OLC (statik + predictor) ----------------
    # predictor'lari model cihazina/fp32'ye tasi (cuda ckpt uyumu)
    mdev = next(model.parameters()).device
    preds = {L: (V.to(mdev).float(), U.to(mdev).float(), b.to(mdev).float())
             for L, (V, U, b) in preds.items()}
    BUDGET = {"frac": None, "mode": "pred"}
    xstash = {}
    static_mask = {}                                 # (L, m) -> [d_ffn] bool
    active = set(range(nL)) if args.only_layers == "all" else \
             set(int(x) for x in args.only_layers.split(","))
    if len(active) < nL:
        print(f"[i] TESHIS modu: maske sadece {sorted(active)} katman(lar)inda")

    def stash_hook(L):
        def h(module, inp):
            if BUDGET["frac"] is not None and BUDGET["mode"] == "pred":
                xstash[L] = inp[0]
        return h

    def mask_hook(L):
        def h(module, inp):
            frac = BUDGET["frac"]
            if frac is None or L not in active:
                return None
            a = inp[0]                               # [T, d_ffn] (OPT duz)
            m = max(1, int(a.shape[-1] * frac))
            if BUDGET["mode"] == "oracle":           # referans: gercek |a| top-m
                thr = a.abs().topk(m, dim=-1).values[..., -1:]
                return (a * (a.abs() >= thr),)
            if BUDGET["mode"] == "static":
                key = (L, m)
                if key not in static_mask:
                    v = torch.zeros(a.shape[-1], dtype=torch.bool, device=a.device)
                    v[static_order[L][:m].to(a.device)] = True
                    static_mask[key] = v
                return (a * static_mask[key],)
            V, U, b = preds[L]
            logits = (xstash[L].float() @ V.T) @ U.T + b   # dusuk-rank predictor
            thr = logits.topk(m, dim=-1).values[..., -1:]
            return (a * (logits >= thr),)
        return h

    handles = []
    for L in range(nL):
        handles.append(layers[L].fc1.register_forward_pre_hook(stash_hook(L)))
        handles.append(layers[L].fc2.register_forward_pre_hook(mask_hook(L)))

    report = {"rank": args.rank, "n_layers": nL, "eval_prompts": len(eval_prompts)}
    budgets = [float(x) for x in args.budgets.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    runs = [(None, "pred")] + [(b, m) for b in budgets for m in modes]
    for bud, mode in runs:
        BUDGET["frac"] = bud; BUDGET["mode"] = mode
        t1 = time.time()
        ppl = perplexity(model, tok, eval_prompts, args.device, args.max_tokens)
        name = "baseline" if bud is None else f"{mode}_{bud:.2f}"
        report[name] = round(ppl, 4)
        base = report.get("baseline", ppl)
        print(f"[{name:>12}]  ppl={ppl:8.3f}   delta={(ppl/base-1)*100:+6.1f}%   "
              f"({time.time()-t1:.0f}s)")
    for h in handles:
        h.remove()

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("Okuma: UC egri yan yana -> oracle (oracle_quality_opt13b.json) / statik / predictor.")
    print("       Predictor'in KATKISI = ayni butcede statikle arasindaki fark.")
    print("       Statik zaten tasiyorsa predictor'a gerek yok (Stage A dersi, ppl'de).")


if __name__ == "__main__":
    main()
