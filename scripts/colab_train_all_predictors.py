"""
colab_train_all_predictors.py  -  Kart 13 hazirlik (Colab H100)
================================================================
300k wikitext token'iyla 24 katmanin HEPSINDE predictor egit,
predictor_quality.py'nin --load-preds formatinda ckpt kaydet +
held-out belgeleri jsonl'e dok.

Bellek dostu: katman basina ayri toplama gecisi (24 gecis, H100'de ~1s/belge
degil — toplam ~25-35 dk). Tek katmanin verisi ~4.5GB, gecis sonunda birakilir.

Sonra (ayni Colab oturumunda, predictor_quality.py'yi de yukleyip):
  python predictor_quality.py --prompts wikitext_eval.jsonl --device cuda \
      --load-preds predictor_weights_wt300k_r512.pt \
      --train-limit 0 --eval-limit 999 \
      --budgets 0.40,0.30,0.25,0.20 --modes static,pred \
      --out predictor_quality_wt300k.json
"""

import json, os, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL   = "facebook/opt-1.3b"
TRAIN_TOKENS = 300_000
EVAL_DOCS    = 200                  # jsonl'e dokulacek held-out belge sayisi
RANK    = 512                       # Kart 12: r128 kapasiteye takildi
EPOCHS  = 8
MAX_TOK = 256
SEED    = 0
DEV     = "cuda" if torch.cuda.is_available() else "cpu"


def get_texts(approx_tokens):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    texts, est = [], 0
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 200:
            continue
        texts.append(t)
        est += min(len(t) // 4, MAX_TOK)
        if est > approx_tokens:
            break
    return texts


def train_lowrank(X, M, rank, epochs=EPOCHS, bs=8192, lr=1e-3):
    d_in, d_out = X.shape[1], M.shape[1]
    torch.manual_seed(SEED)
    net = nn.Linear(d_in, d_out).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    N = len(X)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(X[b].float().to(DEV)), M[b].float().to(DEV))
            loss.backward(); opt.step()
    with torch.no_grad():
        W = net.weight.detach().float()
        U_, S_, Vt_ = torch.linalg.svd(W, full_matrices=False)
        U = (U_[:, :rank] * S_[:rank]).cpu()      # [d_out, r]
        V = Vt_[:rank].cpu()                      # [r, d_in]
        b = net.bias.detach().cpu()
    del net
    torch.cuda.empty_cache()
    return V.contiguous(), U.contiguous(), b


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32,
        device_map=DEV).eval()
    layers = model.model.decoder.layers
    nL = len(layers)

    texts = get_texts(int((TRAIN_TOKENS + 60_000) * 1.3))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(texts))
    eval_texts = [texts[i] for i in perm[:EVAL_DOCS]]
    train_texts = [texts[i] for i in perm[EVAL_DOCS:]]
    print(f"[i] {len(train_texts)} train / {len(eval_texts)} eval belge "
          f"(belge-seviyesi ayrim, seed={SEED})")

    with open("wikitext_eval.jsonl", "w", encoding="utf-8") as f:
        for t in eval_texts:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"[ok] eval belgeleri -> wikitext_eval.jsonl ({len(eval_texts)})")

    out = f"predictor_weights_wt300k_r{RANK}.pt"
    preds, static_order = {}, {}
    if os.path.exists(out):                    # kaldigi yerden devam (Colab kopmasi)
        ck = torch.load(out)
        preds, static_order = ck["preds"], ck["static_order"]
        print(f"[i] {out} bulundu: {sorted(preds.keys())} hazir, devam ediliyor")
    cap = {"x": [], "m": []}
    state = {"cur": -1}

    def x_hook(L):
        def h(module, inp):
            if L == state["cur"]:
                cap["x"].append(inp[0].detach().to(torch.float16).cpu())
        return h

    def m_hook(L):
        def h(module, inp):
            if L == state["cur"]:
                cap["m"].append((inp[0] > 0).cpu())
        return h

    handles = []
    for L in range(nL):
        handles.append(layers[L].fc1.register_forward_pre_hook(x_hook(L)))
        handles.append(layers[L].fc2.register_forward_pre_hook(m_hook(L)))

    t0 = time.time()
    for L in range(nL):
        if L in preds:
            continue                           # onceki oturumda bitmis
        state["cur"] = L
        cap["x"].clear(); cap["m"].clear()
        total = 0
        with torch.no_grad():
            for p in train_texts:
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=MAX_TOK).input_ids.to(DEV)
                model(ids)
                total += ids.shape[1]
                if total >= TRAIN_TOKENS:
                    break
        X = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap["x"]])
        M = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap["m"]])
        static_order[L] = torch.tensor(
            np.argsort(-M.float().mean(0).numpy()).copy())
        V, U, b = train_lowrank(X, M, RANK)
        preds[L] = (V, U, b)
        print(f"  L{L:>2}: N={len(X)} canli={M.float().mean():.1%} "
              f"rank={RANK} ({time.time()-t0:.0f}s)")
        del X, M
        # her katmandan sonra kaydet -> Colab koparsa kaldigi yerden devam
        torch.save({"preds": preds, "static_order": static_order,
                    "seed": SEED, "progressive": False,
                    "train_budget": None, "corpus": "wikitext-103/300k"}, out)

    for h in handles:
        h.remove()
    print(f"\n[ok] ckpt -> {out} ({len(preds)}/{nL} katman)")
    print("Simdi: predictor_quality.py ile in-loop ppl (docstring'deki komut).")


if __name__ == "__main__":
    main()
