"""
scaling_probe_colab.py  -  Kart 12: VERI OLCEKLEME EGRISI (Colab H100)
=======================================================================
Son buyuk test. Soru: recall@butce, veri buyudukce 0.99'a GIDIYOR mu,
yoksa DUZLESIYOR mu?

  Gidiyorsa   -> predictor'lari buyuk veriyle egit, Kart 9 protokolunu tekrarla
  Duzlesiyorsa -> kuyruk x'ten OGRENILEMIYOR -> ilkesel STOP, kapanis karti

Kurulum farklari (bilerek):
  - Korpus: wikitext-103 (CESITLI; synth'e bagli kalmiyoruz — Kart 9 v1
    domain dersi). Belge-seviyesi split, ayni sizinti disiplini.
  - Olcek noktalari IC ICE (6k c 30k c 100k c 300k): temiz egri.
  - Ayni test seti tum olceklerde -> noktalar dogrudan karsilastirilir.

Colab kullanimi:
  !pip -q install transformers datasets accelerate
  !python scaling_probe_colab.py
Cikti: scaling_probe_results.json + tablo. Sure: H100'de ~10-15 dk.
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL   = "facebook/opt-1.3b"
LAYERS  = [4, 12, 20]                      # erken / orta / gec (Kart 8 ile ayni)
SCALES  = [6_000, 30_000, 100_000, 300_000]  # train token hedefleri
TEST_TOKENS = 30_000
BUDGETS = [0.20, 0.30, 0.40, 0.50, 0.60]
TARGETS = [0.90, 0.95, 0.99]
RANK    = 128
MAX_TOK = 256
SEED    = 0
DEV     = "cuda" if torch.cuda.is_available() else "cpu"   # local CPU'da da calisir
# Sure: H100 ~15 dk | CPU ~3-4 saat (toplama ~2s, egitim+eval ~1-1.5s). RAM ~15GB.


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


def collect(model, tok, texts, need_tokens):
    layers = model.model.decoder.layers
    cap = {L: {"x": [], "m": []} for L in LAYERS}

    def x_hook(L):
        def h(module, inp):
            cap[L]["x"].append(inp[0].detach().to(torch.float16).cpu())
        return h

    def m_hook(L):
        def h(module, inp):
            cap[L]["m"].append((inp[0] > 0).cpu())
        return h

    handles = []
    for L in LAYERS:
        handles.append(layers[L].fc1.register_forward_pre_hook(x_hook(L)))
        handles.append(layers[L].fc2.register_forward_pre_hook(m_hook(L)))

    D, total, t0 = [], 0, time.time()
    with torch.no_grad():
        for i, p in enumerate(texts):
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=MAX_TOK).input_ids.to(DEV)
            model(ids)
            n = ids.shape[1]
            D.append(np.full(n, i, dtype=np.int32))
            total += n
            if (i + 1) % 500 == 0:
                print(f"  topla {i+1} belge, {total} token ({time.time()-t0:.0f}s)")
            if total >= need_tokens:
                break
    for h in handles:
        h.remove()

    out = {}
    for L in LAYERS:
        X = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap[L]["x"]])
        M = torch.cat([t.reshape(-1, t.shape[-1]) for t in cap[L]["m"]])
        cap[L] = None
        out[L] = (X, M)                     # fp16 cpu, bool cpu
    return out, np.concatenate(D)


def train_probe(X_tr, M_tr, epochs=8, bs=8192, lr=1e-3):
    d_in, d_out = X_tr.shape[1], M_tr.shape[1]
    torch.manual_seed(SEED)
    net = nn.Linear(d_in, d_out).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    N = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            xb = X_tr[b].float().to(DEV)
            yb = M_tr[b].float().to(DEV)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
    return net


@torch.no_grad()
def recall_curve(logits, M_te):
    """logits, M_te: GPU float. Doner: recall@m egrisi (d_ffn uzunlugunda)."""
    order = torch.argsort(logits, dim=1, descending=True)
    hits = torch.gather(M_te, 1, order)
    cum = hits.cumsum(1) / (M_te.sum(1, keepdim=True) + 1e-9)
    return cum.mean(0).cpu().numpy()


@torch.no_grad()
def eval_probe(net, X_te, M_te, rank):
    d_out, d_in = net.weight.shape
    Xg = X_te.float().to(DEV)
    Mg = M_te.float().to(DEV)
    res = {}
    for tag, W in (("full", net.weight), ("r%d" % rank, None)):
        if W is None:                       # SVD kesme (deploy adayi)
            U_, S_, Vt_ = torch.linalg.svd(net.weight.float(), full_matrices=False)
            W = (U_[:, :rank] * S_[:rank]) @ Vt_[:rank]
        logits = Xg @ W.T + net.bias
        curve = recall_curve(logits, Mg)
        res[tag] = {
            "budget_recall": {f"{b:.2f}": round(float(curve[int(d_out*b)-1]), 4)
                              for b in BUDGETS},
            "frac_for": {f"{t:.2f}": round(min(
                (int(np.searchsorted(curve, t)) + 1) / d_out, 1.0), 4)
                for t in TARGETS},
        }
    return res


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32,
        device_map=DEV).eval()
    assert hasattr(model.model, "decoder"), "OPT bekleniyor"

    need = max(SCALES) + TEST_TOKENS + 20_000
    texts = get_texts(int(need * 1.3))
    print(f"[i] {len(texts)} belge yuklendi (hedef ~{need} token)")
    data, D = collect(model, tok, texts, need)
    print(f"[i] toplam {len(D)} token toplandi")

    # belge-seviyesi split: test sabit, train havuzu ic ice buyur
    rng = np.random.default_rng(SEED)
    docs = rng.permutation(np.unique(D))
    counts = {d: 0 for d in docs}
    for d in D:
        counts[d] += 1
    test_docs, acc = set(), 0
    for d in docs:
        test_docs.add(d); acc += counts[d]
        if acc >= TEST_TOKENS:
            break
    train_docs = [d for d in docs if d not in test_docs]
    is_test = np.isin(D, list(test_docs))
    te_idx = np.where(is_test)[0]
    # ic ice train indeksleri
    train_pools = {}
    order = []
    for d in train_docs:
        order.extend(np.where(D == d)[0].tolist())
    order = np.array(order)
    for S in SCALES:
        train_pools[S] = order[:S]
    print(f"[i] test={len(te_idx)} token ({len(test_docs)} belge), "
          f"train havuzu={len(order)} token")

    report = {"model": MODEL, "corpus": "wikitext-103", "rank": RANK,
              "test_tokens": int(len(te_idx)), "layers": {}}
    te_t = torch.from_numpy(te_idx)
    for L in LAYERS:
        X, M = data[L]
        X_te, M_te = X[te_t], M[te_t]
        report["layers"][L] = {}
        for S in SCALES:
            tr = torch.from_numpy(train_pools[S])
            if len(tr) < S * 0.9:
                print(f"[uyari] L{L} S={S}: havuz yetersiz ({len(tr)})")
            t1 = time.time()
            net = train_probe(X[tr], M[tr])
            res = eval_probe(net, X_te, M_te, RANK)
            report["layers"][L][S] = res
            f = res["full"]["frac_for"]
            r = res[f"r{RANK}"]["frac_for"]
            print(f"[L{L:>2} N={S:>6}] frac@.90/.95/.99 = "
                  f"{f['0.90']:.2f}/{f['0.95']:.2f}/{f['0.99']:.2f}   "
                  f"r{RANK}: {r['0.90']:.2f}/{r['0.99']:.2f}   "
                  f"rec@0.40={res['full']['budget_recall']['0.40']:.3f}  "
                  f"({time.time()-t1:.0f}s)")
            del net; torch.cuda.empty_cache()

    with open("scaling_probe_results.json", "w") as fj:
        json.dump(report, fj, indent=2)
    print("\n[ok] rapor -> scaling_probe_results.json")
    print("Okuma: N buyudukce frac@0.99 DUSUYOR mu (egri canli) yoksa")
    print("       duzlesiyor mu (kuyruk ogrenilemiyor -> ilkesel STOP)?")
    print("       Deploy karari r%d sutunundan okunur." % RANK)


if __name__ == "__main__":
    main()
