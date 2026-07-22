"""
prefill_sparse_proto.py  -  Kart 16: PREFILL entegrasyon prototipi
===================================================================
Kart 14: T=64'te naive gather bile kazaniyordu -> prefill'de compute-skip
BUGUN mumkun mu, gercek modelde, uctan uca?

Tasarim:
  - Prefill C token'lik parcalarla islenir (KV cache ile) -> her parcada
    katman basina UNION maske: parca icindeki her token'in predictor top-B'si
    union'a girer. Her token kendi maskesini union icinde bulur ->
    kalite Kart 13'ten KOTU OLAMAZ (union sadece noron ekler).
  - fc1/fc2, sekil-esnek gather modulleriyle degistirilir:
      GatherFC1: [T,d] -> [T,k_union]   (W1[idx] ile, predictor dahil)
      GatherFC2: [T,k_union] -> [T,d]   (W2[:,idx] ile)
    Aradaki ReLU/dropout elementwise -> HF katmanina dokunmuyoruz.
  - Union butcesi parca buyudukce siser (maskeler token-basina degisiyor,
    Kart 15) -> C kucuk = dar maske ama cok cagri; C buyuk = genis maske.
    Tatli nokta olculecek sey.

Olculenler: uctan uca prefill tokens/s (dense vs sparse, ayni parcalama),
ppl delta, katman-ortalama union butcesi.

Gereksinim: predictor_weights_wt300k_r512.pt (Kart 13 ckpt, wikitext-egitimli)
ve wikitext_eval.jsonl (Kart 13'ten; yoksa --stream-wikitext ile indirir).

Kosu (CPU):
  python prefill_sparse_proto.py --ckpt predictor_weights_wt300k_r512.pt \
      --prompts wikitext_eval.jsonl --limit 30 --budget 0.30 --chunks 8,32,128
"""

import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# parca basina paylasilan durum: GatherFC1 idx'i yazar, GatherFC2 okur
SHARED = {}


class GatherFC1(nn.Module):
    def __init__(self, fc1, V, U, b, budget, layer_id, stats):
        super().__init__()
        self.W1, self.b1 = fc1.weight, fc1.bias          # [f,d], [f]
        self.V, self.U, self.bp = V, U, b                # predictor (r512)
        self.budget, self.L, self.stats = budget, layer_id, stats
        self.f = self.W1.shape[0]

    def forward(self, x):                                # x: [T, d] (OPT duz)
        k = max(1, int(self.f * self.budget))
        logits = (x @ self.V.T) @ self.U.T + self.bp     # [T, f]
        thr = logits.topk(k, dim=-1).values[..., -1:]
        union = (logits >= thr).any(0)                   # [f] parca union'i
        idx = union.nonzero(as_tuple=True)[0]
        SHARED[self.L] = idx
        self.stats.append(len(idx) / self.f)
        a = x @ self.W1[idx].T + self.b1[idx]            # [T, k_union]
        return a


class GatherFC2(nn.Module):
    def __init__(self, fc2, layer_id):
        super().__init__()
        self.W2, self.b2 = fc2.weight, fc2.bias          # [d,f], [d]
        self.L = layer_id

    def forward(self, a):                                # a: [T, k_union]
        idx = SHARED[self.L]
        return a @ self.W2[:, idx].T + self.b2


def load_prompts(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out.append(o.get("text") or o.get("prompt"))
            if len(out) >= limit:
                break
    return out


@torch.no_grad()
def prefill_ppl(model, tok, texts, chunk, max_tokens, device):
    """Parcali prefill: her belge C token'lik dilimlerle (KV cache) islenir.
    Doner: (toplam nll, toplam tahmin token, toplam sure_s, toplam token)."""
    total_nll, total_pred, total_tok, total_t = 0.0, 0, 0, 0.0
    for txt in texts:
        ids = tok(txt, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(device)[0]
        if len(ids) < 2:
            continue
        logits_all = []
        past = None
        t0 = time.perf_counter()
        for s in range(0, len(ids), chunk):
            part = ids[s:s + chunk].unsqueeze(0)
            out = model(part, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_all.append(out.logits[0])
        total_t += time.perf_counter() - t0
        logits = torch.cat(logits_all, 0)                # [T, V]
        nll = nn.functional.cross_entropy(
            logits[:-1].float(), ids[1:], reduction="sum")
        total_nll += nll.item()
        total_pred += len(ids) - 1
        total_tok += len(ids)
    return total_nll, total_pred, total_t, total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--ckpt", default="predictor_weights_wt300k_r512.pt")
    ap.add_argument("--prompts", default="wikitext_eval.jsonl")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--chunks", default="8,32,128",
                    help="parca boyutlari (union genisligi vs cagri sayisi)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="prefill_proto_report.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if args.device == "cpu" else torch.float16,
        device_map=args.device).eval()
    assert hasattr(model.model, "decoder"), "OPT bekleniyor"
    layers = model.model.decoder.layers

    ck = torch.load(args.ckpt)
    preds = {L: tuple(t.to(args.device).float() for t in vub)
             for L, vub in ck["preds"].items()}
    texts = load_prompts(args.prompts, args.limit)
    print(f"[i] {len(texts)} belge, budget={args.budget}, "
          f"ckpt={args.ckpt} ({len(preds)} katman)")

    # orijinal fc1/fc2'leri sakla (dense olcum + geri takma icin)
    orig = [(l.fc1, l.fc2) for l in layers]
    report = {"budget": args.budget, "n_docs": len(texts), "runs": {}}

    for chunk in [int(c) for c in args.chunks.split(",")]:
        # ---- DENSE (ayni parcalama - adil karsilastirma) ----
        for l, (f1, f2) in zip(layers, orig):
            l.fc1, l.fc2 = f1, f2
        nll_d, np_d, t_d, ntok = prefill_ppl(model, tok, texts, chunk,
                                             args.max_tokens, args.device)
        ppl_d = math.exp(nll_d / np_d)

        # ---- SPARSE (gather fc1/fc2 + union maske) ----
        stats = []
        for L, l in enumerate(layers):
            f1, f2 = orig[L]
            V, U, b = preds[L]
            l.fc1 = GatherFC1(f1, V, U, b, args.budget, L, stats)
            l.fc2 = GatherFC2(f2, L)
        nll_s, np_s, t_s, _ = prefill_ppl(model, tok, texts, chunk,
                                          args.max_tokens, args.device)
        ppl_s = math.exp(nll_s / np_s)

        r = {"dense_tok_s": round(ntok / t_d, 1),
             "sparse_tok_s": round(ntok / t_s, 1),
             "speedup": round(t_d / t_s, 3),
             "ppl_dense": round(ppl_d, 3), "ppl_sparse": round(ppl_s, 3),
             "ppl_delta_pct": round((ppl_s / ppl_d - 1) * 100, 2),
             "mean_union_budget": round(float(np.mean(stats)), 4)}
        report["runs"][chunk] = r
        print(f"[C={chunk:>4}] dense {r['dense_tok_s']:7.1f} tok/s | "
              f"sparse {r['sparse_tok_s']:7.1f} tok/s | "
              f"speedup {r['speedup']:.2f}x | union {r['mean_union_budget']:.1%} | "
              f"ppl {ppl_d:.2f}->{ppl_s:.2f} ({r['ppl_delta_pct']:+.1f}%)")

    # temizle
    for l, (f1, f2) in zip(layers, orig):
        l.fc1, l.fc2 = f1, f2

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("Okuma: speedup>1 + ppl_delta ~Kart 13 bandinda -> prefill entegrasyonu")
    print("bugun gecerli. union butcesi C ile nasil sisiyor -> tatli nokta.")
    print("speedup<1 ise: union cok genis ya da predictor overhead'i baskin;")
    print("C'yi kucult ya da budget'i dusur, tabloyu yeniden oku.")


if __name__ == "__main__":
    main()
