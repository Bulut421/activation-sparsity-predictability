"""
oracle_quality.py  -  Oracle skip: butce vs perplexity
=======================================================
Soru: MUKEMMEL tahminle bile, FFN noronlarinin sadece top-B%'sini
tutarsak (gerisi sifir) model kalitesi dayaniyor mu?

- Oracle = maske GERCEK aktivasyondan anlik hesaplanir (predictor yok).
- TUM katmanlara uygulanir (gercek skip senaryosu).
- Cikti: budget -> perplexity tablosu.

Okuma:
  ppl(B=0.40) ~ ppl(baseline)  ->  kapi acik, predictor'a deger
  ppl erken patliyorsa         ->  SiLU yamaci kaliteyi tasimiyor -> ReLUfication hatti

Kart 6 adayi. CPU'da calisir (~30-45 dk, --limit 100 ile).
"""

import argparse, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# global butce; hook'lar bunu okur. None = maske yok (baseline)
BUDGET = {"frac": None}
# baseline gecisinde dogal seyreklik olcumu (ReLU modelde kritik sayi)
STATS = {"zero_sum": 0.0, "n": 0}


def make_oracle_hook():
    """FFN son projeksiyonunun girdisini (a) anlik top-k maskele.
    Qwen/LLaMA: a = silu(gate)*up   |   OPT: a = relu(fc1(x))"""
    def hook(module, inp):
        frac = BUDGET["frac"]
        if frac is None:                      # baseline: dokunma, sadece olc
            STATS["zero_sum"] += (inp[0] == 0).float().mean().item()
            STATS["n"] += 1
            return None
        a = inp[0]                            # [B, T, d_ffn]
        k = max(1, int(a.shape[-1] * frac))
        # her token icin |a| top-k disini sifirla
        thresh = a.abs().topk(k, dim=-1).values[..., -1:]  # [B,T,1]
        mask = (a.abs() >= thresh)
        return (a * mask,)
    return hook


@torch.no_grad()
def perplexity(model, tok, texts, device, max_tokens):
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1                  # tahmin edilen token sayisi
        total_nll += out.loss.item() * n
        total_tok += n
    return math.exp(total_nll / total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--limit", type=int, default=100, help="ppl icin prompt sayisi")
    ap.add_argument("--skip", type=int, default=0, help="dosyanin basindan atla (farkli set icin)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--budgets", default="0.5,0.4,0.3,0.2",
                    help="virgullu butce listesi (tutulan noron orani)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="oracle_quality_report.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if args.device == "cpu" else torch.float16,
        device_map=args.device
    ).eval()

    # TUM katmanlara oracle hook tak (mimari algilama)
    # Qwen/LLaMA: layer.mlp.down_proj girdisi = silu(gate)*up
    # OPT      : layer.fc2 girdisi           = relu(fc1(x))  (dogustan ReLU)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):      # OPT
        mods = [l.fc2 for l in model.model.decoder.layers]
        arch = "OPT (fc1->ReLU->fc2)"
    else:                                                                # Qwen/LLaMA
        mods = [l.mlp.down_proj for l in model.model.layers]
        arch = "Qwen/LLaMA (gate/up/down)"
    handles = [m.register_forward_pre_hook(make_oracle_hook()) for m in mods]
    print(f"[i] {len(handles)} katmana oracle hook takildi  [{arch}]")

    # prompt'lari yukle
    texts = []
    with open(args.prompts, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < args.skip:
                continue
            o = json.loads(line)
            for key in ("prompt", "text"):
                if key in o:
                    texts.append(o[key]); break
            if len(texts) >= args.limit:
                break
    print(f"[i] {len(texts)} prompt (skip={args.skip})")

    budgets = [None] + [float(b) for b in args.budgets.split(",")]
    report = {}
    for b in budgets:
        BUDGET["frac"] = b
        t0 = time.time()
        ppl = perplexity(model, tok, texts, args.device, args.max_tokens)
        name = "baseline" if b is None else f"{b:.2f}"
        report[name] = round(ppl, 4)
        base = report.get("baseline", ppl)
        delta = (ppl / base - 1) * 100
        print(f"[B={name:>8}]  ppl={ppl:8.3f}   delta={delta:+6.1f}%   ({time.time()-t0:.0f}s)")
        if b is None and STATS["n"]:
            zf = STATS["zero_sum"] / STATS["n"]
            report["natural_zero_frac"] = round(zf, 4)
            print(f"[i] dogal sifir orani (a==0): {zf:.1%}  "
                  f"(ReLU modelde yuksek beklenir; SiLU'da ~0)")

    for h in handles:
        h.remove()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] rapor -> {args.out}")
    print("Okuma: delta <%2-3 -> butce dayaniyor. Erken patlama -> ReLUfication hatti.")


if __name__ == "__main__":
    main()