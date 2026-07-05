"""
collect_sparsity.py  -  PARAZIT veri toplama
=============================================
Amac: her token icin (predictor girdisi x  ->  hangi noronlar sicak) ciftlerini topla.

Adim eslesmesi (onceki konusmadan):
  - Adim 0/1 -> FFN'e giren x  = PREDICTOR GIRDISI
  - Adim 3   -> a = silu(gate)*up = NORON ATESLEMESI = ETIKET kaynagi

Parazit = forward hook. Model calisirken (prompt'lar gecerken) pasif okur.
Etiket BEDAVA: modelin kendi forward'i uretiyor, elle etiketleme yok.

Cikti: her secili katman icin .npz  ->  X [N, d_model], A [N, d_ffn]
       (N = tum prompt'lardaki toplam token sayisi)
"""

import argparse, json, time, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------
# 1) Hangi katmanlardan okuyacagiz
#    Ilk olcum icin 3 katman yeter: erken / orta / gec.
#    Hepsini okumak gereksiz - sinyal var mi sorusuna 3 katman cevap verir.
# -------------------------------------------------------------------
def pick_layers(n_layers):
    return sorted({n_layers // 6, n_layers // 2, (5 * n_layers) // 6})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prompts", required=True, help="jsonl, her satir {'text': ...} ya da {'messages': ...}")
    ap.add_argument("--limit", type=int, default=560, help="kac prompt kullanilsin")
    ap.add_argument("--max-tokens", type=int, default=256, help="prompt basina max token (uzunlari kirp)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="sparsity_data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        device_map=args.device
    ).eval()

    # mimari algilama (oracle_quality.py ile ayni)
    # Qwen/LLaMA: x = mlp girdisi,  a = down_proj girdisi (silu(gate)*up)
    # OPT      : x = fc1 girdisi,  a = fc2 girdisi (relu(fc1(x)))
    if hasattr(model, "model") and hasattr(model.model, "decoder"):      # OPT
        layers = model.model.decoder.layers
        get_mods = lambda l: (l.fc1, l.fc2)
        arch = "OPT (fc1->ReLU->fc2)"
    else:                                                                # Qwen/LLaMA
        layers = model.model.layers
        get_mods = lambda l: (l.mlp, l.mlp.down_proj)
        arch = "Qwen/LLaMA (gate/up/down)"
    sel = pick_layers(len(layers))
    print(f"[i] {len(layers)} katman [{arch}], secilenler: {sel}")

    # -----------------------------------------------------------------
    # 2) PARAZITLERI KUR
    #    x  : mlp'nin girdisi (down_proj oncesi degil, FFN blogunun girdisi)
    #    a  : down_proj'un GIRDISI = silu(gate)*up  (= adim 3, noron atesleme)
    #    Iki ayri hook: biri mlp.__call__ girdisi, biri down_proj girdisi.
    # -----------------------------------------------------------------
    cap = {L: {"x": [], "a": []} for L in sel}

    def make_mlp_pre_hook(L):
        # mlp(x) cagrilmadan x'i yakala  (forward_pre_hook)
        def hook(module, inp):
            cap[L]["x"].append(inp[0].detach().float().cpu())  # [B, T, d_model]
        return hook

    def make_down_pre_hook(L):
        # down_proj(a) cagrilmadan a'yi yakala  -> a = silu(gate)*up
        def hook(module, inp):
            cap[L]["a"].append(inp[0].detach().float().cpu())  # [B, T, d_ffn]
        return hook

    handles = []
    for L in sel:
        m_x, m_a = get_mods(layers[L])
        handles.append(m_x.register_forward_pre_hook(make_mlp_pre_hook(L)))
        handles.append(m_a.register_forward_pre_hook(make_down_pre_hook(L)))

    # -----------------------------------------------------------------
    # 3) PROMPT'LARI YUKLE  (senin Stage A formatina uyumlu)
    # -----------------------------------------------------------------
    prompts = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if "prompt" in o:
                prompts.append(o["prompt"])
            elif "text" in o:
                prompts.append(o["text"])
            elif "messages" in o:
                prompts.append(tok.apply_chat_template(o["messages"], tokenize=False))
            if len(prompts) >= args.limit:
                break
    print(f"[i] {len(prompts)} prompt yuklendi")

    # -----------------------------------------------------------------
    # 4) MODELI CALISTIR  -> parazitler pasif okur
    #    Her prompt ayri forward (basit; batch sonra optimize edilir).
    # -----------------------------------------------------------------
    Xs = {L: [] for L in sel}
    As = {L: [] for L in sel}
    Ps = {L: [] for L in sel}          # her token'in geldigi prompt id'si
    t0 = time.time()
    for i, p in enumerate(prompts):
        for L in sel:
            cap[L]["x"].clear(); cap[L]["a"].clear()
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=args.max_tokens).input_ids.to(args.device)
        with torch.no_grad():
            model(ids)
        # token boyutunu duzlestir: [B,T,d] -> [T,d]
        for L in sel:
            # Qwen mlp girdisi [B,T,d]; OPT fc1 girdisi DUZLESTIRILMIS [B*T,d].
            # Ikisini de [T,d]'ye normalize et (B=1 varsayimi).
            x = torch.cat(cap[L]["x"], 0); x = x.reshape(-1, x.shape[-1])
            a = torch.cat(cap[L]["a"], 0); a = a.reshape(-1, a.shape[-1])
            Xs[L].append(x); As[L].append(a)
            Ps[L].append(torch.full((x.shape[0],), i, dtype=torch.int32))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)}  ({time.time()-t0:.0f}s)")

    for h in handles:
        h.remove()

    # -----------------------------------------------------------------
    # 5) KAYDET  (katman basina ayri dosya)
    # -----------------------------------------------------------------
    for L in sel:
        X = torch.cat(Xs[L], 0).numpy().astype(np.float16)  # [N, d_model]
        A = torch.cat(As[L], 0).numpy().astype(np.float16)  # [N, d_ffn]
        P = torch.cat(Ps[L], 0).numpy()                     # [N] prompt id
        path = os.path.join(args.out, f"layer_{L}.npz")
        np.savez_compressed(path, X=X, A=A, P=P)
        print(f"[ok] L{L}: X{X.shape} A{A.shape} P{P.shape} -> {path}")

    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
