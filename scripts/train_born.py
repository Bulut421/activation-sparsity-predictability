"""
train_born.py  -  Faz 3: DOGUSTAN-GOZ B (blok-router FFN, MoE-tarzi)
=====================================================================
Tezin testi. B = A ile AYNI mimari/veri/butce, TEK fark: FFN'de ogrenilmis
blok-router. Her token, 1536 gizli noronu G bloga bolunmusken sadece k blok
(=%k/G aktif) secer; router modelle BIRLIKTE egitilir. "Dogustan goz" —
maske egitim dongusunun icinde, taklit edilecek oracle YOK.

Router (Switch/Mixtral tarzi):
  logits = router(x) [.,G] -> softmax -> top-k blok -> gate = renorm(top-k prob)
  h = relu(fc1(x)) * neuron_gate(blok->noron)  ->  fc2(h)
  gradyan router'a gate uzerinden akar. Load-balancing aux (cokusu onler, Kart 2).

Egitim (research): DENSE hesap + maske (ppl dogru butceyi yansitir; verim
kernel isi, Kart 17). Sifirdan, A ile ayni 20k step / ayni veri.

Olcum: val ppl (A baseline'i ile karsilastir) + router COKUS izleme
(blok kullanim entropisi, max pay — Kart 2 patolojisi).

Kosu: python train_born.py --G 16 --k 2   (=%12.5 aktif; agresif icin --k 1)
"""

import argparse, math, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from train_baby import CausalSelfAttention, get_batch


class RoutedFFN(nn.Module):
    def __init__(self, d, d_ff, G, k):
        super().__init__()
        self.G, self.k, self.bs = G, k, d_ff // G
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.router = nn.Linear(d, G)
        self.aux = None
        self.load = None                              # blok dispatch frac (cokus izleme)
        self.override = None                          # Kart 28 ablasyon: None|random|fixed

    def forward(self, x):                             # x: [B,T,d]
        logits = self.router(x)                       # [B,T,G]
        probs = F.softmax(logits, dim=-1)
        if self.override == "random":                 # rastgele k blok (router yoksay)
            topi = torch.rand_like(probs).topk(self.k, dim=-1).indices
        elif self.override == "fixed":                # sabit ilk-k blok (girdi-bagimsiz)
            B, T, _ = probs.shape
            topi = torch.arange(self.k, device=x.device).view(1, 1, -1).expand(B, T, -1)
        else:                                         # ogrenilmis (varsayilan, birebir ayni)
            topi = probs.topk(self.k, dim=-1).indices
        topv = torch.gather(probs, -1, topi)          # [B,T,k]
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)   # renorm -> olcek korunur
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)   # [B,T,G]
        mask = torch.zeros_like(probs).scatter(-1, topi, 1.0)
        # Switch load-balancing: G * sum_g f_g * P_g  (uniform'da =1, minimum)
        f = mask.mean(dim=(0, 1))                     # dispatch fraction [G]
        P = probs.mean(dim=(0, 1))                    # mean gate [G]
        self.aux = self.G * (f * P).sum()
        self.load = f.detach()
        self.last_mask = mask.detach()                # Kart 28: kosullu cesitlilik teshisi
        neuron_gate = gate.repeat_interleave(self.bs, dim=-1)    # [B,T,d_ff]
        h = F.relu(self.fc1(x)) * neuron_gate         # sadece secili bloklar canli
        return self.fc2(h)


class BornBlock(nn.Module):
    def __init__(self, d, h, d_ff, G, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = RoutedFFN(d, d_ff, G, k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BornGPT(nn.Module):
    def __init__(self, vocab, d=384, n_layer=8, n_head=6, blk=512,
                 d_ff=1536, G=16, k=2):
        super().__init__()
        self.blk, self.G, self.k = blk, G, k
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(blk, d)
        self.blocks = nn.ModuleList(
            [BornBlock(d, n_head, d_ff, G, k) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def aux_loss(self):
        return torch.stack([b.ffn.aux for b in self.blocks]).mean()

    def router_stats(self):
        """cokus izleme: bloklar-arasi kullanim entropisi (1=uniform, 0=cokus)
        + en cok kullanilan blogun pay orani."""
        loads = torch.stack([b.ffn.load for b in self.blocks]).mean(0)  # [G]
        p = loads / (loads.sum() + 1e-9)
        ent = -(p * (p + 1e-9).log()).sum() / math.log(self.G)
        return float(ent), float(p.max())

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            ctx = idx[:, -self.blk:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            p = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx


@torch.no_grad()
def eval_ppl(model, data, blk, bs, dev, iters=60):
    model.eval()
    ls = []
    for _ in range(iters):
        x, y = get_batch(data, blk, bs, dev)
        _, loss = model(x, y)
        ls.append(loss.item())
    model.train()
    return math.exp(float(np.mean(ls)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--G", type=int, default=16, help="blok sayisi")
    ap.add_argument("--k", type=int, default=2, help="aktif blok (aktif%=k/G)")
    ap.add_argument("--alpha", type=float, default=0.01, help="load-balancing agirligi")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--router", choices=["learned", "random", "fixed"], default="learned",
                    help="Kart 28b ablasyon: random=donuk-random router (girdi-bagimli "
                         "ama ogrenilmemis), fixed=sabit ilk-k blok (girdi-bagimsiz)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tok = ByteLevelBPETokenizer(os.path.join(args.data, "vocab.json"),
                                os.path.join(args.data, "merges.txt"))
    vocab = tok.get_vocab_size()
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    model = BornGPT(vocab, args.d_model, args.n_layer, args.n_head, args.block,
                    args.d_ff, args.G, args.k).to(dev)
    # Kart 28b ablasyon kollari (sifirdan egitim, tek fark router):
    if args.router == "random":               # donuk-random router: girdi-bagimli ama ogrenilmemis
        for b in model.blocks:
            b.ffn.router.weight.requires_grad_(False)
            b.ffn.router.bias.requires_grad_(False)
    elif args.router == "fixed":              # sabit ilk-k blok: girdi-bagimsiz
        for b in model.blocks:
            b.ffn.override = "fixed"
    npar = sum(p.numel() for p in model.parameters())
    active = args.k / args.G
    rtag = "" if args.router == "learned" else "_" + args.router   # dosya adi (overwrite onleme)
    print(f"[i] cihaz={dev} seed={args.seed}  BornGPT ~{npar/1e6:.1f}M  "
          f"G={args.G} k={args.k} -> aktif {active:.1%}  router={args.router}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    use_amp = (dev == "cuda")
    amp_dt = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        r = (s - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.1 * args.lr + 0.5 * 0.9 * args.lr * (1 + math.cos(math.pi * r))

    hist, t0 = [], time.time()
    model.train()
    try:
        for step in range(args.max_steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            if step % args.eval_every == 0:
                vl = eval_ppl(model, val, args.block, args.batch, dev)
                ent, mx = model.router_stats()
                hist.append({"step": step, "ppl": round(vl, 3),
                             "route_entropy": round(ent, 3), "max_block": round(mx, 3)})
                print(f"[{step:6d}] val ppl {vl:.3f}  route-entropi {ent:.3f} "
                      f"(1=uniform)  max-blok {mx:.1%}  lr {lr_at(step):.1e}  "
                      f"({time.time()-t0:.0f}s)")
                if step > 0:
                    torch.save({"model": model.state_dict(),
                                "config": {"vocab": vocab, "d": args.d_model,
                                           "L": args.n_layer, "h": args.n_head,
                                           "blk": args.block, "d_ff": args.d_ff,
                                           "G": args.G, "k": args.k},
                                "seed": args.seed, "step": step, "router": args.router},
                               os.path.join(args.data, f"born_G{args.G}k{args.k}_s{args.seed}{rtag}.pt"))

            if step % args.sample_every == 0 and step > 0:
                ids = tok.encode("Once upon a time").ids
                out = model.generate(torch.tensor([ids], device=dev), 55)[0].tolist()
                print("   ornek: " + tok.decode(out).replace("\n", " ")[:190])

            if step == args.max_steps:
                break

            x, y = get_batch(train, args.block, args.batch, dev)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dt):
                    _, ce = model(x, y)
                    loss = ce + args.alpha * model.aux_loss()
            else:
                _, ce = model(x, y)
                loss = ce + args.alpha * model.aux_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    except KeyboardInterrupt:
        print("\n[i] durduruldu — son checkpoint korunuyor")

    ppl = eval_ppl(model, val, args.block, args.batch, dev, iters=120)
    ent, mx = model.router_stats()
    ids = tok.encode("Once upon a time").ids
    out = model.generate(torch.tensor([ids], device=dev), 55)[0].tolist()
    print(f"\n[SONUC] val ppl {ppl:.3f}  aktif {active:.1%}  "
          f"route-entropi {ent:.3f}  max-blok {mx:.1%}")
    print("   ornek: " + tok.decode(out).replace("\n", " ")[:200])
    rep = os.path.join(args.data, f"born_report_G{args.G}k{args.k}_s{args.seed}{rtag}.json")
    with open(rep, "w") as f:
        json.dump({"G": args.G, "k": args.k, "active_frac": active, "router": args.router,
                   "params_M": round(npar/1e6, 2), "final_ppl": round(ppl, 3),
                   "route_entropy": round(ent, 3), "max_block": round(mx, 3),
                   "history": hist}, f, indent=2)
    print(f"[ok] ckpt -> born_G{args.G}k{args.k}_s{args.seed}{rtag}.pt\n[ok] rapor -> {rep}")
    print("Karsilastir: A oracle @B=aktif (oracle_baby_baby_s0.json) + A baseline 4.48.")
    print("Tez: B, A'nin bu butcedeki oracle/predictor'ini geciyorsa dogustan-goz kazandi.")


if __name__ == "__main__":
    main()
