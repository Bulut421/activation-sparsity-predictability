"""
train_born_vark.py  -  Faz 4d adim-1: DEGISKEN-k GOZ (adaptif genislik)
=======================================================================
Kart 27.5'in kopru testi + 4d'nin en kucuk hali. Sabit top-k born'da
"zor token daha cok butce aliyor mu?" TANIMSIZDI (her token tam k blok).
Burada tek goz KENDI butcesini ADAPTIF dagitir: kolay token az blok,
zor token cok blok — ORTALAMA hedef butcede sabit.

Mekanizma (top-k YOK, tam turevlenebilir):
  logits = router(x) [.,G] -> g = sigmoid(logits)   # BAGIMSIZ blok kapilari
  h = relu(fc1(x)) * neuron_gate(g)  ->  fc2(h)
  aktif blok sayisi = (g>0.5).sum  -> token basina DEGISIR (emergent)

Iki ceza butceyi sekillendirir:
  butce : (ortalama_g - hedef)^2   -> ortalama harcamayi hedefe (%12.5) SABITLER
                                       (token-basi dagilim SERBEST = adaptasyon)
  ikili : mean(g*(1-g))            -> kapilari 0/1'e iter (gercek SECIM,
                                       uniform 0.125-olcekleme DEGIL)

Olcum (4d-adim1 sorulari):
  1. ortalama_g ~ hedef mi?  (butce tutuyor mu)
  2. aktif-sayi token-arasi DEGISIYOR mu? (std>0 = adaptasyon var)
  3. corr(aktif-sayi, token-kaybi) > 0 mu? (ZOR token daha cok blok aliyor mu)
  4. val ppl, ayni ortalama butcedeki SABIT-k born'u geciyor mu? (adaptasyon FAYDA mi)

Kosu: python train_born_vark.py --G 16 --k 2   (hedef=%12.5; canli mean_g'yi izle)
Kalibrasyon: mean_g hedefe oturmuyorsa --beta arttir; kapilar 0.5'te takiliysa --gamma arttir.
"""

import argparse, math, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from train_baby import CausalSelfAttention, get_batch


class VarRoutedFFN(nn.Module):
    """Degisken-k: bagimsiz sigmoid kapi/blok, aktif sayi token-basi emergent."""
    def __init__(self, d, d_ff, G, target):
        super().__init__()
        self.G, self.bs, self.target = G, d_ff // G, target
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.router = nn.Linear(d, G)
        self.budget = None        # (mean_g - target)^2  bu katman
        self.binar = None         # mean(g*(1-g))        bu katman
        self.mean_gate = None     # skaler: elde edilen yumusak butce
        self.load = None          # [G] blok-basi ortalama kapi (cokus/olu blok izleme)
        self.last_count = None    # [B,T] token-basi sert aktif sayi (g>0.5)
        self.last_gate = None     # [B,T,G] kapi degerleri (adaptasyon teshisi)

    def forward(self, x):                             # x: [B,T,d]
        g = torch.sigmoid(self.router(x))             # [B,T,G] BAGIMSIZ kapilar
        self.mean_gate = g.mean()
        self.budget = (self.mean_gate - self.target) ** 2
        self.binar = (g * (1.0 - g)).mean()
        self.load = g.mean(dim=(0, 1)).detach()       # [G]
        self.last_count = (g > 0.5).float().sum(-1).detach()   # [B,T]
        self.last_gate = g.detach()
        neuron_gate = g.repeat_interleave(self.bs, dim=-1)     # [B,T,d_ff]
        h = F.relu(self.fc1(x)) * neuron_gate         # kapi olcekli bloklar
        return self.fc2(h)


class BornVarBlock(nn.Module):
    def __init__(self, d, h, d_ff, G, target):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = VarRoutedFFN(d, d_ff, G, target)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BornVarGPT(nn.Module):
    def __init__(self, vocab, d=384, n_layer=8, n_head=6, blk=512,
                 d_ff=1536, G=16, target=0.125):
        super().__init__()
        self.blk, self.G, self.target = blk, G, target
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(blk, d)
        self.blocks = nn.ModuleList(
            [BornVarBlock(d, n_head, d_ff, G, target) for _ in range(n_layer)])
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

    def budget_loss(self):
        return torch.stack([b.ffn.budget for b in self.blocks]).mean()

    def binar_loss(self):
        return torch.stack([b.ffn.binar for b in self.blocks]).mean()

    def spend_stats(self):
        """elde edilen butce + adaptasyon skalerleri (izleme)."""
        mg = float(torch.stack([b.ffn.mean_gate for b in self.blocks]).mean())
        # token-basi aktif sayi: katmanlar-arasi ortalama, sonra token istatistigi
        cnt = torch.stack([b.ffn.last_count for b in self.blocks]).mean(0)  # [B,T]
        # kararlilik: 0.1<g<0.9 disindaki kapi orani (ikili basari)
        gates = torch.stack([b.ffn.last_gate for b in self.blocks])          # [L,B,T,G]
        decided = ((gates < 0.1) | (gates > 0.9)).float().mean().item()
        # olu/hep-acik blok izleme: blok-basi yuk entropisi
        loads = torch.stack([b.ffn.load for b in self.blocks]).mean(0)       # [G]
        p = loads / (loads.sum() + 1e-9)
        ent = float(-(p * (p + 1e-9).log()).sum() / math.log(self.G))
        return {"mean_gate": mg, "count_mean": float(cnt.mean()),
                "count_std": float(cnt.std()), "decided": decided,
                "load_entropy": ent, "max_block": float(p.max())}

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


@torch.no_grad()
def measure_adaptation(model, data, blk, bs, dev, iters=40):
    """4d-adim1 cekirdegi: ZOR token daha cok blok mu aliyor?
    Token-basi (aktif-sayi) vs (kayip) korelasyonu + kolay/zor tersil karsilastirmasi."""
    model.eval()
    counts, losses = [], []
    for _ in range(iters):
        x, y = get_batch(data, blk, bs, dev)
        logits, _ = model(x, y)
        # token-basi CE (indirgemesiz)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                             reduction="none")                 # [B*T]
        # token-basi aktif sayi: katmanlar-arasi ortalama
        cnt = torch.stack([b.ffn.last_count for b in model.blocks]).mean(0)  # [B,T]
        counts.append(cnt.reshape(-1).cpu())
        losses.append(ce.cpu())
    model.train()
    c = torch.cat(counts).numpy()
    l = torch.cat(losses).numpy()
    # Pearson korelasyon
    corr = float(np.corrcoef(c, l)[0, 1])
    # kolay (alt %33 kayip) vs zor (ust %33 kayip) token'larin ortalama blok sayisi
    lo, hi = np.quantile(l, 1/3), np.quantile(l, 2/3)
    easy_c = float(c[l <= lo].mean())
    hard_c = float(c[l >= hi].mean())
    return {"corr_count_loss": round(corr, 4),
            "easy_count": round(easy_c, 3), "hard_count": round(hard_c, 3),
            "hard_over_easy": round(hard_c / (easy_c + 1e-9), 3),
            "count_mean": round(float(c.mean()), 3),
            "count_std": round(float(c.std()), 3),
            "n_tokens": int(len(c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--G", type=int, default=16, help="blok sayisi")
    ap.add_argument("--k", type=int, default=2, help="ORTALAMA aktif blok -> hedef=k/G")
    ap.add_argument("--target", type=float, default=None, help="hedef butce (yoksa k/G)")
    ap.add_argument("--beta", type=float, default=30.0, help="butce cezasi (mean_g'yi hedefe sabitler)")
    ap.add_argument("--gamma", type=float, default=0.2, help="ikili cezasi (kapilari 0/1'e iter; "
                    "dusukse UNIFORM-CHEAT: model tum kapilari 0.125'e koyar = gizli-dense)")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    target = args.target if args.target is not None else args.k / args.G
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tok = ByteLevelBPETokenizer(os.path.join(args.data, "vocab.json"),
                                os.path.join(args.data, "merges.txt"))
    vocab = tok.get_vocab_size()
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    model = BornVarGPT(vocab, args.d_model, args.n_layer, args.n_head, args.block,
                       args.d_ff, args.G, target).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[i] cihaz={dev} seed={args.seed}  BornVarGPT ~{npar/1e6:.1f}M  "
          f"G={args.G} hedef-butce {target:.1%}  beta={args.beta} gamma={args.gamma}")
    print(f"[i] DEGISKEN-k: aktif blok token-basi degisir; ortalama -> hedef")
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
                s = model.spend_stats()
                hist.append({"step": step, "ppl": round(vl, 3), **{k: round(v, 4)
                             for k, v in s.items()}})
                print(f"[{step:6d}] ppl {vl:.3f}  mean_g {s['mean_gate']:.3f}"
                      f"(hed {target:.3f})  sayi {s['count_mean']:.2f}±{s['count_std']:.2f}"
                      f"  ikili {s['decided']:.0%}  yuk-ent {s['load_entropy']:.2f}"
                      f"  ({time.time()-t0:.0f}s)")
                if step > 0:
                    torch.save({"model": model.state_dict(),
                                "config": {"vocab": vocab, "d": args.d_model,
                                           "L": args.n_layer, "h": args.n_head,
                                           "blk": args.block, "d_ff": args.d_ff,
                                           "G": args.G, "target": target},
                                "seed": args.seed, "step": step},
                               os.path.join(args.data, f"bornvar_G{args.G}t{target:.3f}_s{args.seed}.pt"))

            if step % args.sample_every == 0 and step > 0:
                ids = tok.encode("Once upon a time").ids
                out = model.generate(torch.tensor([ids], device=dev), 55)[0].tolist()
                print("   ornek: " + tok.decode(out).replace("\n", " ")[:180])

            if step == args.max_steps:
                break

            x, y = get_batch(train, args.block, args.batch, dev)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dt):
                    _, ce = model(x, y)
                    loss = ce + args.beta * model.budget_loss() + args.gamma * model.binar_loss()
            else:
                _, ce = model(x, y)
                loss = ce + args.beta * model.budget_loss() + args.gamma * model.binar_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    except KeyboardInterrupt:
        print("\n[i] durduruldu — son checkpoint korunuyor")

    ppl = eval_ppl(model, val, args.block, args.batch, dev, iters=120)
    s = model.spend_stats()
    adapt = measure_adaptation(model, val, args.block, args.batch, dev)
    ids = tok.encode("Once upon a time").ids
    out = model.generate(torch.tensor([ids], device=dev), 55)[0].tolist()
    print(f"\n[SONUC] val ppl {ppl:.3f}  mean_g {s['mean_gate']:.3f} (hedef {target:.3f})"
          f"  ikili {s['decided']:.0%}")
    print(f"[ADAPTASYON] aktif-sayi {adapt['count_mean']}±{adapt['count_std']}  "
          f"corr(sayi,kayip) {adapt['corr_count_loss']:+.3f}")
    print(f"   kolay token -> {adapt['easy_count']} blok   |   "
          f"zor token -> {adapt['hard_count']} blok   (zor/kolay {adapt['hard_over_easy']}x)")
    print("   ornek: " + tok.decode(out).replace("\n", " ")[:200])
    rep = os.path.join(args.data, f"bornvar_report_G{args.G}t{target:.3f}_s{args.seed}.json")
    with open(rep, "w") as f:
        json.dump({"G": args.G, "target": target, "beta": args.beta, "gamma": args.gamma,
                   "params_M": round(npar/1e6, 2), "final_ppl": round(ppl, 3),
                   "spend": s, "adaptation": adapt, "history": hist}, f, indent=2)
    print(f"[ok] ckpt -> bornvar_G{args.G}t{target:.3f}_s{args.seed}.pt\n[ok] rapor -> {rep}")
    print("\nOKUMA (4d-adim1):")
    print(" 1. mean_g ~ hedef      -> butce tutuyor (degilse --beta ayarla)")
    print(" 2. ikili yuksek (>%80) -> gercek SECIM (dusukse --gamma arttir, uniform-olcekleme tuzagi)")
    print(" 3. corr>0 & zor/kolay>1 -> ZOR token daha cok blok = ADAPTIF BUTCE calisiyor")
    print(" 4. ppl <= sabit-k born  -> adaptasyon FAYDALI -> iki-goz (4d tam) denemeye deger")
    print("    ppl > sabit-k born   -> adaptif dagitim bedava degil -> koprude dur, raporla")


if __name__ == "__main__":
    main()
