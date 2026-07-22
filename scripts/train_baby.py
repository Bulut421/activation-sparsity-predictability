"""
train_baby.py  -  Faz 1: sifirdan dense ReLU bebek-model (TinyStories)
=======================================================================
Iki amac: (1) pretrain kasi (sifirdan egitim — veri akisi, LR schedule,
loss egrisi, checkpoint). (2) Faz 3'un KONTROL MODELI A (dogustan-goz
deneyinin dense ikizi).

Mimari (bilerek): decoder-only GPT, FFN = fc1 -> ReLU -> fc2 (OPT/Kart 7
ile AYNI; Faz 2-3 zemini — SiLU degil). ~17M param (d=384, 8 kat, 6 head).

Yan olcum (ROADMAP acik sorusu): egitim sonunda dogal sifir orani + katman
deseni — OPT'ye (erken dusuk, gec yuksek) benziyor mu?

Iki seed calistir (--seed 0, --seed 1): Faz 3 "B ~ A" iddiasinin gurultu cubugu.

Gereksinim: torch, numpy, tokenizers  +  prepare_tinystories.py ciktisi.
Cihaz: CUDA varsa otomatik kullanir (RTX 5060 = Blackwell, guncel torch sart);
       yoksa CPU (yavas — token butcesini dusur).
Kosu: python train_baby.py --seed 0
"""

import argparse, math, os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer


# ---------------------------------------------------------------- model
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.d = h, d
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash/causal
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)     # FFN girdisi = fc1 cikisinin ReLU'su
        self.fc2 = nn.Linear(4 * d, d)     # a = relu(fc1(x)) -> fc2  (OPT yapisi)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class BabyGPT(nn.Module):
    def __init__(self, vocab, d=384, n_layer=8, n_head=6, blk=512):
        super().__init__()
        self.blk = blk
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(blk, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight          # tied embeddings
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

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


# ---------------------------------------------------------------- data
def get_batch(data, blk, bs, device):
    ix = torch.randint(len(data) - blk - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i + blk].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + blk].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, blk, bs, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, blk, bs, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def measure_natural_zeros(model, data, blk, bs, device, iters=20):
    """Egitim sonu yan olcum: her katmanda FFN'in relu(fc1(x)) sifir orani.
    ROADMAP acik sorusu — OPT deseni (erken dusuk, gec yuksek) cikiyor mu?"""
    model.eval()
    stats = {i: [] for i in range(len(model.blocks))}
    hooks = []

    def mk(i):
        def hook(mod, inp, out):        # out = fc1(x); relu(out)==0 <=> out<=0
            stats[i].append((out <= 0).float().mean().item())
        return hook
    for i, b in enumerate(model.blocks):
        hooks.append(b.fc1.register_forward_hook(mk(i)))
    for _ in range(iters):
        x, _ = get_batch(data, blk, bs, device)
        model(x)
    for h in hooks:
        h.remove()
    model.train()
    return {i: round(float(np.mean(v)), 4) for i, v in stats.items()}


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinystories_data")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32, help="OOM olursa dusur (16/8)")
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation")
    ap.add_argument("--max-steps", type=int, default=20000, help="~16k token/step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="None=otomatik (cuda>cpu)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[i] cihaz={dev}  seed={args.seed}"
          + ("" if dev == "cuda" else "  [UYARI: CPU — token butcesini dusur]"))

    # tokenizer + veri
    tok = ByteLevelBPETokenizer(os.path.join(args.data, "vocab.json"),
                                os.path.join(args.data, "merges.txt"))
    vocab = tok.get_vocab_size()
    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    print(f"[i] vocab={vocab}  train={len(train):,} tok  val={len(val):,} tok")

    model = BabyGPT(vocab, args.d_model, args.n_layer, args.n_head, args.block).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[i] model ~{nparam/1e6:.1f}M param (d={args.d_model} L={args.n_layer} "
          f"h={args.n_head} blk={args.block})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.wd)
    use_amp = (dev == "cuda")
    amp_dt = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        r = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.1 * args.lr + 0.5 * (0.9 * args.lr) * (1 + math.cos(math.pi * r))

    hist = []
    t0 = time.time()
    model.train()
    for step in range(args.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if step % args.eval_every == 0:
            vl = eval_loss(model, val, args.block, args.batch, dev)
            tl = eval_loss(model, train, args.block, args.batch, dev, iters=20)
            hist.append({"step": step, "train": round(tl, 4), "val": round(vl, 4)})
            print(f"[{step:6d}] train {tl:.3f}  val {vl:.3f}  "
                  f"ppl {math.exp(vl):.2f}  lr {lr_at(step):.1e}  "
                  f"({time.time()-t0:.0f}s)")
            # periyodik checkpoint: Ctrl-C guvenli, en fazla eval-every step kaybi
            if step > 0:
                torch.save({"model": model.state_dict(),
                            "config": {"vocab": vocab, "d": args.d_model,
                                       "L": args.n_layer, "h": args.n_head,
                                       "blk": args.block},
                            "seed": args.seed, "step": step},
                           os.path.join(args.data, f"baby_s{args.seed}.pt"))

        if step % args.sample_every == 0 and step > 0:
            ids = tok.encode("Once upon a time").ids
            x = torch.tensor([ids], device=dev)
            out = model.generate(x, 60)[0].tolist()
            print("   ornek: " + tok.decode(out).replace("\n", " ")[:200])

        if step == args.max_steps:
            break

        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x, y = get_batch(train, args.block, args.batch, dev)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dt):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # yan olcum + kayit
    zeros = measure_natural_zeros(model, val, args.block, args.batch, dev)
    print(f"[i] dogal sifir orani (katman: relu(fc1) sifir%): {zeros}")
    ckpt = os.path.join(args.data, f"baby_s{args.seed}.pt")
    torch.save({"model": model.state_dict(),
                "config": {"vocab": vocab, "d": args.d_model, "L": args.n_layer,
                           "h": args.n_head, "blk": args.block},
                "seed": args.seed}, ckpt)
    rep = os.path.join(args.data, f"baby_report_s{args.seed}.json")
    with open(rep, "w") as f:
        json.dump({"seed": args.seed, "params_M": round(nparam/1e6, 2),
                   "final_val": hist[-1]["val"], "history": hist,
                   "natural_zero_by_layer": zeros}, f, indent=2)
    print(f"[ok] ckpt -> {ckpt}\n[ok] rapor -> {rep}")
    print("Simdi: --seed 1 ile tekrar (gurultu cubugu). Sonra Faz 2 (ProSparse-mini).")


if __name__ == "__main__":
    main()
