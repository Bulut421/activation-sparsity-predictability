"""
born_eye.py  -  "Dogustan goz": blok-router FFN referans implementasyonu
=========================================================================
Bir FFN, her token icin gizli noronlarin sadece kucuk bir kismini (bir kac
"blok") hesaplar; HANGI bloklar sorusuna ogrenilmis bir router karar verir.
Router modelle BIRLIKTE egitilir -> secim, egitimin ICINDE. "Dogustan goz."

Neden onemli (olculmus, arastirma-olcegi — 17M param, TinyStories):
  Dense bir modele SONRADAN takilan bir seyreklik-predictor'i, butce dogal
  canli orana yaklastikca cokuyor: tahmin hatasi katmanlar boyunca birikiyor.
  Ayni butcede dogustan-goz TAHMIN etmez, KARAR verir -> hata birikmesi yok.

  Ayni model/veri/butce, ayni eval (TinyStories, %12.5 ve %6.25 aktif FFN):
    butce   post-hoc predictor(en iyi)   dogustan-goz   oracle(ulasilamaz)
    %12.5   +32.0% ppl                    +5.2% ppl      +2.1% ppl
    %6.25   +158%  ppl                    +22.4% ppl     +12.4% ppl
  Dogustan-goz, "taklit yukunun" ~%90'ini kaldiriyor; post-hoc'un yuku
  butce daraldikca patliyor. (Detay: SPARSITY_NOTES.md Kart 21-27.)

UYARI: Bu bir ARASTIRMA-OLCEGI referans implementasyon (kucuk model, tek
korpus, iki butce). Uretim kutuphanesi DEGIL; buyuk-olcek dogrulama acik is.

Mimari: decoder-only GPT, dikkat standart (nedensel), FARK sadece FFN'de:
  RoutedFFN = router(softmax top-k blok) + gate-agirlikli blok-FFN.
  Egitimde load-balancing aux kaybi (Switch tarzi) router cokusunu onler.

Kendine yeter (harici bagimlilik yok, torch disinda). Egitim ornegi icin
train_born.py; A vs post-hoc vs born karsilastirmasi icin faz3_compare.py.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Standart nedensel cok-kafali dikkat (flash/SDPA)."""
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.h, self.d = n_head, d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class RoutedFFN(nn.Module):
    """DOGUSTAN GOZ. Gizli boyut (d_ff) G bloga bolunur; her token top-k blok
    secer (aktif oran = k/G). Router logit'leri softmax, secili bloklarin
    renormalize gate'i ile agirliklanir (Switch/Mixtral). Gradyan router'a
    gate uzerinden akar. Egitimde self.aux (load-balancing) kaybi toplanmali.

    NOT (verim): bu referans egitim/olcum icin DENSE hesaplar sonra maskeler
    (ppl dogru butceyi yansitir). Gercek hizlanma seyrek kernel/gather ister
    (bkz. SPARSITY_NOTES Kart 14/17: bitisik-blok CPU'da ~3.9x tavan olculdu).
    """
    def __init__(self, d_model, d_ff, G, k):
        super().__init__()
        assert d_ff % G == 0 and 1 <= k <= G
        self.G, self.k, self.block = G, k, d_ff // G
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.router = nn.Linear(d_model, G)
        self.aux = torch.zeros(())       # son forward'in load-balancing kaybi
        self.load = None                 # blok dispatch fraction (cokus izleme)

    def forward(self, x):                # x: [B, T, d_model]
        probs = F.softmax(self.router(x), dim=-1)          # [B,T,G]
        topv, topi = probs.topk(self.k, dim=-1)            # [B,T,k]
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)  # renorm -> olcek korunur
        gate = torch.zeros_like(probs).scatter(-1, topi, topv)   # [B,T,G]
        mask = torch.zeros_like(probs).scatter(-1, topi, 1.0)
        # Switch load-balancing: G * sum_g f_g * P_g  (uniform'da minimum=1)
        f = mask.mean(dim=(0, 1)); P = probs.mean(dim=(0, 1))
        self.aux = self.G * (f * P).sum()
        self.load = f.detach()
        neuron_gate = gate.repeat_interleave(self.block, dim=-1)  # [B,T,d_ff]
        h = F.relu(self.fc1(x)) * neuron_gate               # sadece secili bloklar
        return self.fc2(h)


class BornBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, G, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = RoutedFFN(d_model, d_ff, G, k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BornGPT(nn.Module):
    """Blok-router FFN'li decoder-only GPT. A ile ayni govde, tek fark FFN.

    aux_loss(): egitimde  total = cross_entropy + alpha * aux_loss()  (~alpha=0.01)
    router_stats(): (entropi[0..1], max_blok_pay) -> cokus izleme.
    """
    def __init__(self, vocab, d_model=384, n_layer=8, n_head=6, block_size=512,
                 d_ff=1536, G=16, k=2):
        super().__init__()
        self.block_size, self.G, self.k = block_size, G, k
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [BornBlock(d_model, n_head, d_ff, G, k) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.wte.weight               # tied embeddings
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
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def aux_loss(self):
        return torch.stack([b.ffn.aux for b in self.blocks]).mean()

    def router_stats(self):
        loads = torch.stack([b.ffn.load for b in self.blocks]).mean(0)
        p = loads / (loads.sum() + 1e-9)
        ent = float(-(p * (p + 1e-9).log()).sum() / math.log(self.G))
        return ent, float(p.max())


if __name__ == "__main__":
    # smoke test: sekil + aktif oran dogrulamasi (egitim degil)
    m = BornGPT(vocab=8192, G=16, k=2)
    npar = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, 8192, (2, 64))
    logits, loss = m(x, x)
    ent, mx = m.router_stats()
    print(f"BornGPT ~{npar/1e6:.1f}M param  aktif={m.k/m.G:.1%}  "
          f"logits {tuple(logits.shape)}  loss {loss.item():.2f}  "
          f"aux {m.aux_loss().item():.3f}  route-entropi {ent:.3f} max-blok {mx:.1%}")
