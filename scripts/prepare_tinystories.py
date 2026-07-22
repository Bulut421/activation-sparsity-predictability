"""
prepare_tinystories.py  -  Faz 1 hazirlik: 8k BPE tokenizer + tokenlenmis .bin
================================================================================
Sifirdan pretrain'in ilk adimi (nanoGPT-tarzi). TinyStories'i indirir,
uzerine kucuk BPE tokenizer egitir, tum korpusu tokenler ve train/val.bin
(uint16) yazar. Bir kere calisir; train_baby.py bunlari memmap'ler.

Cikti (tinystories_data/):
  vocab.json, merges.txt   - tokenizer (train_baby uretim/decode icin yukler)
  train.bin, val.bin       - uint16 token akisi

Gereksinim: pip install datasets tokenizers numpy
Sure: indirme ~2GB + tokenleme ~birkac dk. Bellek: bounded (~1M id tampon).
"""

import os
import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

VOCAB = 8192
OUT = "tinystories_data"
EOT = "<|endoftext|>"


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[i] TinyStories indiriliyor / cache'ten aciliyor...")
    ds = load_dataset("roneneldan/TinyStories")

    # --- BPE tokenizer egit (byte-level -> OOV yok) ---
    vj, mf = os.path.join(OUT, "vocab.json"), os.path.join(OUT, "merges.txt")
    if os.path.exists(vj) and os.path.exists(mf):
        print("[i] tokenizer zaten var, egitim atlandi")
    else:
        print(f"[i] {VOCAB}-vocab BPE egitiliyor (train uzerinde)...")
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(
            (ex["text"] for ex in ds["train"]),
            vocab_size=VOCAB, min_frequency=2, special_tokens=[EOT])
        tok.save_model(OUT)
        print(f"[ok] tokenizer -> {OUT}/vocab.json, merges.txt")

    tok = ByteLevelBPETokenizer(vj, mf)
    eot_id = tok.token_to_id(EOT)

    # --- tokenle + .bin yaz (bounded bellek: 1M id tampon, dosyaya append) ---
    for split, name in (("train", "train"), ("validation", "val")):
        path = os.path.join(OUT, f"{name}.bin")
        if os.path.exists(path):
            print(f"[i] {name}.bin zaten var, atlandi")
            continue
        BATCH = 20_000                         # encode_batch = paralel (Rust), hizli
        total, done, n_ex = 0, 0, len(ds[split])
        batch = []
        with open(path, "wb") as f:
            def flush_batch():
                nonlocal total
                encs = tok.encode_batch(batch)
                ids = []
                for e in encs:
                    ids.extend(e.ids); ids.append(eot_id)
                np.array(ids, dtype=np.uint16).tofile(f); f.flush()
                total += len(ids)
            for ex in ds[split]:
                batch.append(ex["text"])
                if len(batch) >= BATCH:
                    flush_batch(); done += len(batch); batch = []
                    print(f"  {name}: {done:,}/{n_ex:,} hikaye, {total:,} token")
            if batch:
                flush_batch(); done += len(batch)
        print(f"[ok] {name}.bin <- {total:,} token")

    print("\n[done] Simdi: python train_baby.py --seed 0  (ve --seed 1)")


if __name__ == "__main__":
    main()
