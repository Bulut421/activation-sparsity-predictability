"""wikitext_eval.jsonl'i yeniden uret (colab_train_all_predictors.py ile
BIREBIR ayni mantik + seed -> ayni 200 held-out belge).
Gereksinim: pip install datasets
"""
import json
import numpy as np

TRAIN_TOKENS = 300_000
EVAL_DOCS = 200
MAX_TOK = 256
SEED = 0


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


texts = get_texts(int((TRAIN_TOKENS + 60_000) * 1.3))
perm = np.random.default_rng(SEED).permutation(len(texts))
eval_texts = [texts[i] for i in perm[:EVAL_DOCS]]
with open("wikitext_eval.jsonl", "w", encoding="utf-8") as f:
    for t in eval_texts:
        f.write(json.dumps({"text": t}) + "\n")
print(f"[ok] wikitext_eval.jsonl <- {len(eval_texts)} belge "
      f"(toplam havuz {len(texts)}, seed={SEED})")
