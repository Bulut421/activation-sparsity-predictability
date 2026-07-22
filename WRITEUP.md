# Born Sparse: a model that *decides* what not to compute beats one that *predicts* it

*A three-week, measurement-first investigation into activation sparsity — from "can we predict which FFN neurons to skip?" to "what if the model is trained to decide instead?" Everything here is reproducible at hobby scale (a single RTX 5060 + a few Colab hours). Scripts and per-experiment notes are in this repo.*

---

## TL;DR

Large language models already waste most of their feed-forward compute: with a ReLU activation, ~90–96% of FFN neurons output exactly zero on any given token. The dream (PowerInfer, DejaVu) is to **predict** that sparsity cheaply and skip the dead neurons. It works — but only under two conditions, and it hits a wall:

1. **The activation function is a hard boundary.** On SiLU/SwiGLU models (Qwen2.5-3B) even a *perfect oracle* mask wrecks quality below a 50% budget. On born-ReLU models (OPT-1.3b) the oracle is free down to ~10%. No predictor can beat its oracle ceiling — check the ceiling first.
2. **"Unpredictable" often just means "data-starved."** With 6k training tokens every deployment variant blew up (+400–2000% perplexity). At 300k–1M tokens the same pipeline hit +1–3%. The scaling curve never plateaued.

But the deeper result is about **who does the selecting.** A predictor bolted onto a trained dense model has to *imitate* an oracle it can never see, and its errors compound across layers. So we built the alternative: a small model whose FFN has a **learned block router** — it *decides* which neurons fire, trained end-to-end. At matched budget and matched training, on the same eval:

| FFN budget | post-hoc predictor (best case) | **born-eye (learned router)** | oracle (unreachable) |
|---|---|---|---|
| 12.5% active | +32.0% ppl | **+5.2% ppl** | +2.1% ppl |
| 6.25% active | +158% ppl | **+22.4% ppl** | +12.4% ppl |

The born-eye removes ~90% of what we call the **imitation burden** (predictor cost minus oracle cost), *consistently* — while the post-hoc predictor's burden explodes as the budget tightens. Being born knowing what to skip beats learning to guess it.

**Caveat up front:** this is validated at *research scale* — a 17.5M-parameter model on TinyStories, at two budgets. The mechanism is clean and the trend is consistent, but large-scale confirmation is open work.

---

## The question

Frame inference as an economy of attention. The eye doesn't send the whole visual field to the brain — the retina takes in ~10 Gbit/s, the optic nerve carries ~10 Mbit/s, conscious perception gets ~40 bit/s. Vision is an aggressive, *learned* selector, and it's a big reason the brain runs on ~20 watts. "Knowing where to look" is a million times cheaper than "seeing everything."

An LLM's FFN is the same opportunity. For each token, which hidden neurons actually matter? If we knew *before* computing them, we could skip the rest — a compute win, not a memory win. This is the line **beyond MoE**: instead of routing to a few big experts, select at neuron granularity.

The whole project is a chain of small, falsifiable measurements ("cards"). Each one asks a single question, reports one headline number, and tries to *break* any positive result before believing it. Here's the arc.

## Part 1 — How far does *predicting* sparsity go?

**Domain identity knows nothing; token activation knows a lot.** Conditioning a sparsity mask on the prompt's domain barely helped (+0.037 coverage lift). But a linear probe from the FFN *input* `x` to the hot-neuron mask lifted +0.12–0.18 over a static baseline, and survived a prompt-level held-out split — real generalization, not leakage. The signal lives in the per-token activation, not the topic.

**But SiLU won't carry a skip.** On Qwen2.5-3B (SiLU/SwiGLU), we masked FFN neurons to the top-B% by magnitude across all layers — an *oracle*, using the true activations — and measured perplexity:

| budget | Qwen2.5-3B (SiLU) | ReluLLaMA-7B (ReLUfied) | OPT-1.3b (born ReLU) |
|---|---|---|---|
| natural zero rate | ~0% | 67% | 96.1% |
| B=0.50 | +2.4% | +0.0% | −0.0% |
| B=0.30 | +20.7% | +0.1% | −0.0% |
| B=0.20 | +64.8% | +0.8% | −0.0% |
| B=0.05 | — | +22.1% | +1.1% |

SiLU has a soft tail: nothing is exactly zero, so every neuron you drop costs something. Born-ReLU has a hard floor: 96% of neurons are *already* zero, so skipping them is free down to a ~10% budget. ReLUfied models (a SiLU model fine-tuned to ReLU) sit in between — 67% natural zeros, a plateau that's real but shorter. **The natural-zero rate is monotone in how baked-in the sparsity is: born (96%) > retrofitted (67%) > never (0%).** That ordering is the whole thesis in one number, and it's the first hint that *when* the selectivity is learned matters more than *how* well it's predicted.

**The negatives were data starvation.** On OPT, per-layer recall looked fine but every in-loop deployment variant at 6k training tokens exploded (+400–2000% ppl). Independent error would predict ~+24% across 24 layers; we saw 50× worse — errors *compound superlinearly* because each masked layer shifts the input distribution the next predictor was trained on. Progressive (error-aware) training halved it; nothing fixed it — until we drew the scaling curve. The budget needed for 99% recall fell from 0.95 (at 6k tokens) to **0.04–0.12 (at 1M tokens)**, no plateau in sight. With enough data and rank-256+, a low-rank linear predictor delivers a measured **44–54% net FFN compute reduction on OPT for +1.1–2.9% perplexity**, and a copy-free gather-matvec kernel turns that into a real **~1.8× decode speedup** on CPU (masking ≠ speedup; you have to actually *not read* the skipped rows).

So post-hoc prediction is real and useful. But notice where it fails: **at aggressive budgets, near the natural live rate, where you'd most want to skip.** That failure has a name.

## Part 2 — The imitation burden

A post-hoc predictor is a *guesser*. There exists a "true" set of neurons the trained model will fire, and the predictor must reproduce it from `x`. Every neuron it misses is an error, because it didn't define the truth — the frozen weights did. When the budget is near the live rate, there's no slack: the predictor has to nail almost every live neuron, imperfectly, across every layer, and the misses compound.

The oracle — perfect post-hoc selection — is the ceiling no predictor can pass. So the honest measure of the guessing tax is **predictor cost minus oracle cost**. That's the imitation burden. Part 3 measures it directly, and shows how to remove it.

## Part 3 — Born vs bolted-on

To compare fairly we needed a controlled setting, so we went small and clean: train models from scratch on TinyStories.

**The control model (A).** A 17.5M-parameter ReLU GPT (8 layers, d=384), trained from scratch. It reaches coherent TinyStories generation (val ppl ~4.48). The free finding: with *zero* sparsity regularization, standard training produced **~89% natural activation sparsity**, reproducible across seeds. ReLU FFNs sparsify themselves — born selectivity, at from-scratch tiny scale.

**The half-measure (ProSparse-mini).** Fine-tuning A with a progressive L1 penalty on activations pushed sparsity from 89% to **95.6% at +0.1% ppl** — essentially free, given enough steps. Interesting on its own: *pushing* sparsity post-hoc is cheap in quality. So the born-vs-bolted-on difference isn't about the sparsity level — it's about who selects.

**The born-eye (B).** Same architecture, data, and training budget as A, with one change: each FFN's 1536 hidden neurons are split into 16 blocks, and a learned router picks the top-k blocks per token (softmax gating, renormalized, Switch-style load-balancing aux loss). The selection is trained *in the loop*. At k=2 that's 12.5% active FFN; at k=1, 6.25%.

It trained cleanly — **the router never collapsed** (usage entropy 0.999–1.000, perfectly balanced blocks) even at k=1, and generation stayed coherent. Then the decisive comparison, all five arms on the *same* eval batches at the *same* budget:

**At 12.5% active:**

| arm | ppl | Δ vs dense |
|---|---|---|
| A dense (100% FFN) | 4.481 | — |
| A + oracle (unreachable) | 4.573 | +2.1% |
| A + static top-B | 1338 | +29763% |
| A + **best real predictor** (full-rank, 500k tok) | 5.916 | +32.0% |
| **born-eye** | **4.712** | **+5.2%** |

The best post-hoc predictor we could train — full-rank, half a million tokens — still costs +32%. The born-eye costs +5.2%, near the oracle. Born beats the best achievable predictor by 20% perplexity. (A first, under-trained predictor scored +225%; we deliberately gave the post-hoc arm its strongest form, and it still loses.)

**At 6.25% active** — below the natural live rate, the aggressive regime:

| arm | ppl | Δ vs dense |
|---|---|---|
| A + oracle | 5.035 | +12.4% |
| A + **best real predictor** | 11.572 | +158% |
| **born-eye** | **5.482** | **+22.4% (−53% vs predictor)** |

And the trend across budgets is the real result — not a point, a slope:

| budget | oracle | predictor | born | imitation burden (pred − oracle) | born's gap above oracle |
|---|---|---|---|---|---|
| 12.5% | +2.1% | +32% | +5.2% | ~30 pts | ~3 pts |
| 6.25% | +12.4% | +158% | +22.4% | ~146 pts | ~10 pts |

As the budget tightens, the post-hoc predictor's imitation burden *explodes* (30 → 146 points), while the born-eye stays within ~3–10 points of the unreachable oracle — **removing ~90% of the burden, consistently.** The born-eye's two-seed noise bar is 0.6%, so the −20% and −53% wins aren't luck.

The mechanism, measured: at a budget near the live rate, the predictor must guess with no recall slack, and its errors compound across layers. The born-eye doesn't guess — it decides, and its weights are organized around that decision. There is no true activation to miss, because the selection *is* the computation. (The static baseline's +29763% is the same lesson in reverse: a fixed neuron set that ignores the input is catastrophic — the input-conditioning is the whole game.)

## What it means

The three-week chain lands on one sentence: **post-hoc selectivity is a costly imitation of built-in selectivity, and the cost is measurable.** It showed up three times, on three different axes:

- *Activation function:* skip works on ReLU (born-sparse), fails on SiLU (never learned to be sparse). Retrofitting (ReLUfication) lands in between.
- *Data:* a predictor at 6k tokens is a caricature of the same predictor at 1M.
- *Selection:* a bolted-on predictor pays a +30-to-+146-point imitation burden; a born-eye that decides pays ~1/10th of it.

This is the most honest formulation we found of "beyond MoE": the win isn't a cleverer predictor stapled to a dense model — it's training the model to *decide what not to compute* in the first place. A fine-grained MoE, at neuron-block granularity, born with its eye.

## Honest caveats & what's open

- **Scale.** 17.5M parameters, TinyStories, two budgets, a block-router of one particular design. The mechanism is clean and the trend is consistent, but nothing here proves it survives at 1B+ parameters or on hard tasks. That's the headline open item.
- **Efficiency ≠ measured yet for the born-eye.** We measured *quality* at a budget; realized wall-clock speedup needs the sparse kernel path (which we prototyped separately: a copy-free gather-matvec hits ~1.8× decode on CPU, with a ~3.9× ceiling from contiguous blocks — and block-granularity routing maps directly onto it).
- **Router health at scale.** Load-balancing held perfectly here, including at k=1; large models are where routers actually collapse.

Natural next steps: the aggressive-budget trend at a larger model; a unified selector that reads one signal for several "eyes" (neuron-skip, KV-cache eviction, early-exit) trained together; and folding the born-eye onto the sparse-kernel path for a real end-to-end speedup.

## Reproduce it

Everything runs on a single consumer GPU plus a few Colab hours.

- `born_eye.py` — the born-eye reference implementation (block-router FFN + GPT), self-contained and documented.
- `train_baby.py` / `prepare_tinystories.py` — control model A (from-scratch ReLU GPT on TinyStories).
- `train_prosparse.py` — the ProSparse-mini half-measure.
- `train_born.py` — the born-eye, from scratch with load-balancing.
- `oracle_baby.py` / `faz3_compare.py` — the oracle ceiling and the five-arm decision table.
- Part-1 tooling (`collect_sparsity_v2.py`, `analyze_sparsity_v2.py`, `oracle_quality.py`, `predictor_quality.py`, `scaling_probe_colab.py`, `bench_kernel_kart17.py`) — the OPT/Qwen/ReluLLaMA post-hoc line and the CPU kernel micro-benchmarks.

The full per-experiment log (27 "cards", question → setup → number → verdict → lesson) is in `SPARSITY_NOTES.md` (and its English condensation in `README.md`).

*Method notes that kept us honest: try to break every positive result before believing it; recall is a proxy, end-to-end perplexity is the metric; measure in-loop, not per-layer; never trust a sequential train/test split; one variable at a time; and check the oracle ceiling before building any predictor.*
