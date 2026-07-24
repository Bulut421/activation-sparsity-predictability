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

The born-eye removes ~90% of what we call the **imitation burden** (predictor cost minus oracle cost), *consistently* — while the post-hoc predictor's burden explodes as the budget tightens. Being born knowing what to skip beats learning to guess it. And it shows up in wall-clock: because the born-eye's active blocks are *contiguous*, ordinary matmul on zero-copy slices turns the 12.5% budget into a measured **~2.1× end-to-end decode speedup** at 1.3B-parameter shape on CPU — where the same post-hoc predictor, stuck gathering scattered neurons, manages ~1.2×.

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

### Why it wins — it's co-adaptation, not the router

We tried to break our own result. Entropy near 1.0 shows the router doesn't collapse, but it doesn't show the router is *useful* — a random router gives entropy 1.0 too. So we trained three born-eyes from scratch, identical except the router: **learned** (normal), **random** (a frozen random projection — input-dependent but never trained), and **fixed** (input-independent, always the same blocks). Same eval:

| variant | ppl | what it adds |
|---|---|---|
| born, learned router | 4.723 | — |
| born, random (frozen) router | 4.827 | learning the router: **+0.10** |
| born, fixed (input-independent) | 5.265 | input-conditioning: **+0.44** |
| best post-hoc predictor | 5.916 | co-adaptation: **+0.65** |

The decomposition is humbling and clarifying. In order of importance: **co-adaptation ≫ input-conditioning ≫ learning the router.** The learned router — the part that looks clever — contributes the *least* (~2%). What matters is (1) training the body knowing it will be sparse, and (2) giving it a *consistent, input-conditional* partition to organize around — even a random one. Strikingly, even the *fixed* born-eye (the dumbest variant) beats the best post-hoc predictor. So born-eye's edge over bolt-on prediction is almost entirely co-adaptation, not routing intelligence: the body and its (arbitrary) sparse structure grow together, so there is no foreign pattern to imitate.

This doesn't weaken the thesis — born still beats post-hoc decisively — it corrects the *why*. And the caveat matters: at this tiny scale (16 blocks, k=2), learned routing may simply have little to add; at scale, with many blocks and harder tasks, learning *where* to look could dominate. Treat "learning the router barely helps" as a small-scale finding, not a law — the same discipline that caught the data-starvation artifacts earlier.

One more control completes the picture: sweeping the *granularity* at fixed budget (8, 16, and 32 blocks). The partition has to be fine enough — 8 coarse blocks cost ~15% more perplexity — but past a threshold it saturates: 16 and 32 blocks give *identical* perplexity. So the born-eye needs three things — co-adaptive sparse training, an input-conditional partition, and enough granularity (block count past a floor) — and needs neither a *learned* selection nor ever-finer blocks. That's richer than a single fixed sparsity pattern, but simpler than a smart router.

### Does it actually run faster?

Compute saved isn't latency saved until you actually *don't read* the skipped weights. We built a full KV-cached decode loop — every layer, attention, head, one token at a time — and timed it on CPU (single-thread, the clean small-matvec regime), dense vs born vs post-hoc, across four scales:

| model | FFN share of decode | born end-to-end | post-hoc end-to-end |
|---|---|---|---|
| 17M | 49% | 1.24× | 0.75× |
| 91M | 55% | 1.64× | 0.91× |
| 693M | 61% | 2.00× | 1.08× |
| 1.3B-shape | 63% | **2.12×** | 1.23× |

Three things. (1) The born-eye's isolated ~7× FFN kernel survives end-to-end but is **Amdahl-bounded** by the FFN's share of decode time (49→63%, rising with scale), so the honest deployable figure is ~2×, not 8× — the rest is attention, the head, and norms, which the born-eye doesn't touch. (2) It **grows with scale**, as the FFN comes to dominate. (3) The post-hoc predictor is *slower than dense* at small scale (0.75×) and only 1.2× even at 1.3B — its scattered gather plus predictor overhead eats the saving. Born beats post-hoc end-to-end by ~1.7–1.85× at every scale. The contiguity that lets born's blocks be sliced is the whole reason the compute win turns into a wall-clock win.

## What it means

The three-week chain lands on one sentence: **post-hoc selectivity is a costly imitation of built-in selectivity, and the cost is measurable.** It showed up three times, on three different axes:

- *Activation function:* skip works on ReLU (born-sparse), fails on SiLU (never learned to be sparse). Retrofitting (ReLUfication) lands in between.
- *Data:* a predictor at 6k tokens is a caricature of the same predictor at 1M.
- *Selection:* a bolted-on predictor pays a +30-to-+146-point imitation burden; a born-eye that decides pays ~1/10th of it.

This is the most honest formulation we found of "beyond MoE": the win isn't a cleverer predictor stapled to a dense model — it's training the model to *decide what not to compute* in the first place. A fine-grained MoE, at neuron-block granularity, born with its eye.

## Honest caveats & what's open

- **Scale — and a sobering signal.** Most results are at 17.5M parameters on TinyStories. A controlled 2× scale-up (to 36M), measured at two budgets, sharpens the picture honestly. Two axes separate cleanly. (1) *Budget:* the born-eye's edge is far larger at aggressive budgets — at 36M it beats the best post-hoc predictor by 31% at a 6.25% budget versus 10% at 12.5% — and this holds at both scales, so the effect is real, not an artifact. (2) *Scale:* at *every* budget, doubling the model roughly *halves* the imitation burden (post-hoc's cost above the oracle drops from ~30 to ~16 points at 12.5%, and from ~146 to ~71 at 6.25%), because the bigger model is naturally sparser (92% vs 89% zeros), which makes prediction easier — exactly the way more training data did earlier. The born-eye's own cost stays flat. So it still wins in every cell, but the margin erodes with scale. The honest claim is therefore narrower than "born-eye wins": its advantage is **concentrated in the constrained regime** — aggressive budgets, smaller models, less data. If the halving-per-doubling trend continued, post-hoc could eventually catch it even at aggressive budgets; whether that happens or the gap plateaus at 1B+ is the headline open question, and two points can't answer it. (The learned-vs-random finding, by contrast, held steady across the scale-up.)
- **The decode speedup is realizable with stock matmul — but the born-eye's contiguity is what makes it so.** A single-FFN decode micro-benchmark (CPU, T=1, 12.5% budget) shows the born-eye's *contiguous* blocks can be read as zero-copy slice-views and fed to ordinary matmul, reaching ~7× (near the ~8.75× prebuilt ceiling) — about 6× faster than the equivalent scattered-neuron gather a post-hoc predictor is stuck with (~1.2×). The catch is purely implementational: a naive fancy-index gather (`W[idx]`) copies and throws the advantage away (falling back to the post-hoc number); you have to slice per block. Post-hoc sparsity can't do this at all — its live neurons are scattered, so it needs a custom kernel and still pays a penalty. Born sparsity needs no custom kernel. (End-to-end numbers are in "Does it actually run faster?" above: the ~7× kernel becomes a ~2× full-decode speedup once attention and the head are counted.)
- **Adaptive width doesn't help — and difficulty isn't the axis.** The smallest step toward budget-*trading* is to let one born-eye spend a variable number of blocks per token (average pinned at 12.5%), trained with a straight-through hard gate so the selection is genuine, not a soft rescaling. (A soft-gate version *looked* like it worked — until we thresholded it for deployment and perplexity jumped from 4.6 to 551; the quality was hiding in fractional gates that skip nothing. Measure the deployed behavior, not the training proxy.) The honest version did *not* beat fixed-k (+3.1%), and the correlation between spend and token difficulty came out **negative** — hard, high-loss tokens got *fewer* blocks. High cross-entropy is often irreducible surprise (a name, the start of a sentence) that no extra FFN compute fixes, so the model rationally spends where compute has traction. Fixed-k is enough here, and token loss is the wrong signal to route a budget on.
- **Router health at scale.** Load-balancing held perfectly here, including at k=1; large models are where routers actually collapse.

Natural next steps: the aggressive-budget trend at a larger model; a unified selector that reads one signal for several "eyes" (neuron-skip, KV-cache eviction, early-exit) trained together; and a second eye on the attention/KV side, where the decode breakdown shows the remaining time goes at long context. (The adaptive-width negative above is a caution: budget-trading on token difficulty is not the obvious way in.)

## Reproduce it

Everything runs on a single consumer GPU plus a few Colab hours.

- `born_eye.py` — the born-eye reference implementation (block-router FFN + GPT), self-contained and documented.
- `train_baby.py` / `prepare_tinystories.py` — control model A (from-scratch ReLU GPT on TinyStories).
- `train_prosparse.py` — the ProSparse-mini half-measure.
- `train_born.py` — the born-eye, from scratch with load-balancing.
- `oracle_baby.py` / `faz3_compare.py` — the oracle ceiling and the five-arm decision table.
- `train_born_vark_ste.py` — the adaptive-width (variable-k) born-eye with a straight-through hard gate (the honest negative).
- `bench_decode_e2e.py` — the end-to-end KV-cached decode benchmark (dense vs born vs post-hoc, scale sweep).
- Part-1 tooling (`collect_sparsity_v2.py`, `analyze_sparsity_v2.py`, `oracle_quality.py`, `predictor_quality.py`, `scaling_probe_colab.py`, `bench_kernel_kart17.py`) — the OPT/Qwen/ReluLLaMA post-hoc line and the CPU kernel micro-benchmarks.

The full per-experiment log (33 "cards", question → setup → number → verdict → lesson) is in `NOTES_TR.md` (and its English condensation in `README.md`).

*Method notes that kept us honest: try to break every positive result before believing it; recall is a proxy, end-to-end perplexity is the metric; measure in-loop, not per-layer; never trust a sequential train/test split; one variable at a time; and check the oracle ceiling before building any predictor.*
