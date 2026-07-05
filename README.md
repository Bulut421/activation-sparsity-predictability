# Activation Sparsity Predictability — Research Log

**Last updated:** 2026-07-05
**Author:** Bulut (Kler3) | **Hardware:** RTX 5060 8GB / 32GB RAM (CPU-torch venv), Colab H100/A100 for training
**Status:** Value proof achieved (Card 13). Research phase closed; engineering phase open.

## TL;DR

Can FFN neuron sparsity be predicted *before* computing it, cheaply enough to skip the computation? (The PowerInfer/DejaVu mechanism, rebuilt and measured from scratch.)

**Answer: yes — but only on ReLU-family models, and only with enough predictor training data.**

Final measured result (OPT-1.3b, 24 layers, in-loop, wikitext held-out):

| Budget B | static mask | learned predictor (r512) | net FFN compute saved* |
|----------|-------------|--------------------------|------------------------|
| 0.40     | +1151% ppl  | **+1.1% ppl**            | 44%                    |
| 0.30     | +3213% ppl  | **+2.9% ppl**            | 54%                    |
| 0.25     | +5629% ppl  | **+4.6% ppl**            | 59%                    |
| 0.20     | +9095% ppl  | **+8.3% ppl**            | 64%                    |

\* net = 1 − B − 0.156 (rank-512 predictor cost relative to OPT's 2-matmul FFN)

Two findings we consider the most transferable:

1. **The activation function is a hard boundary.** On SiLU models (Qwen2.5-3B) even a *perfect* oracle mask destroys quality below 50% budget. On born-ReLU models (OPT) the oracle is free down to ~10%. No predictor can beat its oracle ceiling — check the ceiling first.
2. **Per-layer metrics do not compose, and apparent "unpredictability" may be data starvation.** With 6k training tokens every deployment variant failed catastrophically (+400–2000% ppl) despite decent per-layer recall. The scaling curve (6k → 300k tokens) showed no plateau; at 300k tokens the same pipeline reached +1.1% ppl. Every "principled-looking" negative at small data was an artifact.

## Goal

Predict FFN neuron sparsity before it is computed, to skip computation → compute/energy savings ("beyond MoE", the 20W paradox). This is the *compute-less* line, not the memory-fitting line.

## Main narrative (chronological chain summary)

- Domain identity does **not** know sparsity (Card 1: +0.037 coverage lift → STOP). Token activation **does** (Card 3: LIFT +0.12–0.18, survives prompt-level held-out split).
- But on SiLU, even with signal, the soft tail means skipping hurts (Cards 5–6): oracle skip at B=0.40 already +7.2% ppl. Architectural boundary, not an engineering failure.
- ReLU control (Card 7): same protocol on OPT-1.3b → 96.1% natural zeros, oracle free down to B=0.10. Economics inverted.
- Predictor line on OPT (Cards 8–11): signal confirmed, low-rank works offline — but every in-loop variant exploded at 6k training tokens. Compounding across 24 layers dominates; error *consistency* matters as much as recall (a fixed static mask beat an adaptive predictor over 24 layers, and lost to it over 6).
- Scaling test (Card 12): frac-for-recall-0.99 fell 0.95 → 0.18 going 6k → 300k tokens, no plateau. Not "chaotic tail", just a starving predictor.
- Value proof (Card 13): 300k tokens, rank-512, all 24 layers in-loop → table above.

## Oracle comparison (same protocol, budget → ppl delta)

| Budget | Qwen2.5-3B (SiLU) | OPT-1.3b (ReLU) |
|--------|-------------------|------------------|
| natural zero rate | ~0 | 96.1% |
| B=0.50 | +2.4%  | −0.0% |
| B=0.30 | +20.7% | −0.0% |
| B=0.20 | +64.8% | −0.0% |
| B=0.10 | —      | +0.1% |
| B=0.05 | —      | +1.1% |

SiLU: smooth slope, every slice hurts. ReLU: plateau, late break.

## Experiment cards

### Card 1 — Stage A: dense domain-static sparsity → STOP (+0.037)
Qwen2.5-3B, 560 prompts, domain-conditioned top-k mask. Coverage lift +0.037 (MoE side had +0.186). Global top-20% already covers ~90% → no ceiling. Block/layer/k sweeps all negative.

### Card 2 — MoE routing spike: L20 CV=2.77, not a domain pattern → paused
Routing is imbalanced but does not map to human-readable domains.

### Card 3 — x → hot-mask prediction: SIGNAL EXISTS (verified)
Qwen2.5-3B, L6/18/30, k=20% magnitude, linear probe. LIFT +0.176/+0.124/+0.164 over static baseline. Identical under prompt-level held-out split → generalization, not leakage. Domain signal +0.037 vs token-activation signal ~+0.15 (4–5×).

### Card 4 — k-sweep (0.10/0.20/0.30): LIFT independent of k
Same LIFT at every k; layer order stable (L6 > L30 > L18 across 3 measurements). k becomes an engineering choice.

### Card 5 — Neuron-level predictor + budget/cost analysis: door half-open
MLP could **not** beat the linear probe (rule confirmed 4×) → signal is linear and saturated at this data scale. Recall 0.9 needs 48% of neurons at best layer; net ceiling ~32% FFN.

### Card 6 — Oracle skip on Qwen (36 layers, budget → ppl): early explosion
B=0.50 +2.4% / 0.40 +7.2% / 0.30 +20.7% / 0.20 +64.8%. No plateau → SiLU signature. Combined with Card 5: realistic gain ~30% FFN for +4–6% ppl → weak trade. **Vanilla-SiLU skip line: STOP (architectural).**

### Card 7 — ReLU control (OPT-1.3b, same protocol): PLATEAU
96.1% natural zeros (mean live ~3.9%). Delta exactly 0 down to B=0.20; +0.1% at 0.10; +1.1% at 0.05. Two-model controlled comparison confirms the SiLU-vs-ReLU hypothesis. Cards 3–5 infrastructure wasn't wasted — it was pointed at the wrong model.

### Card 8 — Live-mask predictor on OPT (x → a>0): signal in ReLU too
Live rates L4/12/20 = 0.7/4.5/6.8%. Linear LIFT up to **+0.312** (largest measured). Layer order *reversed* vs Qwen (L20 > L12 > L4). SVD rank sweep: signal is low-dimensional — rank 64 cuts predictor cost 50% → 1.9% with almost no budget loss. w-recall diagnostic added (missing small-magnitude live neurons is cheap; missing large ones is not).

### Card 9 — Predictor-in-the-loop ppl: single layer cheap, 24 layers explode
Mechanics validated (oracle path bit-identical to baseline; B=0.95 sanity +0.4%). Single layer ≤+1.7%. All 24 layers: +467…+2029%. Independent compounding predicts ~+24%; observed 50× worse → **superlinear compounding**: predictors trained on clean x degrade on the drifted x they themselves cause. Also: sequential prompt split silently measured domain-holdout (baseline 42.8 vs 26.4) — random split is mandatory; and under domain shift both static and predictor collapse → train predictors on mixed-domain data.

### Card 10 — Error-aware (progressive) training: halved, not fixed
Training each layer's predictor on the masked forward of previous layers cut explosions ~2× (+1264 → +536% at B=0.30). Distribution shift is real but secondary. **Surprise:** in-loop, static beat the predictor (fixed error → consistent pruned subnetwork; per-token variable error → compounding noise). "Error consistency" matters as much as recall — consistent with PowerInfer's static-heavy hybrid design.

### Card 11 — Partial-layer skip: both blocks fail (at 6k tokens)
Early block (L1–8, live 0.6–1.5%) at B=0.10–0.15: static +86/+214%, predictor +615/+1372%. "Dead layer = safe layer" is **false**: the few live neurons in early layers are per-token chaotic and large (the low w-recall had warned exactly this). Mid block (L12–17) at B=0.30–0.40: predictor beats static 3–4× (dose–response confirms Card 10), but absolute level still bad (~14% of FFN for +16% ppl). Design lesson: per-layer budget must scale with *predictability*, not liveness.

### Card 12 — Data scaling curve (Colab, wikitext): CURVE ALIVE, NO PLATEAU
Fixed 30k-token test set (document-level split), nested train sets. frac@0.99 (full linear):

| N tokens | L4 | L12 | L20 |
|----------|-----|-----|-----|
| 6k   | 0.96 | 0.95 | 0.96 |
| 30k  | 0.86 | 0.79 | 0.81 |
| 100k | 0.52 | 0.43 | 0.40 |
| 300k | **0.29** | **0.18** | **0.21** |

recall@budget-0.40: 0.75 → 0.92 → 0.98 → **0.994–0.999**. Every 3.3× data helps; no saturation. Cards 9–11 negatives were data starvation. New bottleneck: rank-128 hits capacity at large data → rank 256–512 (still cheap on OPT: r512 = 15.6% of FFN).

### Card 13 — VALUE PROOF: 300k-token r512 predictors, in-loop (table in TL;DR)
All 24 layers, wikitext held-out. pred B=0.40 → **+1.1%** ppl (was +467% at 6k). Static dead (+1151%). At recall ≈0.999 per layer, compounding is manageable — the "error consistency" concern dissolves at high recall. Measured closing statement: *low-rank linear predictor (r512, 15.6% FFN cost) + budget 0.30–0.40 = 44–54% net FFN compute reduction for +1.1–2.9% ppl on OPT-1.3b.*

## Closed doors (do not reopen)

- domain → neuron/expert mapping (Cards 1–2; absent in both architectures)
- "splitting" a dense model by labels (virtual position ≠ physical separation)
- skip-based gains on SiLU models (Cards 5+6+7: the boundary is architectural)
- growing/shrinking the predictor on SiLU (Card 5: MLP never beat linear; signal saturated)

## Open questions

- Why is the predictability order L6 > L30 > L18 on Qwen but reversed (L20 > L12 > L4) on OPT? (Stable across all measurements within each model.)
- Where does the scaling curve actually saturate? (1M+ tokens not yet measured.)
- Per-layer budgets: early OPT layers are chaotic (Card 11) — does a predictability-weighted budget beat uniform B?

## Next

Research phase closed. Options, in rough order of value:

1. **Engineering:** masking ≠ speedup. Real gains need not-computing the skipped rows (gather/sparse matmul; row-skip fc1/fc2 in a CPU/GGUF context). First measurement: naive PyTorch gather wall-clock. Plus rank/budget Pareto sweep and per-layer budgets.
2. **Generalization:** same chain on another ReLU/ReLUfied model (Bamboo-7B / ReluLLaMA-7B) — closes the "is this OPT-specific?" question (~1 hour on Colab).
3. **The fork, now justified:** ReLUfying a SiLU model (ProSparse recipe) is now a defensible investment — the payoff chain is measured on ReLU. Note: billions of tokens of continued pretraining, not LoRA scale.
4. **Writing it up:** 13 cards, 2 models, every failure mode mapped.

## Method notes (invariant)

- Every measurement: question / setup / headline number / verdict / what we learned.
- Try to *break* positive results (split control, k-sweep, rank sweep) — write the card only if it survives.
- Proxy-metric warning: recall is a proxy; the real metric is end-to-end quality (ppl). It fooled us once (Cards 8 vs 9).
- Same protocol, different model = controlled comparison; one variable at a time.
- Sequential splits are forbidden: files may be domain-ordered → random prompt/document-level split, seeded.
- Per-layer metrics do not compose: in-loop measurement is mandatory (Card 9 lesson).
- Check the oracle ceiling before building predictors (Card 6 lesson: no predictor beats its oracle).

## Repository layout & reproduction

```
scripts/
  collect_sparsity_v2.py         # parasite data collection (Qwen/OPT auto-detect)
  analyze_sparsity_v2.py         # signal test; --full predictor; --mask-mode live; SVD sweep
  oracle_quality.py              # oracle budget→ppl (Cards 6–7)
  predictor_quality.py           # in-loop 3-curve eval (Cards 9–11, 13); --progressive; --load-preds
  scaling_probe_colab.py         # data scaling curve (Card 12)
  colab_train_all_predictors.py  # 24-layer predictor training, resumable (Card 13)
results/                         # JSON reports from all cards
NOTES_TR.md                      # original Turkish research notes
```

Key reproduction path (GPU, ~1 hour total):

```bash
pip install transformers datasets accelerate torch
python scripts/scaling_probe_colab.py                    # Card 12 (~15 min on H100)
python scripts/colab_train_all_predictors.py             # Card 13 training (~30 min)
python scripts/predictor_quality.py \
    --prompts wikitext_eval.jsonl --device cuda \
    --load-preds predictor_weights_wt300k_r512.pt \
    --train-limit 0 --eval-limit 999 \
    --budgets 0.40,0.30,0.25,0.20 --modes static,pred     # Card 13 eval (~5 min)
```

The trained predictor weights (~400MB) are not committed; Card 13's script regenerates them in ~30 minutes.
