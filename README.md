# Hermes — Steering Vector → System Prompt

Can a steering vector's behavioral effect be reproduced by a **natural-language system prompt**, and
can a model learn to **generate** that prompt from the vector?

We train a model `T` that takes a steering vector `w` and emits a system prompt `S`, such that the
*unsteered* subject model conditioned on `S` behaves like the *steered* model. The objective is a forward
KL over WildChat prompts, optimized with GRPO (reward = log-likelihood the unsteered+`S` model assigns to
rollouts from the steered model — context distillation in token space). See `CLAUDE.md` for the full
problem statement, prior work, and the wedge vs. LatentQA/Patchscopes.

Subject model: **Qwen2.5-3B-Instruct**. Prompt distribution: **WildChat**.

## Headline result (honest version)

- **Phase 1/1b** — for a fixed vector, search for the best `S`. A per-vector **floor** (matched-norm
  random control) shows a large part of every vector's KL is just the cost of adding *any* vector of that
  norm, which no prompt can close. French best prompt closes ~67% of the *closable* gap; pirate ~20%;
  hedging: no prompt beats an empty system prompt (its structured signal is smaller than the cost of
  having a system prompt at all).
- **Phase 2** — GRPO-train `T` on 6 vectors. In-distribution it **beats a 48-candidate LLM search** on
  4 of 6 and ties on French, at a single forward generation. It learns intensity calibration that search
  could not.
- **Phase 3/3b** — generalization to vectors `T` never saw. **`T` does not read arbitrary `w`.** Its margin
  over a search is **monotone in the vector's cosine to the nearest training vector**: it wins only on
  near-duplicates (spanish 0.89, emoji 0.77) and loses on every genuinely novel vector (japanese 0.69 →
  emits Chinese and fails; korean 0.65 → partial; uppercase 0.51 → worse than an empty prompt, mirroring
  the one training vector it failed on, lowercase). The Phase-2 win is in-distribution + near-duplicate
  interpolation, not conditioning on `w`.

Full numbers and per-run detail live in the `PHASE*_STATUS.md` files.

## Repository layout

| File(s) | What |
|---|---|
| `CLAUDE.md` | Problem statement, build order, known failure modes, prior work, evaluation criteria |
| `PHASE{0,1,1B,2,3B}_SPEC.md` | Per-phase specifications |
| `PHASE{0,1,2,3,3B}_STATUS.md` | Per-phase results and findings (the lab notebook) |
| `common.py` | Model loading, steering hook, WildChat prompts, pinned ancestral sampling, KL estimators |
| `behaviors.py` | Behavior definitions (contrast elicitation, candidate-prompt ladder, behavior metrics) |
| `oracle.py` | Phase 0: steering/KL infrastructure sanity |
| `build_vectors.py`, `oracle_search.py` | Phase 1: difference-of-means vectors, α calibration, candidate ladder |
| `matched_norm_control.py`, `floor_validation.py`, `token_attribution.py`, `pirate_layer_sweep.py` | Phase 1b diagnostics |
| `readability.py`, `gate_redteam.py` | Hardened readability gate (Phi-4 classifier + payload check + base-LM perplexity) and its red-team |
| `build_vectors_p2.py`, `train_T.py`, `eval_T.py` | Phase 2: vector screening, GRPO training of `T`, evaluation |
| `rung4.py` | LLM-proposed candidate-prompt search baseline |
| `lang_id.py`, `phase3b_build.py`, `phase3_eval.py` | Phase 3/3b: language ID, low-cosine holdout construction, holdout evaluation |
| `vectors/*.pt` | The difference-of-means steering vectors (+ contrast data), per layer |
| `runs/` | Experiment logs and results (JSON + logs). Trained LoRA checkpoints and model/rollout binaries are **not** committed (size); regenerate with `train_T.py`. |

## Reproducing

```bash
source env.sh                 # sets HF_HOME, PYTHONPATH (deps in /workspace/.pylib), etc.
# deps: transformers, accelerate, peft, torch, lingua-language-detector, pyarrow
python oracle.py              # Phase 0 sanity
python build_vectors.py       # build + calibrate Phase 1 vectors
python oracle_search.py       # Phase 1 ladder
python train_T.py --tag main6 # Phase 2 GRPO (writes a LoRA checkpoint under runs/)
python eval_T.py --lora runs/<run>/lora_step200 --vectors french:18,...
python phase3_eval.py --no-autostop --only japanese,uppercase,korean
```

The WildChat prompt split (`data/wildchat_prompts.json`) is rebuilt from `allenai/WildChat-1M` on first
use and is not redistributed here.

## Safety / dual use

`T` is a trained generator of system prompts that reproduce arbitrary steering directions; for a harmful
direction this is a jailbreak factory (the same dual-use structure LatentQA notes). Do not release a
trained `T` together with harmful vectors without gating. See the safety note in `CLAUDE.md`.
