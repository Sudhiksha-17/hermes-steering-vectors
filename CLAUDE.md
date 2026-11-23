# Steering Vector → System Prompt

## What we are building

A model `T` that takes a steering vector `w` as input and outputs a **system
prompt `S` in natural language**, such that the unsteered subject model
conditioned on `S` behaves like the steered subject model.

The loss is a KL divergence over output distributions, averaged over a
distribution of user prompts `X` drawn from WildChat:

```
L(S) = E_{X ~ WildChat} [ KL( p_steer(Y | X, w) || q(Y | X, S) ) ]
```

where
- `p_steer` = subject model with `w` added to the residual stream at a fixed layer
- `q`       = same subject model, no steering, with system prompt `S` prepended

Because `S` is discrete text, `T` is optimized with policy gradient (GRPO),
not backprop through the loss.

Using the identity `KL(p||q) = E_{Y~p}[log p(Y) - log q(Y)]`, and noting that
`log p_steer` does not depend on `T`'s parameters, the gradient reduces to:

> **maximize the log-likelihood that the unsteered+`S` model assigns to
> rollouts sampled from the steered model.**

So: sample rollouts from `p_steer`, score them under `q(·|S)`, and that
cross-entropy is the reward. This is GRPO over system prompts with a
distillation reward.

## Why it matters

Standard autointerp produces max-activating examples plus an LLM summary, and
the summary is never tested for sufficiency. This objective forces the
explanation to **behaviorally reproduce** the intervention. The explanation is
not a vibe; it is the thing that, when you condition on it, makes the unsteered
model behave like the steered one.

Secondary payoff: `S` is a black-box-deployable artifact. It needs no
white-box access, unlike the steering vector itself.

## Build order (do not skip ahead)

**Phase 0 — infrastructure sanity.** `oracle.py`. Load the subject model,
apply a random Gaussian steering vector at a middle layer, generate steered vs
baseline rollouts, compute `KL(p_steer || p_baseline)`. Confirms steering has a
measurable effect and the KL computation is correct.

Gate to pass:
- steered and baseline outputs visibly differ, and steered text is still coherent
- `KL(steered || baseline)` is clearly positive (roughly 2–20)
- `KL(zero-steer || baseline)` is approximately 0 (this is the correctness check)

**Phase 1 — the per-vector oracle.** `oracle_search.py`. Fix one vector `w`.
Directly search for a system prompt `S` minimizing
`KL(p_steer || q(·|S))` averaged over a WildChat batch. Start with a handful of
hand-written candidate prompts, then an LLM-proposed candidate loop.

This is the single most important phase. It yields:
- the achievable KL floor for a given vector (is there *any* prompt equivalent?)
- the baseline the amortized model must beat
- a way to separate "T failed" from "impossible in principle"

Do not start Phase 2 until Phase 1 shows a real KL reduction over the
no-system-prompt baseline on held-out prompts.

**Phase 2 — amortize.** Train `T` with GRPO. Reward = negative cross-entropy of
steered rollouts under `q(·|S)`, grouped over candidate prompts per vector.

**Phase 3 — generalization.** Does `T` produce useful prompts for vectors it was
never trained on?

## Known failure modes — design against these from the start

**Gibberish hack (the main one).** Discrete prompt optimization reliably finds
adversarial token strings that behave like soft prompts: they match the
distribution while being unreadable. If this happens we have reinvented soft
prompts in token space and lost the interpretability that motivated the project.
Mitigation: constrain `S` toward a base LM (fluency reward / KL-to-base-LM
penalty), or gate on a readability check. Treat "is `S` readable English?" as a
first-class metric reported alongside KL, never as an afterthought.

**Degenerate vector distribution.** Isotropic Gaussian `w` mostly yields
semantically null directions. A prompt that reproduces noise is a verbalization
of noise. Gaussian is fine for Phase 0 infrastructure testing, but the training
distribution for Phase 2 should be *meaningful* directions: SAE features,
difference-of-means behavior vectors, or contrastive vectors. Note that RepE
(linear steering) produces visibly degraded, gibberish-adjacent text on complex
behaviors; expect random directions to be worse.

**Expressiveness gap.** Steering is additive at a layer on every token position;
a system prompt acts only through attention over a fixed prefix. These are
different intervention classes, so some directions have no prompt equivalent and
the achievable KL has a per-vector floor. This is why the Phase 1 oracle exists.
High KL is otherwise ambiguous between "T is bad" and "no such prompt exists."

**Steering norm miscalibration.** Too small and the intervention does nothing;
too large and outputs degenerate and the task becomes trivial. `ALPHA` must be
calibrated per layer before any result is diagnostic.

**Shortcut learning.** LatentQA found that a decoder trained on both control and
stimulus activations learns to cheat by reading control token embeddings
directly out of the residual stream rather than inferring the control's
behavioral effect. Watch for the analogous shortcut here.

## KL direction

Forward KL (as specified above) is mode-covering: risks a vague, over-broad `S`.
Reverse KL is mode-seeking: risks an `S` capturing only one mode of `w`'s effect.
Forward is the default (it is the convenient one and it penalizes missing
behaviors), but state the tradeoff in any writeup.

## Prior work — read before claiming novelty

- **LatentQA** (Pan, Chen, Steinhardt, ICLR 2026, arXiv 2412.08686). The closest
  work. Trains a decoder LLM on activation/QA pairs via patching. **Crucially,
  its "control" is a differentiable loss backpropagated into a LoRA, not
  generated text.** It also already does "uncover hidden system prompts from
  activations" at 92% accuracy on hard personas. Read it before writing anything
  that claims novelty.
- **Patchscopes** (Ghandeharioun et al., ICML 2024) and **SelfIE** — patch an
  activation into a target prompt and let the model verbalize it. Readouts, not
  behaviorally-equivalent prompts.
- **Creating Steering Vectors via Input Optimisation** (Akar, arXiv 2611.17481).
  ActGCG: optimizes an input to match the *activations* a steering vector would
  induce. The inverse of our Phase 1 oracle, and the baseline for it.
- **ActAdd** (Turner et al., arXiv 2308.10248). Frames the prompt-vs-steering
  equivalence question and uses KL-div to measure steering effects.
- **Context distillation** (Askell et al.). Our loss is mechanically this. Cite
  it as precedent for the objective.

**The wedge:** LatentQA controls via gradients into weights; we generate
readable text and test it by output-distribution KL over a prompt distribution.
LatentQA never closes the sufficiency loop; the KL objective does. The output is
also black-box deployable. Those two differences are the contribution. If we
cannot state the wedge in two sentences, the project is not ready to pitch.

## Evaluation

The regression/KL loss is the training signal, **not** the headline result.
Report:
1. KL reduction vs the no-system-prompt baseline, on **held-out** WildChat prompts
2. KL vs the Phase 1 oracle floor for that vector (how much of the achievable gap did `T` close?)
3. Readability of `S` (guards the gibberish hack)
4. Generalization to unseen vectors

## Safety note

LatentQA's own results show a trained activation interpreter elicits harmful
knowledge on 100% of harmful prompts and 87% of benign ones once steered toward
a "rogue biologist" persona. A system-prompt generator for arbitrary steering
vectors has the same dual-use structure: for a harmful direction, it is a
jailbreak factory. Name this in any writeup. Do not publicly release a trained
`T` plus harmful vectors without gating.

## Environment

- RunPod, A100 80GB (current pod: H200 141GB). **All work lives in `/workspace/`** — the pod root
  filesystem is ephemeral and is wiped on restart.
- Subject model: Qwen2.5-3B-Instruct (swap up if memory allows).
- Prompt distribution: WildChat.
- Config lives at the top of each script as module-level constants. No hidden
  magic numbers buried in function bodies.

## Working style

- Phased and iterative. Get Phase 0 passing its gate before writing Phase 1.
- Every phase ends in a runnable script with a printed sanity check, not a
  notebook of fragments.
- Prefer a small, fast, obviously-correct experiment over a large one that
  takes an hour to reveal a bug.
- When a result looks good, check the trivial baseline before believing it.
  Specifically: does an empty system prompt, or a generic one, get the same KL?
  If so, the result is confounded.
- Log every run's config (layer, alpha, seed, vector) alongside its numbers.
  Results without their config are not reproducible.
## Implementation notes (conventions established in code)

- `HF_HOME=/workspace/.cache/huggingface`. Prompts cached at `data/wildchat_prompts.json` (fixed train/heldout split).
- "No system prompt" baseline = Qwen chat template with no system message, which injects Qwen's
  default system prompt. `p_steer` uses that same context; `q(.|S)` replaces it with `S`.
- Sampling must be pure ancestral (temperature=1, top_k=0, top_p=1, no repetition penalty); Qwen's
  generation_config defaults are NOT that, and would bias every KL estimate.
- Runs log to `runs/<phase>_<timestamp>.json` with full config.
