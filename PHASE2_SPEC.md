# Phase 2 — Amortize: train `T`

Status: DONE 2026-11-23 (results in `PHASE2_STATUS.md`). Prerequisites: Phase 1b DONE (`PHASE1_STATUS.md`), plus
prerequisite P0 below.

## What Phase 2 builds

A model `T` that takes a steering vector `w` and emits a system prompt `S`,
trained with GRPO. Reward = the log-likelihood the unsteered+`S` model assigns
to rollouts sampled from the steered model (the forward-KL gradient, per
`CLAUDE.md`), minus the penalties in "Reward" below.

## What Phase 1b changed about this phase

Read this before designing anything. Three Phase 1b results bound what `T` can
achieve, and one of them changes the project's framing.

**The limits are in the intervention class, not in search.** Rung 4 found
intensity-matched hedging prompts (1.28 vs 1.37 hedges per 100 words) that
still lose to the empty prompt, because a generic system prompt alone costs
4.1 nats, about 40% of hedging's ~10-nat structured component. Pirate at L8,
with the norm confound removed, closes 17–20% of a 147-nat gap. Run 2 showed
French's residual is language-boundary artifacts (fused English–French words,
mid-sentence code-switching), not unreproduced content. A prefix system prompt
has an irreducible presence cost and a granularity floor, and additive steering
produces artifacts no fluent instruction imitates. **`T` inherits all three.**

**The marginal value of search over a one-liner is small and measured.** Versus
rung 0: French 36% → 48%, pirate 13% → 18%, hedging none. That is the band `T`
has to beat, and rung 4 (an LLM proposer with 6 rounds and 48 candidates per
target) is the honest baseline, not the one-liner.

**The floor is solid.** Run 1 plus the rung-4 random negative control: a
proposer given 6 rounds against a random direction converged toward blandness
and never reached rung 1. No tested prompt, hand-written or searched, closes any
part of F.

---

## Prerequisite P0 — harden the readability gate (blocking)

**Gate item 4 of Phase 1b is marked DONE. For Phase 2 it is not.** In rung 4
the gate faced no adversarial pressure: the proposer produced no gibberish,
all 5 rejections were false rejections of sensible sentences, and JSON-wrapped
strings passed at ~7.2. That was harmless because no KL-winning candidate was
rejected.

Under GRPO the gate becomes a reward term, and policy gradient is adversarial
pressure by construction. A judge with a noisy boundary is exactly what a policy
learns to sit on. Harden before training starts:

1. **Fix the false-rejection boundary.** Five sensible hedging prompts were
   rejected (judge 4.5 on "Pause briefly in your thoughts before providing an
   answer."; subject perplexity 2,538 on another). Recalibrate thresholds
   against a larger set of known-good prompts.
2. **Reject structural artifacts.** JSON wrappers passed. Strip-and-revalidate,
   or fail any candidate that is not plain prose.
3. **Red-team it.** Hand-write or search for 20+ candidates designed to pass the
   judge while being degenerate (fluent nonsense, instruction-shaped strings
   with no semantic content, prompts that smuggle format control). The fluent-
   nonsense control already in the calibration set is one example and the judge
   caught it; find ones it does not.
4. **Decide the reward shape now.** A hard pass/fail gate is a cliff that GRPO
   will probe; a continuous penalty is smoother but lets a policy trade
   readability for KL. Recommendation: continuous penalty **plus** a hard floor
   below which reward is zero, so there is no gradient toward the cliff edge.

Phi-4 as judge and OLMo-2-0425-1B as base-perplexity model carry over. Note that
both live in an ephemeral root-disk cache (`/root/hf_ephemeral`) and are
re-downloaded on pod restart; move them to `/workspace` if there is room, since
a training run is long enough to span a restart.

---

## Protocol rules (carried and amended)

**R1 (AMENDED).** An `S` helps only if it beats **the better of rung 0 and
rung 1**. The original wording assumed rung 1 is the harder baseline. Hedging
shows it is not: rung 1 is worse than silence there, so "clears rung 1" (+11%)
was meaningless while rung 0 went uncleared. The reward function must encode
this, not the old wording.

**R2.** Report closure against the floor alongside closure against the better
baseline.

**R3 (as revised).** Closure-vs-floor rests on an additivity assumption and is
not a point estimate. Evidence so far suggests it is *conservative*: run 1 and
the rung-4 control found F not closable by any tested prompt, and run 2 found
the off-target cost of a structured vector exceeding F.

**R4.** Do not report a closure fraction when the best `S` falls inside the
spread of the random floor. Report it as indeterminate and say why.

**R5 (NEW).** Degeneration is a reward penalty, not just a reported diagnostic.
Carried from Phase 0: KL is non-monotonic in steering strength and *falls* when
the steered model degenerates into repetition the baseline predicts easily. The
mirror-image hack in Phase 2 is an `S` that makes the unsteered model degenerate
similarly, scoring well on KL while explaining nothing. Distinct-token fraction
(or token entropy) enters the reward as a penalty from the first training run.

---

## Vector distribution

**Selection criterion.** A vector is a usable training target only if its
structured component (KL0 − F) **exceeds roughly 3× the rung-1 presence cost**
for that vector. Below that, the prefix intervention's own side effects are as
large as the behavior being imitated, and the optimal `S` is the empty string.

**Training set.** Build 8–12 difference-of-means vectors meeting that criterion.
Include French (L18, α 0.20) and pirate (L8, α 0.5) as the two characterized
anchors. Span registers, languages, formats and personas rather than three
variants of one thing; log pairwise cosines, as in Step 2.

**Hedging is held out as a diagnostic, not trained on.** Its structured
component (~10 nats) is smaller than the cost of any system prompt. Training on
a vector whose correct output is the empty string teaches `T` that silence wins,
and GRPO will find that quickly. Keep it in the eval set as the known-negative
case.

**But keep subtlety in training.** Do not let the criterion collapse the
distribution to loud vectors only. Find at least two subtle vectors whose
structured component clears the threshold, or the Phase 3 generalization claim
is only about strong behaviors.

**Per-vector calibration.** Every vector gets the Step 3 treatment (α sweep,
coherence by eye, behavior metric) and a matched-norm floor from 8 directions.
A vector with F/KL0 near 75%, as pirate had at L18, is unusable regardless of
how strong the behavior looks.

---

## Architecture

Per `CLAUDE.md`: `T` is a fine-tune of the subject model family. `w` enters as a
conditioning input; the natural choice is to inject it at the residual stream of
an early layer, mirroring how the subject model receives it, but an adapter that
maps `w` to a soft prefix is the obvious alternative. **Pick one and record
why**; this is the first place where an architecture choice could silently
become the result.

A `T` that ignores `w` entirely and emits a generic prompt is a real failure
mode. Test for it directly: feed `T` two unrelated vectors and check the outputs
differ in substance, not just wording.

---

## Reward

```
reward = logprob of steered rollouts under q(·|S)      # the KL term
       − degeneration penalty                          # R5
       − readability penalty                           # P0
```

Grouped over candidates per vector, GRPO-style.

**Do not reward-shape toward naming the behavior.** The point is to find what
minimizes KL, and Phase 1b showed the KL-optimal prompt for French *loosens* the
instruction ("sauf si le langage technique l'exige") rather than strengthening
it.

**Known hacks to watch, all with Phase 1 precedent:**
- Degenerate-match (R5).
- Fluent-adversarial `S` that passes the judge (P0).
- Collapse to the empty string on any vector where presence cost dominates. If
  `T` emits empty or near-empty prompts across the board, the vector
  distribution is wrong, not the training.

---

## Evaluation

Held-out WildChat prompts throughout (`heldout[0:200]`), pinned ancestral
sampling, same as Phases 0–1.

**The baseline `T` must beat is rung 4, not the one-liner.** Rung 4 is an LLM
proposer with 6 rounds and 48 candidates per target. If `T` matches rung 4 at
a fraction of the inference cost, that is the amortization claim and it is worth
making. If `T` only beats the one-liner, say so plainly: the measured room above
a one-liner is 12 points on French and 5 on pirate.

Report per vector:

| Quantity | Why |
|---|---|
| KL vs better of rung 0/1 (R1) | the headline |
| KL vs floor (R2, R3 wording) | removes the norm effect |
| KL vs rung-4 best for that vector | the real baseline |
| distinct-token fraction | R5 |
| judge score + base perplexity of `S` | P0 |
| behavior metric of q(·\|S) samples | catches intensity overshoot |
| the emitted `S` itself | readable, or the project has lost its point |

**Diagnostics beyond the aggregate.** Run 2's token-level attribution for French
found that the best prompt wins all its gain in the first few tokens and gives
back ~2.7 nats over the tail. A `T` that learns the same trick is doing
something real; a `T` whose gain is uniformly diffuse may be exploiting
something else. Run per-token attribution on at least one trained output.

**Negative controls, both required:**
1. A random direction in the eval set. `T` should emit something near-empty or
   fail to beat rung 0, matching the rung-4 control.
2. Hedging. `T` should not beat the empty prompt. If it does, check for a reward
   hack before celebrating.

---

## Gate to Phase 3

1. P0 done: gate hardened and red-teamed.
2. `T` beats the better of rung 0/1 on held-out prompts for a majority of
   training vectors, without degeneration.
3. `T` is compared against rung-4 per vector, favorable or not.
4. Both negative controls pass.
5. Emitted prompts are readable, with examples in the writeup.

If (2) fails, that is a reportable result given Phase 1b: it would say the
achievable gains are small enough that amortization cannot capture them, which
follows from the measured marginal value of search.

---

## Writeup note

Phase 1b's finding about the intervention class is more interesting than
amortization alone and should lead any writeup, with Phase 2 as the test of
whether a learned generator closes the remaining gap. Two things to state
plainly: the L8 pirate vector has cosine 0.74 with the L18 one, so the layer
sweep produced a related but not identical behavior; and the prompt-matched
floor metric was computed and dropped because F_S mixes in behavior mismatch
between `S` and random-direction text, which is not unstructured cost. A
reviewer will propose that metric, so the methods section should pre-empt it.

The dual-use note in `CLAUDE.md` still applies and becomes more concrete here:
`T` is a trained generator of system prompts that reproduce arbitrary steering
directions.
