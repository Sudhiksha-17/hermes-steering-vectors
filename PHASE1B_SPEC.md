# Phase 1b — Post-control runs

Status: DONE 2026-11-23 (runs 1–4; results in `PHASE1_STATUS.md`, Phase 1b). Prerequisite: Step 5 (matched-norm control) DONE, see
`PHASE1_STATUS.md`.

Four runs, **in this order**. The order matters: run 4 (rung 4) is last because
its target depends on what runs 1–3 establish. Do not reorder.

---

## Protocol rules adopted from Phase 1 (apply to every run below and to Phase 2)

These are promoted from findings to rules. They are not optional.

**R1. Rung 1 is the baseline for subtle vectors, not rung 0.** A generic system
prompt alone moves the KL: −3.5% (French), +1.7% (pirate), −20.6% (hedging). For
hedging the generic prompt costs 41% of the closable gap. Any claim that an `S`
helps must clear rung 1, not just rung 0. Report both.

**R2. Report closure against the floor alongside closure against rung 0.** Raw
closure understates what prompts achieve on high-norm vectors and is
uninformative where the structured component is small.

**R3. Closure-vs-floor is not a point estimate, and the evidence says it is conservative.**
*(Revised 2026-11-22. The original wording called it an "upper bound" because F plausibly
underestimates the off-target cost of a structured vector. That reasoning gives a lower bound:
a larger true floor F′ shrinks the denominator (KL0 − F′), which raises true closure. See
`PHASE1_STATUS.md`, Phase 1b run 2 note.)* The figure rests on structured and unstructured KL
adding, which is not established. Evidence so far: run 1 found F not closable by any tested
prompt, and run 2 found the off-target cost of a structured vector (French) exceeding F. Both
suggest closure-vs-floor *understates* closure. Word every closure-vs-floor number as
conservative under the additivity assumption, never as a point estimate.

**R4. Do not report a closure fraction when the best `S` falls inside the
spread of the random floor.** Report it as indeterminate and say why. See run 3.

---

## Run 1 — Validate the floor (do this first)

**This is not optional.** It was listed as "optional, cheap" in the Phase 1
next steps. It is the load-bearing assumption of all of Step 5: `F` is defined
as unclosable, and every closure-vs-floor denominator `(KL0 − F)` rests on that.

**Question.** Do the ladder prompts reduce `KL(p_rand ‖ q_S)` for random
directions?

**Method.** Reuse the 8 random directions from
`runs/phase1_randfloor_20261122_231649.json` and the rollouts already generated
for them. For each direction, score rungs 0–3 exactly as in Step 4. No new
generation is needed for the steered side.

**Interpretation.**
- If no prompt meaningfully reduces KL against a random direction, the floor is
  genuinely unclosable and Step 5's numbers stand.
- If prompts *do* reduce it, `F` is partly closable, every closure-vs-floor
  figure in Step 5 is inflated, and they must be recomputed or withdrawn.
  Pirate's 84% is the most exposed.

**Gate.** Do not circulate any closure-vs-floor number outside the team until
this run is reported, pass or fail.

---

## Run 2 — Per-token KL attribution for French

Promoted from optional. French is the one vector where the behavior is fully
reproduced (99% French, reverse KL down 97%) and ~10 nats of structured KL
remain unexplained. That residual is the most interpretable quantity in Phase 1
and it directly informs Phase 2.

**Question.** Are the residual ~10 nats concentrated on content tokens, or
spread uniformly across the sequence?

**Method.** For the best French `S` (rung 3a or 3b), decompose the per-token KL
over held-out rollouts. Report the distribution, not just the mean: top-k
highest-KL token positions, KL by position index, and KL grouped by a coarse
token class (content word, function word, punctuation, number).

**Interpretation.**
- **Concentrated on content tokens** → steering changes *what* the model says,
  not only *how*. That is a substantive claim and gives Phase 2 a target that no
  style instruction reaches.
- **Spread uniformly** → more likely residual off-target cost, which means `F`
  underestimates the true unstructured component for structured vectors. This is
  exactly the R3 caveat, and it would weaken the closure-vs-floor framing.

Both outcomes are informative. Report whichever occurs.

---

## Run 3 — Pirate layer sweep

Pirate at layer 18 is not a usable test vector and **its 84% closure figure
should not be reported** (rule R4). The best prompt lands at 127 nats inside a
random floor spanning 87–147: for some random directions, the pirate prompt is
further from the steered model than noise is. The quantity is close to
undefined, not merely imprecise. Report pirate at layer 18 as **indeterminate:
floor too large and too variable to measure closure**.

**Method.** Rebuild the pirate vector at layers 8, 10, 12, 14. For each, run the
α calibration from Step 3 and pick α by the Step 3 criterion (behavior clearly
present, text coherent). Look for a layer where the pirate lexicon appears
*before* fluency degrades and before the norm cost dominates.

**Selection criterion for the layer.** Maximise the closable gap `KL0 − F`
relative to `KL0`, subject to coherent steered text. A layer where `F/KL0` is
near 75% is unusable regardless of how strong the behavior looks.

**Then** re-run the Step 4 ladder and the Step 5 control at the chosen layer.

If no layer in 8–14 gives a workable window, say so and drop pirate as a test
vector rather than reporting a number from layer 18.

---

## Run 4 — Rung 4: LLM-proposed candidates

Last, because the target depends on runs 1–3.

**Proposer.** Local Qwen2.5-14B-Instruct, alongside the 3B subject model on the
H200. It sees steered samples, neutral samples, and previous candidates with
their train-set KL.

**Selection.** Candidates scored on train (`train[64:192]`), best reported on
held-out (`heldout[0:200]`). The proposer never sees held-out data.

**Primary target: intensity calibration.** This is the Phase 1 finding that
gives rung 4 a concrete job. Prompts overshoot the steered behavior's intensity
on all three vectors: prompted pirate 6–9 words per 100 against steered 3.7;
prompted hedging up to 2.9 against steered 1.4. For hedging specifically, the
one-liner costs 108% of the closable gap, so **the task is not to name the
behavior but to dial it down**. Probe hedged, partial and indirect instructions
("occasionally", "slightly", "when it is natural to do so"), and prompts that
name the behavior without commanding it.

**Hedging is the vector to prioritise.** Its structured component is ~10 nats
and every hand-written prompt lands further away than silence. A prompt that
beats rung 0 there would be the first evidence that the gap is search failure
rather than an expressiveness limit. A thorough negative is equally publishable:
it would show prompts cannot address a small structured effect at all.

### Readability gate — decide BEFORE this run

Rung 4 is the first phase whose candidates are not human-authored, so the
perplexity gate becomes load-bearing here.

The current gate (perplexity under the subject model with a framing prefix,
threshold 2,000) catches crude token soup and nothing else. A fluent adversarial
`S` passes it, and that is the space an LLM proposer and later GRPO will search.

**Required before run 4 starts:** a second gate that does not reduce to subject-
model perplexity. Options: an LLM judge scoring "is this a sensible instruction
a person might write", or a fluency penalty against a *separate* base model.
Report both the gate score and the KL for every rung-4 candidate, so a candidate
that wins on KL while failing readability is visible rather than silently
selected.

---

## Reporting

Update `PHASE1_STATUS.md` after each run, not at the end.

For each run, record: the question, the method, the result, and whether it
changes any earlier finding. Step 5 revised findings 4 and 5 from Step 4; that
kind of revision is the normal output here and should be stated explicitly
rather than silently overwriting the earlier text.

Log config and seeds with every run, as in Phases 0 and 1.

---

## Gate to Phase 2

1. Run 1 reported. If the floor fails validation, Step 5's closure-vs-floor
   numbers are recomputed or withdrawn before anything proceeds.
2. Pirate either has a workable layer (run 3) or is dropped as a test vector.
   No layer-18 closure figure is reported.
3. Rung 4 reported for hedging, pass or fail, with a read on search failure
   versus expressiveness limit.
4. The stronger readability gate exists and has been applied to rung-4 output.

Phase 2's reward must include a degeneration penalty (carried from Phase 0) and
must be measured against rung 1 (R1). Both are already known; neither is
optional.
