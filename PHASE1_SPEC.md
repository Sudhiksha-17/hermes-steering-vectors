# Phase 1 — The Per-Vector Oracle

Status: not started. Phase 0 PASSED 2026-11-20 (see `PHASE0_REPORT.md`).

## The question Phase 1 answers

For a **fixed, meaningful** steering vector `w`, does there exist a system
prompt `S` such that the unsteered model conditioned on `S` behaves like the
steered model? And how close can it get?

This is the single most important phase. Without it, a high KL in Phase 2 is
ambiguous between "the amortized model `T` is bad" and "no such prompt exists
for this vector." Phase 1 produces the floor that disambiguates them.

Phase 1 does **not** train anything. It is a search over candidate prompts for
one vector at a time.

## Carried over from Phase 0 — read before writing code

Three Phase 0 findings change how Phase 1 must be built.

**1. KL is non-monotonic in steering strength.** KL peaked near α≈2 and *fell*
to 79 nats at α=4, because the steered model degenerated into repetition that
the baseline predicts easily (distinct-token fraction 0.03 vs baseline 0.68).
KL measures surprise, not intervention strength. Consequence: **every KL number
in this phase is reported alongside a degeneration metric** (distinct-token
fraction, or token entropy). A KL improvement accompanied by a collapse in
distinct-token fraction is not an improvement.

**2. The system-prompt baseline is confounded and must be fixed first.**
Phase 0's "no system prompt" condition used Qwen's chat template with no system
message, which silently inserts Qwen's *default* system prompt. If `p_steer`
runs with that default and `q(·|S)` replaces it with `S`, then the measured KL
reduction conflates two changes: removing the steering, and swapping one system
prompt for another.

> **First task of Phase 1, before any oracle work:** make the system-prompt
> slot explicit and identical across `p_steer` and `q`. `p_steer` should run
> with an explicitly empty (or explicitly fixed neutral) system prompt, and
> `q(·|S)` should differ from it in exactly one respect: the content of `S`.
> Re-run the Phase 0 α sweep after this change and confirm the numbers are
> stable. Document any shift.

**3. Gate α = 0.25 does not transfer.** That value was calibrated for an
isotropic Gaussian vector, which is near-orthogonal to anything the model
represents. Difference-of-means vectors live in a structured subspace and will
be far more potent per unit norm. **Calibrate α per vector**, targeting a
KL-per-token in the 0.02–0.15 range (Phase 0's length-invariant restatement of
the 2–20 nat gate), and confirm coherence by eye at the chosen α.

## Vectors to build

Use difference-of-means (CAA-style) over contrast prompt pairs. Three vectors,
chosen to span the difficulty range:

| # | Behavior | Why this one |
|---|---|---|
| 1 | Respond in French | **Sanity check.** Near-deterministic, obviously nameable, and a one-line prompt should nail it. If the oracle fails here, the oracle is broken. |
| 2 | Pirate speak / persona | Strong style transform, known to work in the steering literature, moderate difficulty. |
| 3 | **A subtle one** — hedging/uncertainty, verbosity, or formality gradient | **The real test.** Graded rather than binary, harder to name in one line. This is where the expressiveness gap between additive steering and a prefix system prompt should show up, if it shows up anywhere. |

Vectors 1 and 2 are easy wins and will likely both succeed. Do not let that be
the headline. **Vector 3 is the informative one** — report it with equal
prominence even if the result is negative.

Store each vector with its contrast pairs, layer, and calibrated α, in
`vectors/`, alongside the Phase 0 Gaussian for reference.

## Candidate prompts — the ladder

For each vector, evaluate this ladder in order. The ladder *is* the trivial
baseline check; it is not optional.

| Rung | Candidate `S` | Purpose |
|---|---|---|
| 0 | Empty / fixed neutral system prompt | The floor. Any `S` must beat this. |
| 1 | A generic unrelated instruction (e.g. "You are a helpful assistant.") | Controls for *having* a system prompt at all, separately from its content. |
| 2 | **The obvious one-liner** naming the behavior ("Respond in French.") | **The key baseline.** |
| 3 | 2–3 hand-written elaborate prompts | Does engineering effort buy anything over rung 2? |
| 4 | LLM-proposed candidates (loop) | Only after rungs 0–3 are measured. |

**The rung-2 result is a finding in its own right.** If the obvious one-liner
captures most of the achievable KL reduction, that bounds how much the
amortized `T` in Phase 2 could ever add, and it needs to be stated plainly
rather than buried. A Phase 2 that beats rung 0 but not rung 2 has not
demonstrated anything.

## Protocol

- **Held-out prompts only** for all reported numbers. Use the 500-prompt
  held-out split from `data/wildchat_prompts.json`. Tune on train, report on
  held-out.
- Same pinned ancestral sampling as Phase 0 (temperature 1, top_k 0, top_p 1,
  repetition penalty 1). Never inherit Qwen's generation defaults.
- Report mean ± SE over rollouts, as in Phase 0.
- Report KL both sequence-level and per-token.
- Report distinct-token fraction for every condition (see carried-over
  finding 1).
- Report readability of `S` from the very first run, not retrofitted later.
  For hand-written candidates this is trivially "yes"; the column exists so
  that it is already in place when rung 4 and Phase 2 start producing text that
  may not be readable.
- Log every run's full config and dump samples, same as Phase 0.

## Reporting format

One table per vector, rows = ladder rungs:

| Rung | `S` | KL (seq) ± SE | KL/token | distinct-tok frac | readable? |
|---|---|---|---|---|---|

Plus, per vector, a single headline number: **fraction of the rung-0 KL closed
by the best `S`**, with the rung-2 one-liner's fraction stated alongside it.

## Gate to pass before Phase 2

1. The system-prompt confound (carried-over finding 2) is fixed and documented.
2. For at least one vector, some `S` achieves a substantial KL reduction versus
   rung 0 **on held-out prompts**, without degeneration.
3. The rung-2 one-liner baseline is measured for every vector, so the marginal
   value of search is known.
4. Vector 3 (the subtle behavior) is reported, pass or fail, with an honest
   read on whether the gap is a search failure or an expressiveness limit.

If (2) fails on all three vectors, do not proceed to Phase 2. That result says
the prompt-equivalence hypothesis is wrong for this model and layer, which is
itself worth writing up and worth telling Francisco quickly.

## Note for Phase 2 (record now, act later)

The non-monotonic KL finding implies a reward hack in Phase 2: GRPO may find an
`S` that makes the *unsteered* model degenerate in the same way an over-steered
model does, scoring well on KL while explaining nothing. The degeneration
metric must enter the Phase 2 **reward as a penalty**, not merely the reported
diagnostics. This is the same class of problem as the gibberish hack, by a
different mechanism.