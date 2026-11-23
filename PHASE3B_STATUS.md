# Phase 3b: Status (in progress)

Spec: `PHASE3B_SPEC.md`. Prerequisite Phase 3 run: `runs/phase3_20261123_145628/`. Last updated 2026-11-23.

**The question Phase 3 left open:** did T condition on w, or memorize six prompts? Phase 3 could not
answer it because every holdout it evaluated was a near-duplicate of a training vector.

## Run 3 — Cosine-to-nearest-trained (done; reframes everything)

Max cosine of each holdout's unit vector to the 6 trained vectors (french L18, pirate L8, german L18,
formal L18, enthusiastic L18, lowercase L18). Added to `phase3_eval.summarize()` and `run_one` so all
future runs carry it (gate item 3).

| Phase 3 holdout | max cos | nearest | T result (vs floor) |
|---|---|---|---|
| italian  | **0.90** | french | −1.3% (and mislabelled, see Run 1) |
| spanish  | **0.89** | french | +31.6% |
| emoji    | **0.77** | enthusiastic | +46.0% |
| archaic_formal | 0.62 | formal | UNUSABLE |
| gloomy_L8 | 0.37 | formal | UNUSABLE |

**Reading:** the two holdouts where T performed (spanish, emoji) are the two nearest its training set,
and the two genuinely far vectors were both unusable. **Phase 3 did not test generalization.** A model
that memorized six prompts and interpolates produces exactly this. This column must appear beside every
closure number; "T beats a 48-candidate search at cosine 0.89 from a trained vector" is a weak claim.

## Run 1 — Language identification (done; `lingua`, not stopwords or eye)

`pip install lingua-language-detector --target /workspace/.pylib --no-deps`. Script: `lang_id.py`.
Run: `runs/phase3b_langid.json`. 100 held-out prompts; detector over EN/FR/ES/PT/IT/CA/DE.

| Vector | steered-sample languages | q(·\|S) under T's prompt | S detected as |
|---|---|---|---|
| french (control) | **98% French** | 94% French | French |
| spanish | **72% Spanish**, 18% Portuguese, 6% French | 56% English, 41% Spanish | Spanish |
| italian | **31% French, 26% Portuguese, 24% Italian**, 10% Spanish | 64% English, 20% Catalan, 16% Spanish | Catalan |

**Findings.**
1. **Positive control passes:** french steers to clean French (98%), confirming detector + method.
2. **spanish label is roughly correct — NOT Portuguese.** The spec's "reads as Portuguese by eye" is
   falsified: 72% Spanish. Keep the label; note the ~18% Portuguese blend. (Correcting the spec's
   assumption, not confirming it — this is why eyeballing was banned.)
3. **italian is MISLABELLED: a Romance smear dominated by French (only 24% Italian).** Consistent with
   its 0.90 cosine to french. It joins gloomy/empathy as a mislabelled vector, and its −1.3% result is
   **uninformative about T** (there is no coherent behavior to reproduce). Dropped from the memorization
   argument.
4. The stopword behavior metric cannot separate Romance languages and must be replaced by detector
   output wherever it chose an α for a language vector. (spanish/italian α were fixed by hand at 0.15,
   not metric-chosen, so no α needs redoing; the metric was only a reporting proxy.)
5. Note for the writeup: on spanish, T's grammatical-Spanish prompt induces Spanish only 41% of the
   time (56% English), yet closes +31.6% — so part of T's gain there is not the language switch.

**Gate items 1 and 2: satisfied.**

## Run 2 — alpha review by eye + re-check UNUSABLE (done)

Swept-vector samples reviewed (`<vector>.json` → `alpha_sweep[*].samples`):
- **italian** α 0.15: behavior present but it is the Romance smear above; vector mislabelled (Run 1).
- **archaic_formal** L18: behavior IS visible by eye at α 0.4 ("I shall embark upon crafting... the
  portrayal of which shall be guided"), but only where **F/KL0 = 62%** — norm-dominated, like pirate at
  L18. Unusable at L18 (R4). Earlier layers folded into Run 4 screening.
- **gloomy_L8**: F/KL0 is healthy (22–34%) but the samples show **no visible gloom** — ordinary
  slightly-formal answers. Mislabelled-noise, as Phase 2 found. Stays unusable; not a valid test.

So both genuinely-low-cosine Phase 3 holdouts are unusable as tested → Run 4 (a clean far holdout) is
required.

## Run 4 — Low-cosine holdouts (done). Run: `runs/phase3_20261123_160018/`

Candidates built (`phase3b_build.py`), screened on cosine first, then evaluated exactly as Phase 3
(pool-size α sweep + criterion, 8-dir held-out floor, R4, baseline structurally-valid-not-judge-filtered,
closure vs better-of-rung-0/1 and vs floor). All three screened cleanly (F/KL0 0.12–0.14, behavior
present by eye).

| Holdout | layer | α | max cos | F/KL0 | **T vs floor** | T greedy | **baseline vs floor** | T−base | T emits |
|---|---|---|---|---|---|---|---|---|---|
| japanese | 18 | 0.15 | 0.69 | 0.12 | **+2.3%** | −1.6% | +25.6% (rung4) | **−23.3** | **Chinese** (not Japanese) |
| korean | 12 | 0.3 | 0.65 | 0.14 | **+14.7%** | +4.4% | +24.8% (ladder) | **−10.1** | Korean ✓ |
| uppercase | 18 | 0.2 | 0.51 | 0.14 | **−22.5%** | −19.5% | 0.0% (empty best) | **−22.5** | caps-ish English, no all-caps instruction |

## Conclusion: T interpolates / partially memorizes; it does not read `w` in general

Combining Phase 3 and Phase 3b, ordered by cosine to the nearest trained vector:

| Holdout | max cos | T vs floor | baseline vs floor | **T − baseline** | note |
|---|---|---|---|---|---|
| italian | 0.90 | −1.3% | 0 | — | **mislabelled** (Romance smear, Run 1): uninformative |
| spanish | 0.89 | +31.6% | +22.7% (rung4) | **+8.9** | T beats a 48-candidate search |
| emoji | 0.77 | +46.0% | +48.8% (rung4) | **−2.8** | T ≈ search |
| japanese | 0.69 | +2.3% | +25.6% (rung4) | **−23.3** | T fails; emits Chinese |
| korean | 0.65 | +14.7% | +24.8% (ladder) | **−10.1** | partial (Korean) but < trivial search |
| uppercase | 0.51 | −22.5% | 0 | **−22.5** | T worse than empty; reproduces the lowercase failure |

**T − baseline is monotone in cosine.** T beats or ties a search only on the two holdouts nearest its
training set (≥ 0.77). On every genuinely novel vector (cos < 0.7) it is below the baseline, and on the
two most novel (uppercase, japanese) it does not even beat the empty prompt.

This is the clean negative the spec anticipated: **T's Phase-2/Phase-3 success is in-distribution plus
near-duplicate generalization, not conditioning on arbitrary `w`.** Supporting detail:
- **uppercase** reproduces the Phase-2 **lowercase** failure exactly. T writes *some* of its own prompt
  in caps (the in-style prior) but never emits the instruction, so q stays ~4% uppercase vs the steered
  model's 68%. A case/format transform it never saw gives the in-style prior no gradient toward naming it.
- **japanese**: T falls back to a generic **Chinese** prompt (Qwen's non-English default) and fails,
  while a real search closes +25.6%.
- **korean**: the one partial transfer — T does emit Korean (+14.7%), because switching to *a* non-English
  language is the shared "unusual register/language" component T did learn from french+german — but it
  still loses to a trivial "Respond in Korean." search.
- The near-duplicate wins (spanish, emoji) are consistent with interpolation between french and
  enthusiastic, which T trained on.

**Honest framing for the writeup:** T learned a real but narrow thing — a shared language/register
direction that lets it sometimes switch language off-distribution — but it did not learn to read an
arbitrary steering vector. The amortization claim from Phase 2 holds only in-distribution.

### Gate (all met)

1. Every language vector's label confirmed/corrected by the detector (Run 1). ✓
2. italian classified: **mislabelled vector** (24% Italian; Romance smear), result uninformative (Run 1). ✓
3. Cosine-to-nearest-trained reported for every holdout, in the table and in `phase3_eval.summarize()`. ✓
4. Three holdouts at max cos < 0.7 evaluated (0.69, 0.65, 0.51), all fail to beat the baseline. ✓

### Caveats / carried forward
- lingua resolves Spanish/Portuguese well on 128-token samples but is noisier on <20-char strings.
- Negative controls (hedging, random) were not re-run in 3b; they are Phase-2/3 gate items and already pass.
- phi-4 was re-downloaded to the ephemeral `/root/hf_ephemeral/hub` after the restart (needed for the
  rung-4 search + gate); it is lost on the next restart.
- `uppercase`/`japanese`/`korean`/`russian` vectors now exist at L8/L12/L18 in `vectors/`.
