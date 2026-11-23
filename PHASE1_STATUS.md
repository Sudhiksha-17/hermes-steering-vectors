# Phase 1: Status. Rungs 0–3 DONE; matched-norm control DONE; Phase 1b DONE (runs 1–4); Phase 2 gate met, with open questions

**Bottom line.** Part of every vector's KL comes from adding *any* vector of that norm, and no tested prompt closes that part (Phase 1b run 1). Measured against that unstructured-perturbation floor (step 5; validated in Phase 1b run 1; conservative under the additivity assumption, per revised R3):
- **French:** the best prompt closes **59% ± 3%** of the closable gap (42% of raw KL).
- **Pirate:** indeterminate at layer 18 (rule R4; the 84% figure is withdrawn). At **layer 8** (Phase 1b run 3) the floor is only 10% of KL0, and the best prompt closes **17% ± 1%** of the closable gap: a determinate negative. The steered register is variable and theatrical, and pirate prompts overshoot it.
- **Hedging:** half of the KL is the norm effect. The other ~10 nats are structured, and every prompt moves *away* from them: the one-liner adds more KL than the entire structured component.

Across all three vectors, prompts overshoot the steered behavior's intensity.

**After rung 4 (Phase 1b run 4):**
- French best 67% vs floor (47.5% vs rung 0).
- Pirate (L8) best 20%.
- Hedging: still no prompt beats silence, even intensity-matched ones; having any system prompt costs ~40% of its signal.
- A random-direction negative control could not be fitted by the search.

Spec: `PHASE1_SPEC.md`, `PHASE1B_SPEC.md`. Last updated 2026-11-23.

## Gate checklist

| # | Gate item | Status |
|---|---|---|
| 1 | System-prompt confound fixed and documented | **DONE.** See `PHASE0_STATUS.md` addendum. |
| 2 | Some `S` substantially reduces KL against rung 0 on held-out prompts, without degeneration | **PASS (French).** Forward KL closed 41.6% ± 2.3% (best `S` chosen on train); reverse KL 695 → 17; 99% of responses French; distinct fraction 0.76, the same as the steered model. Pirate closes 21%. |
| 3 | Rung-2 one-liner measured for every vector | **DONE.** French 36.3%, pirate 20.1%, hedging −54.5% |
| 4 | Vector 3 (hedging) reported, with a read on search failure versus expressiveness limit | **Reported: NEGATIVE.** The matched-norm control (step 5) shows it is not pure noise: about 10 nats are structured and no hand-written prompt captures them. Rung 4 is still needed to separate search failure from an expressiveness limit. |

The gate is technically met. The matched-norm control is done (step 5). **Run rung 4 before Phase 2.**

## Step 1: Confound fix (done)

- `p_steer` and `q(·|S)` now share an explicit system block. The neutral prompt is the empty string, and the two contexts differ only in the system text.
- The Phase 0 sweep was re-run with this setup and is stable. At the gate α, KL went from 15.1 ± 0.8 to 16.2 ± 0.9, and no row shifted by more than about 9%.
- Run: `runs/phase0_20261120_234449.json`.

## Step 2: Vectors (done)

**Code:** `build_vectors.py` (construction and α sweep) and `behaviors.py` (elicitation prompts, ladder prompts, behavior metrics).

**Method.** Difference of means, CAA-style, at layer 18 of 36.

1. For each of 128 train prompts (`train[500:628]`), sample a 96-token response twice:
   - once under an eliciting system prompt (positive);
   - once under the empty system prompt (negative).
2. Re-encode both responses under the **empty** system prompt.
3. Average the layer-18 residual stream over **response tokens only**. The vector is mean(positive) − mean(negative).

**Circularity caveat.** The positive responses come from prompting, so the steered behavior is distilled from prompted behavior. A behavior-matching prompt (rung 2) has a home-field advantage. To reduce this, the eliciting prompts are worded differently from the rung-2 one-liners; for French, the eliciting prompt is itself written in French. Even so, rung-2 success is partly expected and must be read with this in mind.

| Vector | File | Raw ‖v‖ (× resid norm 71.9) | Behavior metric: positive / negative responses |
|---|---|---|---|
| French | `vectors/french_L18.pt` | 14.8 (0.21) | fraction in French: 0.94 / 0.00 |
| Pirate | `vectors/pirate_L18.pt` | 16.9 (0.23) | pirate words per 100: 9.25 / 0.07 |
| Hedging | `vectors/hedging_L18.pt` | 4.1 (0.06) | hedges per 100 words: 2.45 / 0.91 |

**Cosine similarities.**

- French and pirate: +0.31. Both encode "non-standard register or language".
- All other pairs: |cos| ≤ 0.05, including each vector against the Phase 0 Gaussian.

**Behavior metrics** are lexical heuristics, used for calibration and sanity checks only:

- French: fraction of responses where French stopwords outnumber English ones.
- Pirate: pirate-lexicon hits per 100 words.
- Hedging: hedge expressions per 100 words.

## Step 3: α calibration (done)

**Setup.**

- Run: `runs/phase1_calib_20261120_234651.json` (stdout: `runs/phase1_calib_stdout.log`).
- 64 train prompts (`train[0:64]`), 128 tokens each, empty system prompt.
- α is a multiple of the median residual norm at layer 18 (71.9).
- Baseline distinct-token fraction: 0.700.

**French** (baseline metric: 0.00)

| α | KL (seq) ± SE | KL/tok | distinct | frac French |
|---|---|---|---|---|
| 0.05 | 0.56 ± 0.03 | 0.005 | 0.691 | 0.00 |
| 0.10 | 17.82 ± 2.00 | 0.147 | 0.725 | 0.25 |
| 0.15 | 29.69 ± 1.44 | 0.245 | 0.758 | 0.95 |
| **0.20** | **34.62 ± 1.55** | **0.280** | **0.755** | **1.00** |
| 0.30 | 43.52 ± 2.06 | 0.356 | 0.747 | 1.00 |
| 0.40 | 55.33 ± 2.07 | 0.445 | 0.753 | 1.00 |
| 0.60 | 89.31 ± 2.91 | 0.721 | 0.750 | 1.00 |
| 0.80 | 144.93 ± 3.90 | 1.160 | 0.768 | 0.98 |

**Pirate** (baseline metric: 0.07)

| α | KL (seq) ± SE | KL/tok | distinct | pirate / 100w |
|---|---|---|---|---|
| 0.05 | 0.98 ± 0.05 | 0.008 | 0.696 | 0.07 |
| 0.10 | 3.85 ± 0.20 | 0.032 | 0.694 | 0.13 |
| 0.15 | 9.34 ± 0.44 | 0.078 | 0.699 | 0.09 |
| 0.20 | 17.59 ± 0.97 | 0.149 | 0.701 | 0.18 |
| 0.30 | 41.49 ± 2.07 | 0.355 | 0.711 | 0.16 |
| 0.40 | 79.97 ± 3.91 | 0.678 | 0.723 | 0.74 |
| **0.60** | **153.98 ± 5.61** | **1.357** | **0.744** | **4.36** |
| 0.80 | 210.00 ± 5.78 | 1.774 | 0.742 | 5.86 |

**Hedging** (baseline metric: 1.06)

| α | KL (seq) ± SE | KL/tok | distinct | hedges / 100w |
|---|---|---|---|---|
| 0.05 | 1.17 ± 0.07 | 0.010 | 0.697 | 1.30 |
| 0.10 | 4.67 ± 0.22 | 0.038 | 0.697 | 1.42 |
| 0.15 | 11.18 ± 0.58 | 0.091 | 0.705 | 1.72 |
| **0.20** | **22.29 ± 1.07** | **0.177** | **0.702** | **1.86** |
| 0.30 | 63.83 ± 2.29 | 0.506 | 0.716 | 2.23 |
| 0.40 | 126.35 ± 3.64 | 1.000 | 0.711 | 2.35 |
| 0.60 | 221.11 ± 4.94 | 1.821 | 0.707 | 3.67 |
| 0.80 | 300.25 ± 6.05 | 2.397 | 0.634 | 4.24 |

### Chosen α (recorded in each vector file and in `build_vectors.CALIBRATED_ALPHA`)

| Vector | α | ‖w‖ | KL/tok | In 0.02–0.15 band? | Why |
|---|---|---|---|---|---|
| French | 0.20 | 14.4 | 0.280 | **no**, above | Smallest α with 100% French and fluent text. No α inside the band produces the behavior: at 0.10, only 25% of responses are French. |
| Pirate | 0.60 | 43.2 | 1.357 | **no**, far above | Pirate speech appears only at α ≥ 0.6. At 0.4 the steering only adds enthusiasm ("It's super simple!"). **Coherence is borderline**: some words garble ("taon'tinear", "hoardsaft"). α = 0.8 is gibberish. |
| Hedging | 0.20 | 14.4 | 0.177 | slightly above | Coherent, and the hedge rate rises 75% over baseline. From α = 0.3, an artifact appears: responses start with a lowercase letter. |

### Calibration findings

1. **The 0.02–0.15 KL/tok band does not fit meaningful vectors.** The band came from a random vector, where KL is the only signal. For a real behavior, the KL needed to express it depends on the behavior: switching language costs about 0.28 nats per token. Calibration was therefore done on "behavior clearly present and text coherent", with KL/tok recorded. This departs from the spec, deliberately.
2. **The pirate vector is weak at layer 18.** Its usable window is narrow or absent: by the time the lexicon appears, fluency is degrading. Expect a large KL floor for pirate that partly reflects degraded steered text rather than an expressiveness gap. **Follow-up:** sweep earlier layers (for example 8–14) for pirate.
3. **Hedging is subtle, as intended.** The raw difference-of-means norm is about 4× smaller than for the other two vectors. In the samples, the visible effect at α = 0.2 is mild: longer, more qualified sentences.
4. **No degeneration at the chosen α values.** Distinct-token fractions are 0.70–0.76 against a baseline of 0.70.

### Samples at the chosen α

- **French, 0.20:** *"Crowdstrike Falcon est un produit de sécurité antivirus avancé intégré au système de logiciels antivirus FireEye. Voici cinq alternatives…"*. The text is fluent, though the facts are hallucinated.
- **Pirate, 0.60:** *"Sure, here be a ye olden way to do it, say in ye Lua: …"*, and also *"…this fine oak-built ship's lantern of a Billy-boat still find the fiddly hoardsaft of 'barth…"*.
- **Hedging, 0.20:** *"Retiring a product like CrowdStrike Falcon will require a careful assessment of the specific security measures and features you aim to preserve. However, given that…"*

## Step 4: Ladder evaluation, rungs 0–3 (done)

**Runs.** `runs/phase1_ladder_20261120_235111.json` (stdout: `runs/phase1_ladder_stdout.log`). Code: `oracle_search.py`.

**Prompts.**
- Every number below uses 200 held-out prompts (`heldout[0:200]`), 128 tokens each.
- The "best S" is chosen on forward KL over a separate train set (`train[64:192]`), so the choice never sees held-out data.

**Columns.**
- **KL fwd** = KL(p_steer ‖ q_S), on one shared set of steered rollouts. Because the rollouts are shared, the "closed" column is a paired difference against rung 0.
- **KL rev** = KL(q_S ‖ p_steer), on q_S's own samples.
- **distinct** and the behavior metric are computed on q_S's own samples.
- **ppl(S)** is the perplexity of S, conditioned on a framing prefix. The perplexity values were recomputed after the run with the fixed readability metric; the KL numbers are unaffected (see the readability note below).

### Vector 1: French (α 0.20). Steered target: distinct 0.753, frac French 0.98

| Rung | S | KL fwd (seq) ± SE | KL/tok | closed ± SE | KL rev (seq) ± SE | distinct | frac French | ppl(S) | readable? |
|---|---|---|---|---|---|---|---|---|---|
| 0 neutral | `""` | 34.19 ± 1.16 | 0.279 | 0 | 695.0 ± 18.7 | 0.700 | 0.01 | — | yes |
| 1 generic | You are a helpful assistant. | 35.39 ± 1.12 | 0.289 | −3.5% ± 0.6% | 686.7 ± 17.3 | 0.696 | 0.01 | 14 | yes |
| 2 one-liner | Respond in French. | 21.78 ± 1.10 | 0.178 | **+36.3% ± 2.3%** | 31.5 ± 6.4 | 0.767 | 0.97 | 118 | yes |
| 3a elaborate | You are a helpful assistant. Always write your entire response in French… | **19.69 ± 1.00** | 0.161 | +42.4% ± 2.3% | **16.7 ± 0.6** | 0.762 | 0.99 | 60 | yes |
| 3b elaborate | Réponds toujours en français, quelle que soit la langue… | 19.96 ± 1.11 | 0.163 | +41.6% ± 2.3% | 59.3 ± 11.8 | 0.759 | 0.91 | 17 | yes |

**Headline (French).** The best S, rung 3b (selected on train), closes **41.6% ± 2.3%** of the rung-0 forward KL. The rung-2 one-liner closes **36.3% ± 2.3%**.

### Vector 2: Pirate (α 0.60). Steered target: distinct 0.740, pirate words per 100 words 3.73

| Rung | S | KL fwd (seq) ± SE | KL/tok | closed ± SE | KL rev (seq) ± SE | distinct | pirate/100w | ppl(S) | readable? |
|---|---|---|---|---|---|---|---|---|---|
| 0 neutral | `""` | 161.75 ± 3.31 | 1.388 | 0 | 146.3 ± 4.1 | 0.700 | 0.06 | — | yes |
| 1 generic | You are a helpful assistant. | 158.95 ± 3.18 | 1.364 | +1.7% ± 0.3% | 151.8 ± 4.1 | 0.696 | 0.02 | 14 | yes |
| 2 one-liner | Talk like a pirate. | 129.26 ± 2.61 | 1.109 | **+20.1% ± 1.0%** | 126.4 ± 2.5 | 0.730 | 5.85 | 474 | yes |
| 3a elaborate | You are a helpful assistant who speaks like a swashbuckling pirate captain… | 133.01 ± 2.62 | 1.141 | +17.8% ± 1.2% | 134.3 ± 2.2 | 0.724 | 8.97 | 9 | yes |
| 3b elaborate | Ye be a pirate, matey! Answer every question as a salty sea dog would… | **127.18 ± 2.56** | 1.091 | +21.4% ± 1.1% | **126.0 ± 2.3** | 0.721 | 7.04 | 48 | yes |

**Headline (pirate).** The best S, rung 3b, closes **21.4% ± 1.1%**. The rung-2 one-liner closes **20.1% ± 1.0%**.

### Vector 3: Hedging (α 0.20). Steered target: distinct 0.715, hedges per 100 words 1.37

| Rung | S | KL fwd (seq) ± SE | KL/tok | closed ± SE | KL rev (seq) ± SE | distinct | hedges/100w | ppl(S) | readable? |
|---|---|---|---|---|---|---|---|---|---|
| 0 neutral | `""` | **19.96 ± 0.46** | 0.164 | 0 | **17.0 ± 0.4** | 0.700 | 0.83 | — | yes |
| 1 generic | You are a helpful assistant. | 24.08 ± 0.51 | 0.198 | −20.6% ± 1.0% | 20.0 ± 0.5 | 0.696 | 0.83 | 14 | yes |
| 2 one-liner | Express uncertainty and hedge your claims. | 30.84 ± 0.94 | 0.253 | **−54.5% ± 4.3%** | 39.1 ± 1.9 | 0.717 | 1.34 | 509 | yes |
| 3a elaborate | You are a careful, humble assistant. You are frequently unsure… | 28.94 ± 0.71 | 0.238 | −45.0% ± 2.8% | 34.8 ± 1.2 | 0.694 | 2.88 | 27 | yes |
| 3b elaborate | Answer helpfully but tentatively. Acknowledge that you may be mistaken… | 29.54 ± 0.81 | 0.243 | −48.0% ± 3.2% | 29.0 ± 0.9 | 0.696 | 1.49 | 130 | yes |

**Headline (hedging).** The best S (selected on train) is **rung 0, the empty prompt**: no candidate closes any of the gap. The rung-2 one-liner *widens* the gap by **54.5% ± 4.3%**.

### Example: "give the working principle for hamming code used for error detection"

- **French, steered:** *"Leonsde Hamming a contribué au développement des technologies de codage correctionnel…"*
- **French, 3a:** *"Le code de Hamming, nommé aussi code de Hamming, est une technologie basée sur du codage…"*
- **French, 3b:** *"Le code de Hamming est un code纠错码，它主要用于检测错误…"*. The prompt written in French sometimes triggers a switch into Chinese, which explains 3b's high, noisy reverse KL.
- **Pirate, steered:** *"First, yeet your data onto the heap… (You could yahttwo 7 + 2 for yer parity bits if ye wish.)"*. This is mild slang with garbled words.
- **Pirate, one-liner:** *"Ahoy matey! The Hamming code is a clever method… for our sailing buccaneers."*. This is theatrical, full-costume pirate.
- **Hedging, steered:** *"…The key idea behind Hamming codes lies in arbitrarily chosen positions of the data bits…"*. There is no visible hedging.
- **Hedging, 3a:** *"…allows some kind of error detection."*. The hedging is explicit and overshoots.

### Findings

1. **The confound control matters.** Simply having a system prompt moves the KL: "You are a helpful assistant." shifts it by −3.5% (French), +1.7% (pirate) and **−20.6% (hedging)**. For hedging, the effect of the generic prompt (+4 nats) is a fifth of the whole steering effect (20 nats). Any Phase 2 result on subtle vectors must be measured against rung 1 as well as rung 0.
2. **Prompts overshoot the intensity of the steered behavior, for all three vectors.** Prompted pirate text has 6–9 pirate words per 100, against 3.7 for the steered model. Prompted hedging reaches up to 2.9 hedges per 100 words, against 1.4. Prompted French is cleaner French than the steered output. Steering at a calibrated α is a *light-touch* shift with side effects. A system prompt naming the behavior produces the *full-costume* version. This is the main qualitative result so far, and it gives rung 4 a concrete target: prompts that set intensity ("occasionally", "slightly").
3. **For French, the behavior is reproduced but the forward KL plateaus at about 20 nats (0.16 per token).** Reverse KL falls 97%, from 695 to 17: the prompted model produces almost nothing the steered model would not. But the steered model's French has its own content: different word choices and more hallucination ("Leonsde Hamming"), which no instruction reproduces. One candidate explanation is that part of the residual KL is a **non-semantic perturbation from injecting a vector of that norm at all**. In Phase 0, a random vector of similar norm (α 0.20 × 71.5 ≈ 14.3) gave about 10 nats. This is untested; the matched-norm control in Next steps tests it.
4. **Pirate's small closure is confounded by its calibration.** At α 0.6 the steered text is partly garbled, and no readable prompt can reproduce garbling. The pirate floor is therefore mostly a property of the vector and layer, not evidence about prompt expressiveness.
5. **Hedging: search failure or expressiveness limit? Provisional read: mainly neither. The vector carries little hedging and the effect is close to the noise scale.**
   - At α 0.2 the steered hedge rate is 1.37 per 100 words, against 0.83 for the empty prompt. That increase is small, and the samples show no visible hedging.
   - The steering KL (20 nats) is about what a *random* vector of the same norm would produce (about 10–16 nats in Phase 0).
   - So most of the hedging vector's effect at this α may be unstructured perturbation, which by construction no prompt can match. This would be the "degenerate vector" failure mode in a subtle disguise.
   - Hand-written prompts also overshoot the intensity (finding 2), so a mild-hedging prompt from rung 4 might still beat rung 0. **Until the matched-norm control and rung 4 are run, the hedging result cannot be attributed to an expressiveness gap.**
6. **No degeneration** in any condition. Distinct-token fractions range from 0.69 to 0.77 everywhere, against a steered target of 0.72–0.75.

### Readability metric: fixed during this step

- **Problem.** The first version scored S as raw text with no context. It wrongly flagged "Respond in French." (perplexity 3420) and "Express uncertainty and hedge your claims." (831) as unreadable: short strings get high perplexity when nothing precedes them.
- **Fix.** Perplexity is now conditioned on a framing prefix ("Below is a system prompt that configures the behavior of an AI assistant. System prompt: ") with threshold 2,000.
- **Check.** The 10 hand-written prompts score between 9 and 509. Three adversarial or junk strings score between 66k and 444k, including a token-soup example of the kind GCG-style optimizers produce.
- **Limit.** This filters only crude gibberish. A fluent-but-adversarial S would pass it, so the gibberish hack needs a stronger gate before Phase 2, for example an LLM judge or a fluency penalty against a base model.

## Step 5: Matched-norm random control (done)

**Run.** `runs/phase1_randfloor_20261122_231649.json` (stdout: `runs/phase1_randfloor_stdout.log`). Code: `matched_norm_control.py`.

**Method.**
- For each vector, draw 8 isotropic Gaussian directions (seeds 5000–5007) and rescale each to exactly ‖w‖. French and hedging have the same norm, so they share the same 8 runs.
- Steer with each direction at layer 18 on the same held-out set and seed as the ladder (`heldout[0:200]`, 128 tokens), then measure forward KL against the neutral prompt.
- **Floor F** = the mean over directions.
- Rungs 0–3 were re-scored per prompt on the same rollouts, so closure has paired uncertainty. Every KL reproduces the step-4 run to two decimals.
- **closed vs floor** = (KL0 − KL_S) / (KL0 − F). Its SE comes from a two-level bootstrap over prompts and directions.
- **Sanity checks.** |cos(random, w)| ≤ 0.06. The MC and Rao-Blackwellized estimates agree for every direction.

| Vector | ‖w‖ | KL0 = KL(steer ‖ neutral) | Floor F (mean ± SE over 8 dirs) | Range over dirs | F / KL0 | Closable gap KL0 − F | Random distinct | Steered distinct |
|---|---|---|---|---|---|---|---|---|
| French | 14.4 | 34.19 | **9.89 ± 0.28** | 9.2–11.7 | 29% | 24.3 | 0.696 | 0.753 |
| Pirate | 43.2 | 161.75 | **120.8 ± 6.9** | 86.8–147.3 | **75%** | 41.0 | 0.673 | 0.740 |
| Hedging | 14.4 | 19.96 | **9.89 ± 0.28** | 9.2–11.7 | **50%** | 10.1 | 0.696 | 0.715 |

Random-direction text is fluent and stays on topic, even at pirate's norm: it reads like a mildly different but normal sample. The floor is the cost of a coherent but different trajectory, not of degeneration.

### Closure fractions, recomputed (held-out, forward KL)

| Vector | Rung | S | KL fwd | Closed vs rung 0 | **Closed vs floor** |
|---|---|---|---|---|---|
| French | 1 generic | You are a helpful assistant. | 35.39 | −3.5% ± 0.6% | −4.9% ± 0.9% |
| French | 2 one-liner | Respond in French. | 21.78 | +36.3% ± 2.3% | **+51.1% ± 3.1%** |
| French | 3a | You are a helpful assistant. Always write… | 19.69 | +42.4% ± 2.3% | +59.7% ± 2.9% |
| French | 3b (best on train) | Réponds toujours en français… | 19.96 | +41.6% ± 2.3% | **+58.6% ± 3.2%** |
| Pirate | 1 generic | You are a helpful assistant. | 158.95 | +1.7% ± 0.3% | +6.8% ± 1.7% |
| Pirate | 2 one-liner | Talk like a pirate. | 129.26 | +20.1% ± 1.0% | **+79.3% ± 15.3%** |
| Pirate | 3a | You are a helpful assistant who speaks like… | 133.01 | +17.8% ± 1.2% | +70.1% ± 13.7% |
| Pirate | 3b (best on train) | Ye be a pirate, matey!… | 127.18 | +21.4% ± 1.1% | **+84.3% ± 16.5%** |
| Hedging | 1 generic | You are a helpful assistant. | 24.08 | −20.6% ± 1.0% | −40.9% ± 2.6% |
| Hedging | 2 one-liner | Express uncertainty and hedge your claims. | 30.84 | −54.5% ± 4.3% | **−108.0% ± 11.1%** |
| Hedging | 3a | You are a careful, humble assistant… | 28.94 | −45.0% ± 2.8% | −89.2% ± 7.7% |
| Hedging | 3b | Answer helpfully but tentatively… | 29.54 | −48.0% ± 3.2% | −95.2% ± 8.6% |
| Hedging | best on train | rung 0 (empty) | 19.96 | 0 | **0** |

### Findings

1. **French.** About 10 of the 20-nat plateau (finding 3 above) is unstructured perturbation. The best prompt still leaves about **10 nats of structured KL** unexplained: the steered model's French has its own content, beyond "is French". The one-liner gets 51% of the closable gap and engineering adds 8 points. That bounds how much rung 4 or Phase 2 could add on top of a one-liner, unless they reproduce content rather than language.
2. **Pirate.** The low raw closure was mostly a **norm effect, not a prompt-expressiveness gap.** A random vector at α 0.6 costs 121 nats by itself, and the best prompt reaches 127. The best prompt sits inside the spread of the random floor (87–147), so closure versus floor is poorly determined (±15–17%). This revises finding 4 above: the floor is not garbling, since random directions at this norm stay fluent. α = 0.6 simply has a very large perturbation cost. Pirate at layer 18 is a poor test vector. The layer sweep (next steps) matters more than rung 4 for pirate.
3. **Hedging.** This revises finding 5 above. The vector is *not* pure noise: KL0 is twice the floor, leaving a **structured component of about 10 nats**, comparable in size to French's unexplained residual. Every prompt moves away from it:
   - The generic prompt alone costs 41% of the closable gap.
   - The one-liner costs 108%: it adds more KL than the steering's whole structured effect.

   This fits the overshoot finding (finding 2 above). The structured effect is small, and any explicit instruction lands much further away than the empty prompt does. Rung 4 has a concrete, small target: a prompt must shift the model by under about 10 nats in the right direction. Intensity-calibrated or indirect prompts are the thing to test. Expressiveness versus search is still not settled.
4. **Phase 2 implication.** Raw KL reduction understates what prompts achieve on high-norm vectors (pirate), and it is uninformative where the structured component is small (hedging). Report closure against the floor alongside closure against rung 0 from now on.

### Caveats

- **The floor subtraction assumes structured and unstructured KL add.** They need not: an isotropic direction is not "the non-semantic part of w", and KL is not additive across components. Treat F as a *scale* estimate.
- **An untested check would validate it:** if no ladder prompt reduces KL(p_rand ‖ q_S) for a random direction, the floor really is unclosable. This is cheap to add to `matched_norm_control.py`.
- Isotropic directions are near-orthogonal to everything the model represents, while w lives in a structured subspace. The off-target cost of w could therefore be larger or smaller than F.
- **Pirate's floor varies 1.7× across directions** (87 to 147 nats). Eight directions give a ±7-nat SE on F, which dominates the closure SE.

## Next steps (proposed, in order)

1. ~~Matched-norm random control~~ (done, step 5). Optional, cheap: test whether ladder prompts reduce KL(p_rand ‖ q_S), to validate the floor.
2. **Rung 4: LLM-proposed candidates.** The proposer is a local Qwen2.5-14B-Instruct. It sees steered samples, neutral samples, and previous candidates with their train-set KL. Candidates are scored on train and the best is reported on held-out. Specifically probe *intensity-calibrated* prompts (finding 2).
3. **Pirate layer sweep** (layers 8–14) to find a layer where the behavior appears before coherence degrades. Re-run the ladder there.
4. Optionally, **per-token KL attribution** for French: are the remaining 20 nats (about 10 of them unstructured, per step 5) concentrated on content tokens or spread uniformly?

---

# Phase 1b: post-control runs (spec: `PHASE1B_SPEC.md`)

The Next steps above are superseded by `PHASE1B_SPEC.md`: four runs in fixed order, plus protocol rules R1–R4. From here on, every closure-vs-floor number is **conservative under the additivity assumption, not a point estimate** (R3, revised after run 2) and must be reported next to closure against rung 1 (R1).

**Revision to Step 5, under rule R4.** Pirate's layer-18 closure versus floor (79–84%) is **withdrawn and reported as indeterminate**. The best prompt (127 nats) falls inside the random floor's spread across directions (87–147), so the quantity is close to undefined. It is kept in the Step 5 table only as a record of what was computed.

## Run 1: Validate the floor (done, PASS)

**Question.** Do the ladder prompts reduce KL(p_rand ‖ q_S) for the random directions? If they do, F is partly closable and the Step 5 closure-vs-floor figures are inflated.

**Run.** `runs/phase1b_floorval_20261122_233156.json` (stdout: `runs/phase1b_floorval_stdout.log`). Code: `floor_validation.py`. Rollouts saved: `runs/phase1b_randrollouts_20261122_233156.pt`.

**Method.**
- Same 8 directions (seeds 5000–5007), held-out set and sampling seed as Step 5.
- For each direction, rungs 0–3 were scored as in Step 4, paired per prompt. The ‖w‖ = 14.39 directions were scored against both the French and the hedging ladders; the ‖w‖ = 43.16 directions against the pirate ladder.
- **Deviation from the spec.** Step 5 logged the directions but not their rollout token ids. Rollouts were regenerated with the same seed. The script asserts that every direction's rung-0 KL matches the logged Step 5 value to within 0.02 nats, and all 16 passed, so the rollouts are identical. They are saved this time.

**Result.** F_S = mean over directions of KL(p_rand ‖ q_S); "dirs reduced" counts directions where S beats rung 0.

| Norm | Rung | S | F_S | Change vs F0 | Dirs reduced |
|---|---|---|---|---|---|
| 14.39 | 0 | (empty) | 9.89 | 0 | — |
| 14.39 | 1 generic | You are a helpful assistant. | 11.98 | **+2.09** (+21% ± 2%) | 0/8 |
| 14.39 | French 2 / 3a / 3b | (ladder) | 40.95 / 43.08 / 28.86 | +31.1 / +33.2 / +19.0 | 0/8 each |
| 14.39 | hedging 2 / 3a / 3b | (ladder) | 23.98 / 24.45 / 23.76 | +14.1 / +14.6 / +13.9 | 0/8 each |
| 43.16 | 0 | (empty) | 120.76 | 0 | — |
| 43.16 | 1 generic | You are a helpful assistant. | 122.68 | **+1.91** (+1.6% ± 0.5%) | 1/8 |
| 43.16 | pirate 2 / 3a / 3b | (ladder) | 141.8 / 161.4 / 153.9 | +21.0 / +40.6 / +33.1 | 0/8 each |

**No ladder prompt reduces the floor.** Across 8 directions × 13 prompt/norm combinations, only one direction-level reduction occurs: the generic prompt on one pirate-norm direction, with a net change of +1.9 nats. The Step 5 numbers stand, under the R3 wording (revised after run 2: conservative under additivity).

**How strong a pass this is.**
- The behavior prompts are a weak test: they push the model toward French, pirate or hedging, which random-direction rollouts lack, so they were always going to *raise* this KL.
- **The informative row is rung 1**: a generic prompt that names no behavior. It also raises the floor, at both norms.
- So **no tested prompt closes F**. That is weaker than "F is unclosable": a prompt searched specifically against a random direction might do better.
- **Proposed addition to run 4:** include one random direction as a negative control in the rung-4 search. If the proposer can find a readable prompt that substantially lowers a *random* direction's KL, that is direct evidence F is closable, and also a warning that the search fits noise.

**Dropped metric.** The script also computed a "prompt-matched floor" closure, 1 − (KL_S(w) − F_S)/(KL0 − F0). It is ill-posed and is not reported as a finding: F_S mixes in the behavior mismatch between prompt S and random-direction text (for example, +31 nats for "Respond in French." on English rollouts), which is not unstructured cost. Its values exceed 100% (up to 196%), which confirms it measures the wrong thing. It is kept in the JSON only.

**Changes to earlier findings.** None to Step 5's numbers. They now carry the R3 wording (revised after run 2: conservative under additivity, not an upper bound), and pirate's figure is withdrawn under R4 (above).

## Run 2: Per-token KL attribution for French (done)

**Question.** Are the ~10 structured nats left after the best French prompt concentrated on content tokens (steering changes *what* is said), or spread across the sequence (residual off-target cost)?

**Run.** `runs/phase1b_tokattr_20261122_233246.json` (stdout: `runs/phase1b_tokattr_stdout.log`). Code: `token_attribution.py`.

**Method.**
- Per-token Rao-Blackwellized KL(p_steer(·|y<t) ‖ q_S(·|y<t)) on the Step 4 held-out steered rollouts (`heldout[0:200]`, seed 101). Totals reproduce Step 4 exactly: 34.19 / 19.69 / 19.96.
- Conditions: rung 0 (full KL); 3b, the best prompt selected on train (the residual); 3a, for robustness.
- Reference profile: the 8 random directions against neutral, on run 1's saved rollouts, as the shape of pure off-target cost.
- Token classes are word-level: BPE pieces are merged into words, and each word is tagged as French/English function word, content word, punctuation, number, whitespace, end-of-sequence, or other (CJK, mixed scripts).

### Where the KL sits, by position (nats per sequence)

| Region | Rung 0 (neutral) | 3b residual | Random floor |
|---|---|---|---|
| Token 0 | 14.6 (43%) | 0.8 | 0.3 |
| Tokens 0–7 | 23.2 | **5.9** | 1.0 |
| Tokens 8–15 | 1.3 | 1.5 | 0.7 |
| Tokens 16+ | 9.8 | **12.5** | 8.2 |
| Total | 34.2 | 20.0 | 9.9 |

At the random floor's per-token rate, the 3b tail (tokens 16+) would cost 8.5 nats; it actually costs 12.5, an **excess of 4.0 nats**. Tokens 0–7 add about 5 nats over the floor. Together these account for the ~10 structured nats from Step 5.

### Concentration and token class

| | Rung 0 | 3b residual | Random floor |
|---|---|---|---|
| Share of KL in top 1% / 5% / 10% of tokens | 57% / 81% / 86% | 33% / 55% / 67% | 16% / 37% / 52% |
| Gini | 0.90 | 0.79 | 0.69 |
| Content words: KL share / token share | 48% / 58% | **56% / 58%** | 56% / 50% |
| Function words: KL share / token share | 45% / 27% | 30% / 27% | 27% / 29% |

**Top residual tokens** (full list of 30 in the log). Almost all are failures *at the language boundary*, not content:
- **The steered model fuses a French first token onto an English word:** "Le|aving", "Le|aky", "Le|ons…" (the "Leonsde Hamming" seen earlier), "Il|leg…", "Non|-standard". The prompted model reads "Le" and continues in French ("Le code"), so it gives the steered continuation ~0 probability.
- **Code-switching inside a sentence:** "Sorry, mais", "The distribution du", "the-person-en", plus one drift into Spanish ("no es general|mente").

### Findings

1. **Neither of the spec's two outcomes holds cleanly. The residual has two parts of roughly equal size.**
   - **About 5 nats, concentrated in the first 8 tokens**: language-boundary artifacts of additive steering, namely fused English–French words and mid-sentence code-switching. This is *how* the steered model produces French, and it is degraded; it is not *what* it says. No readable instruction asks for it.
   - **About 4–5 nats, spread through the tail** at about 1.5× the random floor's per-token rate (0.09–0.20 vs 0.07–0.09 per token).
2. **No content-word concentration.** Content words carry 56% of the residual KL on 58% of tokens, the same proportion as in the random-floor profile. The substantive claim "steering changes what the model says" is **not supported** at this resolution.
3. **The prompt overshoots at token level as well.** In the tail (16+), 3b costs **more** than the empty prompt (12.5 vs 9.8 nats). Once the steered text is in French, the unprompted model follows that context and tracks the steered model's continuations better than the explicitly instructed model does. So 3b wins all of its 17-nat gain in the first few tokens and gives back about 2.7 nats over the rest of the response, which matches the overshoot finding from Step 4. **Rung-4 target for French:** a prompt that makes the response *start* in French and otherwise constrains it as little as possible.
4. **Effect on Step 5's French figure.** Most of the ~10 "structured" nats look like steering-specific off-target cost: boundary artifacts plus an elevated diffuse tail. This is the R3 scenario, where F underestimates the off-target cost of a structured vector.
   - The correct reading of the **direction** of the resulting bias differs from the rule's wording; see the note below.
   - The French residual should not be described as "unreproduced semantic content". It is better described as **artifacts of steering a language switch, which a fluent prompt cannot imitate**.

### Note on rule R3 (needs a decision)

R3 says closure-vs-floor is an **upper bound** because F plausibly *underestimates* the off-target cost. The direction is reversed.
- Closure-vs-floor = (KL0 − KL_S) / (KL0 − F).
- If the true unclosable part F′ is larger than F, the true denominator (KL0 − F′) is *smaller*, and the true closure is *larger*. Underestimating F makes the reported figure a **lower** bound.
- It would be an upper bound if F *overestimated* the unclosable part, for example if F were partly closable. Run 1 found no evidence of that among tested prompts.
- Non-additivity can push either way.

**Recommended rewording:** "closure-vs-floor rests on an additivity assumption and is not a point estimate. The evidence so far (run 1: F not closable by tested prompts; run 2: off-target cost of a structured vector exceeds F) suggests it is *conservative*."

R3's practical instruction ("don't treat it as a point estimate") stands either way.

**Resolved 2026-11-22:** R3 is reworded in `PHASE1B_SPEC.md` along these lines. Closure-vs-floor figures are now described as **conservative under the additivity assumption**, not as upper bounds. The earlier mention in this file of "upper bounds per R3" (bottom line, Phase 1b preamble) is superseded by this.

## Run 3: Pirate layer sweep (done; workable layer found: L8, α 0.5)

**Question.** Is there a layer where the pirate behavior appears before fluency degrades and before the norm cost dominates? Layer 18 is indeterminate (rule R4).

**Runs.**
- Coarse sweep: `runs/phase1b_piratesweep_20261122_233946.json`.
- Fine sweep: `runs/phase1b_piratesweep_20261122_234204.json`.
- Code: `pirate_layer_sweep.py`.
- Ladder at L8: `runs/phase1_ladder_20261122_234338.json` (`oracle_search.py --only pirate --layer pirate=8`).
- Control at L8: `runs/phase1_randfloor_20261122_234450.json` (`matched_norm_control.py --only pirate --layer pirate=8`).
- Vectors: `vectors/pirate_L{8,10,12,14}.pt`. Only L8 has a calibrated α (0.5, with `alpha_source` recorded in the file).

**Method.**
- Same difference-of-means recipe and contrast data as Step 2. The contrast responses are layer-independent and were sampled once.
- The α sweep uses the Step 3 calibration prompts (`train[0:64]`). For every (layer, α) cell, 3 matched-norm random directions (seeds 7000–7002) were also run, giving F/KL0 for the run-3 criterion: maximize (KL0 − F)/KL0, subject to coherent text.
- Coherence was judged by eye on 3 samples per cell (all printed in the stdout logs).
- Cosine of each layer's unit vector with the L18 unit vector: L8 0.74, L10 0.78, L12 0.82, L14 0.89.

### Sweep summary (calibration prompts; pirate words per 100 in steered text; neutral baseline 0.07)

| Layer | α | KL0 | F/KL0 | Pirate/100w | Coherent? |
|---|---|---|---|---|---|
| 8 | 0.40 | 60.5 | 19% | 0.40 | yes; barely pirate |
| 8 | 0.45 | 101.2 | 14% | 1.14 | yes; no visible pirate in the samples |
| **8** | **0.50** | **181.2** | **10%** | **2.47** | **yes; mild dialect ("a bit o' faveery… we'll need ta' tinker wi' teh styling")** |
| 8 | 0.55 | 230.2 | 9% | 5.58 | borderline; garbling returns ("askowuld", "becklicks") |
| 8 | 0.60 | 303.9 | 8% | 8.43 | no; thick dialect, does not answer |
| 10 | 0.60 | 185.2 | 17% | 1.41 | no; garbled code and words |
| 12 | 0.70 | 141.1 | 56% | 2.31 | mostly, with some garbling |
| 14 | 0.60 | 153.6 | 53% | 3.22 | yes; good mild pirate |
| 14 | 0.70 | 186.8 | 71% | 6.09 | borderline |
| 18 (Step 3) | 0.60 | 154.0 | 75% | 4.36 | borderline |

**Choice: L8, α 0.5.** It has the largest closable fraction among coherent cells: about 90%, against 47% for the next-best coherent cell (L14, α 0.6). At L12–14 the floor climbs steeply with α, so any α with visible behavior is norm-dominated.

### Ladder and control at L8, α 0.5 (held-out). Steered target: distinct 0.736, pirate 2.34 per 100 words

Floor **F = 15.89 ± 0.51** (8 directions, range 13.9–18.0; |cos(random, w)| ≤ 0.045). That is **10% of KL0**, and the closable gap is 146.8 nats.

| Rung | S | KL fwd ± SE | KL rev | Pirate/100w | Closed vs rung 0 | Closed vs rung 1 | Closed vs floor |
|---|---|---|---|---|---|---|---|
| 0 | (empty) | 162.71 ± 4.00 | 76.2 | 0.06 | 0 | — | 0 |
| 1 | You are a helpful assistant. | 161.48 ± 3.78 | 76.8 | 0.02 | +0.8% ± 0.3% | 0 | +0.8% ± 0.4% |
| 2 | Talk like a pirate. | 141.58 ± 3.44 | 121.8 | 5.85 | +13.0% ± 0.9% | +12.3% | +14.4% ± 0.8% |
| 3a | You are a helpful assistant who speaks like… | 147.88 ± 3.25 | 147.2 | 8.97 | +9.1% ± 1.2% | +8.4% | +10.1% ± 1.2% |
| **3b (best on train)** | Ye be a pirate, matey!… | **137.35 ± 3.16** | 129.6 | 7.04 | **+15.6% ± 1.1%** | **+14.9%** | **+17.3% ± 1.0%** |

Closed vs rung 1 = (KL₁ − KL_S)/KL₁, from the means. R4 is satisfied: the best prompt (137) lies far outside the floor's spread (14–18), so this closure is **determinate**.

### Findings

1. **Pirate now has a usable test vector (gate item 2), and the result is a determinate negative.** About 147 nats of structured KL exist, and the best readable prompt closes **about 17%** of it.
2. **Revision to Step 5 finding 2.** That finding said pirate's low layer-18 closure was "mostly a norm effect, not a prompt-expressiveness gap". With the norm cost removed (F/KL0 75% → 10%), prompts still close only about 17%. The layer-18 figure of 84% was an artifact of a huge, noisy denominator, as R4 anticipated. **Prompts do not reproduce pirate steering at either layer.**
3. **Why, from the samples: the vector is not "pirate".**
   - The steered L8 model speaks in a *variable, theatrical, nonstandard register*: archaic diction ("I shall heed the instructions and set pen to paper", "tome of lore"), Scots-like dialect ("tae", "wi'", "an'"), occasional pirate words, and some drift into Chinese.
   - The intensity varies a lot from prompt to prompt: some answers are plain English with one "o'".
   - Every pirate prompt produces the stereotyped costume ("Shiver me timbers", "Ahoy bearded matey") at 6–9 pirate words per 100, against 2.3 for steering.
4. **Overshoot again, and stronger.** Reverse KL *rises* under every pirate prompt (76 → 122–147). The prompted model's text is *less* plausible to the steered model than the unprompted model's text is. This is the same pattern as hedging. Rung 4's intensity-calibration target applies to pirate as well as hedging, and a register description ("slightly old-fashioned, theatrical, occasionally dialectal") is the obvious candidate.
5. **Caveats.**
   - L8's vector differs from L18's (cosine 0.74), so the two layers' results are not strictly the same behavior.
   - α was chosen by eye on 3 calibration samples. The held-out samples confirm coherence.

## Run 4: Rung 4, LLM-proposed candidates (done)

**Run.** `runs/phase1b_rung4_20261123_001216.json` (stdout: `runs/phase1b_rung4_stdout.log`). Code: `rung4.py`.
Smoke test: `runs/phase1b_rung4_20261123_001026.json` (random control, 1 round; not a result).

### Readability gate (decided before the run, as the spec requires)

Two gates that do not reduce to subject-model perplexity, on top of the Phase 1 subject-perplexity gate (< 2000):
- **Judge:** Phi-4, from a non-Qwen family, so the proposer never grades itself. It answers "Is this a sensible, readable instruction a person might plausibly write?" on a 1–9 scale, scored as the expected rating from the digit logits. Pass if ≥ 6.
- **Base-model perplexity:** OLMo-2-0425-1B base (non-Qwen), using the same framing prefix. Pass if below 3× the maximum over the hand-written prompts, which works out to 11,019.
  - **Deviation:** OLMo-2-7B was the plan, but it ships in fp32 (~29 GB) and did not fit on disk beside Phi-4, so the 1B base model is used.
  - Judge and base model live in an ephemeral cache on the root disk (`/root/hf_ephemeral`), because `/workspace` lacked room. They are re-downloaded if the pod restarts.

**Calibration** (in the log): all 10 hand-written ladder prompts pass (judge 7.2–8.0) and all 4 junk strings fail (judge 1.6–2.3).
- The judge does the real work. It rejects the fluent-nonsense control "The purple calendar negotiates with the assistant's elbows…" (judge 1.8), whose base-model perplexity (967) is inside the threshold.
- The base-model threshold is loose because "Respond in French." already scores 3,673 under OLMo.

**Applied to the 192 rung-4 candidates.**
- The proposer produced **no gibberish**: 187 passed.
- **All 5 rejections are false rejections** of sensible hedging sentences, for example "Pause briefly in your thoughts before providing an answer." (judge 4.5) and "Suggest possible interpretations while noting uncertainties." (subject perplexity 2,538).
- **8 hedging candidates were JSON-wrapped** (`{"prompt": "..."}`) because of a proposer parsing defect, and the judge passed them at about 7.2. The defect is fixed in `rung4.py` for future runs.
- **No KL-winning candidate was rejected by the gate.**
- So the gate is untested against real adversarial pressure, and noisy at its boundary in both directions. That matters for Phase 2, where the gate becomes a reward term.

### Method

- **Proposer:** Qwen2.5-14B-Instruct, 6 rounds × 8 candidates per target, temperature 0.9. Pure ancestral sampling is still used for every *subject-model* rollout.
- **What the proposer sees:**
  - 6 steered-vs-neutral response pairs from train;
  - all previous candidates with their train KL, ladder prompts included (marked REJECTED where they failed the gate);
  - the rung-0 and rung-1 scores;
  - the general Phase 1 lesson (strong instructions overshoot; try mild, partial or indirect wording).
- **What it is not told:** which behavior the vector encodes. For the real vectors it can infer this from the ladder prompts in its history.
- **Scoring:** forward KL on `train[64:192]`. The best gate-passing candidate and the next two go to held-out (`heldout[0:200]`), with rung 0/1 recomputed for paired closure. The held-out rung 0/1 values reproduce Steps 4, 5 and run 1 exactly.
- **Negative control:** a random direction (seed 5000) at the hedging norm and layer, given exactly the same treatment, with the proposer blind to it.

### Results (held-out)

| Target | Rung 0 | Rung 1 | Best ladder S | **Best rung-4 S** (chosen on train) | Rung 4 vs rung 0 | vs rung 1 | vs floor (conservative) | Behavior metric (steered / rung 4) |
|---|---|---|---|---|---|---|---|---|
| **Hedging** | **19.96** | 24.08 | rung 0 (empty) | 21.43 ("Sometimes consider alternative views subtly in your replies.", JSON-wrapped); runner-up 21.39 "Suggest answers cautiously, noting areas of uncertainty." | **−7.4% ± 1.5%** | +11.0% ± 1.4% | −14.6% | hedges/100w 1.37 / 0.98 (runner-up **1.28**) |
| Pirate (L8) | 162.71 | 161.48 | 137.35 (3b) | 133.98 "Speak with a slight pirate influence, using occasional phrases like 'arr', 'matey'…" | +17.7% ± 1.1% | +17.0% ± 1.0% | **+19.6%** (ladder: 17.3%) | pirate/100w 2.34 / 6.13 |
| French | 34.19 | 35.39 | 19.96 (3b) | 17.94 "Écrivez vos réponses en français, sauf si le langage technique l'exige." | +47.5% ± 2.2% | +49.3% ± 2.1% | **+66.9%** (ladder: 58.6%) | frac French 0.98 / 0.90 |
| Random control | **9.55** | 11.51 | — | 12.63 "Strive for a balance between simplicity and depth." | **−32.4% ± 2.3%** | −9.8% ± 1.8% | n/a | — |

### Findings

1. **Hedging (gate item 3): NEGATIVE. No prompt beats the empty system prompt, on train or held-out.** 48 candidates; the best on train was 22.83 against 21.61 for rung 0.
   - Rung 4 did fix the overshoot: the best candidates are 8–9 nats better than the hand-written hedging prompts (21.4 vs 28.9–30.8).
   - The runner-up matches the steered hedge rate almost exactly (1.28 vs 1.37 per 100 words).
   - **But an intensity-matched prompt still costs about 1.4 nats more than silence.**
   - **Read: the gap is mainly not a search failure.** The search found the right *kind* of prompt, at the right lexical intensity. What it cannot remove is the cost of *having a system prompt at all*: rung 1 costs +4.1 nats, about 40% of hedging's ~10-nat structured component. Rung 4 recovers about two thirds of that presence cost, not all of it.
   - At this signal size, the prefix intervention's own side effects are as large as the behavior being imitated. That is an expressiveness limit of the intervention class, in the sense of CLAUDE.md, though not necessarily a limit on verbalizing the behavior.
   - **Caveats:** a single proposer and 48 candidates per target; the search plateaued after round 2.
2. **Rule R1 needs a refinement.** R1 assumes rung 1 is the harder baseline. For hedging it is the *easier* one: rung 1 is worse than silence, so "clears rung 1" (+11%) is meaningless while rung 0 is not cleared. **Proposed wording:** an S helps only if it beats *the better of rung 0 and rung 1*.
3. **Pirate: rung 4 adds about 2 points** (closure vs floor 17.3% → 19.6%).
   - The winner is still a pirate-lexicon prompt that overshoots (6.1 vs 2.3 pirate words per 100) despite asking for a "slight" influence.
   - Register-only prompts with no pirate words ("as if writing from a character's perspective in a novel", "storytelling around a campfire") close about 10% with the pirate rate near baseline (0.1). The theatrical register is roughly half of what prompts can capture, and the lexicon is the other half.
   - Pirate stays a **determinate negative**: about 80% of the closable gap remains.
4. **French: rung 4 adds about 8 points** (58.6% → 66.9% vs floor; 41.6% → 47.5% vs rung 0).
   - The winning prompt *loosens* the instruction ("unless technical language requires it"), and its French rate drops to 0.90.
   - This matches run 2: the strict instruction overshoots after the opening tokens, and loosening it helps.
5. **Random negative control: PASS, a strengthened run-1 result.** The search does not fit noise.
   - Over 6 rounds the proposer improved steadily (train 21.4 → 14.1), but only by converging toward short, bland prompts that approach the generic prompt. No readable prompt reached rung 1, let alone rung 0.
   - **No tested prompt, hand-written or searched, closes any part of the random floor.** This is stronger evidence that F is unclosable than run 1's fixed ladder.
6. **Marginal value of search over the one-liner** (vs rung 0): French 36% → 48%, pirate 13% → 18%, hedging none. That bounds what Phase 2's amortized T can add on top of a one-liner for these vectors.

### Gate to Phase 2 (PHASE1B_SPEC.md)

| # | Item | Status |
|---|---|---|
| 1 | Run 1 reported; Step 5 numbers recomputed or withdrawn if the floor fails | **DONE.** The floor passed in run 1, and the rung-4 random control confirms it. Pirate's L18 figure is withdrawn under R4. |
| 2 | Pirate has a workable layer or is dropped; no L18 closure reported | **DONE.** L8, α 0.5: F/KL0 10%; closure determinate at 17% (ladder) and 20% (rung 4). |
| 3 | Rung 4 reported for hedging, with a read on search failure vs expressiveness | **DONE: NEGATIVE.** Intensity-matched prompts were found and still do not beat silence. The cost of having any system prompt is about 40% of the signal. Read: expressiveness limit of the intervention class at this signal size, not search failure (with caveats). |
| 4 | Stronger readability gate exists and is applied to rung-4 output | **DONE**, with limits: no adversarial candidates arose to test it; 5 false rejections; JSON-wrapped strings pass. Needs hardening before it becomes a Phase 2 reward term. |

**Open questions for the team before Phase 2:**
- R1 refinement (finding 2).
- Whether hedging remains a training-distribution vector, given that its structured signal is smaller than the cost of any system prompt.
- Gate hardening (false rejections at the judge boundary).
