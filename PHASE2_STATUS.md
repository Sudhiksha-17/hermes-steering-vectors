# Phase 2: Status. DONE. Phase 3 gate met on all 5 items (with caveats)

Spec: `PHASE2_SPEC.md`. Last updated 2026-11-23.

| Step | Status |
|---|---|
| Environment: persistent deps and model caches | DONE, but Phi-4 is still on the ephemeral disk; see below |
| **P0: harden the readability gate** | **DONE.** One class is not caught: evocative-but-meaningless personas. It is monitored, not gated. |
| Vector distribution (8–12 training vectors, hedging held out) | **DONE.** 8 training vectors, 1 Phase 3 holdout, 2 eval-only negatives. Mean-centering was tried and rejected. |
| Architecture | **CHOSEN:** residual injection of w at its own layer and α into a LoRA fine-tune of the subject model |
| GRPO training | **DONE:** 6 vectors × 200 steps (`runs/phase2_train_20261123_023935_main6`). An 8-vector run was aborted at step 58 after 2 vectors failed the criterion on the training pool. |
| Evaluation and negative controls | **DONE.** T beats the search baseline on 4 of 6 vectors and ties on French; both negative controls pass; per-token attribution done. |

## Environment (done)

- **Python dependencies.** A pod restart on 2026-11-22 wiped every pip package on the root disk. The dependencies now live in `/workspace/.pylib`:
  - transformers 5.24.0, accelerate 1.16.0, peft 0.22.0, huggingface_hub 2.4.0, pyarrow, and their small dependencies.
  - Installed with `--no-deps` on purpose: a plain `--target` install pulls in torch 2.15 with CUDA 13, which would shadow the base image's torch 2.8.
  - `env.sh` exports `PYTHONPATH=/workspace/.pylib`. torch 2.8 and numpy 2.5.4 still come from the base image, so every Phase 0–1b number was produced with the same torch and numpy.
- **Model caches.**
  - In `/workspace/.cache/huggingface` (persistent): Qwen2.5-3B-Instruct (subject), Qwen2.5-14B-Instruct (rung-4 proposer), and OLMo-2-0425-1B (moved here from the ephemeral cache).
  - **Phi-4 (judge, ~29 GB) is still in the ephemeral `/root/hf_ephemeral/hub`.** The 47 GB network volume has 3.7 GB free.
  - **A pod restart during training would lose the judge.** Options: grow the volume by about 40 GB, or drop the 14B proposer from disk after the per-vector rung-4 baselines are computed.

## P0: Harden the readability gate (done)

**Code:** `readability.py` (the gate, shared by Phase 2 and any future rung-4 runs) and `gate_redteam.py` (calibration and red-team).
**Runs:**
- `runs/phase2_gate_20261123_005618.json` — v1: the 1–9 rating judge.
- `runs/phase2_gate_20261123_010336.json` — final.
- `runs/phase2_gate_q23.json` — design study for the extra judge questions.
- `runs/phase2_gate_redwave2.json` — red-team wave 2.
- Stdout for each in `runs/phase2_gate_*_stdout.log`.

### Test sets

Splits keep tuning data and evaluation data apart.

| Set | n | Content | Role |
|---|---|---|---|
| good_dev | 202 | Phase 1 ladder prompts plus the rung-4 candidates (JSON wrappers unwrapped) | calibration |
| good_test | 40 | New diverse system prompts: 7 languages, personas, formats, odd-but-fine tones | held-out |
| red_dev | 37 | 30 hand-written attacks in 6 categories, plus the 8 JSON-wrapped rung-4 originals | design |
| red_test | 84 | Qwen-14B told to write strings that fool a readability judge | held-out; see note |
| wave2 | 21 | Harder hand-written attacks, used to design the payload check | design |
| **wave3** | 20 | **12 attacks + 8 hard-but-legitimate prompts, written before the gate was run on them** | **held-out** |

### What changed, and why

1. **The judge is now a classifier, not a rating.**
   - v1 asked Phi-4 for a 1–9 "is this a readable system prompt" rating. It passed 4 of 6 fluent-nonsense attacks (about 7.0–7.6) and pasted in-character sample text (7.9), and rejected "You are a helpful assistant." at 4.5.
   - The final judge classifies each S as **A** instruction or persona, **B** sample content (written *in* a style rather than describing it), **C** nonsense, or **D** empty. The score is P(A), renormalized over the four letter tokens.
   - P(A) is sharply bimodal (good-set median 1.00; red-team median ≤ 0.001), so the floor of 0.5 sits in a wide gap.
2. **Payload check (new).** A second Phi-4 question: "Does it contain a sample of the reply itself…?"
   - It targets the most dangerous hack for GRPO: **an instruction prefix plus pasted steered text**, which is a soft prompt written out as text. Wave 2 found this passing the classifier 4 times out of 6.
   - Good-set maximum P(Yes) is 0.068, so the hard threshold is 0.10.
   - Backed by a structural rule: a cue phrase plus colon ("like this:", "for example:", "sound like:"…) followed by more than 8 words is rejected.
3. **Structural rules (hard fail, no strip-and-revalidate).** Under GRPO, silently stripping a wrapper would still reward emitting it. Rejected:
   - markup, JSON or code characters; chat-template tokens; URLs;
   - role-marker lines (fake transcripts);
   - mixed-script words (homoglyphs); invisible characters;
   - low letter ratio; word runs, repeated sentences, low distinct-word ratio;
   - quoted spans over 12 words; fewer than 2 or more than 80 words.
   - Word counts treat CJK characters as half a word, because v1 rejected a valid Chinese prompt as a single word.
4. **The empty string is valid.** It is rung 0, and the correct answer for a random direction. It is never scored by the judge and carries zero penalty.
5. **Base-model perplexity** (OLMo-2-1B): hard cap 10,000; the good-set maximum is 3,673, for "Respond in French.". Continuous penalty starts at 1,000. This is a weak signal; the judge does the work.
6. **Subject-model perplexity is dropped from the gate.** It produced one of the original false rejections (2,538 on a sensible sentence), and it is the self-referential metric the spec asked to replace. It is still reported.
7. **An "actionability" judge question was tried and dropped.** The question: "Would two readers agree what the assistant should do differently?"
   - It separates meaningless personas (0.06–0.25).
   - But it also fails many legitimate prompts: "Sound genuinely excited about whatever the user asks." scores 0.25, and many subtle rung-4 hedging prompts score below 0.45.
   - **As a reward term it would push T toward loud, explicit prompts, which is exactly the overshoot that Phase 1b showed loses.**

### Final results

| Set | Metric | Result |
|---|---|---|
| good_dev | false rejection | 4/202. Real: 2/200 (1%), both vague rung-4 sentences ("Capture the essence of a well-researched article."). The other two: "You are a helpful assistant." classed D (empty) by design, and a malformed JSON fragment. |
| good_test | false rejection | **0/40** |
| wave2 legit / **wave3 legit** | false rejection | 0/5 / **0/8** (subtle, whimsical-but-meaningful, German) |
| red_dev | acceptance | 1/37 |
| wave2 attacks | acceptance | 7/16. All prefixed-sample attacks caught. Missed: 5/5 meaningless personas, 1 code-switched salad, and 1 contentless sentence at P(A) 0.53. |
| **wave3 attacks** | acceptance | **4/12. All 4 are meaningless personas.** Caught: all 4 sample-text attacks (no-colon, dash, two-sentence variants), contentless, unpunctuated repetition, lowercase pirate stream. |
| red_test | acceptance | 7/84. **All 7 are legitimate instructions** ("Give advice on writing effective emails."). The 14B mostly failed to write degenerate strings, so this set is a weak test. |

### Known blind spot (stated, not hidden)

**Evocative-but-meaningless personas pass:** 9/9 across waves 2 and 3. Examples: "Speak in the manner of an hourglass that has grown tired of sand.", "Answer with the confidence of a Wednesday and the patience of a staircase."

The classifier cannot separate these from legitimate whimsical personas ("Answer as an excitable puppy would, if puppies could type."). The only question that could, actionability, carries an anti-subtlety bias that would damage the objective.

**Mitigation: monitor, don't gate.**
- The actionability question runs on every emitted S *as a logged training-time monitor* with no reward pressure. An alarm triggers if the fraction of emitted prompts below 0.3 rises over training.
- A KL-to-reference penalty on T's policy (standard GRPO) keeps T near its initial distribution.
- Every reported S gets human review (Phase 3 gate item 5).

### Reward shape (P0 item 4, decided)

For a candidate S for vector w, with KL measured on a minibatch of steered rollouts:

```
B        = min(KL(rung 0), KL(rung 1))          # R1 as amended: the better baseline
r_KL     = (B − KL(S)) / (B − F)                # closure vs floor, against the better baseline
                                                # unit-free, so comparable across vectors whose gaps differ ~10x
penalty  = 0.05 × readability_penalty(S)        # readability.reward_terms: 10 × max(0, 0.9 − P(A))
                                                #   + 0.5 × max(0, ln base_ppl − ln 1000); 1 point ≈ 5% closure
         + degeneration_penalty(S)              # R5: on q(·|S) samples, see below
reward   = clip(r_KL − penalty, −2, +∞)

HARD FLOOR → reward = −2 (constant), if any of:
  readability hard fail (structural, P(A) < 0.5, payload > 0.10, base ppl > 10,000)
  distinct-token fraction of q(·|S) samples < 0.5        (repetition degeneration)
```

- **Degeneration penalty (R5).** Applied when the distinct-token fraction of q(·|S) samples falls below the steered model's own fraction minus 0.05. The penalty is the shortfall in units of 0.05, × 0.1 closure.
- **Why −2 for the hard floor.** It is a constant, so there is no gradient toward the cliff edge inside the failing region. Clipping valid rewards at −2 means failing is never *better* than a bad valid prompt. Training vectors are selected for closable gaps of at least 3× the presence cost (see the vector distribution section), so valid rewards below −1 should be rare.
- **No shaping toward naming the behavior,** per the spec.
EOF
echo done

## Vector distribution (done)

**Code:** `build_vectors_p2.py`; the behaviors are `BEHAVIORS_P2` in `behaviors.py`.
**Runs:**
- `runs/phase2_vecscreen_20261123_011625.json` — uncentered.
- `runs/phase2_vecscreen_centered_20261123_012357.json` — mean-centered.
- Stdout for both in `runs/phase2_vecscreen*_stdout.log`, with 3 samples per (vector, α) cell.

**Method.**
- 16 candidate behaviors spanning languages (German, Spanish), registers (formal, casual, archaic), personas (cowboy), affect (enthusiastic, gloomy, empathy), formats (bullets, lowercase, emoji, concise, verbose) and topics (cats, cooking).
- Same difference-of-means recipe and contrast prompts as Phase 1, at layer 18. Eliciting prompts are worded differently from the one-liners.
- α swept on the Step 3 calibration prompts. Each (vector, α) cell gets KL₀, KL₁ (rung 1), presence cost c = KL₁ − KL₀, and the floor F(α). F is computed once per α from 3 matched-norm random directions, because the residual norm at L18 is the same for all vectors.

**Screening criterion** (the spec's, with references from Phase 1). A cell passes if all of these hold:
- gap = KL₀ − F ≥ 3·max(c, 0);
- gap ≥ 15 nats (hedging's ~10 is the known-unusable reference);
- F/KL₀ ≤ 0.6 (pirate L18's 0.75 is the known-unusable reference);
- distinct fraction ≥ 0.65;
- the behavior metric moves by at least 50% of the contrast's positive−negative difference.

Coherence was then checked by eye at the chosen α.

### Results (uncentered; "passes at" = α values meeting every criterion)

| Vector | Passes at | Chosen α | F/KL₀ there | Behavior by eye at chosen α | Decision |
|---|---|---|---|---|---|
| german | 0.15–0.3 | 0.20 | 24% | fluent German | **train** |
| spanish | 0.15–0.2 | 0.15 | 19% | **Portuguese, not Spanish** (the stopword metric cannot tell them apart) | **Phase 3 holdout** (unseen language); relabel or recheck before use |
| formal | 0.3–0.4 | 0.30 | 32% | clearly formal ("necessitates", "facilitating") | **train** (subtle) |
| enthusiastic | 0.3–0.4 | 0.30 | 48% | mild ("You're going to love this", "a fantastic adventure!") | **train** (subtle) |
| gloomy | 0.3 | 0.30 | 50% | **weak**: faintly downbeat ("isn't an easy process… no free solution") | **train** (subtle; behavior weak) |
| lowercase | 0.3–0.4 | 0.30 | 35% | sentence starts and names in lowercase | **train** |
| empathy | 0.3 | 0.30 | 59% | **reads as warmth/positivity rather than empathy** (cosine 0.76 with enthusiastic) | **train** (subtle; label approximate) |
| verbose | 0.3–0.4 | — | — | — | **dropped**: metric invalid. Under the 96-token cap the contrast barely differed in length (72.7 vs 67.6 words), so "behavior shift" passed trivially. |
| casual, concise, bullets, emoji, archaic, cowboy, cats, cooking | none | — | 45–65% at any α with visible behavior | Bullets never produces lists. The others show behavior only where the norm cost dominates or the gap criterion fails. | **not usable at L18** |

**Training set (8):**
- **french** L18 α 0.20 and **pirate** L8 α 0.50 (the Phase 1 anchors, unchanged);
- **german** 0.20, **formal** 0.30, **enthusiastic** 0.30, **gloomy** 0.30, **lowercase** 0.30, **empathy** 0.30, all L18.

Subtle or graded vectors: formal, enthusiastic, gloomy, empathy (the spec requires at least 2). Gloomy and empathy pass the criterion (a structured KL gap) but show weak or mislabeled behavior by eye. That is acceptable for the objective, because T has to reproduce the steering whatever it is, but they should not be described as "gloom" and "empathy" without that caveat.

**Held out:** spanish (Phase 3, unseen vector). Eval-only negatives: hedging L18 α 0.2 (known-negative), and a random direction (seed 5000 at hedging's norm and layer).

### Findings from screening

1. **Phase 1's layer-18 problem is general.** For 11 of 16 behaviors, F/KL₀ is 45–65% at every α where the behavior is visible: more than half the KL is the norm effect. Only languages and lowercase have low floors (13–35%). Styles at L18 are expensive per nat of behavior, consistent with pirate L18.
2. **Difference-of-means vectors share a large common component.** Pairwise cosines at L18 run 0.72–0.90 (cowboy/pirate 0.90, spanish/french 0.89, enthusiastic/emoji 0.85, cats/cooking 0.80). Contrasting "eliciting system prompt" with "empty system prompt" evidently captures a generic "behave as if under an unusual system prompt" direction as well as the specific behavior.
3. **Mean-centering was tried and rejected.** It subtracts the mean of the 16 raw vectors.
   - It makes the vectors more distinct: cosines become mixed-sign, and among the centered vectors most are below 0.7.
   - But it **weakens behavior per nat**: formal loses its behavior shift, and gloomy and casual need larger α with floors near 60%. So the shared component carries part of the behavior signal.
   - Uncentered vectors are used throughout, with the same recipe as the anchors.
   - **Consequence for T:** its conditioning inputs are highly correlated (for example enthusiastic/empathy 0.76), so the "T ignores w" test matters more.

## Architecture (chosen)

**T = Qwen2.5-3B-Instruct + LoRA (r 16, all attention and MLP projections). w is injected into T's residual stream at w's own layer, scaled by w's own calibrated α, at every position, exactly as the subject model receives it.** T reads a fixed instruction that never names a behavior (`T_INSTRUCTION` in `train_T.py`) and writes S while steered.

**Why this, over a w → soft-prefix adapter:**
1. **It mirrors the intervention exactly:** same model, layer, vector and α. T experiences the steering it has to describe, and mixed-layer vectors (French L18, pirate L8) need no special handling.
2. **There is a zero-shot prior, and it is measured as the step-0 baseline (T₀ = the untrained, steered model).** This is the Patchscopes/SelfIE idea: the steered model writing a system prompt is already a behaviorally grounded readout.
3. **The alternative would decide the outcome.** A soft-prefix adapter (an MLP from 2048 to P×2048) trained on 8 vectors would almost certainly memorize one prompt per vector. Phase 3's unseen-vector test would then fail by construction, the "architecture silently becomes the result" risk the spec names. Residual injection reuses the model's own computation and adds no new conditioning parameters to learn from 8 data points.

**Risks, and how each is checked:**
- **T writes *in* the steered style instead of describing it.** Long in-style text is caught by the payload check; short in-style prompts such as "Ahoy! Answer like a pirate" are legitimate.
- **T ignores w.** `eval_T.py` prints each vector's greedy S side by side, and the correlated inputs above raise this risk.
- **Shortcut learning.** T may learn to recognize each training vector and recall a memorized prompt. Phase 2 cannot tell this apart from real understanding; Phase 3 can.

One base copy of the weights serves as T (LoRA on), as T_ref for the KL regularizer (LoRA off, steered), and as the subject model for p and q (LoRA off).


### Revision: the training set is 6 vectors, not 8 (2026-11-23)

`train_T.py` re-measures each vector on its 256-prompt training pool (`train[192:448]`). On that larger sample two vectors fail the criterion they had passed on the 64 calibration prompts:

| Vector | Pool KL₀ | Pool KL₁ | Presence cost | F | Closable gap | Criterion |
|---|---|---|---|---|---|---|
| gloomy | 47.80 | 55.65 | 7.85 | 28.84 | 18.96 | **fails**: 18.96 < 3 × 7.85 = 23.6 |
| empathy | 43.65 | 48.38 | 4.73 | 28.84 | 14.81 | **fails**: 14.81 < 15-nat minimum (3 × cost = 14.2, marginal) |
| enthusiastic | 47.50 | 52.37 | 4.87 | 28.84 | 18.66 | passes, barely (3 × cost = 14.6) |

**Decision (user, 2026-11-23):**
- Train on the 6 vectors that pass: french, pirate L8, german, formal, enthusiastic, lowercase. Subtle vectors: formal and enthusiastic, which meets the minimum of 2.
- In parallel, screen the failed styles at L8 and L12 to try to get back to 8 or more for a final run.

**Partial run kept for the record:** the 8-vector run was aborted at step 58 (`runs/phase2_train_20261123_014335_main_ABORTED_8vec`). Rewards were already rising for french, pirate, german and formal.

**Lesson.** A 64-prompt screen is too noisy for a criterion with a 3× margin; screen on the pool size used for training.

## Step 0: T₀, the untrained steered model (done)

**Run.** `runs/phase2_eval_T0_*_step0.json` (stdout: `runs/phase2_eval_T0_stdout.log`). Code: `eval_T.py`.
An earlier T₀ run used a first-draft `T_INSTRUCTION` that made T₀ prefix every prompt with "System:", which the gate rejects. It is voided (`runs/phase2_eval_T0_OLDINSTRUCTION_void_stdout.log`); only its cached floors were kept, and they reproduce Step 5 exactly.

**Method.**
- T₀ = the base model steered by w, given `T_INSTRUCTION`.
- 8 sampled candidates plus 1 greedy S per vector. The best gate-passing candidate is picked by forward KL on `train[64:192]`.
- Held-out evaluation on `heldout[0:200]`, paired across conditions.
- Closure is measured against the better of rung 0 and rung 1 (R1), and against an 8-direction held-out floor.

| Vector | Better baseline | T₀ best-of-8 vs better | **vs floor (conservative)** | Rung-4 best vs floor | T₀ best-of-8 S |
|---|---|---|---|---|---|
| french | rung 0 | +21.3% ± 2.1% | +30.0% | **+66.9%** | "Compose en français avec un ton amical et convivial, adoptant un style…" |
| pirate L8 | rung 1 | +18.8% ± 0.8% | **+20.9%** | +18.9% | "Write quick and keep it plain, jesters. Speake as a man would talk…" (pirate rate 0.25, vs rung 4's 6.1) |
| german | rung 0 | +9.8% ± 2.2% | +12.7% | (running) | "Folgen Sie meinem stilen, dem Ton und dem Sprachgebrauch…" |
| formal | rung 0 | +9.2% ± 0.8% | +13.4% | (running) | "engage in discourse as an articulate and accessible academic expert…" |
| enthusiastic | rung 0 | +16.7% ± 2.0% | +31.7% | (running) | "Write like you're your friend just chatting - friendly, funny…" |
| lowercase | rung 1 | **−26.7% ± 1.2%** | −38.6% | (running) | "as yourself, with a friendly and informative tone…" (written in lowercase; q's output stays 94% lowercase, unchanged) |
| gloomy (dropped) | rung 0 | +12.1% | +23.7% | — | |
| empathy (dropped) | rung 0 | +19.0% | +43.1% | — | |
| **hedging (negative)** | rung 0 | **−20.0%** | −39.6% | −14.6% | passes: does not beat the empty prompt |
| **random (negative)** | rung 0 | **−93.5%** | n/a | — | passes: does not beat rung 0 |

**Findings.**
1. **The zero-shot prior is real.** Without any training, T₀ beats both baselines on 5 of 6 training vectors (lowercase is the exception).
   - On **pirate it already beats the rung-4 search** (+20.9% vs +18.9% against the floor). It does this by capturing the *register* without the pirate lexicon (pirate rate 0.25, against 6.1 for rung 4). That is the intensity calibration rung 4 could not achieve.
2. **Both negative controls pass at step 0.**
3. **"Ignores w" check, mixed.**
   - The greedy outputs differ in language and register: French, German, dialect, formal-ese, lowercase.
   - Their *content* is nearly identical, a variant of "a friendly, conversational tone" for almost every vector, including hedging and random.
   - **T₀ conveys w mainly by writing in-style, not by describing it.** For lowercase this fails outright: a prompt written in lowercase does not make q write in lowercase.
   - Whether training turns in-style writing into description is the key question for the trained T.
4. **The greedy output is much worse than best-of-8** on most vectors (formal greedy −16%, gloomy greedy −35%). Sample-and-select matters at step 0.


### Extra screening at L8 and L12 (done): only 1 addition, so the 8-vector target is not reachable

**Runs:** `runs/phase2_vecscreen_L8_*.json` and `runs/phase2_vecscreen_L12_*.json` (stdout `runs/phase2_vecscreen_L{8,12}_stdout.log`).

**Method.** The saved L8/L12 vectors for the 10 behaviors that failed at L18 were reused. α swept over 0.2–0.8, with the same criteria and checks by eye.

**Results.**
- **Earlier layers are much cheaper per nat:** F/KL₀ is 14–40%, against 45–65% at L18.
- **But by eye, the behavior appears only as fluency collapses**, the pirate-L18 pattern:
  - cats, casual and cooking at L8 α 0.8 produce mixed-script soup and garbled words ("juicychef falcon");
  - concise at L12 α 0.6 is a Title-Case-Capitalization artifact, not brevity; at L8 α 0.6 it is gibberish;
  - gloomy (L12) and empathy (L8, L12) show no visible behavior at any coherent α.
- **The only coherent cell with visible behavior is emoji at L12 α 0.6:** enthusiastic text with emoji (2.7 per 100 words, some with broken encoding), F/KL₀ 39%. It overlaps heavily with enthusiastic (cosine 0.85 at L18).

**Finding.** For topic and format behaviors (cats, cooking, bullets, concise, casual), difference-of-means steering at L8, L12 and L18 expresses the behavior only once fluency breaks down. Pirate L18 was the first instance. With this vector recipe, the usable behaviors are languages, register (formal), mild affect (enthusiastic), and case (lowercase). That bounds what the steering → prompt objective can be tested on.

**Decision.**
- The 6-vector run stays the Phase 2 main result. One more vector, heavily overlapping enthusiastic, does not justify a second training run.
- **emoji L12 α 0.6 is held out for Phase 3** (unseen vector) alongside spanish, which is really Portuguese.
- Its training-pool criterion check runs when Phase 3 evaluates it.


## Rung-4 baselines for the new training vectors (done)

**Run.** `runs/phase1b_rung4_20261123_023404.json` (stdout: `runs/phase2_rung4_stdout.log`). Same method as Phase 1b rung 4: blind Qwen-14B proposer, 6 rounds × 8, selected on train. It still used the *Phase 1b* gate, for comparability.

Held-out closure against the floor (conservative):

| Vector | Best LLM proposal | Best ladder prompt | T₀ best-of-8 |
|---|---|---|---|
| german | +20.5% "Occasionally switch to German, ensuring responses remain comprehensible…" | **+25.4%** "Respond in German." | +12.7% |
| formal | **+36.6%** "Occasionally favor a tone that is both precise and slightly academic." | +20.4% (3a) | +13.4% |
| enthusiastic | **+52.6%** "Maintain a slightly informal chat-like style." | +35.7% (3a) | +31.7% |
| lowercase | −9.4% "Be friendly and helpful, but stay on topic." | **+32.4%** "Write in all lowercase." | −38.6% |

**Findings.**
1. **The intensity-calibration pattern repeats.** The winning proposals for formal and enthusiastic are *hedged* ("occasionally", "slightly"), and they beat the explicit ladder prompts by 16–17 points.
2. **The Phase 1b gate sabotaged the lowercase search.** It falsely rejected "Write in all lowercase." (train KL 57.5, by far the best). The proposer saw it marked REJECTED and abandoned case instructions entirely. This is a concrete instance of the false-rejection cost P0 was meant to fix.
3. **The search baseline used in evaluation** is therefore the best prompt *on train* among everything rung 4 scored (ladder rungs plus proposals), re-checked with the **hardened** gate (`eval_T.rung4_best`). For lowercase that is "Write in all lowercase."; for hedging and the random control it is the empty prompt.
4. **In this run the old gate's base-perplexity threshold was effectively off** (223,653). The judge was still active, and every reported prompt is re-scored with the hardened gate in `eval_T`.


## Gate v2 (2026-11-23): fix for a false-rejection gap on short prompts

**Found in use, not in calibration.** The v1 gate rejected legitimate one-liners, which emptied the lowercase search baseline:
- "Write in all lowercase.": P(A) 0.4996, just under the floor;
- "Mention cats.": P(A) 0.00;
- "Be casual.": base perplexity 15,852, over the cap;
- "Write formally.": base perplexity 74,551, over the cap.

None of the calibration sets had prompts this short.

**Changes (`readability.py`, `GATE_VERSION = 2`):**
1. The base-perplexity cap and penalty apply only to S with at least 5 words. Perplexity of a 2–4 token string is meaningless.
2. For S under 5 words, the judge score is P(A) + P(D). The judge classes ultra-short commands as D (empty); the contentless attacks D exists for are all long, and a short contentless prompt is harmless. B (sample content) and C (nonsense) still fail at every length.
3. **Tried and reverted:** adding short-instruction few-shot examples to the judge and raising the cap to 50,000. That made the judge accept the 3 wave-2 contentless attacks and roughly doubled red_test acceptance, while still rejecting "Use bullet points.".

**Validation** (`runs/phase2_gate_20261123_033816.json`, stdout `runs/phase2_gate_v7_gatev2final_stdout.log`):
- Every v1 set is unchanged: good_dev 4/202 (same items), good_test 0/40, red_dev 1/37, wave2 7/16, wave3 attacks 4/12 (the persona blind spot), wave3 legit 0/8.
- red_test acceptance 7 → 8; the new one is "Please provide interesting information.", a vague but legitimate instruction.
- **New held-out set of 15 short legitimate prompts: 0/15 false rejections** (3/15 before fix 2).
- Phase 2 ladder one-liners: 1/33, and that one is "You are a helpful assistant.", rejected as D by design.

**Which gate is used where:**
- The 6-vector training run uses **v1**, loaded at startup (`runs/readability_v1_used_in_main6_training.py`). The gap mainly limited exploration of very short prompts, which T rarely writes.
- **Evaluation uses v2.**
- **Search baseline:** the best prompt on train among everything rung 4 scored, required to be structurally valid but **not** filtered by the judge. It is a hand-written or proposed comparison, not T's output, so a judge false rejection must not handicap it. Its gate status is still reported.


## Interim checkpoint evaluations (held-out, gate v2)

**Runs.**
- `runs/phase2_eval_T_*_step50.json`. Its gate pass/fail column used gate v1; the KL numbers do not depend on the gate.
- `runs/phase2_eval_T_*_step100.json`.
- Stdout for both in `runs/phase2_eval_T_step{50,100}_stdout.log`.

**Closure against the floor (conservative):**

| Vector | T₀ best-of-8 | Step 50 best-of-8 | **Step 100 best-of-8 / greedy** | Search best (ladder + rung 4) |
|---|---|---|---|---|
| french | +30.0% | +52.4% | +50.1% / +23.9% | **+66.9%** |
| pirate L8 | +20.9% | +24.5% | **+33.0% / +27.6%** | +18.9% |
| german | +12.7% | +16.1% | **+33.1% / +26.7%** | +25.4% |
| formal | +13.4% | +44.8% | **+49.3% / +40.8%** | +36.6% |
| enthusiastic | +31.7% | +64.4% | **+65.4%** / +55.7% | +57.0% |
| lowercase | −38.6% | −17.8% | −6.8% / −7.9% | **+31.6%** |
| hedging (negative) | −39.6% | −36.5% | −44.0% / −32.5%: **passes** | empty prompt |
| random (negative) | −94% vs rung 0 | −39% | −44% / −72% vs rung 0: **passes** | empty prompt |

**Step-100 readout:**
- T beats the better of rung 0/1 on **5 of 6** training vectors, and the search baseline on **4 of 6** (pirate, german, formal, enthusiastic). Greedy alone beats it on pirate, german and formal.
- Both negative controls pass. No output fails gate v2, and no distinct-fraction collapse.
- **French and lowercase fail the same way: T never *names* the behavior.** It writes S *in* French, or *in* lowercase, but never instructs the language or the case.
  - For French, a French-language prompt alone gives about 51% French output, against 90% for the search best's explicit "en français".
  - For lowercase, writing in lowercase does not transfer to q at all.
  - This is the "conveys w by writing in-style" pattern from step 0, which training has not broken for these two vectors.
- For german, the same strategy (German-language prompt, register only, 34% German output) *beats* "Respond in German." (94%). The steered model is only 91% German, and its register matters.
- **The "ignores w" check is improving.** Greedy outputs now differ in content: formal asks for "academic precision and scholarly erudition", pirate for "a talkin' Scots voice", enthusiastic for a "friendly… Go on!" style. Hedging, random and lowercase all get near-identical generic prompts, which is correct for the negatives.
- **Overshoot is returning on pirate:** 4.85 pirate words per 100 against 2.34 steered. KL is still much better than the search baseline (113 vs 134).


## Final result: T at step 200 (held-out, gate v2)

**Runs.**
- Training: `runs/phase2_train_20261123_023935_main6`. It contains `config.json`, `candidates.jsonl` (every candidate with its reward components) and the `lora_step{50,100,150,200}` checkpoints. Stdout: `runs/phase2_train_main6_stdout.log`.
- Evaluation: `runs/phase2_eval_T_20261123_050945_step200.json` and `runs/phase2_eval_T_*_step150.json` (stdout `runs/phase2_eval_T_step{150,200}_stdout.log`).

**Training config:** 6 vectors; G = 8 candidates per vector per step; 32 steered rollouts per candidate per step, from the 256-prompt pool `train[192:448]`; LoRA r 16; learning rate 3e-5; β_ref 0.02; seed 0; gate v1.

### Headline table

Closure is against the better of rung 0/1 (R1) and against the floor (R2; conservative under the additivity assumption, R3). No result falls inside the floor's spread (R4).

| Vector | Better baseline | **T best-of-8** vs better / vs floor | **T greedy** vs floor | Search best vs floor (ladder + rung 4) | T vs search | Behavior metric: steered / T / search |
|---|---|---|---|---|---|---|
| french | rung 0 | +48.2% ± 2.0% / **+67.8%** | **+68.4%** | +66.9% | tie (+0.9 pts, within SE) | frac French 0.98 / 0.92 / 0.90 |
| pirate L8 | rung 1 | +30.3% ± 1.0% / **+33.6%** | +24.8% | +18.9% | **+14.7 pts** | pirate/100w 2.34 / **2.69** / 6.13 |
| german | rung 0 | +30.9% ± 2.3% / **+40.2%** | +37.9% | +25.4% | **+14.8 pts** | frac German 0.91 / 0.77 / 0.94 |
| formal | rung 0 | +39.1% ± 1.2% / **+56.8%** | +56.5% | +36.6% | **+20.2 pts** | formal/100w 0.46 / 0.28 / 0.24 |
| enthusiastic | rung 0 | +39.3% ± 1.9% / **+74.4%** | +74.4% | +57.0% | **+17.4 pts** | enthus/100w 3.22 / **3.11** / 3.07 |
| lowercase | rung 1 | −3.7% ± 0.8% / −5.4% | −5.4% | +31.6% | **−37.0 pts** | frac lowercase 0.98 / 0.94 / 0.98 |
| **hedging (negative)** | rung 0 | −14.9% / −29.6% | −31.1% | rung 0 (empty) | **passes: does not beat the empty prompt** | |
| **random (negative)** | rung 0 | −67.2% vs rung 0 | −84.1% | rung 0 (empty) | **passes: does not beat rung 0** | |

Distinct-token fractions of q(·|S) samples for T's prompts are 0.70–0.78, against a steered range of 0.67–0.80: no degeneration (R5). Every emitted S passes gate v2.

### Emitted prompts (greedy, step 200)

| Vector | S |
|---|---|
| french | "Veuillez écrire des réponses précises et détaillées en français avec un ton courtois et professionnel, comme celui que vous utilisez habituellement." |
| pirate | "Tell th' AI ta write in a hearty Scots voice, wi' a laddie's tone, an' a bit o' humour. Make it breeelalid wi' folks, an' be careful tae be too formal." |
| german | "Bitte schreibe deine Antworten mit einem sachlichen und freundlichen Tonfall wie dieser in Deutsch." |
| formal | "Generate responses in an academic, rigorous manner, imbued with a tone of authoritative erudition and an unwavering commitment to precision and depth. Adhere meticulously to the lexicon of formal discourse, eschewing colloqu…" |
| enthusiastic | "Let your replies be lively and informative, in a lively and enthusiastic tone, just as you do!" |
| lowercase | "as yourself, continue to write replies in a helpful, informative tone using simple and clear language." |
| hedging (neg.) | "Please generate responses in an informative and detailed manner, maintaining a helpful and knowledgeable tone akin to this conversation, using precise and clear language." |
| random (neg.) | "Please write replies in an informative and helpful tone, using a knowledgeable and friendly style." |

### Findings

1. **T beats the search baseline (best of ladder + rung 4, selected on train) on 4 of 6 training vectors by 15–20 points, and ties on French.**
   - Rung 4 is a 14B proposer run for 6 rounds × 8 candidates. T is a 3B LoRA, and its **greedy single output** matches its best-of-8 on french, formal and enthusiastic.
   - So the amortization claim holds on these vectors: one forward generation replaces a 48-candidate search. This is the spec's "matches rung 4 at a fraction of the inference cost" case, and better.
2. **T learned intensity calibration, which search could not.**
   - Behavior rates of T's prompts track the steered model: pirate 2.69 per 100 words against 2.34 steered (search: 6.13), enthusiastic 3.11 against 3.22.
   - This is the overshoot problem Phase 1b identified and rung 4 could only partly fix with "occasionally/slightly" wording.
   - Formal *undershoots* the lexical metric (0.28 against 0.46) while having the best KL, so the metric is a coarse proxy here.
3. **French changed between step 150 and step 200.**
   - Until step 150, T wrote prompts *in* French without instructing French: about 50% French output, 52% closure.
   - At step 200 it says "en français" (92% French output) and ties the best search prompt. German moves the same way ("in Deutsch").
   - The in-style prior from step 0 was eventually overcome on the language vectors.
4. **Per-token attribution (spec diagnostic):** `runs/phase1b_tokattr_20261123_051112.json`, stdout `runs/phase2_tokattr_T200_stdout.log`. Code: `token_attribution.py --extra-s`.

   | Nats per sequence | Tokens 0–7 | Tokens 8–15 | Tokens 16+ | Total |
   |---|---|---|---|---|
   | rung 0 | 23.15 | 1.25 | 9.79 | 34.19 |
   | hand-written 3b | 5.93 | 1.54 | 12.49 | 19.96 |
   | **T step 200** | **5.44** | 1.40 | **10.74** | **17.58** |
   | random floor | 0.95 | 0.70 | 8.24 | 9.89 |

   - T's prompt has the same shape as the best hand-written one: nearly all the gain at the language boundary in the first tokens, and the same token-class profile.
   - **Its advantage over 3b is concentrated in the tail.** It cuts 3b's tail give-back over the unprompted model from +2.7 to +0.95 nats.
   - That is the "start in French, constrain nothing else" target from Phase 1b run 2. Per the spec, a T that learns this trick is doing something real, and the gain is not diffuse.
5. **Lowercase failure: a GRPO exploration failure, not a gate or objective failure.**
   - By about step 100 all 8 candidates per step had collapsed onto one generic prompt ("as yourself, continue to write replies…"). Group variance was zero, so the advantage was zero and there was no gradient.
   - T never tried mentioning case. The in-style prior (writing the prompt in lowercase) gives q no lowercase signal, so it had no gradient toward discovering the instruction.
   - Gate v1's false rejection of short case instructions may have contributed early, but T's candidates were long sentences that never mentioned case.
   - **Fix for a rerun:** an entropy bonus or a per-vector temperature floor when group variance collapses.
6. **"Ignores w" check: passed in substance on the vectors where T succeeds.** Outputs differ in content, not just wording: French vs German language; academic erudition (formal) vs lively enthusiasm vs Scots humour (pirate). The three vectors with no reproducible prompt signal for T (lowercase, hedging, random) all get near-identical generic "informative and helpful" prompts, which is the right answer for the two negatives.
7. **Readability caveats.**
   - Pirate prompts contain dialect non-words ("breeelalid", "blacktharett", "stannin'"). They pass gate v2 and read as Scots-flavoured English, but they are the closest thing to the gibberish hack in the outputs and should be watched in Phase 3.
   - The formal prompts are long and florid (some truncated at 64 tokens).
   - No evocative-nonsense personas (the known blind spot) appeared among the best prompts; the actionability monitor stayed at 0–25%.

### Gate to Phase 3 (PHASE2_SPEC.md)

| # | Item | Status |
|---|---|---|
| 1 | P0: gate hardened and red-teamed | **DONE.** Gate v2. Blind spot (evocative personas) documented and monitored. |
| 2 | T beats the better of rung 0/1 on held-out for a majority of training vectors, without degeneration | **PASS: 5 of 6** (lowercase fails). Distinct fractions 0.70–0.78. |
| 3 | T compared against rung 4 per vector | **DONE:** T > search on 4, tie on 1, below on 1. |
| 4 | Both negative controls pass | **PASS:** hedging −30% and random −67% vs the empty prompt. |
| 5 | Emitted prompts readable, with examples in the writeup | **PASS**, examples above; caveat on pirate dialect non-words. |

### Caveats for any writeup

- **This is in-distribution only.** T is evaluated on held-out *prompts* for its 6 *training* vectors. With 6 vectors, memorizing one prompt per vector is indistinguishable from understanding w. Phase 3 (spanish, which is really Portuguese, and emoji L12, both unseen) is the test.
- **The training set is smaller than the spec's 8–12,** for the reasons recorded in the vector-distribution sections. Topic and format behaviors could not be steered coherently with this recipe.
- **Enthusiastic at 74% closure against the floor sits near the additivity assumption's limit** (R3); minibatch rewards reached +1.0 (KL below F).
- **Gates differ by stage:** training used gate v1 and evaluation used gate v2.
- **Dual use (CLAUDE.md):** T is now a trained generator of system prompts that reproduce steering directions. It must not be released with harmful vectors without gating.
