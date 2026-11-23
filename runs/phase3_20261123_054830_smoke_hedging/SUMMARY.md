# Phase 3: holdout results

| Vector | alpha | F/KL0 | T vs better (%) | T vs floor (%) | T greedy vs floor (%) | Baseline | Baseline vs floor (%) | T - baseline (pts) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| hedging | 0.2 | 0.496 | -14.9 | -29.6 | -31.1 | rung4 (Phase 1b record) | 0.0 | -29.6 | NEGATIVE CONTROL PASSES (T does not beat the better baseline) |

## Emitted prompts (greedy)

- **hedging**: Please generate responses in an informative and detailed manner, maintaining a helpful and knowledgeable tone akin to this conversation, using precise and clear language.

## Reading this

The question is whether T conditions on w or memorized six prompts.
T succeeding on vectors it never saw is the only evidence that
separates these. Note in the writeup which baselines were full rung-4
searches and which were ladder-only: ladder-only is conservative
against T, since Phase 2 measured rung 4 beating the ladder by 16-17
points on register vectors.

Both negative controls must still fail to beat the empty prompt. If
either now 'succeeds', suspect a reward or evaluation bug before
believing it.

alpha for swept vectors was chosen by automated criteria; review the saved
samples (alpha_sweep[*].samples in <vector>.json) by eye before trusting it.