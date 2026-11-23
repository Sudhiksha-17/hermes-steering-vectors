# Phase 3: holdout results

| Vector | alpha | F/KL0 | T vs better (%) | T vs floor (%) | T greedy vs floor (%) | Baseline | Baseline vs floor (%) | T - baseline (pts) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| spanish | 0.15 | 0.181 | 25.9 | 31.6 | 28.6 | rung4 | 22.7 | +8.9 |  |
| emoji | 0.6 | 0.387 | 28.2 | 46.0 | 32.1 | rung4 | 48.8 | -2.8 |  |
| italian | 0.15 | 0.121 | -1.1 | -1.3 | -11.6 | ladder | 0.0 | -1.3 |  |
| archaic_formal |  |  | UNUSABLE: no alpha on the grid meets the Step 3 / Phase 2 cr | | | | | | |
| gloomy_L8 |  |  | UNUSABLE: no alpha on the grid meets the Step 3 / Phase 2 cr | | | | | | |
| hedging | 0.2 | 0.496 | -14.9 | -29.6 | -31.1 | rung4 (Phase 1b record) | 0.0 | -29.6 | NEGATIVE CONTROL PASSES (T does not beat the better baseline) |
| random_seed5000 | 0.2 | 1.036 | -67.2 | None | None | rung4 (Phase 1b record) | 0.0 |  | NEGATIVE CONTROL PASSES (T does not beat the better baseline) |

## Emitted prompts (greedy)

- **spanish**: Por favor, escribe respuestas detalladas y amplias en un tono claro y amigable, utilizando un estilo informado y respetuoso.
- **emoji**: Respond in a friendly, helpful tone! Just like you do!
- **italian**: Please escribe respostes en un llengua clara i natural, amb un ton educatiu i assistentiu, similar al meu propi estil de resposta.
- **hedging**: Please generate responses in an informative and detailed manner, maintaining a helpful and knowledgeable tone akin to this conversation, using precise and clear language.
- **random_seed5000**: Please write replies in an informative and helpful tone, using a knowledgeable and friendly style.

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