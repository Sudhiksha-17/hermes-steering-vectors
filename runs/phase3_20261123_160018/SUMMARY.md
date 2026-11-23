# Phase 3: holdout results

| Vector | alpha | max cos (to) | F/KL0 | T vs better (%) | T vs floor (%) | T greedy vs floor (%) | Baseline (type) vs floor (%) | T - baseline (pts) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| japanese | 0.15 | 0.692 (german) | 0.123 | 2.0 | 2.3 | -1.6 | rung4 25.6 | -23.3 |  |
| uppercase | 0.2 | 0.512 (german) | 0.142 | -19.3 | -22.5 | -19.5 | ladder 0.0 | -22.5 |  |
| korean | 0.3 | 0.651 (german) | 0.142 | 12.6 | 14.7 | 4.4 | ladder 24.8 | -10.1 |  |

## Emitted prompts (greedy)

- **japanese**: 请以正式且友好的语气撰写回复，尽量使用准确且详细的语言。
- **uppercase**: Please write replies in a knowledgeable, helpful, and PERSONAL tone using ACADEMIC LANGUAGE and EXAMPLES where APPROPRIATE.
- **korean**: 네, 당신이 원하는 대로 답변하겠습니다. 상세하고 명료하며 친절한 tone으로 작성하겠습니다.

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