# Phase 0 — Status: PASS

Date: 2026-11-20
Scripts: `oracle.py` (Phase 0 entry point), `common.py` (shared: steering hook, prompts, sampling, KL)
Run log: `runs/phase0_20261120_233645.json` (full config + numbers + samples)
Vector: `vectors/gauss_seed0_d2048.pt` (unit-norm; scaled by α × residual norm at run time)

## Config

| Setting | Value |
|---|---|
| Subject model | Qwen/Qwen2.5-3B-Instruct (bf16) |
| Steering layer | 18 of 36 (output of the decoder block, all token positions) |
| Vector | isotropic Gaussian, seed 0, unit-normalized |
| α units | × median residual-stream norm at layer 18 = **71.5** (position 0 / attention sink excluded) |
| Gate α | **0.25** (‖w‖ = 17.9) |
| Prompts | 32 WildChat train prompts × 2 samples = 64 rollouts |
| Rollout length | max 128 new tokens (EOS included in scoring) |
| Sampling | temperature 1, top_k 0, top_p 1, repetition penalty 1 (pure ancestral) |
| Gen seed | 1234 (baseline), 1235 (steered) |
| Hardware | H200 141 GB |

## Gate

| Check | Result | Detail |
|---|---|---|
| (a) Steered differs from baseline, still coherent | PASS (eyeballed) | Coherent at α=0.25; differences are small drifts (see below) |
| (b) KL(steer ‖ base) in 2–20 nats at gate α | PASS | 15.12 ± 0.77 |
| (c) KL(zero-steer ‖ base) ≈ 0 | PASS | 0.000 exactly (hook installed, adding zeros) |
| (d) Teacher-forced scoring reproduces `generate()` logprobs | PASS | mean \|Δ\| 0.023/token, max 0.44 (bf16 noise) |
| (e) Monte Carlo and Rao-Blackwellized KL estimators agree | PASS | 15.16 vs 15.12, within 3 SE |

## α sweep

KL is sequence-level (nats, summed over ≤128 generated tokens), mean ± SE over 64 rollouts.

| α | ‖w‖ | KL (RB) | ± SE | KL (MC) | KL / token | mean len | distinct-token frac | log p_base / tok |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 124.0 | 0.665 | −1.156 |
| 0.10 | 7.1 | 2.35 | 0.13 | 2.30 | 0.019 | 121.7 | 0.682 | −1.150 |
| 0.15 | 10.7 | 5.15 | 0.27 | 5.04 | 0.042 | 122.3 | 0.681 | −1.103 |
| **0.25** | **17.9** | **15.12** | **0.77** | **15.16** | **0.124** | 122.1 | 0.672 | −1.218 |
| 0.35 | 25.0 | 33.18 | 1.76 | 32.22 | 0.267 | 124.1 | 0.672 | −1.326 |
| 0.50 | 35.7 | 73.59 | 2.77 | 73.89 | 0.578 | 127.4 | 0.664 | −1.791 |
| 1.00 | 71.5 | 218.50 | 4.43 | 219.06 | 1.707 | 128.0 | 0.686 | −3.964 |
| 4.00 | 285.9 | 78.79 | 3.03 | 78.42 | 0.616 | 128.0 | 0.031 | −1.229 |

Baseline distinct-token fraction: 0.680.

## Qualitative samples

**α = 0.25 (gate)** — coherent, on-task, small drifts:
- *"get pack compose button for input text that hide and shows password"* → answers in **SwiftUI** instead of Jetpack Compose.
- *"i bought a car from 1972 …"* → normal answer, but a stray Chinese token (`老化`) appears mid-list.

**α = 1.0** — fluent but off-topic:
- Jetpack Compose question → *"When working with the PyPA announcement board, it's helpful to consider whether to use the `cryptography` module…"*
- Crowdstrike replacement question → *"Let's consider the spectra of cognitive systems used by language models like GPT-4…"*

**α = 4.0** — collapses into repetition (distinct-token fraction 0.03).

## Findings and caveats

1. **The pipeline is correct.** Zero-steer gives exactly 0, scoring matches generation, and the two KL estimators agree.
2. **The random vector is noise, not a behavior.** At α=0.25 it produces 15 nats of KL from scattered drifts (framework swaps, stray tokens) with no behavior that can be named. A system prompt that reproduces this would be describing noise. This is the degenerate-vector failure mode from the brief, and it is why Phase 1 should use meaningful vectors.
3. **The 2–20 nat gate depends on rollout length.** It applies to 128-token rollouts. The comparable measure across lengths is roughly 0.02–0.15 nats per token.
4. **KL alone does not show over-steering.** KL peaks near α≈2 and then falls (79 at α=4) because the steered model degenerates into repetition that the baseline predicts easily. Later phases need a degeneration metric (for example, the distinct-token fraction) reported alongside KL.
5. **Baseline context.** "No system prompt" means the Qwen chat template with no system message, which inserts Qwen's default system prompt. `p_steer` uses this same context, and `q(·|S)` replaces it with `S`.
6. **Sampling.** Qwen's default generation config (temperature 0.7, top_k 20, top_p 0.8, repetition penalty 1.05) would bias every KL estimate, so sampling is pinned to pure ancestral settings.

## Addendum (2026-11-20): system-prompt confound fixed

**Problem.** The original run used the chat template with no system message, so Qwen inserted its
default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.").
Any later `q(·|S)` that replaced that default would differ from `p_steer` in two ways at once: the
steering and the system text.

**Fix.** `common.chat_ids` now requires an explicit `system` argument. `NEUTRAL_SYSTEM = ""` emits
an explicit empty system block (`<|im_start|>system\n<|im_end|>`). `p_steer` always uses it, and
`q(·|S)` differs only in the text inside that block. `oracle.py` has `SYSTEM = NEUTRAL_SYSTEM`.

**Re-run:** `runs/phase0_20261120_234449.json`. The numbers are stable, and all checks still pass.

| α | KL, Qwen default system | KL, empty system | KL/tok, empty system | distinct, empty system |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.683 |
| 0.10 | 2.35 ± 0.13 | 2.52 ± 0.12 | 0.021 | 0.676 |
| 0.15 | 5.15 ± 0.27 | 5.60 ± 0.29 | 0.046 | 0.684 |
| **0.25** | **15.12 ± 0.77** | **16.24 ± 0.89** | **0.133** | 0.679 |
| 0.35 | 33.18 ± 1.76 | 34.78 ± 1.68 | 0.278 | 0.666 |
| 0.50 | 73.59 ± 2.77 | 77.87 ± 3.77 | 0.611 | 0.672 |
| 1.00 | 218.50 ± 4.43 | 223.81 ± 4.60 | 1.748 | 0.697 |
| 4.00 | 78.79 ± 3.03 | 85.88 ± 2.68 | 0.671 | 0.029 |

- The shift is at most about 9%. At the gate α it is +1.1 nats, within the combined SE (~1.2).
- The median residual norm is unchanged (71.5).
- The generate-vs-score check now gives mean |Δ| 0.024.
- With the empty system prompt, the steering looks slightly more potent at every α, but no row
  changes qualitatively.

## Data

- **Source:** WildChat-1M shard `data/train-00000-of-00014.parquet`.
- **Filter:** English, non-toxic, unredacted, first user turn only, 20–600 characters, deduplicated, shuffled with seed 0.
- **Split:** fixed, 1,000 train / 500 held-out, cached at `data/wildchat_prompts.json`. Phase 0 used train[0:32].

## Next: Phase 1 (proposed)

- Build 2–3 difference-of-means vectors for clear behaviors (e.g. respond in French, pirate speak, extreme formality).
- Calibrate α per vector using the same sweep.
- Score hand-written candidate prompts by KL on **held-out** prompts. Compare against the no-system-prompt baseline and a generic system prompt.
- Then add the LLM-proposed candidate loop.
- Report readability of `S` alongside KL from the first run.
