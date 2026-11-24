# Unsteered baseline outputs — Hamming code prompt

- **Held-out set index:** 1 (`data/wildchat_prompts.json` → `heldout[1]`)
- **Prompt (verbatim):** `give the working principle for hamming code used for error detetction`
  *(the "detetction" typo is in the source prompt and was fed to the model as-is)*
- **Source run:** `runs/phase1_ladder_20261120_235111.json`
- **Model:** Qwen/Qwen2.5-3B-Instruct

All outputs below were **read from the stored run**, not regenerated. The run dumped
rollout text for `heldout[0:4]`; this prompt is the second dump (`samples[1]`).

Each output is 128 new tokens (`MAX_NEW_TOKENS=128`) and therefore ends mid-sentence.
Sampling followed the project convention: pure ancestral (temperature 1, top_k 0,
top_p 1, repetition penalty 1.0). The baseline rollouts are `q(·|S)` generations with
seed `SEED_Q=202`.

---

## Condition: unsteered baseline (rung 0 — empty system prompt)

- **Source:** stored run (`results[*].rows[0]`, rung `"0 neutral"`, `S=""`)
- **Config:** empty system prompt through the Qwen chat template, unsteered, 128 new
  tokens, pure ancestral sampling, `SEED_Q=202`

> Hamming code is a set of error-correction codes that can be used for error detection and correction. The key behind Hamming code is to use additional parity bits (or check bits) to detect and correct single-bit errors.
>
> ### Working Principle:
>
> 1. **Adding Parity Bits**:
>    - Suppose the sender wants to send a data block of size 7 bits (including parity bits). Let's denote this data block as D = D6D5D4D3D2D1D0.
>    - The 7-bit block is divided into k groups, where k+1 is the number of parity bits (

---

## Condition: pirate, rung 0

- **Source:** stored run (`results[1]` = "pirate", `rows[0]`, rung `"0 neutral"`, `S=""`)
- **Note:** Rung 0 is the empty system prompt with **no steering**, so it does not
  depend on the steering vector. This output is **byte-identical** to the unsteered
  baseline above (verified: same text, same SHA-256). There is no separate
  pirate-specific rung-0 rollout.

## Condition: hedging, rung 0

- **Source:** stored run (`results[2]` = "hedging", `rows[0]`, rung `"0 neutral"`, `S=""`)
- **Note:** Same as above — **byte-identical** to the unsteered baseline. Rung 0 is
  condition-independent (unsteered, empty system, same `SEED_Q=202`).

---

*The "pirate" and "hedging" steered rollouts (the `p_steer` targets, not rung-0
baselines) are stored separately under `results[*].target.samples[1]` in the same run
file, if a steered comparison column is needed later.*
