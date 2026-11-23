# Phase 3: Status. RUNNING (started 2026-11-23, detached, with pod auto-stop)

**Run dir:** `runs/phase3_20261123_145628/`. Contents:
- `run.log` — full log;
- `<vector>.json` — written after each vector;
- `all.json`;
- `SUMMARY.md` — written at the end;
- `rung4_*.json` and `ladder_*.json` — baseline records.

**Code:** `phase3_eval.py`. It is the user's draft with the adapter implemented against the repo; the deliberate changes are listed in its docstring. Smoke test, which reproduced Phase 2 hedging exactly: `runs/phase3_20261123_054830_smoke_hedging/`.

**Question.** Does T (`runs/phase2_train_20261123_023935_main6/lora_step200`) condition on w, or did it memorize its 6 training prompts?

**Plan:** 5 unseen vectors plus the 2 Phase 2 negative controls.

| Vector | Layer | α | Baseline | Note |
|---|---|---|---|---|
| spanish | 18 | 0.15 | rung 4 | reads as **Portuguese** by eye |
| emoji | 12 | 0.6 | rung 4 | cosine 0.85 with `enthusiastic` (a training vector) |
| italian | 18 | swept | ladder | built in this run |
| archaic_formal | 18 | swept | ladder | built in this run; **interpretation:** Victorian formal English (behaviors.BEHAVIORS_P3) |
| gloomy_L8 | 8 | swept | ladder | |
| hedging | 18 | 0.2 | Phase 1b rung-4 record | negative control |
| random seed 5000 | 18 | 0.2 | Phase 1b rung-4 record | negative control |

Ladder-only baselines are conservative *against* T: rung 4 beat the ladder by 16–17 points on register vectors in Phase 2.

**When the pod comes back, check:**
1. `SUMMARY.md`, and any `FAILED` lines in `run.log`.
2. **Review by eye the α chosen for the swept vectors.** It was picked by automated criteria; samples are in `<vector>.json` → `alpha_sweep[*].samples`.
3. Both negative controls must still fail to beat the empty prompt.
4. **Re-download Phi-4 into `/root/hf_ephemeral/hub` before any further gate or rung-4 work.** The pod stop wipes the ephemeral disk.
