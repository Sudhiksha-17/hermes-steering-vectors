# Phase 3b — Diagnose the holdouts, then close the memorization question

Status: DONE 2026-11-23. All 4 gate items met. Conclusion: T interpolates/memorizes; does not read arbitrary w (see PHASE3B_STATUS.md).

(Full spec as provided by the user 2026-11-23; see conversation. Runs:
 1. Language identification via lingua-language-detector (blocks the rest).
 2. alpha review by eye for swept vectors; re-check UNUSABLE verdicts with a wider grid.
 3. Cosine-to-nearest-trained, reported in every table and in phase3_eval.summarize().
 4. Low-cosine (<0.7 to all 6 trained vectors) holdouts: the actual generalization test.

Gate:
 1. Every language vector's label confirmed/corrected by detector output.
 2. italian classified as mislabelled-vector or real-conditioning-failure, with evidence.
 3. Cosine-to-nearest-trained reported for every holdout.
 4. >= 1 holdout at max cosine < 0.7 evaluated, pass or fail.
)
