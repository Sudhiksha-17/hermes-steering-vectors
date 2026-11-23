"""Phase 0 -- infrastructure sanity.

Apply a random Gaussian steering vector at a middle layer of the subject model, sample steered
vs baseline rollouts on WildChat prompts, and measure KL(p_steer || p_baseline).

Gate:
  (a) steered outputs visibly differ from baseline and stay coherent        -> eyeball printed samples
  (b) KL(steer || baseline) at ALPHA is clearly positive, roughly 2-20 nats -> checked
  (c) KL(zero-steer || baseline) ~ 0                                        -> checked
Extra correctness checks:
  (d) teacher-forced scoring reproduces generate()'s own logprobs           -> checked
  (e) MC and Rao-Blackwell KL estimators agree within noise                 -> checked

ALPHA is expressed relative to the median residual-stream norm at LAYER (excluding position 0,
which carries Qwen's massive attention-sink activation).
"""
import datetime
import json
import os
import statistics as st

import torch

from common import (NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, steering, summarize,
                    wildchat_prompts)

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = NEUTRAL_SYSTEM          # explicit empty system block (Phase 0 original run used None = Qwen default)
LAYER = 18                      # of 36 decoder layers
VECTOR_SEED = 0
ALPHA_SWEEP = [0.1, 0.15, 0.25, 0.35, 0.5, 1.0, 4.0]   # x median residual norm
ALPHA = 0.25                    # value the gate is evaluated at (calibrated: run phase0_20261120_233529)
N_PROMPTS = 32
N_SAMPLES_PER_PROMPT = 2
MAX_NEW_TOKENS = 128
GEN_SEED = 1234
GEN_BATCH = 64
SCORE_BATCH = 16
N_SHOW = 3                      # samples printed per condition

KL_GATE = (2.0, 20.0)            # sequence-level nats over MAX_NEW_TOKENS=128 tokens
ZERO_KL_TOL = 1e-3
LOGPROB_MATCH_TOL = 0.05        # mean |logp_generate - logp_score| per token (bf16 noise)
OUT_DIR = "/workspace/runs"
VEC_DIR = "/workspace/vectors"


def residual_norm(model, ctxs, layer):
    norms = []

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        norms.append(h[0, 1:].float().norm(dim=-1).cpu())

    hd = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        for c in ctxs:
            model.model(input_ids=torch.tensor([c]).cuda())
    hd.remove()
    return torch.cat(norms).median().item()


def distinct_frac(comps):
    """Fraction of unique tokens per rollout: low values flag repetition/degeneration."""
    return st.mean(len(set(y)) / max(1, len(y)) for y in comps)


def main():
    model, tok = load_model(MODEL_NAME)
    prompts = wildchat_prompts("train", N_PROMPTS)
    ctxs = [chat_ids(tok, x, SYSTEM) for x in prompts]
    ctxs_rep = [c for c in ctxs for _ in range(N_SAMPLES_PER_PROMPT)]
    d = model.config.hidden_size

    rnorm = residual_norm(model, ctxs, LAYER)
    g = torch.Generator().manual_seed(VECTOR_SEED)
    unit = torch.randn(d, generator=g)
    unit = (unit / unit.norm()).cuda()
    os.makedirs(VEC_DIR, exist_ok=True)
    vec_path = f"{VEC_DIR}/gauss_seed{VECTOR_SEED}_d{d}.pt"
    torch.save(unit.cpu(), vec_path)
    print(f"layer {LAYER}: median residual norm {rnorm:.1f}; unit vector -> {vec_path}")

    def show(name, comps):
        print(f"\n--- {name}")
        for k in range(N_SHOW):
            i = k * N_SAMPLES_PER_PROMPT
            print(f"[USER] {prompts[k][:150]!r}\n[ASSISTANT] {tok.decode(comps[i])[:400]!r}\n")

    # ---- baseline rollouts + scoring-consistency check (d)
    base, base_logits = sample(model, tok, ctxs, MAX_NEW_TOKENS, N_SAMPLES_PER_PROMPT,
                               seed=GEN_SEED, batch_size=GEN_BATCH, return_logits=True)
    show("baseline", base)
    from common import completion_logprobs
    diffs = []
    for i in range(0, len(base), SCORE_BATCH):
        lp, m = completion_logprobs(model, ctxs_rep[i:i + SCORE_BATCH], base[i:i + SCORE_BATCH])
        for j, y in enumerate(base[i:i + SCORE_BATCH]):
            gen_lp = torch.log_softmax(base_logits[i + j], -1)
            idx = torch.tensor(y)
            a = gen_lp.gather(-1, idx[:, None]).squeeze(-1)
            b = lp[j, :len(y)].cpu().gather(-1, idx[:, None]).squeeze(-1)
            diffs.append((a - b).abs())
    diffs = torch.cat(diffs)
    match = dict(mean_abs=diffs.mean().item(), max_abs=diffs.max().item())
    print(f"\n(d) generate-vs-score logprob |diff| per token: mean {match['mean_abs']:.4f}, "
          f"max {match['max_abs']:.4f}")

    # ---- alpha sweep
    results = {}
    conds = [0.0] + ALPHA_SWEEP
    for a in conds:
        vec = unit * (a * rnorm)
        comps = sample(model, tok, ctxs, MAX_NEW_TOKENS, N_SAMPLES_PER_PROMPT, vec=vec, layer=LAYER,
                       seed=GEN_SEED + 1, batch_size=GEN_BATCH)
        r = kl_on_rollouts(model, ctxs_rep, ctxs_rep, comps, vec_p=vec, vec_q=None, layer=LAYER,
                           batch_size=SCORE_BATCH)
        s = summarize(r)
        s["distinct_frac"] = distinct_frac(comps)
        s["logp_base_per_tok"] = sum(r["logq"]) / sum(r["ntok"])  # steered text NLL under baseline
        s["samples"] = [tok.decode(comps[k * N_SAMPLES_PER_PROMPT]) for k in range(N_SHOW)]
        results[a] = s
        if a in (0.0, ALPHA):
            show(f"alpha={a}" + (" (zero-steer)" if a == 0 else " (gate alpha)"), comps)

    base_s = {"distinct_frac": distinct_frac(base)}
    print(f"\nbaseline distinct_frac {base_s['distinct_frac']:.3f}")
    print(f"\n{'alpha':>6} {'|w|':>8} {'KL_rb':>9} {'±se':>7} {'KL_mc':>9} {'±se':>7} "
          f"{'KL/tok':>7} {'len':>6} {'distinct':>8} {'logp_base/tok':>13}")
    for a, s in results.items():
        print(f"{a:6.2f} {a * rnorm:8.1f} {s['rb']:9.3f} {s['rb_se']:7.3f} {s['mc']:9.3f} "
              f"{s['mc_se']:7.3f} {s['rb_per_tok']:7.3f} {s['mean_len']:6.1f} "
              f"{s['distinct_frac']:8.3f} {s['logp_base_per_tok']:13.3f}")

    # ---- gate
    z, g_ = results[0.0], results[ALPHA]
    checks = {
        "(b) KL(steer||base) in range at ALPHA": KL_GATE[0] <= g_["rb"] <= KL_GATE[1],
        "(c) KL(zero||base) ~ 0": abs(z["rb"]) < ZERO_KL_TOL and abs(z["mc"]) < ZERO_KL_TOL,
        "(d) generate/score logprobs match": match["mean_abs"] < LOGPROB_MATCH_TOL,
        "(e) MC ~ RB at ALPHA (within 3 se)":
            abs(g_["mc"] - g_["rb"]) < 3 * (g_["mc_se"] ** 2 + g_["rb_se"] ** 2) ** 0.5,
    }
    print("\nGATE  (a) coherence/difference: inspect samples above")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("PHASE 0 GATE:", "PASS (pending eyeball of (a))" if all(checks.values()) else "FAIL")

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{OUT_DIR}/phase0_{ts}.json"
    cfg = {k: v for k, v in globals().items() if k.isupper()}
    json.dump({"config": cfg, "vector_path": vec_path, "residual_norm": rnorm,
               "logprob_match": match, "baseline": base_s,
               "baseline_samples": [tok.decode(base[k * N_SAMPLES_PER_PROMPT]) for k in range(N_SHOW)],
               "sweep": {str(a): s for a, s in results.items()},
               "checks": {k: bool(v) for k, v in checks.items()}},
              open(path, "w"), indent=1)
    print(f"logged -> {path}")


if __name__ == "__main__":
    main()
