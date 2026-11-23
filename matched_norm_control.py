"""Phase 1, step 4b: matched-norm random control -- the unstructured-perturbation floor.

Question: how much of KL(p_steer || q_neutral) for a meaningful vector w is just "a vector of norm
||w|| was added at layer 18", which no system prompt can reproduce?

For each vector w (French, pirate, hedging; calibrated alpha from vectors/*.pt):
  1. Draw N_DIRS isotropic Gaussian directions, rescale each to exactly ||w||.
  2. For each random direction r: sample rollouts from p_r = model + r (neutral system prompt) on the
     SAME held-out prompts / seed as the ladder, and compute fwd KL(p_r || q_neutral).
     Floor F(w) = mean over directions. This is the unstructured-perturbation floor.
  3. Re-run the ladder (rungs 0-3) fwd KL for w itself on the same held-out set, keeping per-prompt
     values, so closure can be recomputed with paired uncertainty:
         closed_vs_rung0 = (KL0 - KL_S) / KL0                 (original definition)
         closed_vs_floor = (KL0 - KL_S) / (KL0 - F)           (fraction of the CLOSABLE gap)
     SE by two-level bootstrap: resample prompts (paired across conditions) and random directions.

Caveat (stated in output): floor subtraction assumes the structured and unstructured parts of the KL
add. They need not; a random direction is not "the noise component of w". Treat F as an estimate of
scale, not an exact decomposition.
"""
import datetime
import json
import os
import statistics as st

import numpy as np
import torch

from behaviors import ALL_BEHAVIORS as BEHAVIORS
from common import NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize, wildchat_prompts
from oracle_search import distinct_frac, ladder

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
VECTORS = ["french", "pirate", "hedging"]
VEC_LAYER = {"french": 18, "pirate": 18, "hedging": 18}   # which vectors/<name>_L<layer>.pt to load
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
LADDER_RUN = "/workspace/runs/phase1_ladder_20261120_235111.json"   # for reproduction cross-check
HELDOUT_OFFSET, N_HELDOUT = 0, 200        # identical to oracle_search.py
MAX_NEW_TOKENS = 128
SEED_P = 101                               # identical to oracle_search.py (steered rollouts)
N_DIRS = 8
DIR_SEED_BASE = 5000                       # direction k uses torch seed DIR_SEED_BASE + k
GEN_BATCH, SCORE_BATCH = 100, 16
N_BOOT = 4000
BOOT_SEED = 0
N_DUMP = 2


def random_dirs(dim, norm):
    out = []
    for k in range(N_DIRS):
        g = torch.Generator().manual_seed(DIR_SEED_BASE + k)
        v = torch.randn(dim, generator=g)
        out.append(v / v.norm() * norm)
    return out


def steered_fwd(model, tok, ctx, vec, layer):
    """Rollouts from model+vec under neutral context, scored against unsteered neutral."""
    y = sample(model, tok, ctx, MAX_NEW_TOKENS, vec=vec, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
    res = kl_on_rollouts(model, ctx, ctx, y, vec_p=vec, layer=layer, batch_size=SCORE_BATCH)
    return y, res


def bootstrap_closure(kl0, kls, floor_mat, rng):
    """kl0, kls: [n_prompts] per-prompt fwd KL for rung 0 and rung S (paired).
    floor_mat: [n_dirs, n_prompts] per-prompt fwd KL under each random direction.
    Returns (point, se) of (mean kl0 - mean kls) / (mean kl0 - mean floor)."""
    D = kl0 - kls
    point = D.mean() / (kl0.mean() - floor_mat.mean())
    n, K = len(kl0), floor_mat.shape[0]
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        k = rng.integers(0, K, K)
        bs[b] = D[i].mean() / (kl0[i].mean() - floor_mat[np.ix_(k, i)].mean())
    return float(point), float(bs.std(ddof=1))


def main():
    global VECTORS, LADDER_RUN
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--layer", default=None, help="override VEC_LAYER, e.g. pirate=14")
    ap.add_argument("--ladder-run", default=None, help="ladder json for the repro cross-check")
    args = ap.parse_args()
    if args.only:
        VECTORS = args.only.split(",")
    if args.layer:
        for kv in args.layer.split(","):
            k, v = kv.split("=")
            VEC_LAYER[k] = int(v)
    if args.ladder_run:
        LADDER_RUN = args.ladder_run
    model, tok = load_model(MODEL_NAME)
    xs = wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    prior = {r["name"]: {row["rung"]: row for row in r["rows"]} for r in json.load(open(LADDER_RUN))["results"]}
    rng = np.random.default_rng(BOOT_SEED)

    floor_cache, results = {}, []
    for name in VECTORS:
        vd = torch.load(f"{VEC_DIR}/{name}_L{VEC_LAYER[name]}.pt")
        layer = vd["layer"]
        norm = vd["alpha"] * vd["residual_norm"]
        w = (vd["unit"] * norm).cuda()
        print(f"\n===== {name}: layer {layer}, alpha {vd['alpha']}, ||w|| = {norm:.2f}", flush=True)

        # ---- random-direction floor (shared across vectors with identical norm)
        key = (layer, round(norm, 4))
        if key not in floor_cache:
            dirs = random_dirs(vd["unit"].numel(), norm)
            per_dir = []
            for k, r in enumerate(dirs):
                y, res = steered_fwd(model, tok, ctx0, r.cuda(), layer)
                s = summarize(res)
                txt = [tok.decode(t) for t in y]
                per_dir.append(dict(k=k, seed=DIR_SEED_BASE + k, fwd=s["rb"], fwd_se=s["rb_se"],
                                    fwd_mc=s["mc"], fwd_tok=s["rb_per_tok"], mean_len=s["mean_len"],
                                    distinct=distinct_frac(y), per_prompt=res["rb"], samples=txt[:N_DUMP],
                                    unit=(r / r.norm()).tolist()))
                print(f"  rand dir {k}: fwd {s['rb']:6.2f}±{s['rb_se']:.2f} (mc {s['mc']:6.2f})  "
                      f"KL/tok {s['rb_per_tok']:.3f}  distinct {per_dir[-1]['distinct']:.3f}", flush=True)
            floor_cache[key] = per_dir
        per_dir = floor_cache[key]
        cos = [float(torch.dot(torch.tensor(d["unit"]), vd["unit"].float())) for d in per_dir]
        floor_mat = np.array([d["per_prompt"] for d in per_dir])
        dir_means = floor_mat.mean(1)
        F, F_sd = float(dir_means.mean()), float(dir_means.std(ddof=1))
        F_se = F_sd / len(dir_means) ** 0.5

        # ---- ladder on the real vector, per-prompt
        y_w, res0 = steered_fwd(model, tok, ctx0, w, layer)
        rows = []
        per_rung = {}
        for rung, S in ladder(name):
            ctx_q = [chat_ids(tok, x, S) for x in xs]
            res = res0 if S == NEUTRAL_SYSTEM else kl_on_rollouts(model, ctx0, ctx_q, y_w, vec_p=w, layer=layer,
                                                                   batch_size=SCORE_BATCH)
            per_rung[rung] = np.array(res["rb"])
        kl0 = per_rung["0 neutral"]
        for rung, S in ladder(name):
            v = per_rung[rung]
            d = kl0 - v
            c0, c0_se = d.mean() / kl0.mean(), d.std(ddof=1) / len(d) ** 0.5 / kl0.mean()
            cf, cf_se = bootstrap_closure(kl0, v, floor_mat, rng)
            old = prior.get(name, {}).get(rung, {})
            rows.append(dict(rung=rung, S=S, fwd=float(v.mean()), fwd_se=float(v.std(ddof=1) / len(v) ** 0.5),
                             closed_vs_rung0=float(c0), closed_vs_rung0_se=float(c0_se),
                             closed_vs_floor=cf, closed_vs_floor_se=cf_se,
                             prior_fwd=old.get("fwd"), prior_closed=old.get("closed")))
        gap = float(kl0.mean()) - F
        results.append(dict(name=name, layer=layer, alpha=vd["alpha"], norm=norm,
                            steered_distinct=distinct_frac(y_w), kl0=float(kl0.mean()),
                            floor=F, floor_sd_across_dirs=F_sd, floor_se=F_se,
                            floor_min=float(dir_means.min()), floor_max=float(dir_means.max()),
                            floor_frac_of_kl0=F / float(kl0.mean()), closable_gap=gap,
                            cos_rand_vs_w=cos, rows=rows,
                            random_dirs=[{k: v for k, v in d.items() if k not in ("per_prompt", "unit")}
                                         for d in per_dir]))

    # ---------------------------------------------------------------- report
    print("\n\n## Unstructured-perturbation floor (held-out, fwd KL vs neutral, nats/seq)\n")
    print("| Vector | ‖w‖ | KL(steer‖neutral) | floor F: mean ± SE over dirs | dir range | F / KL0 | "
          "closable gap KL0−F | random distinct | steered distinct |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        rd = st.mean(d["distinct"] for d in r["random_dirs"])
        print(f"| {r['name']} | {r['norm']:.1f} | {r['kl0']:.2f} | {r['floor']:.2f} ± {r['floor_se']:.2f} | "
              f"{r['floor_min']:.1f}–{r['floor_max']:.1f} | {r['floor_frac_of_kl0']:.0%} | {r['closable_gap']:.2f} | "
              f"{rd:.3f} | {r['steered_distinct']:.3f} |")
    for r in results:
        print(f"\n### {r['name']}  (max |cos(random, w)| = {max(abs(c) for c in r['cos_rand_vs_w']):.3f})\n")
        print("| Rung | S | KL fwd ± SE | (prior run) | closed vs rung 0 | closed vs floor |")
        print("|---|---|---|---|---|---|")
        for x in r["rows"]:
            s = (x["S"][:50] + "…") if len(x["S"]) > 50 else x["S"]
            pf = f"{x['prior_fwd']:.2f}" if x["prior_fwd"] is not None else "—"
            print(f"| {x['rung']} | {s!r} | {x['fwd']:.2f} ± {x['fwd_se']:.2f} | {pf} | "
                  f"{x['closed_vs_rung0']:+.1%} ± {x['closed_vs_rung0_se']:.1%} | "
                  f"{x['closed_vs_floor']:+.1%} ± {x['closed_vs_floor_se']:.1%} |")
    print("\nSanity: KL0 should reproduce the prior ladder run's rung-0 value (same prompts, seed); "
          "|cos(random, w)| should be ~1/sqrt(2048) ≈ 0.02.")
    print("Caveat: closed-vs-floor assumes structured and unstructured KL add; F is a scale estimate.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/phase1_randfloor_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    cfg = {k: v for k, v in globals().items() if k.isupper()}
    json.dump({"config": cfg, "results": results}, open(out, "w"), indent=1, default=str)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
