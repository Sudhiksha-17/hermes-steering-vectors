"""Phase 1b, run 3 (part 1): pirate layer sweep -- build + calibrate pirate vectors at earlier layers.

Pirate at layer 18 is indeterminate (F/KL0 = 75%, best prompt inside the floor's spread). Look for a
layer where the pirate lexicon appears before fluency degrades and before the norm cost dominates.

For each layer in LAYERS:
  1. Difference-of-means pirate vector, same recipe as build_vectors.py (contrast responses are
     layer-independent, so they are sampled once and re-encoded at every layer in one pass).
  2. alpha sweep (x median residual norm AT THAT LAYER) on calibration prompts train[0:64]:
     KL0 = KL(steer || neutral), distinct fraction, pirate words/100w, samples.
  3. Matched-norm random control per cell: N_RAND isotropic directions at the same norm, same prompts
     -> F and F/KL0, the run-3 selection criterion (maximise (KL0-F)/KL0 subject to coherent text).
Vectors saved to vectors/pirate_L<layer>.pt with alpha=None; alpha is chosen by inspecting this
sweep (samples printed) and written into the vector file (alpha, alpha_source).
"""
import argparse
import datetime
import json
import os
import statistics as st

import torch

from behaviors import BEHAVIORS
from common import NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize, wildchat_prompts

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NAME = "pirate"
LAYERS = [8, 10, 12, 14]
CONTRAST_OFFSET, N_CONTRAST = 500, 128      # identical to build_vectors.py
CONTRAST_MAX_TOKENS = 96
CONTRAST_SEED = 7
CALIB_OFFSET, N_CALIB = 0, 64               # identical to build_vectors.py
CALIB_MAX_TOKENS = 128
CALIB_SEED = 11
ALPHA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5]
N_RAND = 3
RAND_SEED_BASE = 7000
GEN_BATCH, SCORE_BATCH = 64, 16
N_SHOW = 3
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"


def distinct_frac(comps):
    return st.mean(len(set(y)) / max(1, len(y)) for y in comps)


@torch.no_grad()
def mean_response_acts(model, ctxs, comps, layers):
    cap, hs = {}, []
    for L in layers:
        hs.append(model.model.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
    acc, n = {L: 0 for L in layers}, 0
    for c, y in zip(ctxs, comps):
        if not y:
            continue
        model.model(input_ids=torch.tensor([c + y]).cuda())
        for L in layers:
            acc[L] = acc[L] + cap[L][0, len(c):].float().sum(0)
        n += len(y)
    for h in hs:
        h.remove()
    return {L: (acc[L] / n).cpu() for L in layers}


@torch.no_grad()
def residual_norms(model, ctxs, layers):
    cap = {L: [] for L in layers}
    hs = [model.model.layers[L].register_forward_hook(
        lambda _m, _i, o, L=L: cap[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu()))
        for L in layers]
    for c in ctxs:
        model.model(input_ids=torch.tensor([c]).cuda())
    for h in hs:
        h.remove()
    return {L: torch.cat(cap[L]).median().item() for L in layers}


def main():
    global LAYERS, ALPHA_SWEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None, help="comma-separated override of LAYERS")
    ap.add_argument("--alphas", default=None, help="'L:a,a;L:a,a' per-layer alpha override")
    args = ap.parse_args()
    per_layer = {}
    if args.alphas:
        for part in args.alphas.split(";"):
            L, al = part.split(":")
            per_layer[int(L)] = [float(a) for a in al.split(",")]
        LAYERS = sorted(per_layer)
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",")]
    model, tok = load_model(MODEL_NAME)
    spec = BEHAVIORS[NAME]
    metric = spec["metric"]

    # ---- contrast data (layer-independent)
    prompts = wildchat_prompts("train", N_CONTRAST, CONTRAST_OFFSET)
    neutral = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in prompts]
    elicit = [chat_ids(tok, x, spec["elicit"]) for x in prompts]
    pos = sample(model, tok, elicit, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
    neg = sample(model, tok, neutral, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
    mp, mn = mean_response_acts(model, neutral, pos, LAYERS), mean_response_acts(model, neutral, neg, LAYERS)
    pos_t, neg_t = [tok.decode(y) for y in pos], [tok.decode(y) for y in neg]
    cm = dict(pos=st.mean(map(metric, pos_t)), neg=st.mean(map(metric, neg_t)))
    print(f"contrast pirate/100w: pos {cm['pos']:.2f} / neg {cm['neg']:.2f}", flush=True)

    calib = wildchat_prompts("train", N_CALIB, CALIB_OFFSET)
    ctxs = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in calib]
    rnorm = residual_norms(model, ctxs, LAYERS)
    base = sample(model, tok, ctxs, CALIB_MAX_TOKENS, seed=CALIB_SEED, batch_size=GEN_BATCH)
    base_t = [tok.decode(y) for y in base]
    log = {"config": {k: v for k, v in globals().items() if k.isupper()}, "alpha_override": per_layer,
           "contrast_metric": cm,
           "baseline_distinct": distinct_frac(base), "baseline_metric": st.mean(map(metric, base_t)), "layers": {}}
    print(f"baseline distinct {log['baseline_distinct']:.3f}, pirate/100w {log['baseline_metric']:.2f}", flush=True)
    l18 = torch.load(f"{VEC_DIR}/pirate_L18.pt")["unit"]

    for L in LAYERS:
        raw = mp[L] - mn[L]
        unit = raw / raw.norm()
        torch.save(dict(name=NAME, layer=L, model=MODEL_NAME, raw=raw, unit=unit, raw_norm=raw.norm().item(),
                        elicit_system=spec["elicit"], contrast_prompts=prompts, pos_responses=pos_t,
                        neg_responses=neg_t, contrast_metric=cm,
                        config=dict(CONTRAST_OFFSET=CONTRAST_OFFSET, N_CONTRAST=N_CONTRAST,
                                    CONTRAST_MAX_TOKENS=CONTRAST_MAX_TOKENS, CONTRAST_SEED=CONTRAST_SEED),
                        alpha=None, residual_norm=rnorm[L]), f"{VEC_DIR}/pirate_L{L}.pt")
        print(f"\n===== layer {L}: median resid {rnorm[L]:.1f}, |raw| {raw.norm():.1f} "
              f"({raw.norm() / rnorm[L]:.2f} x resid), cos(unit, L18 unit) {torch.dot(unit, l18):+.3f}", flush=True)
        print(f"{'alpha':>6} {'|w|':>6} {'KL0':>8} {'F':>7} {'F/KL0':>6} {'gap':>7} {'KL0/tok':>7} "
              f"{'distinct':>8} {'pirate':>7} {'rand_dist':>9}", flush=True)
        rows = {}
        for a in per_layer.get(L, ALPHA_SWEEP):
            norm = a * rnorm[L]
            vec = (unit * norm).cuda()
            y = sample(model, tok, ctxs, CALIB_MAX_TOKENS, vec=vec, layer=L, seed=CALIB_SEED, batch_size=GEN_BATCH)
            s = summarize(kl_on_rollouts(model, ctxs, ctxs, y, vec_p=vec, layer=L, batch_size=SCORE_BATCH))
            txt = [tok.decode(t) for t in y]
            F, rdist = [], []
            for k in range(N_RAND):
                g = torch.Generator().manual_seed(RAND_SEED_BASE + k)
                v = torch.randn(unit.numel(), generator=g)
                r = (v / v.norm() * norm).cuda()
                ry = sample(model, tok, ctxs, CALIB_MAX_TOKENS, vec=r, layer=L, seed=CALIB_SEED, batch_size=GEN_BATCH)
                F.append(summarize(kl_on_rollouts(model, ctxs, ctxs, ry, vec_p=r, layer=L,
                                                  batch_size=SCORE_BATCH))["rb"])
                rdist.append(distinct_frac(ry))
            row = dict(alpha=a, norm=norm, kl0=s["rb"], kl0_se=s["rb_se"], kl0_tok=s["rb_per_tok"],
                       F=st.mean(F), F_dirs=F, F_frac=st.mean(F) / s["rb"], gap=s["rb"] - st.mean(F),
                       distinct=distinct_frac(y), metric=st.mean(map(metric, txt)), rand_distinct=st.mean(rdist),
                       samples=txt[:N_SHOW])
            rows[a] = row
            print(f"{a:6.2f} {norm:6.1f} {row['kl0']:8.2f} {row['F']:7.2f} {row['F_frac']:6.0%} {row['gap']:7.2f} "
                  f"{row['kl0_tok']:7.3f} {row['distinct']:8.3f} {row['metric']:7.2f} {row['rand_distinct']:9.3f}",
                  flush=True)
        for a, row in rows.items():
            print(f"\n  -- layer {L} alpha {a}")
            for k, t in enumerate(row["samples"]):
                print(f"   [{calib[k][:50]!r}] -> {t[:260]!r}")
        log["layers"][str(L)] = dict(residual_norm=rnorm[L], raw_norm=raw.norm().item(),
                                     cos_L18=torch.dot(unit, l18).item(), sweep={str(a): r for a, r in rows.items()})

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/phase1b_piratesweep_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(log, open(out, "w"), indent=1, default=str)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
