"""Phase 1, step 1: build difference-of-means (CAA-style) steering vectors and calibrate alpha.

For each behavior in behaviors.py:
  1. Sample responses to N_CONTRAST train prompts under the eliciting system prompt (positive) and
     under the neutral empty system prompt (negative).
  2. Re-encode BOTH under the neutral system prompt and average the layer-LAYER residual stream
     over response tokens only. vector = mean(pos) - mean(neg).
  3. Sweep alpha (x median residual norm) on CALIB prompts and report KL(steer || neutral), KL/token,
     distinct-token fraction and the behavior metric, plus samples for eyeballing.

Vectors are saved to vectors/<name>_L<layer>.pt with the contrast data, layer and calibrated alpha
(CALIBRATED_ALPHA below; None until chosen by inspecting the sweep).

Prompt usage (train split, disjoint ranges):
  train[0:64]    calibration sweep     train[500:500+N_CONTRAST]  contrast pairs
"""
import argparse
import datetime
import json
import os
import statistics as st

import torch

from behaviors import BEHAVIORS
from common import (NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize,
                    wildchat_prompts)

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 18
CONTRAST_OFFSET, N_CONTRAST = 500, 128
CONTRAST_MAX_TOKENS = 96
CONTRAST_SEED = 7
CALIB_OFFSET, N_CALIB = 0, 64
CALIB_MAX_TOKENS = 128
CALIB_SEED = 11
ALPHA_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]  # x median residual norm
CALIBRATED_ALPHA = {"french": 0.2, "pirate": 0.6, "hedging": 0.2}  # chosen from run phase1_calib_20261120_234651
GEN_BATCH, SCORE_BATCH = 64, 16
N_SHOW = 2
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
GAUSS_PATH = "/workspace/vectors/gauss_seed0_d2048.pt"


def distinct_frac(comps):
    return st.mean(len(set(y)) / max(1, len(y)) for y in comps)


@torch.no_grad()
def mean_response_act(model, ctxs, comps, layer):
    acc, n = None, 0
    cap = {}
    hd = model.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    for c, y in zip(ctxs, comps):
        if not y:
            continue
        model.model(input_ids=torch.tensor([c + y]).cuda())
        h = cap["h"][0, len(c):].float().sum(0)
        acc = h if acc is None else acc + h
        n += len(y)
    hd.remove()
    return acc / n


@torch.no_grad()
def residual_norm(model, ctxs, layer):
    cap = []
    hd = model.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu()))
    for c in ctxs:
        model.model(input_ids=torch.tensor([c]).cuda())
    hd.remove()
    return torch.cat(cap).median().item()


def build(model, tok, name, spec, path):
    prompts = wildchat_prompts("train", N_CONTRAST, CONTRAST_OFFSET)
    neutral = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in prompts]
    elicit = [chat_ids(tok, x, spec["elicit"]) for x in prompts]
    pos = sample(model, tok, elicit, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
    neg = sample(model, tok, neutral, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
    raw = (mean_response_act(model, neutral, pos, LAYER) - mean_response_act(model, neutral, neg, LAYER)).cpu()
    pos_t, neg_t = [tok.decode(y) for y in pos], [tok.decode(y) for y in neg]
    m = spec["metric"]
    data = dict(name=name, layer=LAYER, model=MODEL_NAME, raw=raw, unit=raw / raw.norm(),
                raw_norm=raw.norm().item(), elicit_system=spec["elicit"],
                contrast_prompts=prompts, pos_responses=pos_t, neg_responses=neg_t,
                contrast_metric=dict(pos=st.mean(map(m, pos_t)), neg=st.mean(map(m, neg_t))),
                config=dict(CONTRAST_OFFSET=CONTRAST_OFFSET, N_CONTRAST=N_CONTRAST,
                            CONTRAST_MAX_TOKENS=CONTRAST_MAX_TOKENS, CONTRAST_SEED=CONTRAST_SEED),
                alpha=None, residual_norm=None)
    torch.save(data, path)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated behavior names")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(MODEL_NAME)
    calib_prompts = wildchat_prompts("train", N_CALIB, CALIB_OFFSET)
    ctxs = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in calib_prompts]
    rnorm = residual_norm(model, ctxs, LAYER)
    print(f"layer {LAYER} median residual norm (neutral system): {rnorm:.1f}")
    os.makedirs(VEC_DIR, exist_ok=True)

    base = sample(model, tok, ctxs, CALIB_MAX_TOKENS, seed=CALIB_SEED, batch_size=GEN_BATCH)
    base_t = [tok.decode(y) for y in base]
    log = {"config": {k: v for k, v in globals().items() if k.isupper()}, "residual_norm": rnorm,
           "baseline_distinct": distinct_frac(base), "vectors": {}}
    print(f"baseline distinct_frac {log['baseline_distinct']:.3f}")

    names = args.only.split(",") if args.only else list(BEHAVIORS)
    units = {}
    for name in names:
        spec = BEHAVIORS[name]
        path = f"{VEC_DIR}/{name}_L{LAYER}.pt"
        data = torch.load(path) if os.path.exists(path) and not args.rebuild else \
            build(model, tok, name, spec, path)
        data["alpha"], data["residual_norm"] = CALIBRATED_ALPHA.get(name), rnorm
        torch.save(data, path)
        units[name] = data["unit"]
        cm = data["contrast_metric"]
        print(f"\n===== {name}: |raw| {data['raw_norm']:.1f} ({data['raw_norm'] / rnorm:.2f} x resid); "
              f"contrast {spec['metric_name']} pos {cm['pos']:.2f} / neg {cm['neg']:.2f}; "
              f"calibrated alpha = {data['alpha']}")
        bm = st.mean(map(spec["metric"], base_t))
        rows = {}
        if not args.no_sweep:
            unit = data["unit"].cuda()
            print(f"{'alpha':>6} {'|w|':>6} {'KL_rb':>8} {'±se':>6} {'KL/tok':>7} {'len':>6} "
                  f"{'distinct':>8} {spec['metric_name']:>15}   (baseline metric {bm:.2f})")
            for a in ALPHA_SWEEP:
                vec = unit * a * rnorm
                comps = sample(model, tok, ctxs, CALIB_MAX_TOKENS, vec=vec, layer=LAYER,
                               seed=CALIB_SEED, batch_size=GEN_BATCH)
                s = summarize(kl_on_rollouts(model, ctxs, ctxs, comps, vec_p=vec, layer=LAYER,
                                             batch_size=SCORE_BATCH))
                txt = [tok.decode(y) for y in comps]
                s.update(distinct_frac=distinct_frac(comps), metric=st.mean(map(spec["metric"], txt)),
                         samples=txt[:N_SHOW])
                rows[a] = s
                print(f"{a:6.2f} {a * rnorm:6.1f} {s['rb']:8.2f} {s['rb_se']:6.2f} {s['rb_per_tok']:7.3f} "
                      f"{s['mean_len']:6.1f} {s['distinct_frac']:8.3f} {s['metric']:15.2f}")
            for a, s in rows.items():
                print(f"\n  -- alpha {a}")
                for k, t in enumerate(s["samples"]):
                    print(f"   [{calib_prompts[k][:60]!r}] -> {t[:220]!r}")
        log["vectors"][name] = dict(path=path, raw_norm=data["raw_norm"], contrast_metric=cm,
                                    baseline_metric=bm, alpha=data["alpha"],
                                    sweep={str(a): s for a, s in rows.items()})

    if os.path.exists(GAUSS_PATH):
        units["gauss"] = torch.load(GAUSS_PATH)
    ks = list(units)
    print("\ncosine similarities:")
    for i, a in enumerate(ks):
        print("  " + " ".join(f"{a}/{b}={torch.dot(units[a], units[b]).item():+.3f}" for b in ks[i + 1:]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/phase1_calib_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(log, open(out, "w"), indent=1, default=str)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
