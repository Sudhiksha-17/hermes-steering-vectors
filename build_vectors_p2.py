"""Phase 2: build + screen the candidate training vectors (difference of means, same recipe as Phase 1).

For each behavior in behaviors.BEHAVIORS_P2:
  1. Contrast responses under ELICIT (positive) vs the empty system prompt (negative; sampled ONCE, shared),
     re-encoded under the empty system prompt; mean residual over response tokens at every layer in
     SAVE_LAYERS (one forward pass). Saves vectors/<name>_L<layer>.pt (alpha=None).
  2. alpha sweep at SWEEP_LAYER on the Step 3 calibration prompts train[0:64]:
       KL0 = KL(steer || neutral), KL1 = KL(steer || rung 1), presence cost c = KL1 - KL0,
       F(alpha) = mean over N_RAND matched-norm random directions (shared across vectors: the residual
       norm at a fixed layer is the same for all), distinct fraction, behavior metric, samples.
  3. Screening criterion (PHASE2_SPEC "Vector distribution"):
       gap = KL0 - F >= PRESENCE_MULT * max(c, 0)   and   gap >= MIN_GAP_NATS   and   F/KL0 <= MAX_F_FRAC
     plus coherence (distinct >= MIN_DISTINCT) and behavior shift (metric moved >= MIN_SHIFT of the
     contrast pos-neg difference). The suggested alpha is the smallest cell passing all of these; the
     final choice is confirmed by eye on the printed samples (Step 3 protocol) and recorded separately.
"""
import argparse
import datetime
import json
import os
import statistics as st

import torch

from behaviors import BEHAVIORS_P2, RUNG1
from common import NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize, wildchat_prompts

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAVE_LAYERS = [8, 12, 18]
SWEEP_LAYER = 18
CONTRAST_OFFSET, N_CONTRAST = 500, 128      # identical to build_vectors.py
CONTRAST_MAX_TOKENS = 96
CONTRAST_SEED = 7
CALIB_OFFSET, N_CALIB = 0, 64               # identical to build_vectors.py
CALIB_MAX_TOKENS = 128
CALIB_SEED = 11
ALPHA_SWEEP = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
N_RAND = 3
RAND_SEED_BASE = 7000
PRESENCE_MULT = 3.0                         # spec: structured component > ~3x rung-1 presence cost
MIN_GAP_NATS = 15.0                         # hedging's ~10-nat gap is the known-unusable reference
MAX_F_FRAC = 0.6                            # pirate L18 (0.75) is the known-unusable reference
MIN_DISTINCT = 0.65                         # baseline 0.70
MIN_SHIFT = 0.5                             # metric moved >= 50% of the contrast (pos - neg) difference
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
def residual_norm(model, ctxs, layer):
    cap = []
    h = model.model.layers[layer].register_forward_hook(
        lambda _m, _i, o: cap.append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu()))
    for c in ctxs:
        model.model(input_ids=torch.tensor([c]).cuda())
    h.remove()
    return torch.cat(cap).median().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--sweep-layer", type=int, default=None, help="override SWEEP_LAYER")
    ap.add_argument("--alphas", default=None, help="comma list overriding ALPHA_SWEEP")
    ap.add_argument("--reuse", action="store_true", help="load saved raw vectors instead of regenerating contrasts")
    ap.add_argument("--center", action="store_true",
                    help="mean-center: raw_i - mean_j raw_j over all BEHAVIORS_P2 (from saved files); saves <name>_c_L<layer>.pt")
    args = ap.parse_args()
    global SWEEP_LAYER, ALPHA_SWEEP
    if args.sweep_layer is not None:
        SWEEP_LAYER = args.sweep_layer
    if args.alphas:
        ALPHA_SWEEP = [float(a) for a in args.alphas.split(",")]
    names = args.only.split(",") if args.only else list(BEHAVIORS_P2)
    model, tok = load_model(MODEL_NAME)
    centered = {}
    if args.center:
        for Ls in SAVE_LAYERS:
            raws = {n: torch.load(f"{VEC_DIR}/{n}_L{Ls}.pt")["raw"].float() for n in BEHAVIORS_P2}
            mu = torch.stack(list(raws.values())).mean(0)
            for n in BEHAVIORS_P2:
                d = torch.load(f"{VEC_DIR}/{n}_L{Ls}.pt")
                raw = raws[n] - mu
                d.update(name=f"{n}_c", raw=raw, unit=raw / raw.norm(), raw_norm=raw.norm().item(), centered_from=n,
                         center_mean_over=list(BEHAVIORS_P2), alpha=None, residual_norm=None)
                torch.save(d, f"{VEC_DIR}/{n}_c_L{Ls}.pt")
                if Ls == SWEEP_LAYER:
                    centered[n] = raw

    prompts = wildchat_prompts("train", N_CONTRAST, CONTRAST_OFFSET)
    neutral = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in prompts]
    if not (args.center or args.reuse):
        neg = sample(model, tok, neutral, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
        m_neg = mean_response_acts(model, neutral, neg, SAVE_LAYERS)
        neg_t = [tok.decode(y) for y in neg]

    calib = wildchat_prompts("train", N_CALIB, CALIB_OFFSET)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in calib]
    ctx1 = [chat_ids(tok, x, RUNG1) for x in calib]
    L = SWEEP_LAYER
    rnorm = residual_norm(model, ctx0, L)
    base = sample(model, tok, ctx0, CALIB_MAX_TOKENS, seed=CALIB_SEED, batch_size=GEN_BATCH)
    base_t = [tok.decode(y) for y in base]
    print(f"layer {L} median residual norm {rnorm:.2f}; baseline distinct {distinct_frac(base):.3f}", flush=True)

    # shared random floor per alpha
    F = {}
    for a in ALPHA_SWEEP:
        vals = []
        for k in range(N_RAND):
            g = torch.Generator().manual_seed(RAND_SEED_BASE + k)
            v = torch.randn(model.config.hidden_size, generator=g)
            r = (v / v.norm() * a * rnorm).cuda()
            ry = sample(model, tok, ctx0, CALIB_MAX_TOKENS, vec=r, layer=L, seed=CALIB_SEED, batch_size=GEN_BATCH)
            vals.append(summarize(kl_on_rollouts(model, ctx0, ctx0, ry, vec_p=r, layer=L, batch_size=SCORE_BATCH))["rb"])
        F[a] = dict(mean=st.mean(vals), dirs=vals)
        print(f"  F(alpha={a}) = {F[a]['mean']:.2f}  dirs {[round(x, 1) for x in vals]}", flush=True)

    log = {"config": {k: v for k, v in globals().items() if k.isupper()}, "args": vars(args), "residual_norm": rnorm,
           "baseline_distinct": distinct_frac(base), "F": F, "vectors": {}}
    units = {}
    for name in names:
        spec = BEHAVIORS_P2[name]
        metric = spec["metric"]
        if args.center:
            d18 = torch.load(f"{VEC_DIR}/{name}_c_L{L}.pt")
            cm, raw = d18["contrast_metric"], centered[name]
            unit = raw / raw.norm()
            units[f"{name}_c"] = unit
            bm = st.mean(map(metric, base_t))
            need = MIN_SHIFT * (cm["pos"] - cm["neg"])
            print(f"\n===== {name}_c (centered): |raw| {raw.norm():.1f} ({raw.norm() / rnorm:.2f} x resid); cos(centered, "
                  f"uncentered) {torch.dot(unit, torch.load(f'{VEC_DIR}/{name}_L{L}.pt')['unit'].float()):+.3f}", flush=True)
        elif args.reuse:
            d = torch.load(f"{VEC_DIR}/{name}_L{L}.pt")
            cm, raw = d["contrast_metric"], d["raw"].float()
            unit = raw / raw.norm()
            units[name] = unit
            bm = st.mean(map(metric, base_t))
            need = MIN_SHIFT * (cm["pos"] - cm["neg"])
            print(f"\n===== {name} @L{L} (reused): |raw| {raw.norm():.1f} ({raw.norm() / rnorm:.2f} x resid); contrast "
                  f"{spec['metric_name']} pos {cm['pos']:.2f} / neg {cm['neg']:.2f}; baseline {bm:.2f}", flush=True)
        else:
          elicit = [chat_ids(tok, x, spec["elicit"]) for x in prompts]
          pos = sample(model, tok, elicit, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
          m_pos = mean_response_acts(model, neutral, pos, SAVE_LAYERS)
          pos_t = [tok.decode(y) for y in pos]
          cm = dict(pos=st.mean(map(metric, pos_t)), neg=st.mean(map(metric, neg_t)))
          for Ls in SAVE_LAYERS:
            raw = m_pos[Ls] - m_neg[Ls]
            torch.save(dict(name=name, layer=Ls, model=MODEL_NAME, raw=raw, unit=raw / raw.norm(), raw_norm=raw.norm().item(),
                            elicit_system=spec["elicit"], contrast_prompts=prompts, pos_responses=pos_t,
                            neg_responses=neg_t, contrast_metric=cm,
                            config=dict(CONTRAST_OFFSET=CONTRAST_OFFSET, N_CONTRAST=N_CONTRAST,
                                        CONTRAST_MAX_TOKENS=CONTRAST_MAX_TOKENS, CONTRAST_SEED=CONTRAST_SEED),
                            alpha=None, residual_norm=None), f"{VEC_DIR}/{name}_L{Ls}.pt")
          raw = m_pos[L] - m_neg[L]
          unit = (raw / raw.norm())
          units[name] = unit
          bm = st.mean(map(metric, base_t))
          need = MIN_SHIFT * (cm["pos"] - cm["neg"])
          print(f"\n===== {name}: |raw| {raw.norm():.1f} ({raw.norm() / rnorm:.2f} x resid); contrast "
              f"{spec['metric_name']} pos {cm['pos']:.2f} / neg {cm['neg']:.2f}; baseline {bm:.2f}", flush=True)
        print(f"{'alpha':>6} {'KL0':>7} {'KL1':>7} {'c':>6} {'F':>6} {'gap':>7} {'F/KL0':>6} {'distinct':>8} "
              f"{'metric':>8} {'pass':>5}", flush=True)
        rows, suggested = {}, None
        for a in ALPHA_SWEEP:
            vec = (unit * a * rnorm).cuda()
            y = sample(model, tok, ctx0, CALIB_MAX_TOKENS, vec=vec, layer=L, seed=CALIB_SEED, batch_size=GEN_BATCH)
            k0 = summarize(kl_on_rollouts(model, ctx0, ctx0, y, vec_p=vec, layer=L, batch_size=SCORE_BATCH))["rb"]
            k1 = summarize(kl_on_rollouts(model, ctx0, ctx1, y, vec_p=vec, layer=L, batch_size=SCORE_BATCH))["rb"]
            txt = [tok.decode(t) for t in y]
            mt = st.mean(map(metric, txt))
            c, gap, ff = k1 - k0, k0 - F[a]["mean"], F[a]["mean"] / k0
            ok = dict(criterion=gap >= PRESENCE_MULT * max(c, 0) and gap >= MIN_GAP_NATS and ff <= MAX_F_FRAC,
                      coherent=distinct_frac(y) >= MIN_DISTINCT, shift=(mt - bm) >= need)
            row = dict(alpha=a, kl0=k0, kl1=k1, presence=c, F=F[a]["mean"], gap=gap, F_frac=ff,
                       distinct=distinct_frac(y), metric=mt, ok=ok, samples=txt[:N_SHOW])
            rows[a] = row
            allok = all(ok.values())
            if allok and suggested is None:
                suggested = a
            flags = "".join(sym if ok[k] else "." for k, sym in (("criterion", "G"), ("coherent", "D"), ("shift", "B")))
            print(f"{a:6.2f} {k0:7.2f} {k1:7.2f} {c:6.2f} {F[a]['mean']:6.2f} {gap:7.2f} {ff:6.0%} "
                  f"{row['distinct']:8.3f} {mt:8.2f} {flags:>5}", flush=True)
        print(f"  suggested alpha: {suggested}   (flags: G=gap criterion, D=distinct/coherent, B=behavior shift; '.' = fail)")
        for a in ALPHA_SWEEP:
            print(f"  -- alpha {a}")
            for k, t in enumerate(rows[a]["samples"]):
                print(f"     [{calib[k][:40]!r}] -> {t[:240]!r}")
        log["vectors"][f"{name}_c" if args.center else name] = dict(raw_norm=raw.norm().item(), contrast_metric=cm, baseline_metric=bm,
                                    metric_name=spec["metric_name"], suggested_alpha=suggested,
                                    sweep={str(a): r for a, r in rows.items()})

    # cosines at the sweep layer, including the Phase 1 anchors
    for nm, Ls in [("french", 18), ("pirate", 8), ("hedging", 18), ("pirate", 18)]:
        p = f"{VEC_DIR}/{nm}_L{Ls}.pt"
        if os.path.exists(p) and Ls == L:
            units[f"{nm}_L{Ls}"] = torch.load(p)["unit"]
    ks = list(units)
    cos = {f"{a}/{b}": float(torch.dot(units[a].float(), units[b].float())) for i, a in enumerate(ks) for b in ks[i + 1:]}
    log["cosines_L18"] = cos
    print("\nlargest |cos| pairs at L18:")
    for k, v in sorted(cos.items(), key=lambda kv: -abs(kv[1]))[:15]:
        print(f"  {k}: {v:+.3f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/phase2_vecscreen{'_centered' if args.center else ''}_L{SWEEP_LAYER}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(log, open(out, "w"), indent=1, default=str)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
