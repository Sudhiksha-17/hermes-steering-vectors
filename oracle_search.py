"""Phase 1, step 2: the per-vector oracle -- evaluate the candidate-prompt ladder.

For each vector w (built + calibrated by build_vectors.py):
  p = model + alpha*w at LAYER, neutral (empty) system prompt
  q = model, no steering, system prompt S        (contexts differ ONLY in the system text)

Per candidate S we report, on HELD-OUT prompts:
  fwd KL(p || q_S)   on rollouts Y ~ p (shared across S, so differences are paired)
  rev KL(q_S || p)   on rollouts Y ~ q_S
  distinct-token fraction and behavior metric of q_S's own samples (degeneration / behavior check)
  readability of S   (perplexity of S as raw text under the subject model; heuristic threshold)
"Best S" is selected by fwd KL on a separate TRAIN prompt set, then reported on held-out.

Headline per vector: fraction of the rung-0 fwd KL closed by best S, with rung 2 alongside.
"""
import argparse
import datetime
import json
import math
import os
import statistics as st

import torch

from behaviors import ALL_BEHAVIORS as BEHAVIORS, RUNG1
from common import (NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize,
                    wildchat_prompts)

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
VECTORS = ["french", "pirate", "hedging"]
VEC_LAYER = {"french": 18, "pirate": 18, "hedging": 18}   # which vectors/<name>_L<layer>.pt to load
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
HELDOUT_OFFSET, N_HELDOUT = 0, 200
SELECT_OFFSET, N_SELECT = 64, 128          # train[64:192]: disjoint from calib (0:64) and contrast (500:)
MAX_NEW_TOKENS = 128
SEED_P, SEED_Q = 101, 202
GEN_BATCH, SCORE_BATCH = 100, 16
READABLE_PPL = 2000.0                       # coarse gibberish filter; hand-written S: 9-509, junk: 66k+ (phase1 check)
READABILITY_PREFIX = ("Below is a system prompt that configures the behavior of an AI assistant.\n\n"
                      "System prompt: ")
N_DUMP = 4                                  # samples dumped to the log per condition


def distinct_frac(comps):
    return st.mean(len(set(y)) / max(1, len(y)) for y in comps)


@torch.no_grad()
def text_ppl(model, tok, s):
    """Perplexity of s's tokens under the subject model, conditioned on READABILITY_PREFIX.
    (Unconditioned raw-text ppl flags short readable strings like 'Respond in French.' at 3420.)"""
    if not s:
        return float("nan")
    pre = tok.encode(READABILITY_PREFIX)
    x = torch.tensor([pre + tok.encode(s)]).cuda()
    lp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
    nll = -lp.gather(-1, x[0, 1:, None]).squeeze(-1)[len(pre) - 1:].mean().item()
    return math.exp(nll)


def ladder(name):
    b = BEHAVIORS[name]
    return [("0 neutral", NEUTRAL_SYSTEM), ("1 generic", RUNG1), ("2 one-liner", b["rung2"])] + \
           [(f"3{chr(97 + i)} elaborate", s) for i, s in enumerate(b["rung3"])]


def eval_vector(model, tok, name, extra=()):
    vd = torch.load(f"{VEC_DIR}/{name}_L{VEC_LAYER[name]}.pt")
    assert vd["alpha"] is not None, f"{name}: alpha not calibrated (set CALIBRATED_ALPHA, rerun build_vectors)"
    layer = vd["layer"]
    vec = (vd["unit"] * vd["alpha"] * vd["residual_norm"]).cuda()
    spec = BEHAVIORS[name]
    metric = spec["metric"]

    sets = {"heldout": wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET),
            "select": wildchat_prompts("train", N_SELECT, SELECT_OFFSET)}
    P = {}
    for k, xs in sets.items():
        ctx_p = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
        y = sample(model, tok, ctx_p, MAX_NEW_TOKENS, vec=vec, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
        P[k] = (xs, ctx_p, y)
    xs, ctx_p, y_p = P["heldout"]
    yp_txt = [tok.decode(t) for t in y_p]
    target = dict(distinct=distinct_frac(y_p), metric=st.mean(map(metric, yp_txt)), samples=yp_txt[:N_DUMP])

    rows, rb0 = [], None
    for rung, S in list(ladder(name)) + list(extra):
        sel_xs, sel_ctx_p, sel_y = P["select"]
        sel = summarize(kl_on_rollouts(model, sel_ctx_p, [chat_ids(tok, x, S) for x in sel_xs], sel_y,
                                       vec_p=vec, layer=layer, batch_size=SCORE_BATCH))
        ctx_q = [chat_ids(tok, x, S) for x in xs]
        fwd = kl_on_rollouts(model, ctx_p, ctx_q, y_p, vec_p=vec, layer=layer, batch_size=SCORE_BATCH)
        y_q = sample(model, tok, ctx_q, MAX_NEW_TOKENS, seed=SEED_Q, batch_size=GEN_BATCH)
        rev = summarize(kl_on_rollouts(model, ctx_q, ctx_p, y_q, vec_q=vec, layer=layer, batch_size=SCORE_BATCH))
        f = summarize(fwd)
        if rb0 is None:
            rb0 = fwd["rb"]
        diff = [a - b for a, b in zip(rb0, fwd["rb"])]  # paired reduction vs rung 0
        m0 = st.mean(rb0)
        yq_txt = [tok.decode(t) for t in y_q]
        ppl = text_ppl(model, tok, S)
        rows.append(dict(
            rung=rung, S=S, fwd=f["rb"], fwd_se=f["rb_se"], fwd_tok=f["rb_per_tok"], fwd_mc=f["mc"],
            rev=rev["rb"], rev_se=rev["rb_se"], select_fwd=sel["rb"],
            closed=st.mean(diff) / m0, closed_se=st.stdev(diff) / len(diff) ** 0.5 / m0,
            distinct=distinct_frac(y_q), metric=st.mean(map(metric, yq_txt)),
            ppl=ppl, readable=(S == "" or ppl < READABLE_PPL), samples=yq_txt[:N_DUMP]))
        r = rows[-1]
        print(f"  {r['rung']:<14} fwd {r['fwd']:7.2f}±{r['fwd_se']:.2f}  closed {r['closed']:+.3f}±{r['closed_se']:.3f}"
              f"  rev {r['rev']:7.2f}  distinct {r['distinct']:.3f}  {spec['metric_name']} {r['metric']:.2f}"
              f"  ppl {r['ppl']:.0f}", flush=True)
    return dict(name=name, layer=layer, alpha=vd["alpha"], residual_norm=vd["residual_norm"],
                target=target, rows=rows, metric_name=spec["metric_name"],
                heldout_prompts=xs[:N_DUMP])


def report(res):
    t = res["target"]
    print(f"\n### {res['name']}  (layer {res['layer']}, alpha {res['alpha']} x {res['residual_norm']:.1f})")
    print(f"steered target: distinct {t['distinct']:.3f}, {res['metric_name']} {t['metric']:.2f}\n")
    print(f"| Rung | S | KL fwd (seq) ± SE | KL/tok | closed ± SE | KL rev (seq) | distinct | "
          f"{res['metric_name']} | ppl(S) | readable? |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        s = (r["S"][:60] + "…") if len(r["S"]) > 60 else r["S"]
        print(f"| {r['rung']} | {s!r} | {r['fwd']:.2f} ± {r['fwd_se']:.2f} | {r['fwd_tok']:.3f} | "
              f"{r['closed']:+.3f} ± {r['closed_se']:.3f} | {r['rev']:.2f} ± {r['rev_se']:.2f} | "
              f"{r['distinct']:.3f} | {r['metric']:.2f} | {r['ppl']:.0f} | {'yes' if r['readable'] else 'NO'} |")
    best = min(res["rows"], key=lambda r: r["select_fwd"])
    r2 = next(r for r in res["rows"] if r["rung"].startswith("2"))
    print(f"\nHEADLINE {res['name']}: best S (selected on train) = rung {best['rung']}, closes "
          f"{best['closed']:.1%} ± {best['closed_se']:.1%} of rung-0 KL on held-out; "
          f"rung-2 one-liner closes {r2['closed']:.1%} ± {r2['closed_se']:.1%}")
    res["headline"] = dict(best_rung=best["rung"], best_closed=best["closed"], rung2_closed=r2["closed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--layer", default=None, help="override VEC_LAYER, e.g. pirate=14")
    args = ap.parse_args()
    if args.layer:
        for kv in args.layer.split(","):
            k, v = kv.split("=")
            VEC_LAYER[k] = int(v)
    model, tok = load_model(MODEL_NAME)
    results = []
    for name in (args.only.split(",") if args.only else VECTORS):
        print(f"\n===== {name}", flush=True)
        res = eval_vector(model, tok, name)
        report(res)
        results.append(res)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/phase1_ladder_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    cfg = {k: v for k, v in globals().items() if k.isupper() and k not in ("BEHAVIORS",)}
    json.dump({"config": cfg, "results": results}, open(out, "w"), indent=1, default=str)
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
