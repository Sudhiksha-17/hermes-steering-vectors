"""Phase 2: evaluate T (or T_0 = the untrained, steered subject model) on held-out prompts.

Per vector (training vectors, eval-only negatives: hedging + a random direction):
  1. T emits K sampled candidates + 1 greedy S, steered by w (same conditioning as training).
  2. Select the best gate-passing sampled candidate by fwd KL on train[64:192] (Step 4 selection set).
  3. Held-out heldout[0:200]: rung 0, rung 1, T best-of-K, T greedy, and the rung-4 best for that vector when one
     exists, all on the SAME steered rollouts (paired). Reports closure vs the better of rung 0/1 (R1 amended),
     vs floor (R2; conservative per R3; indeterminate per R4 if the best S is inside the floor's spread),
     reverse KL, distinct fraction (R5), behavior metric (overshoot), gate scores (P0), and the S text.
  4. "Ignores w" test: prints each vector's greedy S side by side.
Held-out floors F are computed once with N_FLOOR_DIRS directions and cached in FLOOR_CACHE.
"""
import argparse
import datetime
import json
import os
import statistics as st

import torch
from peft import PeftModel

import readability as R
from behaviors import ALL_BEHAVIORS, RUNG1
from common import NEUTRAL_SYSTEM, chat_ids, eos_ids, kl_on_rollouts, load_model, sample, steering, summarize, wildchat_prompts
from oracle_search import distinct_frac
from train_T import S_MAX_TOKENS, T_INSTRUCTION, t_context

# ---------------------------------------------------------------- config
SUBJECT = "Qwen/Qwen2.5-3B-Instruct"
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
FLOOR_CACHE = "/workspace/runs/phase2_heldout_floors.json"
N_FLOOR_DIRS, FLOOR_SEED_BASE = 8, 5000                 # same seeds as Step 5
RANDOM_CTRL = dict(name="random", seed=5000, like=("hedging", 18))
RUNG4_RUNS = ["/workspace/runs/phase1b_rung4_20261123_001216.json", "/workspace/runs/phase1b_rung4_20261123_023404.json"]
SELECT_OFFSET, N_SELECT = 64, 128
HELDOUT_OFFSET, N_HELDOUT = 0, 200
MAX_NEW_TOKENS = 128
SEED_P, SEED_Q, SEED_T = 101, 202, 303
K = 8
GEN_BATCH, SCORE_BATCH = 100, 16


def target(name, layer):
    if name == "random":
        ln, ll = RANDOM_CTRL["like"]
        like = torch.load(f"{VEC_DIR}/{ln}_L{ll}.pt")
        g = torch.Generator().manual_seed(RANDOM_CTRL["seed"])
        v = torch.randn(like["unit"].numel(), generator=g)
        return dict(name="random", layer=ll, alpha=like["alpha"], w=(v / v.norm() * like["alpha"] * like["residual_norm"]).cuda(),
                    metric=None, metric_name=None)
    vd = torch.load(f"{VEC_DIR}/{name}_L{layer}.pt")
    b = ALL_BEHAVIORS[name]
    return dict(name=name, layer=layer, alpha=vd["alpha"], w=(vd["unit"] * vd["alpha"] * vd["residual_norm"]).cuda(),
                metric=b["metric"], metric_name=b["metric_name"])


@torch.no_grad()
def heldout_floor(model, tok, t, ctx0):
    key = f"{t['name']}_L{t['layer']}_a{t['alpha']}"
    cache = json.load(open(FLOOR_CACHE)) if os.path.exists(FLOOR_CACHE) else {}
    if key not in cache:
        norm, vals = float(t["w"].norm()), []
        for k in range(N_FLOOR_DIRS):
            g = torch.Generator().manual_seed(FLOOR_SEED_BASE + k)
            v = torch.randn(t["w"].numel(), generator=g)
            r = (v / v.norm() * norm).cuda()
            y = sample(model, tok, ctx0, MAX_NEW_TOKENS, vec=r, layer=t["layer"], seed=SEED_P, batch_size=GEN_BATCH)
            vals.append(st.mean(kl_on_rollouts(model, ctx0, ctx0, y, vec_p=r, layer=t["layer"], batch_size=SCORE_BATCH)["rb"]))
        cache[key] = dict(F=st.mean(vals), dirs=vals, norm=norm)
        json.dump(cache, open(FLOOR_CACHE, "w"), indent=1)
    return cache[key]


@torch.no_grad()
def t_outputs(model, tok, t):
    eos = set(eos_ids(model, tok))
    ids = sample(model, tok, [t_context(tok)], S_MAX_TOKENS, n_per_ctx=K, vec=t["w"], layer=t["layer"], seed=SEED_T,
                 batch_size=GEN_BATCH)
    samp = [tok.decode([x for x in s if x not in eos], skip_special_tokens=True).strip() for s in ids]
    ctx = torch.tensor([t_context(tok)]).cuda()
    with steering(model, t["layer"], t["w"]):
        g = model.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), do_sample=False, max_new_tokens=S_MAX_TOKENS,
                           eos_token_id=sorted(eos), pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
    greedy = tok.decode([x for x in g[0, ctx.shape[1]:].tolist() if x not in eos], skip_special_tokens=True).strip()
    return samp, greedy


def rung4_best(name, gate):
    """Search baseline: best prompt on TRAIN among everything rung 4 scored (ladder rungs + LLM proposals). It must be
    STRUCTURALLY valid (no JSON wrappers etc.) but is not filtered by the judge: the baseline is a hand-written or
    proposed comparison, not T's output, and the judge has documented false rejections of short one-liners ("Write in
    all lowercase.") that would handicap the baseline and flatter T. Its gate status is still reported. Never selected
    on held-out."""
    for p in RUNG4_RUNS:
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["results"]:
            if r["name"] == name:
                for h in sorted(r["history"], key=lambda h: h["train_kl"]):
                    if R.structural_check(h["S"])[0]:
                        return h["S"]
    return None


@torch.no_grad()
def eval_vector(model, tok, gate, t, adapter_ctx):
    xs_sel = wildchat_prompts("train", N_SELECT, SELECT_OFFSET)
    xs = wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET)
    with adapter_ctx():          # T emits with LoRA ON
        samp, greedy = t_outputs(model, tok, t)
    sc = gate.score(samp)
    ctx_sel = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs_sel]
    y_sel = sample(model, tok, ctx_sel, MAX_NEW_TOKENS, vec=t["w"], layer=t["layer"], seed=SEED_P, batch_size=GEN_BATCH)
    sel = []
    for S, d in zip(samp, sc):
        if d["hard_fail"]:
            sel.append(float("inf"))
            continue
        sel.append(st.mean(kl_on_rollouts(model, ctx_sel, [chat_ids(tok, x, S) for x in xs_sel], y_sel, vec_p=t["w"],
                                          layer=t["layer"], batch_size=SCORE_BATCH)["rb"]))
    bi = min(range(K), key=lambda i: sel[i])
    best = samp[bi] if sel[bi] < float("inf") else None

    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    fl = heldout_floor(model, tok, t, ctx0)
    y = sample(model, tok, ctx0, MAX_NEW_TOKENS, vec=t["w"], layer=t["layer"], seed=SEED_P, batch_size=GEN_BATCH)
    yp_txt = [tok.decode(z) for z in y]
    conds = [("rung 0", NEUTRAL_SYSTEM), ("rung 1", RUNG1)]
    if best is not None:
        conds.append(("T best-of-K", best))
    conds.append(("T greedy", greedy))
    r4 = rung4_best(t["name"], gate)
    if r4:          # '' (empty prompt best on train, e.g. hedging) is already rung 0
        conds.append(("search best (ladder+rung4)", r4))
    rows, base = [], {}
    for lab, S in conds:
        f = kl_on_rollouts(model, ctx0, [chat_ids(tok, x, S) for x in xs], y, vec_p=t["w"], layer=t["layer"],
                           batch_size=SCORE_BATCH)["rb"]
        ctx_q = [chat_ids(tok, x, S) for x in xs]
        yq = sample(model, tok, ctx_q, MAX_NEW_TOKENS, seed=SEED_Q, batch_size=GEN_BATCH)
        rev = summarize(kl_on_rollouts(model, ctx_q, ctx0, yq, vec_q=t["w"], layer=t["layer"], batch_size=SCORE_BATCH))["rb"]
        yq_txt = [tok.decode(z) for z in yq]
        rows.append(dict(label=lab, S=S, per=f, fwd=st.mean(f), fwd_se=st.stdev(f) / len(f) ** 0.5, rev=rev,
                         distinct=distinct_frac(yq), metric=st.mean(map(t["metric"], yq_txt)) if t["metric"] else None,
                         samples=yq_txt[:2]))
    k0, k1 = rows[0]["per"], rows[1]["per"]
    bname, bper = ("rung 0", k0) if st.mean(k0) <= st.mean(k1) else ("rung 1", k1)
    B = st.mean(bper)
    gs = {r["S"]: d for r in rows for d in gate.score([r["S"]])}
    for r in rows:
        d = [a - c for a, c in zip(bper, r["per"])]
        r["vs_better"], r["vs_better_se"] = st.mean(d) / B, st.stdev(d) / len(d) ** 0.5 / B
        r["vs_floor"] = (B - r["fwd"]) / (B - fl["F"]) if B > fl["F"] else float("nan")
        r["inside_floor_spread"] = min(fl["dirs"]) <= r["fwd"] <= max(fl["dirs"])
        r["gate"] = gs[r["S"]]
        del r["per"]
    return dict(name=t["name"], layer=t["layer"], alpha=t["alpha"], floor=fl, better=bname, candidates=samp,
                cand_select_kl=sel, greedy=greedy, rows=rows, steered_samples=yp_txt[:2],
                target=dict(distinct=distinct_frac(y), metric=st.mean(map(t["metric"], yp_txt)) if t["metric"] else None),
                metric_name=t["metric_name"])


def report(res):
    tm = f", {res['metric_name']} {res['target']['metric']:.2f}" if res["target"]["metric"] is not None else ""
    print(f"\n### {res['name']} (L{res['layer']}, alpha {res['alpha']}); better baseline = {res['better']}; "
          f"F = {res['floor']['F']:.2f} [{min(res['floor']['dirs']):.1f}-{max(res['floor']['dirs']):.1f}]; "
          f"steered distinct {res['target']['distinct']:.3f}{tm}\n")
    print("| | S | KL fwd ± SE | vs better (R1) | vs floor | KL rev | distinct | metric | P(A) | payload | gate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        vf = "indeterminate (R4)" if r["inside_floor_spread"] and r["label"] not in ("rung 0", "rung 1") else f"{r['vs_floor']:+.1%}"
        mt = f"{r['metric']:.2f}" if r["metric"] is not None else "—"
        g = r["gate"]
        print(f"| {r['label']} | {r['S'][:70]!r} | {r['fwd']:.2f} ± {r['fwd_se']:.2f} | {r['vs_better']:+.1%} ± "
              f"{r['vs_better_se']:.1%} | {vf} | {r['rev']:.1f} | {r['distinct']:.3f} | {mt} | {g['judge']:.2f} | "
              f"{g['payload']:.2f} | {'FAIL' if g['hard_fail'] else 'pass'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="LoRA checkpoint dir; omit for T_0")
    ap.add_argument("--vectors", required=True, help="comma list name:layer; 'random:0' for the random control")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import contextlib
    base, tok = load_model(SUBJECT)
    if args.lora:
        pm = PeftModel.from_pretrained(base, args.lora)
        pm.eval()
        on = contextlib.nullcontext
        model = base                                    # LoRA injected in place; disable for subject scoring
        scoring_ctx = pm.disable_adapter
    else:
        model, on, scoring_ctx = base, contextlib.nullcontext, contextlib.nullcontext
    gate = R.Gate()
    out = []
    for spec in args.vectors.split(","):
        name, layer = spec.split(":")
        t = target(name, int(layer))
        with scoring_ctx():
            # eval_vector re-enables the adapter only around T's own generation
            res = eval_vector(model, tok, gate, t, adapter_ctx=(lambda: _enable(args.lora, pm)) if args.lora else on)
        report(res)
        out.append(res)
    print("\n## 'Ignores w' check: greedy S per vector\n")
    for r in out:
        print(f"- {r['name']}: {r['greedy']!r}")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/phase2_eval_{'T0' if not args.lora else 'T'}_{datetime.datetime.now():%Y%m%d_%H%M%S}{('_' + args.tag) if args.tag else ''}.json"
    json.dump({"lora": args.lora, "t_instruction": T_INSTRUCTION, "results": out}, open(path, "w"), indent=1,
              default=str, ensure_ascii=False)
    print(f"logged -> {path}")


import contextlib


@contextlib.contextmanager
def _enable(lora, pm):
    """Temporarily re-enable the adapter inside an outer disable_adapter() block."""
    pm.enable_adapter_layers()
    try:
        yield
    finally:
        pm.disable_adapter_layers()


if __name__ == "__main__":
    main()
