"""Phase 1b, run 2: per-token KL attribution for French.

Question: are the ~10 structured nats left after the best French prompt concentrated on content
tokens (steering changes WHAT is said) or spread uniformly (residual off-target cost)?

Per-token term: KL_t = KL(p_steer(.|y<t) || q_S(.|y<t)), Rao-Blackwellized, on Y ~ p_steer, held-out.
Conditions:
  steer|3b   French vector, q = best French S (rung 3b, selected on train in Step 4)   <- the residual
  steer|3a   same, rung 3a (spec allows either; reported for robustness)
  steer|0    French vector, q = neutral (the full rung-0 KL, for reference)
  rand|0     the 8 random directions at ||w||, q = neutral (pure off-target profile; rollouts from run 1)
Reports, per condition:
  - KL share / token share / KL-per-token by coarse class (content, function, punct, number, space, eos, other)
  - KL by position bin
  - concentration: share of total KL in the top 1/5/10% of tokens, Gini
  - top-K highest-KL tokens (steer|3b) with context, p and q probabilities, and q's top choice
Token classes are word-level: BPE pieces are merged into whitespace/punctuation-delimited words and each
piece inherits its word's class; function-word list is French + English (FUNCTION_WORDS below).
"""
import datetime
import glob
import json
import os
import re
import statistics as st

import numpy as np
import torch

from behaviors import BEHAVIORS
from common import NEUTRAL_SYSTEM, chat_ids, completion_logprobs, eos_ids, load_model, sample, wildchat_prompts

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
VEC_PATH = "/workspace/vectors/french_L18.pt"
RAND_ROLLOUTS_GLOB = "/workspace/runs/phase1b_randrollouts_*.pt"   # latest file from run 1
RAND_GROUP = "14.39"
FLOOR_RUN = "/workspace/runs/phase1_randfloor_20261122_231649.json"
OUT_DIR = "/workspace/runs"
HELDOUT_OFFSET, N_HELDOUT = 0, 200
MAX_NEW_TOKENS = 128
SEED_P = 101
GEN_BATCH, SCORE_BATCH = 100, 16
POS_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 129]
TOP_FRACS = [0.01, 0.05, 0.10]
TOP_K = 30
CTX_TOKENS = 8
EXPECTED_KL = {"0 neutral": 34.19, "3a": 19.69, "3b": 19.96}       # Step 4/5 held-out means (repro check)
REPRO_TOL = 0.02

FUNCTION_WORDS = set("""
le la les l un une des du de d au aux à a et ou mais donc or ni car que qu qui quoi dont où ce cet cette ces
c ça cela ceci il elle ils elles on je j tu nous vous me m te t se s lui leur leurs y en ne n pas plus
moins très trop aussi bien est sont être été suis es sommes êtes était ont avoir ai as avons avez avait
fait faire peut peuvent pour par sur sous dans avec sans entre vers chez comme si son sa ses mon ma mes
ton ta tes notre nos votre vos tout tous toute toutes autre autres même quel quelle quels quelles
the a an and or but if then of to in on at by for with from as is are was were be been being am
it its this that these those there here which who whom whose what when where why how not no do does did
have has had i you he she we they me him her us them my your his our their so than too very can could
will would shall should may might must just also all any some each such into over about up out
""".split())
_ALPHA = re.compile(r"[A-Za-zÀ-ÿ'’]")
_DIGIT = re.compile(r"\d")


def classify_tokens(tok, y, eos):
    """Word-level coarse class for each token in completion y."""
    pieces = [tok.decode([t]) for t in y]
    cls = [None] * len(y)
    i = 0
    while i < len(y):
        if y[i] in eos:
            cls[i] = "eos"; i += 1; continue
        p = pieces[i]
        core = p.strip()
        if not core:
            cls[i] = "space"; i += 1; continue
        if _ALPHA.search(core) and not _DIGIT.search(core):
            # merge continuation pieces (no leading whitespace, alphabetic) into one word
            j, word = i + 1, core
            while j < len(y) and y[j] not in eos and pieces[j] and not pieces[j][0].isspace() \
                    and _ALPHA.match(pieces[j]):
                word += pieces[j]; j += 1
            w = re.sub(r"[^A-Za-zÀ-ÿ'’]", "", word).lower().replace("’", "'")
            parts = [x for x in w.split("'") if x]
            c = "function" if parts and all(x in FUNCTION_WORDS for x in parts) else "content"
            for k in range(i, j):
                cls[k] = c
            i = j; continue
        if _DIGIT.search(core):
            cls[i] = "number"
        elif all(not ch.isalnum() for ch in core):
            cls[i] = "punct"
        else:
            cls[i] = "other"      # CJK, mixed scripts, code-ish symbols
        i += 1
    return cls


@torch.no_grad()
def per_token(model, ctx_p, ctx_q, comps, vec, layer):
    """Per-token RB KL(p||q), log p(y_t), log q(y_t), q argmax, for each sequence."""
    out = []
    for i in range(0, len(comps), SCORE_BATCH):
        sl = slice(i, i + SCORE_BATCH)
        y = comps[sl]
        lp, m = completion_logprobs(model, ctx_p[sl], y, vec, layer)
        lq, _ = completion_logprobs(model, ctx_q[sl], y, None, layer)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        qarg = lq.argmax(-1)
        for j, yy in enumerate(y):
            n = len(yy)
            t = torch.tensor(yy, device=lp.device)
            out.append(dict(kl=kl[j, :n].cpu().numpy(),
                            lp=lp[j, torch.arange(n), t].cpu().numpy(),
                            lq=lq[j, torch.arange(n), t].cpu().numpy(),
                            qarg=qarg[j, :n].cpu().tolist()))
        del lp, lq
    return out


def profile(seqs, classes):
    kl = np.concatenate([s["kl"] for s in seqs])
    cl = np.concatenate([np.array(c) for c in classes])
    pos = np.concatenate([np.arange(len(s["kl"])) for s in seqs])
    tot = kl.sum()
    by_class = {}
    for c in ["content", "function", "punct", "number", "space", "eos", "other"]:
        m = cl == c
        if m.sum():
            by_class[c] = dict(kl_share=float(kl[m].sum() / tot), tok_share=float(m.mean()),
                               kl_per_tok=float(kl[m].mean()), n=int(m.sum()))
    by_pos = []
    for a, b in zip(POS_BINS[:-1], POS_BINS[1:]):
        m = (pos >= a) & (pos < b)
        if m.sum():
            by_pos.append(dict(bin=f"{a}-{b - 1}", kl_per_tok=float(kl[m].mean()),
                               kl_share=float(kl[m].sum() / tot), n=int(m.sum())))
    srt = np.sort(kl)[::-1]
    top = {f"top{int(f * 100)}%": float(srt[:max(1, int(len(srt) * f))].sum() / tot) for f in TOP_FRACS}
    x = np.sort(np.clip(kl, 0, None))
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
    return dict(total_per_seq=float(tot / len(seqs)), per_tok=float(kl.mean()), n_tok=int(len(kl)),
                by_class=by_class, by_pos=by_pos, concentration=dict(**top, gini=gini))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-s", default=None, help="an additional system prompt to attribute (e.g. T's output)")
    ap.add_argument("--extra-label", default="T")
    args = ap.parse_args()
    model, tok = load_model(MODEL_NAME)
    eos = set(eos_ids(model, tok))
    vd = torch.load(VEC_PATH)
    layer = vd["layer"]
    w = (vd["unit"] * vd["alpha"] * vd["residual_norm"]).cuda()
    xs = wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    b = BEHAVIORS["french"]
    prompts = {"0 neutral": NEUTRAL_SYSTEM, "3a": b["rung3"][0], "3b": b["rung3"][1]}
    if args.extra_s:
        prompts[args.extra_label] = args.extra_s

    y = sample(model, tok, ctx0, MAX_NEW_TOKENS, vec=w, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
    cls_w = [classify_tokens(tok, yy, eos) for yy in y]
    res, prof = {}, {}
    for lab, S in prompts.items():
        res[lab] = per_token(model, ctx0, [chat_ids(tok, x, S) for x in xs], y, w, layer)
        tot = st.mean(float(s["kl"].sum()) for s in res[lab])
        if lab in EXPECTED_KL:
            assert abs(tot - EXPECTED_KL[lab]) < REPRO_TOL, f"{lab}: {tot:.3f} vs expected {EXPECTED_KL[lab]}"
        prof[f"steer|{lab}"] = profile(res[lab], cls_w)
        print(f"steer|{lab}: {tot:.2f} nats/seq" + (" (repro ok)" if lab in EXPECTED_KL else ""), flush=True)

    # random-direction reference profile (pure off-target cost), rollouts from run 1
    rpath = sorted(glob.glob(RAND_ROLLOUTS_GLOB))[-1]
    rr = torch.load(rpath)
    seeds = [d["seed"] for d in next(r for r in json.load(open(FLOOR_RUN))["results"]
                                     if r["name"] == "french")["random_dirs"]]
    rseqs, rcls = [], []
    for sd in seeds:
        g = torch.Generator().manual_seed(sd)
        v = torch.randn(vd["unit"].numel(), generator=g)
        r = (v / v.norm() * vd["alpha"] * vd["residual_norm"]).cuda()
        ry = rr[f"{RAND_GROUP}/{sd}"]
        rseqs += per_token(model, ctx0, ctx0, ry, r, layer)
        rcls += [classify_tokens(tok, yy, eos) for yy in ry]
    prof["rand|0"] = profile(rseqs, rcls)
    print(f"rand|0: {prof['rand|0']['total_per_seq']:.2f} nats/seq over {len(seeds)} dirs ({rpath})", flush=True)

    # top-K tokens in the residual (steer|3b)
    flat = [(float(s["kl"][t]), i, t) for i, s in enumerate(res["3b"]) for t in range(len(s["kl"]))]
    flat.sort(reverse=True)
    tops = []
    for v, i, t in flat[:TOP_K]:
        s = res["3b"][i]
        tops.append(dict(kl=v, seq=i, pos=t, cls=cls_w[i][t],
                         context=tok.decode(y[i][max(0, t - CTX_TOKENS):t]), token=tok.decode([y[i][t]]),
                         p=float(np.exp(s["lp"][t])), q=float(np.exp(s["lq"][t])),
                         q_top=tok.decode([s["qarg"][t]])))

    # ---------------------------------------------------------------- report
    conds = ["steer|0 neutral", "steer|3b", "steer|3a"] + ([f"steer|{args.extra_label}"] if args.extra_s else []) + ["rand|0"]
    print("\n## Totals\n\n| Condition | KL/seq | KL/tok | top1% share | top5% | top10% | Gini |\n|---|---|---|---|---|---|---|")
    for c in conds:
        p = prof[c]; cc = p["concentration"]
        print(f"| {c} | {p['total_per_seq']:.2f} | {p['per_tok']:.3f} | {cc['top1%']:.1%} | {cc['top5%']:.1%} | "
              f"{cc['top10%']:.1%} | {cc['gini']:.2f} |")
    print("\n## By token class: KL share / token share (KL per token)\n")
    print("| Class | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds))
    for k in ["content", "function", "punct", "number", "space", "eos", "other"]:
        cells = []
        for c in conds:
            d = prof[c]["by_class"].get(k)
            cells.append(f"{d['kl_share']:.0%} / {d['tok_share']:.0%} ({d['kl_per_tok']:.3f})" if d else "—")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\n## By position: KL per token (share of total)\n")
    print("| Positions | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds))
    for j, bb in enumerate(prof["steer|3b"]["by_pos"]):
        cells = []
        for c in conds:
            d = next((x for x in prof[c]["by_pos"] if x["bin"] == bb["bin"]), None)
            cells.append(f"{d['kl_per_tok']:.3f} ({d['kl_share']:.0%})" if d else "—")
        print(f"| {bb['bin']} | " + " | ".join(cells) + " |")
    print(f"\n## Top {TOP_K} highest-KL tokens, steer|3b\n\n| KL | pos | class | context → token | p | q | q's top |\n|---|---|---|---|---|---|---|")
    for t in tops:
        print(f"| {t['kl']:.2f} | {t['pos']} | {t['cls']} | {t['context']!r} → **{t['token']!r}** | {t['p']:.2f} | "
              f"{t['q']:.3f} | {t['q_top']!r} |")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/phase1b_tokattr_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    cfg = {k: (sorted(v) if isinstance(v, set) else v) for k, v in globals().items() if k.isupper()}
    json.dump({"config": cfg, "extra_s": args.extra_s, "rand_rollouts": rpath, "profiles": prof, "top_tokens": tops}, open(path, "w"),
              indent=1, default=str)
    print(f"logged -> {path}")


if __name__ == "__main__":
    main()
