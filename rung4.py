"""Phase 1b, run 4: rung 4 -- LLM-proposed system prompts, with a two-model readability gate.

Per target (hedging first; then pirate at L8; French; and a RANDOM-direction negative control):
  loop ROUNDS times:
    proposer (Qwen2.5-14B-Instruct) sees steered-vs-neutral sample pairs (train), all previous candidates
    with their train KL, and the general Phase 1 lesson that strong instructions overshoot. It is NOT told
    which behavior the vector encodes (keeps the random control a fair test).
    -> PER_ROUND new candidates, each scored on train[64:192]: fwd KL(p_steer || q_S).
  readability, for every candidate (reported next to KL, never hidden):
    subj_ppl   perplexity under the subject model w/ framing prefix (Phase 1 gate, threshold 2000)
    judge      Phi-4 (non-Qwen) expected rating 1-9: "sensible instruction a person might write"
    base_ppl   perplexity under OLMo-2-1B base (non-Qwen) w/ the same framing prefix
  gate = subj_ppl < 2000 AND judge >= JUDGE_MIN AND base_ppl < BASE_PPL_MAX, where BASE_PPL_MAX is
         calibrated at startup as BASE_PPL_MARGIN x the max over the 13 hand-written ladder prompts.
  selection: best train KL among gate-passing candidates (the best overall is reported alongside).
  held-out (heldout[0:200]): top TOP_HELDOUT passing candidates, with rung 0/1 recomputed for paired
  closure vs rung 0, vs rung 1 (R1) and vs floor (R2/R3), reverse KL, distinct fraction, behavior metric.
Proposer sampling uses temperature PROPOSER_TEMP (it is not a subject rollout; the pure-ancestral rule
applies to subject-model rollouts only, which are unchanged).
"""
import argparse
import datetime
import json
import math
import os
import re
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviors import ALL_BEHAVIORS as BEHAVIORS, RUNG1
from common import NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, summarize, wildchat_prompts
from oracle_search import READABILITY_PREFIX, READABLE_PPL, distinct_frac, ladder, text_ppl

# ---------------------------------------------------------------- config
SUBJECT = "Qwen/Qwen2.5-3B-Instruct"
PROPOSER = "Qwen/Qwen2.5-14B-Instruct"
JUDGE = "microsoft/phi-4"
BASE_LM = "allenai/OLMo-2-0425-1B"   # OLMo-2-7B ships fp32 (~29GB) and did not fit beside phi-4
EPHEMERAL_CACHE = "/root/hf_ephemeral/hub"     # judge + base LM (not enough room on /workspace; re-downloadable)
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
TARGETS = ["hedging", "pirate", "french", "random"]
VEC_LAYER = {"hedging": 18, "pirate": 8, "french": 18}
FLOOR_CACHE = "/workspace/runs/phase2_heldout_floors.json"   # Phase 2 vectors (eval_T.py)
RANDOM_CTRL = dict(seed=5000, like="hedging")  # random direction at the hedging vector's norm and layer
LADDER_RUNS = {18: "/workspace/runs/phase1_ladder_20261120_235111.json",
               8: "/workspace/runs/phase1_ladder_20261122_234338.json"}
FLOOR_RUNS = {18: "/workspace/runs/phase1_randfloor_20261122_231649.json",
              8: "/workspace/runs/phase1_randfloor_20261122_234450.json"}
SELECT_OFFSET, N_SELECT = 64, 128              # train; identical to Step 4 selection set
HELDOUT_OFFSET, N_HELDOUT = 0, 200
MAX_NEW_TOKENS = 128
SEED_P, SEED_Q = 101, 202                      # identical to Step 4
ROUNDS, PER_ROUND = 6, 8
N_SHOW_PAIRS = 6
SHOW_CHARS = 450
HISTORY_SHOW = 30
PROPOSER_TEMP, PROPOSER_TOP_P, PROPOSER_MAX_TOKENS = 0.9, 0.95, 1500
PROPOSER_SEED = 0
MAX_S_CHARS = 600
JUDGE_MIN = 6.0                                # on the 1-9 expected-rating scale
BASE_PPL_MARGIN = 3.0
TOP_HELDOUT = 3
GEN_BATCH, SCORE_BATCH = 100, 16
JUNK = [  # readability-gate calibration negatives (not candidates)
    'describing.\\ + similarlyNow write oppositeley.]( Me giving**ONE please? revert with "\\!--Two',
    "hedge hedge maybe maybe perhaps possibly uncertain uncertain might might seems",
    "The purple calendar negotiates with the assistant's elbows every Tuesday before soup.",
    "ye arr matey ahoy ye ye arr booty plunder arr arr",
]

PROPOSER_TEMPLATE = """A small language model (the "subject") is being modified by adding a steering vector to its hidden activations. We are looking for a SYSTEM PROMPT that, given to the UNMODIFIED subject model, makes it produce text distributed like the MODIFIED model's text, for arbitrary user messages.

Score = average KL divergence (nats per response) from the modified model to the unmodified model with your system prompt, over a fixed set of user messages. LOWER IS BETTER. With an empty system prompt the score is {kl0:.2f}; with "You are a helpful assistant." it is {kl1:.2f}.

Below are paired responses to the same user message: MODIFIED model vs UNMODIFIED model (no system prompt). Study the differences carefully: vocabulary, tone, register, structure, length, language, and how strong and how consistent the difference is.

{pairs}

System prompts tried so far, best first (score; "REJECTED" = failed the readability check and cannot be used):
{history}

Lessons from earlier experiments: explicit, strongly worded instructions tend to OVERSHOOT. The unmodified model then exaggerates the behavior far beyond what the modified model does, which can score worse than saying nothing at all. The modification is often subtle and inconsistent across responses. Consider mild, partial or indirect wording ("occasionally", "slightly", "when it feels natural"), describing a register, mood or persona instead of commanding a behavior, and short prompts. Every system prompt must be plain, readable natural language that a person might actually write: no keyword lists, no token soup.

Propose {n} NEW system prompts, different from those already tried and from each other. Output ONLY a JSON list of {n} strings."""

JUDGE_TEMPLATE = """Here is a piece of text proposed as a system prompt for an AI assistant:

<<<
{s}
>>>

Is this a sensible, readable instruction that a person might plausibly write? Judge only readability and sensibility, not whether it is a good idea. Rate from 1 (gibberish, keyword soup, or nonsensical) to 9 (perfectly natural, clear instruction). Answer with a single digit."""


# ---------------------------------------------------------------- readability models
class Readability:
    def __init__(self, subj, subj_tok):
        self.subj, self.subj_tok = subj, subj_tok
        self.jt = AutoTokenizer.from_pretrained(JUDGE, cache_dir=EPHEMERAL_CACHE)
        self.jm = AutoModelForCausalLM.from_pretrained(JUDGE, cache_dir=EPHEMERAL_CACHE, dtype=torch.bfloat16,
                                                       device_map="cuda").eval()
        self.digits = []
        for d in range(1, 10):
            ids = self.jt.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, f"judge digit {d} is not a single token: {ids}"
            self.digits.append(ids[0])
        self.bt = AutoTokenizer.from_pretrained(BASE_LM, cache_dir=EPHEMERAL_CACHE)
        self.bm = AutoModelForCausalLM.from_pretrained(BASE_LM, cache_dir=EPHEMERAL_CACHE, dtype=torch.bfloat16,
                                                       device_map="cuda").eval()
        self.base_max = None

    @torch.no_grad()
    def judge(self, s):
        msgs = [{"role": "user", "content": JUDGE_TEMPLATE.format(s=s)}]
        ids = self.jt.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids") or isinstance(ids, dict):
            ids = ids["input_ids"]
        logits = self.jm(input_ids=ids.cuda()).logits[0, -1].float()
        p = torch.softmax(logits[self.digits], -1)
        return float((p * torch.arange(1, 10, device=p.device)).sum())

    @torch.no_grad()
    def base_ppl(self, s):
        pre = self.bt.encode(READABILITY_PREFIX)                       # with BOS if the tokenizer adds one
        x = torch.tensor([pre + self.bt.encode(s, add_special_tokens=False)]).cuda()
        lp = torch.log_softmax(self.bm(input_ids=x).logits[0, :-1].float(), -1)
        nll = -lp.gather(-1, x[0, 1:, None]).squeeze(-1)[len(pre) - 1:].mean().item()
        return math.exp(nll)

    def score(self, s):
        r = dict(subj_ppl=text_ppl(self.subj, self.subj_tok, s), judge=self.judge(s), base_ppl=self.base_ppl(s))
        r["gate"] = bool(r["subj_ppl"] < READABLE_PPL and r["judge"] >= JUDGE_MIN and
                         (self.base_max is None or r["base_ppl"] < self.base_max))
        return r

    def calibrate(self):
        hand = sorted({S for nm in BEHAVIORS for _, S in ladder(nm) if S})
        rows = [dict(S=S, kind="hand-written", **self.score(S)) for S in hand] + \
               [dict(S=S, kind="junk", **self.score(S)) for S in JUNK]
        self.base_max = BASE_PPL_MARGIN * max(r["base_ppl"] for r in rows if r["kind"] == "hand-written")
        for r in rows:
            r["gate"] = bool(r["subj_ppl"] < READABLE_PPL and r["judge"] >= JUDGE_MIN and r["base_ppl"] < self.base_max)
        print(f"\n## Readability gate calibration (BASE_PPL_MAX = {self.base_max:.0f}, JUDGE_MIN = {JUDGE_MIN})\n")
        print("| kind | S | subj ppl | judge (1-9) | base ppl | gate |\n|---|---|---|---|---|---|")
        for r in rows:
            s = (r["S"][:55] + "…") if len(r["S"]) > 55 else r["S"]
            print(f"| {r['kind']} | {s!r} | {r['subj_ppl']:.0f} | {r['judge']:.2f} | {r['base_ppl']:.0f} | "
                  f"{'pass' if r['gate'] else 'FAIL'} |", flush=True)
        bad = [r for r in rows if (r["kind"] == "hand-written") != r["gate"]]
        if bad:
            print(f"NOTE: gate disagrees with the label on {len(bad)} calibration strings (see table).")
        return rows


# ---------------------------------------------------------------- proposer
class Proposer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(PROPOSER)
        self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(PROPOSER, dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.no_grad()
    def propose(self, prompt, n, seed):
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           return_tensors="pt")
        if hasattr(ids, "input_ids") or isinstance(ids, dict):
            ids = ids["input_ids"]
        torch.manual_seed(seed)
        out = self.m.generate(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids).cuda(), do_sample=True,
                              temperature=PROPOSER_TEMP, top_p=PROPOSER_TOP_P, top_k=0,
                              max_new_tokens=PROPOSER_MAX_TOKENS, pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        cands = []
        try:
            j = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
            cands = [c for c in j if isinstance(c, str)]
        except (ValueError, json.JSONDecodeError):
            cands = re.findall(r'"((?:[^"\\]|\\.){8,})"', txt)
        out = []
        for c in cands:  # unwrap candidates the proposer nested as JSON, e.g. '{"prompt": "..."}' (seen in run 4)
            c = c.strip()
            try:
                j = json.loads(c)
                if isinstance(j, dict) and len(j) == 1 and isinstance(next(iter(j.values())), str):
                    c = next(iter(j.values())).strip()
            except (ValueError, json.JSONDecodeError):
                pass
            if c:
                out.append(c)
        return out[:n], txt


# ---------------------------------------------------------------- evaluation
def get_target(name):
    """Returns dict(name, vec, layer, metric, metric_name, ladder_rungs, floor)."""
    if name == "random":
        like = torch.load(f"{VEC_DIR}/{RANDOM_CTRL['like']}_L{VEC_LAYER[RANDOM_CTRL['like']]}.pt")
        norm = like["alpha"] * like["residual_norm"]
        g = torch.Generator().manual_seed(RANDOM_CTRL["seed"])
        v = torch.randn(like["unit"].numel(), generator=g)
        return dict(name=name, vec=(v / v.norm() * norm).cuda(), layer=like["layer"], metric=None, metric_name=None,
                    ladder_rungs=[("0 neutral", NEUTRAL_SYSTEM), ("1 generic", RUNG1)], floor=None)
    vd = torch.load(f"{VEC_DIR}/{name}_L{VEC_LAYER.get(name, 18)}.pt")
    L = vd["layer"]
    prior = [r for r in json.load(open(FLOOR_RUNS[L]))["results"] if r["name"] == name] if L in FLOOR_RUNS else []
    if prior:
        floor = prior[0]["floor"]
    else:   # Phase 2 vectors: 8-direction held-out floor cached by eval_T.py (same seeds as Step 5)
        floor = json.load(open(FLOOR_CACHE))[f"{name}_L{L}_a{vd['alpha']}"]["F"]
    return dict(name=name, vec=(vd["unit"] * vd["alpha"] * vd["residual_norm"]).cuda(), layer=L,
                metric=BEHAVIORS[name]["metric"], metric_name=BEHAVIORS[name]["metric_name"],
                ladder_rungs=ladder(name), floor=floor)


def fwd_kl(model, tok, ctx_p, xs, y, S, t):
    return kl_on_rollouts(model, ctx_p, [chat_ids(tok, x, S) for x in xs], y, vec_p=t["vec"],
                          layer=t["layer"], batch_size=SCORE_BATCH)


def heldout_eval(model, tok, t, cands):
    xs = wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET)
    ctx_p = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    y = sample(model, tok, ctx_p, MAX_NEW_TOKENS, vec=t["vec"], layer=t["layer"], seed=SEED_P, batch_size=GEN_BATCH)
    yp_txt = [tok.decode(z) for z in y]
    target = dict(distinct=distinct_frac(y), metric=st.mean(map(t["metric"], yp_txt)) if t["metric"] else None)
    rows = []
    base = {lab: fwd_kl(model, tok, ctx_p, xs, y, S, t)["rb"]
            for lab, S in [("0 neutral", NEUTRAL_SYSTEM), ("1 generic", RUNG1)]}
    for lab, S in [("0 neutral", NEUTRAL_SYSTEM), ("1 generic", RUNG1)] + cands:
        f = base[lab] if lab in base else fwd_kl(model, tok, ctx_p, xs, y, S, t)["rb"]
        ctx_q = [chat_ids(tok, x, S) for x in xs]
        yq = sample(model, tok, ctx_q, MAX_NEW_TOKENS, seed=SEED_Q, batch_size=GEN_BATCH)
        rev = summarize(kl_on_rollouts(model, ctx_q, ctx_p, yq, vec_q=t["vec"], layer=t["layer"], batch_size=SCORE_BATCH))
        yq_txt = [tok.decode(z) for z in yq]
        r = dict(label=lab, S=S, fwd=st.mean(f), fwd_se=st.stdev(f) / len(f) ** 0.5, rev=rev["rb"],
                 distinct=distinct_frac(yq), metric=st.mean(map(t["metric"], yq_txt)) if t["metric"] else None,
                 samples=yq_txt[:3])
        for b, key in [("0 neutral", "vs_r0"), ("1 generic", "vs_r1")]:
            d = [a - c for a, c in zip(base[b], f)]
            r[key], r[key + "_se"] = st.mean(d) / st.mean(base[b]), st.stdev(d) / len(d) ** 0.5 / st.mean(base[b])
        if t["floor"] is not None:
            d = [a - c for a, c in zip(base["0 neutral"], f)]
            r["vs_floor"] = st.mean(d) / (st.mean(base["0 neutral"]) - t["floor"])
        rows.append(r)
    return dict(target=target, rows=rows, steered_samples=yp_txt[:3])


def run_target(model, tok, prop, read, name, seed_base):
    t = get_target(name)
    xs = wildchat_prompts("train", N_SELECT, SELECT_OFFSET)
    ctx_p = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    y = sample(model, tok, ctx_p, MAX_NEW_TOKENS, vec=t["vec"], layer=t["layer"], seed=SEED_P, batch_size=GEN_BATCH)
    y_neu = sample(model, tok, ctx_p, MAX_NEW_TOKENS, seed=SEED_Q, batch_size=GEN_BATCH)
    pairs = "\n\n".join(f"### User message {i + 1}\n{xs[i][:300]}\n\n--- MODIFIED:\n{tok.decode(y[i])[:SHOW_CHARS]}"
                        f"\n\n--- UNMODIFIED:\n{tok.decode(y_neu[i])[:SHOW_CHARS]}" for i in range(N_SHOW_PAIRS))
    hist = []
    for lab, S in t["ladder_rungs"]:
        kl = summarize(fwd_kl(model, tok, ctx_p, xs, y, S, t))["rb"]
        hist.append(dict(S=S, train_kl=kl, source=f"ladder {lab}", round=-1, **read.score(S) if S else
                         dict(subj_ppl=float("nan"), judge=float("nan"), base_ppl=float("nan"), gate=True)))
    kl0 = hist[0]["train_kl"]
    kl1 = next(h["train_kl"] for h in hist if h["S"] == RUNG1)
    print(f"\n===== rung 4: {name} (layer {t['layer']}); train KL rung0 {kl0:.2f}, rung1 {kl1:.2f}", flush=True)
    raw_outputs = []
    for rd in range(ROUNDS):
        shown = sorted([h for h in hist if h["S"]], key=lambda h: h["train_kl"])[:HISTORY_SHOW]
        htxt = "\n".join(f"- {h['train_kl']:.2f}{'' if h['gate'] else ' REJECTED'}: {json.dumps(h['S'])}" for h in shown)
        prompt = PROPOSER_TEMPLATE.format(kl0=kl0, kl1=kl1, pairs=pairs, history=htxt, n=PER_ROUND)
        cands, raw = prop.propose(prompt, PER_ROUND, seed_base + rd)
        raw_outputs.append(raw)
        seen = {h["S"] for h in hist}
        new = [c[:MAX_S_CHARS] for c in dict.fromkeys(cands) if c not in seen]
        for S in new:
            kl = summarize(fwd_kl(model, tok, ctx_p, xs, y, S, t))["rb"]
            hist.append(dict(S=S, train_kl=kl, source="rung4", round=rd, **read.score(S)))
        r4 = [h for h in hist if h["source"] == "rung4"]
        bp = min((h for h in r4 if h["gate"]), key=lambda h: h["train_kl"], default=None)
        print(f"  round {rd}: {len(new)} new; best passing so far "
              f"{bp['train_kl'] if bp else float('nan'):.2f}  ({json.dumps(bp['S'])[:90] if bp else '-'})", flush=True)

    r4 = [h for h in hist if h["source"] == "rung4"]
    passing = sorted([h for h in r4 if h["gate"]], key=lambda h: h["train_kl"])
    best_any = min(r4, key=lambda h: h["train_kl"]) if r4 else None
    ladder_best = min([h for h in hist if h["source"].startswith("ladder")], key=lambda h: h["train_kl"])
    cands = [(f"rung4 #{i + 1}", h["S"]) for i, h in enumerate(passing[:TOP_HELDOUT])]
    if ladder_best["S"] not in (NEUTRAL_SYSTEM, RUNG1):
        cands.append((f"{ladder_best['source']} (ladder best)", ladder_best["S"]))
    ho = heldout_eval(model, tok, t, cands)
    return dict(name=name, layer=t["layer"], floor=t["floor"], train_kl0=kl0, train_kl1=kl1, history=hist,
                n_rung4=len(r4), n_passing=len(passing), best_any=best_any, heldout=ho, proposer_raw=raw_outputs,
                metric_name=t["metric_name"])


def report(res):
    print(f"\n## {res['name']} (layer {res['layer']}): {res['n_rung4']} rung-4 candidates, {res['n_passing']} pass the gate")
    print(f"train KL: rung0 {res['train_kl0']:.2f}, rung1 {res['train_kl1']:.2f}")
    print("\n### All candidates by train KL (top 15)\n\n| train KL | subj ppl | judge | base ppl | gate | source | S |\n|---|---|---|---|---|---|---|")
    for h in sorted(res["history"], key=lambda h: h["train_kl"])[:15]:
        print(f"| {h['train_kl']:.2f} | {h['subj_ppl']:.0f} | {h['judge']:.2f} | {h['base_ppl']:.0f} | "
              f"{'pass' if h['gate'] else 'FAIL'} | {h['source']} | {h['S'][:110]!r} |")
    b = res["best_any"]
    if b and not b["gate"]:
        print(f"\nNOTE: best rung-4 candidate on train FAILS the gate: {b['train_kl']:.2f} {b['S'][:120]!r}")
    ho = res["heldout"]
    tm = f", {res['metric_name']} {ho['target']['metric']:.2f}" if ho["target"]["metric"] is not None else ""
    print(f"\n### Held-out (steered target: distinct {ho['target']['distinct']:.3f}{tm})\n")
    print("| | S | KL fwd ± SE | vs rung 0 | vs rung 1 | vs floor (conservative) | KL rev | distinct | metric |\n|---|---|---|---|---|---|---|---|---|")
    for r in ho["rows"]:
        vf = f"{r['vs_floor']:+.1%}" if "vs_floor" in r else "n/a"
        mt = f"{r['metric']:.2f}" if r["metric"] is not None else "—"
        print(f"| {r['label']} | {r['S'][:70]!r} | {r['fwd']:.2f} ± {r['fwd_se']:.2f} | {r['vs_r0']:+.1%} ± {r['vs_r0_se']:.1%} | "
              f"{r['vs_r1']:+.1%} ± {r['vs_r1_se']:.1%} | {vf} | {r['rev']:.1f} | {r['distinct']:.3f} | {mt} |")


def main():
    global TARGETS, ROUNDS, TOP_HELDOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--rounds", type=int, default=None, help="override ROUNDS (smoke tests)")
    ap.add_argument("--top-heldout", type=int, default=None)
    args = ap.parse_args()
    if args.rounds is not None:
        ROUNDS = args.rounds
    if args.top_heldout is not None:
        TOP_HELDOUT = args.top_heldout
    if args.only:
        TARGETS = args.only.split(",")
    model, tok = load_model(SUBJECT)
    read = Readability(model, tok)
    calib = read.calibrate()
    prop = Proposer()
    results = []
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    path = f"{OUT_DIR}/phase1b_rung4_{stamp}.json"
    cfg = {k: v for k, v in globals().items() if k.isupper() and not k.endswith("TEMPLATE")}
    cfg.update(PROPOSER_TEMPLATE=PROPOSER_TEMPLATE, JUDGE_TEMPLATE=JUDGE_TEMPLATE, BASE_PPL_MAX=read.base_max)
    for i, name in enumerate(TARGETS):
        res = run_target(model, tok, prop, read, name, PROPOSER_SEED + 100 * i)
        report(res)
        results.append(res)
        json.dump({"config": cfg, "gate_calibration": calib, "results": results}, open(path, "w"),
                  indent=1, default=str)
        print(f"logged -> {path}", flush=True)


if __name__ == "__main__":
    main()
