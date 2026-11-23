"""Phase 2: train T (steering vector -> system prompt) with GRPO.

ARCHITECTURE (recorded in PHASE2_STATUS.md): T = the subject model (Qwen2.5-3B-Instruct) + LoRA, conditioned on
w by injecting w into T's residual stream at w's own layer and alpha, at every position, exactly as the
subject model receives it. T reads a fixed instruction (T_INSTRUCTION, which never names a behavior) and
writes S while steered. One set of base weights serves three roles:
  T (policy)      = base + LoRA, steered by w
  T_ref           = base, LoRA disabled, steered by w        (KL-to-reference regularizer; also = T at step 0)
  subject p / q   = base, LoRA disabled; p steered by w, q unsteered with system prompt S

REWARD (PHASE2_STATUS.md, P0 item 4), per candidate S for vector v, on a minibatch of steered rollouts:
  KL(S)  = mean_x [ log p(Y|x) - log q(Y|x,S) ]          (log p precomputed for the rollout pool)
  r_KL   = (B_v - KL(S)) / (B_v - F_v),   B_v = min(KL rung0, KL rung1) on the pool (R1 amended)
  reward = clip(r_KL - READ_W * readability_penalty - degeneration_penalty, R_FLOOR)
  HARD FLOOR reward = R_FLOOR if readability hard-fails or q(.|S) samples have distinct fraction < DEG_HARD.
GRPO: G candidates per vector per step, advantage = (r - mean_group) / (std_group + eps), one on-policy
gradient step per batch (no ratio clipping needed), plus BETA_REF * per-token k3 KL(T || T_ref) on S tokens.
Loss is normalized by the constant S_MAX_TOKENS (no length bias).
"""
import argparse
import datetime
import json
import math
import os
import random
import statistics as st

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import readability as R
from behaviors import ALL_BEHAVIORS, RUNG1
from common import (NEUTRAL_SYSTEM, chat_ids, completion_logprobs, eos_ids, kl_on_rollouts, load_model, sample,
                    steering, wildchat_prompts)

# ---------------------------------------------------------------- config
SUBJECT = "Qwen/Qwen2.5-3B-Instruct"
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
TRAIN_VECTORS = [("french", 18), ("pirate", 8), ("german", 18), ("formal", 18), ("enthusiastic", 18),
                 ("lowercase", 18)]   # 6 passing the criterion on the 256-prompt pool (PHASE2_STATUS.md); gloomy and
                                      # empathy dropped (passed only on 64 calibration prompts)
FLOOR_DIRS = 3                             # random directions for F_v on the pool
FLOOR_SEED_BASE = 9000
POOL_OFFSET, N_POOL = 192, 256             # train[192:448]: disjoint from calib 0:64, select 64:192, contrast 500:
ROLLOUT_TOKENS = 128
SEED_P = 101
G = 8                                      # candidates per vector per step
PROMPTS_PER_STEP = 32                      # steered rollouts scoring each candidate (shared within the group: paired)
DEG_PROMPTS, DEG_TOKENS = 8, 64            # q(.|S) samples for the degeneration term
DEG_HARD = 0.5                             # distinct fraction below this -> hard floor
DEG_MARGIN, DEG_UNIT, DEG_W = 0.05, 0.05, 0.1
READ_W = 0.05                              # 1 readability point ~ 5% closure
R_FLOOR = -2.0
S_MAX_TOKENS = 64
T_TEMP = 1.0
T_INSTRUCTION = ("Write a short system prompt, one or two sentences addressed to an AI assistant, that would make it "
                 "write replies just like the ones you would write right now. Be specific about the language, tone, "
                 "style, or persona it should use. Start directly with the instruction, with no label, heading, or "
                 "quotation marks.")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
LR = 3e-5
BETA_REF = 0.02
STEPS = 200
LOG_EVERY, CKPT_EVERY, MONITOR_EVERY = 1, 50, 10
GEN_BATCH, SCORE_BATCH = 128, 32
SEED = 0

ACTIONABILITY_TEMPLATE = """The text below was proposed as a system prompt for an AI assistant. If two different people read it, would they broadly agree on what the assistant should do differently because of it, for example a language, tone, persona, style, format, length, or topic? Answer No if the text is so metaphorical, surreal, or vague that the intended behavior cannot be identified.

Text:
<<<
{s}
>>>

Answer Yes or No."""


# ---------------------------------------------------------------- vectors and per-vector constants
def load_vec(name, layer):
    vd = torch.load(f"{VEC_DIR}/{name}_L{layer}.pt")
    assert vd["alpha"] is not None, f"{name}_L{layer}: alpha not set"
    return (vd["unit"] * vd["alpha"] * vd["residual_norm"]).cuda(), vd


@torch.no_grad()
def prepare_vector(model, tok, name, layer, xs):
    """Steered rollout pool + log p, KL rung0/rung1, F on the pool, steered distinct fraction."""
    w, vd = load_vec(name, layer)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    y = sample(model, tok, ctx0, ROLLOUT_TOKENS, vec=w, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
    r0 = kl_on_rollouts(model, ctx0, ctx0, y, vec_p=w, layer=layer, batch_size=SCORE_BATCH)
    r1 = kl_on_rollouts(model, ctx0, [chat_ids(tok, x, RUNG1) for x in xs], y, vec_p=w, layer=layer,
                        batch_size=SCORE_BATCH)
    norm = float(w.norm())
    fl = []
    for k in range(FLOOR_DIRS):
        g = torch.Generator().manual_seed(FLOOR_SEED_BASE + k)
        v = torch.randn(w.numel(), generator=g)
        rv = (v / v.norm() * norm).cuda()
        ry = sample(model, tok, ctx0, ROLLOUT_TOKENS, vec=rv, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
        fl.append(st.mean(kl_on_rollouts(model, ctx0, ctx0, ry, vec_p=rv, layer=layer, batch_size=SCORE_BATCH)["rb"]))
    kl0, kl1, Fv = st.mean(r0["rb"]), st.mean(r1["rb"]), st.mean(fl)
    distinct = st.mean(len(set(t)) / max(1, len(t)) for t in y)
    return dict(name=name, layer=layer, w=w, alpha=vd["alpha"], ctx0=ctx0, y=y, logp=r0["logp"], kl0=kl0, kl1=kl1,
                B=min(kl0, kl1), F=Fv, F_dirs=fl, distinct=distinct)


# ---------------------------------------------------------------- T
def t_context(tok):
    return chat_ids(tok, T_INSTRUCTION, NEUTRAL_SYSTEM)


@torch.no_grad()
def t_sample(model, tok, v, n, seed):
    """n candidate system prompts from T (current LoRA), steered by v's vector."""
    ids = sample(model, tok, [t_context(tok)], S_MAX_TOKENS, n_per_ctx=n, vec=v["w"], layer=v["layer"], seed=seed,
                 batch_size=GEN_BATCH)
    eos = set(eos_ids(model, tok))
    texts = [tok.decode([t for t in s if t not in eos], skip_special_tokens=True).strip() for s in ids]
    return ids, texts


def t_logprobs(model, tok, v, ids_list):
    """Per-candidate token log-probs of T's samples under the current model (with grad), steered."""
    ctx = t_context(tok)
    lp, m = completion_logprobs_grad(model, [ctx] * len(ids_list), ids_list, v["w"], v["layer"])
    tgt = torch.zeros(m.shape, dtype=torch.long, device=m.device)
    for j, s in enumerate(ids_list):
        tgt[j, :len(s)] = torch.tensor(s)
    return lp.gather(-1, tgt[..., None]).squeeze(-1), m


def completion_logprobs_grad(model, ctxs, comps, vec, layer):
    """completion_logprobs without no_grad (for the policy update)."""
    B = len(ctxs)
    L = max(len(c) + len(y) for c, y in zip(ctxs, comps))
    M = max(max(len(y) for y in comps), 1)
    ids = torch.zeros((B, L), dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, M), dtype=torch.long)
    cmask = torch.zeros((B, M), dtype=torch.bool)
    for i, (c, y) in enumerate(zip(ctxs, comps)):
        ids[i, :len(c) + len(y)] = torch.tensor(c + y)
        attn[i, :len(c) + len(y)] = 1
        pos[i, :len(y)] = torch.arange(len(c) - 1, len(c) - 1 + len(y))
        cmask[i, :len(y)] = True
    ids, attn, pos, cmask = ids.cuda(), attn.cuda(), pos.cuda(), cmask.cuda()
    with steering(model, layer, vec):
        h = model.model(input_ids=ids, attention_mask=attn).last_hidden_state
    h = torch.gather(h, 1, pos[..., None].expand(-1, -1, h.shape[-1]))
    return F.log_softmax(model.lm_head(h).float(), dim=-1), cmask


# ---------------------------------------------------------------- reward
@torch.no_grad()
def score_candidates(model, tok, gate, v, texts, idx, deg_xs, seed):
    """Reward components for each candidate S (subject model, LoRA disabled by the caller)."""
    read = gate.score(texts)
    y = [v["y"][i] for i in idx]
    logp = [v["logp"][i] for i in idx]
    px = [v["xs"][i] for i in idx]
    live = list(dict.fromkeys(S for S, rd in zip(texts, read) if not rd["hard_fail"]))
    # degeneration: one batched sampling call for all distinct live candidates (same seed: paired)
    dist = {}
    if live:
        dq = sample(model, tok, [chat_ids(tok, x, S) for S in live for x in deg_xs], DEG_TOKENS, seed=seed,
                    batch_size=GEN_BATCH)
        for k, S in enumerate(live):
            chunk = dq[k * len(deg_xs):(k + 1) * len(deg_xs)]
            dist[S] = st.mean(len(set(t)) / max(1, len(t)) for t in chunk)
    kl = {}
    for S in live:
        lq = completion_logprobs_sum(model, [chat_ids(tok, x, S) for x in px], y)
        kl[S] = st.mean(a - b for a, b in zip(logp, lq))
    out = []
    for S, rd in zip(texts, read):
        d = dict(S=S, **{k: rd[k] for k in ("structural_ok", "judge", "payload", "base_ppl", "hard_fail", "penalty")})
        if rd["hard_fail"]:
            d.update(kl=float("nan"), r_kl=float("nan"), distinct=float("nan"), deg_pen=float("nan"), reward=R_FLOOR,
                     hard="readability")
        else:
            r_kl = (v["B"] - kl[S]) / (v["B"] - v["F"])
            deg_pen = DEG_W * max(0.0, (v["distinct"] - DEG_MARGIN) - dist[S]) / DEG_UNIT
            hard = "degeneration" if dist[S] < DEG_HARD else None
            reward = R_FLOOR if hard else max(R_FLOOR, r_kl - READ_W * rd["penalty"] - deg_pen)
            d.update(kl=kl[S], r_kl=r_kl, distinct=dist[S], deg_pen=deg_pen, hard=hard, reward=reward)
        out.append(d)
    return out


@torch.no_grad()
def completion_logprobs_sum(model, ctxs, comps):
    res = []
    for i in range(0, len(comps), SCORE_BATCH):
        lp, m = completion_logprobs(model, ctxs[i:i + SCORE_BATCH], comps[i:i + SCORE_BATCH])
        tgt = torch.zeros(m.shape, dtype=torch.long, device=m.device)
        for j, y in enumerate(comps[i:i + SCORE_BATCH]):
            tgt[j, :len(y)] = torch.tensor(y)
        res += (lp.gather(-1, tgt[..., None]).squeeze(-1) * m).sum(1).tolist()
        del lp
    return res


# ---------------------------------------------------------------- main
def main():
    global TRAIN_VECTORS, STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=None, help="comma list name:layer (overrides TRAIN_VECTORS)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.vectors:
        TRAIN_VECTORS = [(a, int(b)) for a, b in (x.split(":") for x in args.vectors.split(","))]
    if args.steps is not None:
        STEPS = args.steps
    assert TRAIN_VECTORS, "no training vectors"
    random.seed(SEED)
    torch.manual_seed(SEED)

    base, tok = load_model(SUBJECT)
    for p in base.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    # base's modules now carry the LoRA layers in place; `base` is used for hooks/forward, `pm` for adapter control
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=LR)
    gate = R.Gate()
    act_ids = [gate.jt.encode(x, add_special_tokens=False)[0] for x in ("Yes", "No")]

    xs = wildchat_prompts("train", N_POOL, POOL_OFFSET)
    vecs = []
    with pm.disable_adapter():
        for name, layer in TRAIN_VECTORS:
            v = prepare_vector(base, tok, name, layer, xs)
            v["xs"] = xs
            vecs.append(v)
            print(f"[prep] {name}_L{layer}: KL0 {v['kl0']:.2f} KL1 {v['kl1']:.2f} B {v['B']:.2f} F {v['F']:.2f} "
                  f"gap {v['B'] - v['F']:.2f} distinct {v['distinct']:.3f}", flush=True)
    deg_xs = wildchat_prompts("train", DEG_PROMPTS, POOL_OFFSET + N_POOL)

    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}{('_' + args.tag) if args.tag else ''}"
    run_dir = f"{OUT_DIR}/phase2_train_{stamp}"
    os.makedirs(run_dir, exist_ok=True)
    cfg = {k: v for k, v in globals().items() if k.isupper() and not k.endswith("TEMPLATE")}
    json.dump({"config": cfg, "vectors": [{k: v[k] for k in ("name", "layer", "alpha", "kl0", "kl1", "B", "F", "F_dirs",
                                                              "distinct")} for v in vecs]},
              open(f"{run_dir}/config.json", "w"), indent=1, default=str)
    cand_log = open(f"{run_dir}/candidates.jsonl", "a")
    print(f"run dir: {run_dir}", flush=True)

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        stats = []
        for vi, v in enumerate(vecs):
            seed = SEED + 1000 * step + vi
            pm.eval()
            ids, texts = t_sample(base, tok, v, G, seed)
            idx = random.Random(seed).sample(range(N_POOL), PROMPTS_PER_STEP)
            with pm.disable_adapter():
                sc = score_candidates(base, tok, gate, v, texts, idx, deg_xs, seed)
            r = torch.tensor([d["reward"] for d in sc])
            adv = (r - r.mean()) / (r.std() + 1e-6) if r.std() > 1e-6 else torch.zeros_like(r)
            pm.train()
            lp, m = t_logprobs(base, tok, v, ids)
            with torch.no_grad(), pm.disable_adapter():
                lr_, _ = t_logprobs(base, tok, v, ids)
            k3 = (torch.exp(lr_ - lp) - (lr_ - lp) - 1) * m          # per-token KL(T || T_ref) estimator
            pg = -(adv.cuda()[:, None] * lp * m).sum() / (S_MAX_TOKENS * G)
            kl_ref = k3.sum() / (S_MAX_TOKENS * G)
            loss = (pg + BETA_REF * kl_ref) / len(vecs)
            loss.backward()
            for d, a in zip(sc, adv.tolist()):
                cand_log.write(json.dumps(dict(step=step, vector=f"{v['name']}_L{v['layer']}", adv=a, **d),
                                          ensure_ascii=False, default=str) + "\n")
            valid = [d for d in sc if d["hard"] is None]
            best = max(sc, key=lambda d: d["reward"])
            stats.append(dict(v=v["name"], r=float(r.mean()), rmax=float(r.max()),
                              rkl=st.mean(d["r_kl"] for d in valid) if valid else float("nan"),
                              hard=sum(d["hard"] is not None for d in sc) / G,
                              empty=sum(d["S"] == "" for d in sc) / G, kl_ref=float(kl_ref), best=best["S"]))
        torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
        opt.step()
        cand_log.flush()
        if step % LOG_EVERY == 0:
            print(f"step {step:4d} | " + " | ".join(
                f"{s['v'][:7]} r {s['r']:+.2f}/{s['rmax']:+.2f} kl {s['rkl']:+.2f} hard {s['hard']:.2f} "
                f"empty {s['empty']:.2f}" for s in stats), flush=True)
        if step % MONITOR_EVERY == 0:
            # blind-spot monitor (no reward pressure): actionability of this step's best candidates
            bests = [s["best"] for s in stats if s["best"]]
            if bests:
                pa = gate._probs(ACTIONABILITY_TEMPLATE, bests, act_ids)[:, 0].tolist()
                print(f"  [monitor] actionability<0.3 frac {sum(p < 0.3 for p in pa) / len(pa):.2f}; best S: " +
                      " || ".join(f"{s['v']}: {s['best'][:80]!r}" for s in stats), flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step == STEPS - 1:
            pm.save_pretrained(f"{run_dir}/lora_step{step + 1}")
    cand_log.close()
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
