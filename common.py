"""Shared utilities: model loading, steering hook, WildChat prompts, sampling, KL scoring.

Conventions
- Steering adds a fixed vector to the OUTPUT of decoder layer `layer` (residual stream after that
  block) at every token position, prompt and generated alike.
- Rollouts are sampled with pure ancestral sampling (temperature 1, no top-k/top-p/repetition
  penalty) so that they are genuine samples from p and the MC KL estimator is unbiased.
- KL(p || q) is a SEQUENCE-level quantity in nats: sum over generated tokens (including the
  terminating EOS) of per-token terms. Two estimators are reported on rollouts Y ~ p:
    mc : sum_t [log p(y_t|y<t) - log q(y_t|y<t)]          (plain Monte Carlo)
    rb : sum_t KL(p(.|y<t) || q(.|y<t))                    (Rao-Blackwellized, lower variance)
  Both are unbiased for the same quantity; disagreement beyond noise indicates a bug.
"""
import contextlib
import json
import os
import random

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ---------------------------------------------------------------- config (data)
WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_SHARD = "data/train-00000-of-00014.parquet"
PROMPT_CACHE = "/workspace/data/wildchat_prompts.json"
PROMPT_MIN_CHARS = 20
PROMPT_MAX_CHARS = 600
PROMPT_POOL_SEED = 0
N_TRAIN_PROMPTS = 1000
N_HELDOUT_PROMPTS = 500


# ---------------------------------------------------------------- model
def load_model(name, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # for generation; scoring builds its own right-padded batches
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def decoder_layers(model):
    return model.model.layers


@contextlib.contextmanager
def steering(model, layer, vec):
    """Add `vec` to the residual stream output of decoder layer `layer`. vec=None -> no hook."""
    if vec is None:
        yield
        return

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)

    h = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------- prompts
def _build_prompt_pool():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(WILDCHAT_REPO, WILDCHAT_SHARD, repo_type="dataset")
    t = pq.read_table(path, columns=["conversation", "language", "toxic", "redacted"]).to_pylist()
    seen, pool = set(), []
    for row in t:
        if row["language"] != "English" or row["toxic"] or row["redacted"]:
            continue
        conv = row["conversation"]
        if not conv or conv[0]["role"] != "user":
            continue
        text = conv[0]["content"].strip()
        if not (PROMPT_MIN_CHARS <= len(text) <= PROMPT_MAX_CHARS) or text in seen:
            continue
        seen.add(text)
        pool.append(text)
    random.Random(PROMPT_POOL_SEED).shuffle(pool)
    need = N_TRAIN_PROMPTS + N_HELDOUT_PROMPTS
    assert len(pool) >= need, f"only {len(pool)} prompts after filtering"
    out = {
        "source": f"{WILDCHAT_REPO}/{WILDCHAT_SHARD}",
        "filter": dict(lang="English", nontoxic=True, unredacted=True, first_user_turn=True,
                       min_chars=PROMPT_MIN_CHARS, max_chars=PROMPT_MAX_CHARS, seed=PROMPT_POOL_SEED),
        "train": pool[:N_TRAIN_PROMPTS],
        "heldout": pool[N_TRAIN_PROMPTS:need],
    }
    os.makedirs(os.path.dirname(PROMPT_CACHE), exist_ok=True)
    with open(PROMPT_CACHE, "w") as f:
        json.dump(out, f, indent=1)
    return out


def wildchat_prompts(split, n, offset=0):
    """Fixed, cached train/heldout split of first-turn English WildChat prompts."""
    data = json.load(open(PROMPT_CACHE)) if os.path.exists(PROMPT_CACHE) else _build_prompt_pool()
    xs = data[split][offset:offset + n]
    assert len(xs) == n, f"requested {n} from {split}[{offset}:], got {len(xs)}"
    return xs


NEUTRAL_SYSTEM = ""  # rung 0: explicit empty system block. p_steer always uses this.


def chat_ids(tok, user, system):
    """Token ids of the chat context ending with the assistant header.
    `system` is REQUIRED and always emitted as an explicit system block, so contexts differ only in
    the system text. (Omitting it would make Qwen's template inject its default system prompt --
    the Phase 0 confound.) system=None is allowed only to reproduce that legacy behavior."""
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
           [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


# ---------------------------------------------------------------- sampling
def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    e = set(e if isinstance(e, (list, tuple)) else [e])
    e.add(tok.eos_token_id)
    return sorted(e)


@torch.no_grad()
def sample(model, tok, ctxs, max_new_tokens, n_per_ctx=1, vec=None, layer=None, seed=0,
           batch_size=64, return_logits=False):
    """Ancestral samples from (model + steering). Returns list (len(ctxs)*n_per_ctx, ctx-major) of
    completion token-id lists, each truncated after (and including) the first EOS.
    If return_logits, also returns per-sequence raw next-token logits for the kept tokens."""
    eos = eos_ids(model, tok)
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None,
                          repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                          eos_token_id=eos, pad_token_id=tok.pad_token_id,
                          output_logits=return_logits, return_dict_in_generate=True)
    flat = [c for c in ctxs for _ in range(n_per_ctx)]
    torch.manual_seed(seed)
    outs, logit_outs = [], []
    for i in range(0, len(flat), batch_size):
        chunk = flat[i:i + batch_size]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, c in enumerate(chunk):
            ids[j, L - len(c):] = torch.tensor(c)
            mask[j, L - len(c):] = 1
        with steering(model, layer, vec):
            g = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), generation_config=gc)
        gen = g.sequences[:, L:].tolist()
        for j, seq in enumerate(gen):
            k = next((t + 1 for t, x in enumerate(seq) if x in eos), len(seq))
            outs.append(seq[:k])
            if return_logits:
                logit_outs.append(torch.stack([g.logits[t][j] for t in range(k)]).float().cpu())
    return (outs, logit_outs) if return_logits else outs


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def completion_logprobs(model, ctxs, comps, vec=None, layer=None):
    """Teacher-forced full next-token log-distributions at each completion position.
    Right-padded so default position ids are correct. Returns [B, M, V] float32 and mask [B, M]."""
    B = len(ctxs)
    full = [c + y for c, y in zip(ctxs, comps)]
    L = max(len(f) for f in full)
    M = max(len(y) for y in comps)
    ids = torch.zeros((B, L), dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, M), dtype=torch.long)  # hidden-state position predicting each comp token
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
    logp = F.log_softmax(model.lm_head(h).float(), dim=-1)
    return logp, cmask


@torch.no_grad()
def kl_on_rollouts(model, ctxs_p, ctxs_q, comps, vec_p=None, vec_q=None, layer=None, batch_size=16):
    """Per-sequence KL(p || q) estimates on completions sampled from p.
    p = model + vec_p conditioned on ctxs_p; q = model + vec_q conditioned on ctxs_q.
    Returns dict of per-sequence lists: mc, rb, logq (sum log q(Y)), logp, ntok."""
    res = {k: [] for k in ["mc", "rb", "logp", "logq", "ntok"]}
    for i in range(0, len(comps), batch_size):
        sl = slice(i, i + batch_size)
        y = comps[sl]
        lp, m = completion_logprobs(model, ctxs_p[sl], y, vec_p, layer)
        lq, _ = completion_logprobs(model, ctxs_q[sl], y, vec_q, layer)
        M = lp.shape[1]
        tgt = torch.zeros(m.shape, dtype=torch.long, device=m.device)
        for j, yy in enumerate(y):
            tgt[j, :len(yy)] = torch.tensor(yy)
        lp_y = lp.gather(-1, tgt[..., None]).squeeze(-1) * m
        lq_y = lq.gather(-1, tgt[..., None]).squeeze(-1) * m
        tok_kl = (lp.exp() * (lp - lq)).sum(-1) * m
        res["mc"] += (lp_y - lq_y).sum(1).tolist()
        res["rb"] += tok_kl.sum(1).tolist()
        res["logp"] += lp_y.sum(1).tolist()
        res["logq"] += lq_y.sum(1).tolist()
        res["ntok"] += m.sum(1).tolist()
        del lp, lq
    return res


def summarize(res):
    """Mean +/- standard error of per-sequence KL estimates, plus per-token rate."""
    import statistics as st
    n = len(res["rb"])
    out = {}
    for k in ["mc", "rb"]:
        xs = res[k]
        out[k] = st.mean(xs)
        out[k + "_se"] = (st.stdev(xs) / n ** 0.5) if n > 1 else float("nan")
    out["rb_per_tok"] = sum(res["rb"]) / max(1, sum(res["ntok"]))
    out["mean_len"] = sum(res["ntok"]) / n
    return out
