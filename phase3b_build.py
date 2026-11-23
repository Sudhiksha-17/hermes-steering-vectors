"""Phase 3b Run 4: build candidate holdout vectors (diff-of-means, same recipe), report cosine to the
6 trained vectors FIRST (screen on cosine before calibrating), at layers 8/12/18.
"""
import sys, os
sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
import statistics as st
import torch
from behaviors import ALL_BEHAVIORS
from common import NEUTRAL_SYSTEM, chat_ids, load_model, sample, wildchat_prompts
from build_vectors_p2 import (mean_response_acts, residual_norm, N_CONTRAST, CONTRAST_OFFSET,
                              CONTRAST_MAX_TOKENS, CONTRAST_SEED, N_CALIB, CALIB_OFFSET, GEN_BATCH)

CANDIDATES = ["japanese", "russian", "korean", "uppercase"]
LAYERS = [8, 12, 18]
TRAIN = [("french", 18), ("pirate", 8), ("german", 18), ("formal", 18), ("enthusiastic", 18), ("lowercase", 18)]


def main():
    model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
    prompts = wildchat_prompts("train", N_CONTRAST, CONTRAST_OFFSET)
    neutral = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in prompts]
    neg = sample(model, tok, neutral, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
    m_neg = mean_response_acts(model, neutral, neg, LAYERS)
    neg_t = [tok.decode(y) for y in neg]
    tr = {f"{n}_L{l}": torch.load(f"vectors/{n}_L{l}.pt")["unit"].float() for n, l in TRAIN}

    for name in CANDIDATES:
        spec = ALL_BEHAVIORS[name]
        metric = spec["metric"]
        elicit = [chat_ids(tok, x, spec["elicit"]) for x in prompts]
        pos = sample(model, tok, elicit, CONTRAST_MAX_TOKENS, seed=CONTRAST_SEED, batch_size=GEN_BATCH)
        m_pos = mean_response_acts(model, neutral, pos, LAYERS)
        pos_t = [tok.decode(y) for y in pos]
        cm = dict(pos=st.mean(map(metric, pos_t)), neg=st.mean(map(metric, neg_t)))
        print(f"\n===== {name}: contrast {spec['metric_name']} pos {cm['pos']:.1f} / neg {cm['neg']:.1f}", flush=True)
        for L in LAYERS:
            raw = m_pos[L] - m_neg[L]
            unit = raw / raw.norm()
            torch.save(dict(name=name, layer=L, model="Qwen/Qwen2.5-3B-Instruct", raw=raw, unit=unit,
                            raw_norm=raw.norm().item(), elicit_system=spec["elicit"], contrast_prompts=prompts,
                            pos_responses=pos_t, neg_responses=neg_t, contrast_metric=cm,
                            config=dict(source="phase3b_build"), alpha=None, residual_norm=None),
                       f"vectors/{name}_L{L}.pt")
            cs = {k: float(torch.dot(unit.float(), v)) for k, v in tr.items()}
            best = max(cs, key=cs.get)
            print(f"  L{L}: |raw|={raw.norm():.1f}  max cos {cs[best]:+.3f} -> {best}   "
                  + " ".join(f"{k.split('_L')[0]}:{cs[k]:+.2f}" for k in sorted(cs, key=cs.get, reverse=True)[:3]))


if __name__ == "__main__":
    main()
