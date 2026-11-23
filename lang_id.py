"""Phase 3b, Run 1: language identification for the Romance-language vectors.

Replaces the stopword behavior metric (which cannot separate French/Spanish/Portuguese/Italian/Catalan)
with lingua-language-detector. Per vector, on heldout[0:N]:
  - steered samples (base + w, neutral system): what language does the VECTOR produce?
  - q(.|S) samples (base, system = T's emitted greedy S): what language does T's PROMPT induce?
  - the detected language of S itself.
french is the positive control (known to work; detector should say French).

Decision rule (italian): if steered samples are a Romance smear rather than Italian, the vector is
mislabelled and its Phase 3 result is uninformative about T. If steered is clean Italian but T's
prompt induces another language, that is a real conditioning failure.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/workspace/.pylib")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from lingua import Language, LanguageDetectorBuilder

from common import NEUTRAL_SYSTEM, chat_ids, load_model, sample, wildchat_prompts

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 100
HELDOUT_OFFSET = 0
MAX_NEW = 128
SEED_STEER, SEED_Q = 101, 202
GEN_BATCH = 100
LANGS = [Language.ENGLISH, Language.FRENCH, Language.SPANISH, Language.PORTUGUESE,
         Language.ITALIAN, Language.CATALAN, Language.GERMAN]

# vector (name, layer, alpha) and T's emitted greedy S (from Phase 3 SUMMARY / Phase 2 step200)
VECTORS = [
    ("french", 18, 0.20,
     "Veuillez écrire des réponses précises et détaillées en français avec un ton courtois et "
     "professionnel, comme celui que vous utilisez habituellement."),
    ("spanish", 18, 0.15,
     "Por favor, escribe respuestas detalladas y amplias en un tono claro y amigable, utilizando un "
     "estilo informado y respetuoso."),
    ("italian", 18, 0.15,
     "Please escribe respostes en un llengua clara i natural, amb un ton educatiu i assistentiu, "
     "similar al meu propi estil de resposta."),
]
OUT = f"/workspace/runs/phase3b_langid.json"


def dist(det, texts):
    c, conf = Counter(), {}
    for t in texts:
        t = t.strip()
        if len(t) < 10:
            c["<short>"] += 1
            continue
        vals = det.compute_language_confidence_values(t)
        lang = vals[0].language.name
        c[lang] += 1
        conf.setdefault(lang, []).append(vals[0].value)
    n = sum(c.values())
    rows = sorted(c.items(), key=lambda kv: -kv[1])
    return {"n": n, "dist": {k: round(v / n, 3) for k, v in rows},
            "mean_conf": {k: round(sum(conf[k]) / len(conf[k]), 2) for k in conf}}


def main():
    det = LanguageDetectorBuilder.from_languages(*LANGS).build()
    model, tok = load_model(MODEL)
    xs = wildchat_prompts("heldout", N, HELDOUT_OFFSET)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    out = {}
    for name, layer, alpha, S in VECTORS:
        vd = torch.load(f"/workspace/vectors/{name}_L{layer}.pt")
        w = (vd["unit"] * alpha * vd["residual_norm"]).cuda()
        y_steer = sample(model, tok, ctx0, MAX_NEW, vec=w, layer=layer, seed=SEED_STEER, batch_size=GEN_BATCH)
        ctx_q = [chat_ids(tok, x, S) for x in xs]
        y_q = sample(model, tok, ctx_q, MAX_NEW, seed=SEED_Q, batch_size=GEN_BATCH)
        steer = dist(det, [tok.decode(t) for t in y_steer])
        q = dist(det, [tok.decode(t) for t in y_q])
        s_lang = det.detect_language_of(S)
        out[name] = {"layer": layer, "alpha": alpha, "S": S, "S_detected": str(s_lang),
                     "steered": steer, "q_under_T_prompt": q}
        print(f"\n===== {name} (L{layer}, a{alpha})", flush=True)
        print(f"  steered ({steer['n']}):  {steer['dist']}")
        print(f"  q|T-prompt ({q['n']}): {q['dist']}")
        print(f"  S itself detected as: {s_lang}   S={S[:70]!r}")
    json.dump(out, open(OUT, "w"), indent=1, default=str, ensure_ascii=False)
    print(f"\nlogged -> {OUT}")


if __name__ == "__main__":
    main()
