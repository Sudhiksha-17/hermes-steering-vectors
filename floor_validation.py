"""Phase 1b, run 1: validate the unstructured-perturbation floor.

Step 5 defined F = mean_k KL(p_rand_k || q_neutral) and treated it as UNCLOSABLE by any prompt. This
script tests that: for each of the 8 random directions from Step 5, score every ladder rung S:
    KL(p_rand_k || q_S)        on the same rollouts Y ~ p_rand_k (paired across S)
If ladder prompts reduce this below rung 0, F is partly closable and Step 5's closure-vs-floor
numbers are inflated.

Rollouts: Step 5 logged directions + seeds but not token ids. Sampling is seeded and deterministic
(Step 5 reproduced the Step 4 ladder to 2 decimals), so rollouts are regenerated with the same seed
and the per-direction rung-0 KL is ASSERTED to match the logged value. Rollouts are saved this time.

Ladders scored per norm group: ||w||=14.39 -> French and hedging ladders (shared directions);
||w||=43.16 -> pirate ladder.

Also reports, for each vector and rung, a prompt-matched corrected closure:
    floor under S:   F_S = mean_k KL(p_rand_k || q_S)
    corrected:       1 - (KL_S(w) - F_S) / (KL0(w) - F_0)
i.e. structured KL removed, if the unstructured part under prompt S is F_S rather than F_0.
"""
import datetime
import json
import os

import numpy as np
import torch

from common import NEUTRAL_SYSTEM, chat_ids, kl_on_rollouts, load_model, sample, wildchat_prompts
from oracle_search import ladder

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FLOOR_RUN = "/workspace/runs/phase1_randfloor_20261122_231649.json"
VEC_DIR, OUT_DIR = "/workspace/vectors", "/workspace/runs"
GROUPS = {"14.39": ["french", "hedging"], "43.16": ["pirate"]}   # norm group -> ladders to score
HELDOUT_OFFSET, N_HELDOUT = 0, 200        # identical to Step 4/5
MAX_NEW_TOKENS = 128
SEED_P = 101                               # identical to Step 4/5
GEN_BATCH, SCORE_BATCH = 100, 16
REPRO_TOL = 0.02                           # nats: max allowed |KL0 - logged KL0| per direction
N_BOOT = 4000
BOOT_SEED = 0


def boot_frac(base, cond, rng):
    """base, cond: [K, n] per-direction per-prompt KL. Returns point, se of (mean base - mean cond)/mean base,
    two-level bootstrap over directions and prompts."""
    K, n = base.shape
    pt = (base.mean() - cond.mean()) / base.mean()
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        k, i = rng.integers(0, K, K), rng.integers(0, n, n)
        bb, cc = base[np.ix_(k, i)], cond[np.ix_(k, i)]
        bs[b] = (bb.mean() - cc.mean()) / bb.mean()
    return float(pt), float(bs.std(ddof=1))


def main():
    floor = json.load(open(FLOOR_RUN))
    by_name = {r["name"]: r for r in floor["results"]}
    model, tok = load_model(MODEL_NAME)
    xs = wildchat_prompts("heldout", N_HELDOUT, HELDOUT_OFFSET)
    ctx0 = [chat_ids(tok, x, NEUTRAL_SYSTEM) for x in xs]
    rng = np.random.default_rng(BOOT_SEED)
    out = {"config": {k: v for k, v in globals().items() if k.isupper()}, "groups": {}}
    rollouts = {}

    for gkey, names in GROUPS.items():
        ref = by_name[names[0]]
        norm, layer = ref["norm"], ref["layer"]
        dim = torch.load(f"{VEC_DIR}/{names[0]}_L18.pt")["unit"].numel()
        seeds = [d["seed"] for d in ref["random_dirs"]]
        logged = [d["fwd"] for d in ref["random_dirs"]]
        # rungs to score: union of ladders, dedup by S text
        rungs = {}
        for nm in names:
            for rung, S in ladder(nm):
                rungs.setdefault(S, f"{nm}:{rung}" if rung[0] in "23" else rung)
        print(f"\n===== norm group {gkey} (layer {layer}, ||r|| {norm:.2f}): {len(seeds)} dirs x {len(rungs)} rungs",
              flush=True)
        per = {lab: [] for lab in rungs.values()}          # label -> [K][n]
        for k, sd in enumerate(seeds):
            g = torch.Generator().manual_seed(sd)
            v = torch.randn(dim, generator=g)
            r = (v / v.norm() * norm).cuda()
            y = sample(model, tok, ctx0, MAX_NEW_TOKENS, vec=r, layer=layer, seed=SEED_P, batch_size=GEN_BATCH)
            rollouts[f"{gkey}/{sd}"] = y
            line = []
            for S, lab in rungs.items():
                ctx_q = [chat_ids(tok, x, S) for x in xs]
                res = kl_on_rollouts(model, ctx0, ctx_q, y, vec_p=r, layer=layer, batch_size=SCORE_BATCH)
                per[lab].append(res["rb"])
                line.append(f"{lab.split(' ')[0]} {np.mean(res['rb']):.2f}")
            kl0 = float(np.mean(per["0 neutral"][-1]))
            assert abs(kl0 - logged[k]) < REPRO_TOL, f"dir {sd}: KL0 {kl0:.3f} != logged {logged[k]:.3f}; rollouts differ"
            print(f"  dir seed {sd}: " + " | ".join(line), flush=True)

        base = np.array(per["0 neutral"])
        rows = []
        for S, lab in rungs.items():
            a = np.array(per[lab])
            fr, fr_se = boot_frac(base, a, rng)
            dir_means = a.mean(1)
            # per-direction paired change
            dd = (base - a).mean(1)
            rows.append(dict(label=lab, S=S, floor_S=float(a.mean()), floor_S_dir_min=float(dir_means.min()),
                             floor_S_dir_max=float(dir_means.max()), reduction=float(base.mean() - a.mean()),
                             frac_floor_closed=fr, frac_floor_closed_se=fr_se,
                             n_dirs_reduced=int((dd > 0).sum()), per_dir_reduction=dd.tolist(),
                             per_dir_mean=dir_means.tolist()))
        F0 = float(base.mean())
        # corrected closure for each vector in this group
        corr = {}
        for nm in names:
            vr = {x["rung"]: x for x in by_name[nm]["rows"]}
            kl0w = vr["0 neutral"]["fwd"]
            corr[nm] = []
            for rung, S in ladder(nm):
                FS = next(x["floor_S"] for x in rows if x["S"] == S)
                klS = vr[rung]["fwd"]
                corr[nm].append(dict(rung=rung, kl_w=klS, floor_S=FS,
                                     closed_vs_floor_step5=vr[rung]["closed_vs_floor"],
                                     closed_corrected=1 - (klS - FS) / (kl0w - F0)))
        out["groups"][gkey] = dict(norm=norm, layer=layer, seeds=seeds, F0=F0, rows=rows, corrected=corr)

    # ---------------------------------------------------------------- report
    for gkey, g in out["groups"].items():
        print(f"\n## Floor under each ladder prompt, ||r|| = {g['norm']:.2f} (F0 = {g['F0']:.2f}, 8 random dirs)\n")
        print("| Rung | S | F_S = mean KL(p_rand ‖ q_S) | dir range | reduction vs rung 0 | frac of floor closed | dirs reduced |")
        print("|---|---|---|---|---|---|---|")
        for x in g["rows"]:
            s = (x["S"][:45] + "…") if len(x["S"]) > 45 else x["S"]
            print(f"| {x['label']} | {s!r} | {x['floor_S']:.2f} | {x['floor_S_dir_min']:.1f}–{x['floor_S_dir_max']:.1f} | "
                  f"{x['reduction']:+.2f} | {x['frac_floor_closed']:+.1%} ± {x['frac_floor_closed_se']:.1%} | "
                  f"{x['n_dirs_reduced']}/8 |")
        for nm, cs in g["corrected"].items():
            print(f"\n### {nm}: closure vs floor, Step 5 (F0) vs prompt-matched (F_S)\n")
            print("| Rung | KL_S(w) | F_S | closed vs floor (Step 5) | closed, prompt-matched floor |")
            print("|---|---|---|---|---|")
            for c in cs:
                print(f"| {c['rung']} | {c['kl_w']:.2f} | {c['floor_S']:.2f} | {c['closed_vs_floor_step5']:+.1%} | "
                      f"{c['closed_corrected']:+.1%} |")
    print("\nPASS criterion: no ladder prompt meaningfully reduces KL(p_rand || q_S) below rung 0 (frac closed <= ~0).")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    path = f"{OUT_DIR}/phase1b_floorval_{stamp}.json"
    json.dump(out, open(path, "w"), indent=1, default=str)
    torch.save(rollouts, f"{OUT_DIR}/phase1b_randrollouts_{stamp}.pt")
    print(f"logged -> {path}  (rollouts -> phase1b_randrollouts_{stamp}.pt)")


if __name__ == "__main__":
    main()
