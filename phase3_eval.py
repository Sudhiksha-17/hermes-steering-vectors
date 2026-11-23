"""
phase3_eval.py
--------------
Phase 3: does T condition on w, or did it memorize six prompts?

Evaluates the trained T (LoRA step 200) on HOLDOUT vectors it never saw.
Stops the pod when finished, including on crash.

Run detached:
    source /workspace/env.sh
    cd /workspace
    nohup python phase3_eval.py > /dev/null 2>&1 &

Results are written incrementally to runs/phase3_<stamp>/ so a crash at
vector 4 does not lose vectors 1 to 3. The full log is runs/phase3_<stamp>/run.log
(stdout is discarded by the nohup line above, so the log file is the record).

Changes from the user-supplied draft (all deliberate, see PHASE3 notes in the log header):
  - ADAPTER implemented against the real repo (the draft's build_vector / sweep_alpha / floor_for_vector /
    ladder_best / evaluate do not exist).
  - Negative controls are exempt from the R4 "F/KL0 > 0.6 -> UNUSABLE" skip: a random direction has F/KL0 ~ 1 by
    construction, so the draft would never have evaluated it.
  - log() also writes run.log (the draft's nohup line sends stdout to /dev/null).
  - alpha sweep: Step 3 asks for coherence "by eye"; unattended, alpha is chosen by the automated criteria on the
    256-prompt pool (gap >= 3x presence, gap >= 15 nats, F/KL0 <= 0.6, distinct >= 0.65, behavior shift >= 50% of
    contrast) and 3 samples per alpha are saved for post-hoc eye review.
  - "archaic_formal" is defined (behaviors.BEHAVIORS_P3) as Victorian-style formal English: an interpretation.
  - rung-4 holdout searches give the proposer gate-v2 feedback (the Phase 1b gate's false rejections sabotaged the
    lowercase search); baselines are still structurally-valid-only, never judge-filtered, never held-out-selected.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

WORKSPACE   = Path("/workspace")
RUNS        = WORKSPACE / "runs"
LORA_CKPT   = RUNS / "phase2_train_20261123_023935_main6" / "lora_step200"

# Holdout vectors. `search` controls baseline cost:
#   "rung4"  = full 14B proposer search (expensive, ~20-40 min)
#   "ladder" = hand-written ladder only (cheap)
# Phase 2 measured rung 4 beating the ladder by 16-17 points on register
# vectors, so ladder-only baselines are conservative against T. Label which
# is which in the writeup.
# Phase 3b Run 4: low-cosine holdouts (max cos < 0.7 to all 6 trained vectors), the actual
# generalization test. Cosines screened first in phase3b_build.py. Vector files use <name>_L<layer>.pt;
# korean/japanese/russian/uppercase were built at layers 8/12/18 by phase3b_build.py.
HOLDOUTS = [
    # name,          layer, alpha, search    (max cos to trained, from phase3b_build)
    ("japanese",        18,  None, "rung4"),   # 0.69 (german): clean script language, full search
    ("uppercase",       18,  None, "ladder"),  # 0.51 (german): format vector; lowercase was T's failure
    ("korean",          12,  None, "ladder"),  # 0.65 (german): second far language
]
# Phase 3 holdouts, kept for reference / re-run with --only:
HOLDOUTS_PHASE3 = [
    ("spanish", 18, 0.15, "rung4"), ("emoji", 12, 0.60, "rung4"), ("italian", 18, None, "ladder"),
    ("archaic_formal", 18, None, "ladder"), ("gloomy_L8", 8, None, "ladder"),
]

N_FLOOR_DIRS   = 8          # matched-norm random directions per vector (held-out floor)
N_CANDIDATES   = 8          # T best-of-8, as in Phase 2 eval
HELDOUT_SLICE  = (0, 200)   # heldout[0:200]
TRAIN_SELECT   = (64, 192)  # candidate selection pool, never held-out
POOL_SLICE     = (192, 448) # pool used for the alpha sweep / criterion, as in training
MAX_NEW        = 128
SWEEP_GRID     = {18: [0.1, 0.15, 0.2, 0.3, 0.4, 0.6], 12: [0.3, 0.4, 0.5, 0.6], 8: [0.3, 0.4, 0.5, 0.6]}
SWEEP_RAND_DIRS, SWEEP_RAND_SEED = 3, 9000   # same as train_T.prepare_vector

# Negative controls, carried from Phase 2. Both must still pass (not beat the empty prompt).
NEGATIVES = [("hedging", 18, 0.20), ("random_seed5000", 18, 0.20)]

AUTOSTOP       = True       # set False to keep the pod alive after the run
STOP_GRACE_SEC = 30         # time to read the log tail before shutdown

# ──────────────────────────────────────────────────────────────────────────────

LOGFILE = None


def log(msg):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")


# ── AUTO-STOP ─────────────────────────────────────────────────────────────────

def preflight_autostop():
    """
    Verify the pod can actually stop itself. Called BEFORE any long work, so a
    missing credential costs zero minutes instead of the whole run.
    Never actually stops the pod here.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not pod_id:
        return False, "RUNPOD_POD_ID is not set"
    try:
        r = subprocess.run(
            ["runpodctl", "get", "pod", pod_id],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        return False, "runpodctl not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "runpodctl timed out"
    if r.returncode != 0:
        return False, f"runpodctl auth failed: {r.stderr.strip()[:200]}"
    return True, pod_id


def stop_pod(pod_id):
    """Flush to the network volume, pause, then stop. Stop, never remove."""
    log("flushing to /workspace")
    os.sync()
    log(f"stopping pod {pod_id} in {STOP_GRACE_SEC}s")
    time.sleep(STOP_GRACE_SEC)
    subprocess.run(["runpodctl", "stop", "pod", pod_id], timeout=120)


# ── ADAPTER ───────────────────────────────────────────────────────────────────
# Every call into the existing code lives here.

NEG_NAMES = {n for n, _, _ in NEGATIVES}
STATE = {}          # lazily loaded models / caches (nothing is loaded on --dry-run)
OUTDIR = None
TRAINED = [("french", 18), ("pirate", 8), ("german", 18), ("formal", 18), ("enthusiastic", 18), ("lowercase", 18)]


def cosine_to_trained(R, vec):
    """Run 3 / gate item 3: max cosine of this holdout's unit vector to the 6 trained vectors, and which."""
    import torch
    if vec.get("beh") == "random" or vec.get("path") is None:
        return dict(max_cos=None, nearest_trained=None, cos_all={})
    u = torch.load(vec["path"])["unit"].float()
    cos = {f"{n}_L{l}": round(float(torch.dot(u, torch.load(f'{WORKSPACE}/vectors/{n}_L{l}.pt')["unit"].float())), 3)
           for n, l in TRAINED}
    nearest = max(cos, key=cos.get)
    return dict(max_cos=cos[nearest], nearest_trained=nearest, cos_all=cos)


def load_repo():
    """Import the project modules. Returns a namespace of what we need."""
    sys.path.insert(0, str(WORKSPACE))
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    import common, behaviors, build_vectors_p2, eval_T, matched_norm_control, rung4, readability
    return dict(common=common, behaviors=behaviors, build=build_vectors_p2, evalT=eval_T,
                control=matched_norm_control, rung4=rung4, readability=readability)


def _models(R):
    """Subject/T base (+ LoRA in place), gate v2. Loaded once."""
    if "base" not in STATE:
        from peft import PeftModel
        base, tok = R["common"].load_model("Qwen/Qwen2.5-3B-Instruct")
        pm = PeftModel.from_pretrained(base, str(LORA_CKPT))
        pm.eval()
        STATE.update(base=base, tok=tok, pm=pm, gate=R["readability"].Gate())
        assert R["readability"].GATE_VERSION == 2
    return STATE["base"], STATE["tok"], STATE["pm"], STATE["gate"]


def _beh(name):
    """Holdout name -> (behavior name used for metric/ladder/vector file)."""
    return {"gloomy_L8": "gloomy", "random_seed5000": "random"}.get(name, name)


def _xs(R, split, sl):
    return R["common"].wildchat_prompts(split, sl[1] - sl[0], sl[0])


def build_holdout_vector(R, name, layer):
    """Difference-of-means vector, same recipe as Phase 1/2 (Step 2). Builds only if the file is missing."""
    import statistics as st
    import torch
    beh = _beh(name)
    if beh == "random":
        return dict(beh="random", layer=layer, path=None)
    path = WORKSPACE / "vectors" / f"{beh}_L{layer}.pt"
    if path.exists():
        vd = torch.load(path)
        if vd.get("residual_norm") is None:     # Phase 2 saved L8/L12 files without it (no alpha chosen there)
            base, tok, pm, _ = _models(R)
            C, B = R["common"], R["build"]
            with pm.disable_adapter():
                calib = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM)
                         for x in C.wildchat_prompts("train", B.N_CALIB, B.CALIB_OFFSET)]
                vd["residual_norm"] = B.residual_norm(base, calib, layer)
            torch.save(vd, path)
            log(f"  {path.name}: residual norm computed {vd['residual_norm']:.2f}")
        return dict(beh=beh, layer=layer, path=str(path))
    base, tok, pm, _ = _models(R)
    C, B = R["common"], R["build"]
    spec = R["behaviors"].ALL_BEHAVIORS[beh]
    with pm.disable_adapter():
        prompts = C.wildchat_prompts("train", B.N_CONTRAST, B.CONTRAST_OFFSET)
        neutral = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM) for x in prompts]
        if "neg" not in STATE:
            neg = C.sample(base, tok, neutral, B.CONTRAST_MAX_TOKENS, seed=B.CONTRAST_SEED, batch_size=B.GEN_BATCH)
            STATE["neg"] = (neg, [tok.decode(y) for y in neg], {})
        neg, neg_t, mneg = STATE["neg"]
        if layer not in mneg:
            mneg.update(B.mean_response_acts(base, neutral, neg, [layer]))
        pos = C.sample(base, tok, [C.chat_ids(tok, x, spec["elicit"]) for x in prompts], B.CONTRAST_MAX_TOKENS,
                       seed=B.CONTRAST_SEED, batch_size=B.GEN_BATCH)
        mpos = B.mean_response_acts(base, neutral, pos, [layer])
        calib = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM) for x in C.wildchat_prompts("train", B.N_CALIB, B.CALIB_OFFSET)]
        rnorm = B.residual_norm(base, calib, layer)
    pos_t = [tok.decode(y) for y in pos]
    raw = mpos[layer] - mneg[layer]
    torch.save(dict(name=beh, layer=layer, model="Qwen/Qwen2.5-3B-Instruct", raw=raw, unit=raw / raw.norm(),
                    raw_norm=raw.norm().item(), elicit_system=spec["elicit"], contrast_prompts=prompts,
                    pos_responses=pos_t, neg_responses=neg_t,
                    contrast_metric=dict(pos=st.mean(map(spec["metric"], pos_t)), neg=st.mean(map(spec["metric"], neg_t))),
                    config=dict(source="phase3_eval.build_holdout_vector"), alpha=None, residual_norm=rnorm), path)
    log(f"  built {path.name}: |raw| {raw.norm():.2f}, resid norm {rnorm:.2f}")
    return dict(beh=beh, layer=layer, path=str(path))


def calibrate_alpha(R, vec, name, layer):
    """
    Step 3 criterion on the 256-prompt POOL (Phase 2 lesson: a 64-prompt screen is too noisy). Automated stand-in
    for "behavior clearly present and text coherent": the Phase 2 screening criteria. Saves 3 samples per alpha for
    post-hoc eye review. Returns (alpha or None, sweep_rows).
    """
    import statistics as st
    import torch
    base, tok, pm, _ = _models(R)
    C, B = R["common"], R["build"]
    beh = vec["beh"]
    spec = R["behaviors"].ALL_BEHAVIORS[beh]
    vd = torch.load(vec["path"])
    xs = _xs(R, "train", POOL_SLICE)
    rows, chosen = [], None
    with pm.disable_adapter():
        ctx0 = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM) for x in xs]
        ctx1 = [C.chat_ids(tok, x, R["behaviors"].RUNG1) for x in xs]
        if "pool_base" not in STATE:
            yb = C.sample(base, tok, ctx0, MAX_NEW, seed=101, batch_size=100)
            STATE["pool_base"] = [tok.decode(y) for y in yb]
        bm = st.mean(map(spec["metric"], STATE["pool_base"]))
        cm = vd["contrast_metric"]
        need = B.MIN_SHIFT * (cm["pos"] - cm["neg"])
        fcache = STATE.setdefault("pool_floor", {})
        for a in SWEEP_GRID[layer]:
            norm = a * vd["residual_norm"]
            w = (vd["unit"] * norm).cuda()
            y = C.sample(base, tok, ctx0, MAX_NEW, vec=w, layer=layer, seed=101, batch_size=100)
            k0 = st.mean(C.kl_on_rollouts(base, ctx0, ctx0, y, vec_p=w, layer=layer, batch_size=16)["rb"])
            k1 = st.mean(C.kl_on_rollouts(base, ctx0, ctx1, y, vec_p=w, layer=layer, batch_size=16)["rb"])
            key = (layer, round(norm, 4))
            if key not in fcache:
                vals = []
                for k in range(SWEEP_RAND_DIRS):
                    g = torch.Generator().manual_seed(SWEEP_RAND_SEED + k)
                    v = torch.randn(w.numel(), generator=g)
                    rv = (v / v.norm() * norm).cuda()
                    ry = C.sample(base, tok, ctx0, MAX_NEW, vec=rv, layer=layer, seed=101, batch_size=100)
                    vals.append(st.mean(C.kl_on_rollouts(base, ctx0, ctx0, ry, vec_p=rv, layer=layer, batch_size=16)["rb"]))
                fcache[key] = st.mean(vals)
            F = fcache[key]
            txt = [tok.decode(t) for t in y]
            dist = st.mean(len(set(t)) / max(1, len(t)) for t in y)
            mt = st.mean(map(spec["metric"], txt))
            c, gap = k1 - k0, k0 - F
            ok = dict(gap=gap >= B.PRESENCE_MULT * max(c, 0) and gap >= B.MIN_GAP_NATS,
                      f_frac=F / k0 <= B.MAX_F_FRAC, coherent=dist >= B.MIN_DISTINCT, shift=(mt - bm) >= need)
            rows.append(dict(alpha=a, kl0=k0, kl1=k1, presence=c, F=F, gap=gap, F_over_KL0=F / k0, distinct=dist,
                             metric=mt, baseline_metric=bm, need_shift=need, ok=ok, samples=txt[:3]))
            log(f"    alpha {a}: KL0 {k0:.1f} KL1 {k1:.1f} F {F:.1f} gap {gap:.1f} F/KL0 {F / k0:.0%} "
                f"distinct {dist:.3f} {spec['metric_name']} {mt:.2f} pass={all(ok.values())}")
            if chosen is None and all(ok.values()):
                chosen = a
                break                       # smallest passing alpha (Step 3 criterion)
    if chosen is not None:
        vd["alpha"] = chosen
        vd["alpha_source"] = f"phase3_eval automated pool sweep {OUTDIR.name}; samples saved for eye review"
        torch.save(vd, vec["path"])
    return chosen, rows


def _target(R, vec, alpha):
    import torch
    E = R["evalT"]
    if vec["beh"] == "random":
        return E.target("random", 0)
    vd = torch.load(vec["path"])
    if vd["alpha"] != alpha:
        vd["alpha"] = alpha
        torch.save(vd, vec["path"])
    return E.target(vec["beh"], vec["layer"])


def compute_floor(R, vec, name, layer, alpha):
    """
    Matched-norm random floor F on the HELD-OUT set (N_FLOOR_DIRS directions, Step 5 seeds; cached in
    runs/phase2_heldout_floors.json), plus KL0 on the same set so F/KL0 is defined.
    """
    import statistics as st
    base, tok, pm, _ = _models(R)
    C, E = R["common"], R["evalT"]
    assert E.N_FLOOR_DIRS == N_FLOOR_DIRS
    t = _target(R, vec, alpha)
    xs = _xs(R, "heldout", HELDOUT_SLICE)
    with pm.disable_adapter():
        ctx0 = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM) for x in xs]
        fl = E.heldout_floor(base, tok, t, ctx0)
        y = C.sample(base, tok, ctx0, MAX_NEW, vec=t["w"], layer=t["layer"], seed=101, batch_size=100)
        kl0 = st.mean(C.kl_on_rollouts(base, ctx0, ctx0, y, vec_p=t["w"], layer=t["layer"], batch_size=16)["rb"])
    return dict(F=fl["F"], dirs=fl["dirs"], KL0=kl0, F_over_KL0=round(fl["F"] / kl0, 3))


class _V2Read:
    """rung4.py's readability interface, backed by gate v2."""
    def __init__(self, gate):
        self.g = gate

    def score(self, s):
        d = self.g.score([s])[0]
        return dict(subj_ppl=float("nan"), judge=d["judge"], base_ppl=d["base_ppl"], gate=not d["hard_fail"])


def search_baseline(R, name, layer, alpha, mode):
    """
    Baseline to beat. "rung4" runs the 14B proposer (6 rounds x 8, blind to the behavior); "ladder" scores the
    hand-written rungs only. Either way the baseline S is the best STRUCTURALLY VALID prompt on TRAIN_SELECT
    (eval_T.rung4_best), never judge-filtered, never held-out-selected; its held-out numbers are filled in by
    eval_T_on_vector on the same rollouts as T. Negatives reuse the Phase 1b rung-4 records where they exist.
    """
    import statistics as st
    base, tok, pm, gate = _models(R)
    C, E, R4 = R["common"], R["evalT"], R["rung4"]
    beh = _beh(name)
    existing = E.rung4_best(beh, gate)
    if existing is not None and name in NEG_NAMES:
        return dict(mode="rung4 (Phase 1b record)", S=existing)
    if mode == "rung4":
        if "prop" not in STATE:
            STATE["prop"] = R4.Proposer()
        R4.VEC_LAYER[beh] = layer
        R4.TOP_HELDOUT = 1
        with pm.disable_adapter():
            res = R4.run_target(base, tok, STATE["prop"], _V2Read(gate), beh, 7000 + 100 * len(E.RUNG4_RUNS))
        path = OUTDIR / f"rung4_{name}.json"
        path.write_text(json.dumps({"results": [res]}, default=str, ensure_ascii=False))
    else:
        from oracle_search import ladder
        rungs = [("0 neutral", ""), ("1 generic", R["behaviors"].RUNG1)] if beh == "random" else ladder(beh)
        t = _target(R, build_holdout_vector(R, name, layer), alpha)
        xs = _xs(R, "train", TRAIN_SELECT)
        hist = []
        with pm.disable_adapter():
            ctx0 = [C.chat_ids(tok, x, C.NEUTRAL_SYSTEM) for x in xs]
            y = C.sample(base, tok, ctx0, MAX_NEW, vec=t["w"], layer=t["layer"], seed=101, batch_size=100)
            for lab, S in rungs:
                kl = st.mean(C.kl_on_rollouts(base, ctx0, [C.chat_ids(tok, x, S) for x in xs], y, vec_p=t["w"],
                                              layer=t["layer"], batch_size=16)["rb"])
                hist.append(dict(S=S, train_kl=kl, source=f"ladder {lab}"))
        path = OUTDIR / f"ladder_{name}.json"
        path.write_text(json.dumps({"results": [dict(name=beh, history=hist)]}, ensure_ascii=False))
    E.RUNG4_RUNS.insert(0, str(path))            # this vector's record takes precedence
    return dict(mode=mode, S=E.rung4_best(beh, gate), record=str(path))


def eval_T_on_vector(R, name, layer, alpha, floor):
    """
    T best-of-N plus greedy on held-out prompts, via eval_T.eval_vector: closure against the better of rung 0/1 (R1)
    and against the floor (R2, R3 wording), distinct-token fraction (R5), gate v2 status, behavior metric, and the
    emitted S. Percentages are returned in percent units.
    """
    base, tok, pm, gate = _models(R)
    E = R["evalT"]
    assert E.K == N_CANDIDATES and E.MAX_NEW_TOKENS == MAX_NEW
    t = _target(R, build_holdout_vector(R, name, layer), alpha)
    with pm.disable_adapter():
        res = E.eval_vector(base, tok, gate, t, adapter_ctx=lambda: E._enable(str(LORA_CKPT), pm))
    E.report(res)
    rows = {r["label"]: r for r in res["rows"]}

    def pct(r, k):
        v = r.get(k) if r else None
        return None if v is None or v != v else round(100 * v, 1)

    tb = rows.get("T best-of-K") or rows["T greedy"]
    sb = next((r for lab, r in rows.items() if lab.startswith("search best")), None)
    return dict(closed_vs_better=pct(tb, "vs_better"), closed_vs_floor=pct(tb, "vs_floor"),
                greedy_closed_vs_floor=pct(rows["T greedy"], "vs_floor"),
                greedy_closed_vs_better=pct(rows["T greedy"], "vs_better"),
                best_S=tb["S"], greedy_S=rows["T greedy"]["S"],
                inside_floor_spread=tb["inside_floor_spread"],
                baseline_heldout=dict(S=sb["S"] if sb else "", closed_vs_floor=pct(sb, "vs_floor"),
                                      closed_vs_better=pct(sb, "vs_better")) if sb else
                dict(S="", closed_vs_floor=0.0, closed_vs_better=0.0, note="baseline is the empty prompt (rung 0)"),
                full=res)

# ── END ADAPTER ───────────────────────────────────────────────────────────────


def run_one(R, outdir, name, layer, alpha, mode):
    log(f"=== {name} (L{layer}, search={mode}) ===")
    neg = name in NEG_NAMES
    rec = {"vector": name, "layer": layer, "search_mode": mode, "negative_control": neg,
           "started": datetime.now(timezone.utc).isoformat()}

    vec = build_holdout_vector(R, name, layer)
    rec.update(cosine_to_trained(R, vec))
    log(f"  max cos to trained: {rec['max_cos']} -> {rec['nearest_trained']}")

    if alpha is None:
        log("  sweeping alpha")
        alpha, sweep = calibrate_alpha(R, vec, name, layer)
        rec["alpha_sweep"] = sweep
        log(f"  alpha = {alpha}")
        if alpha is None:
            rec["verdict"] = "UNUSABLE: no alpha on the grid meets the Step 3 / Phase 2 criteria on the pool"
            log(f"  {rec['verdict']}")
            return rec
    rec["alpha"] = alpha

    log(f"  floor over {N_FLOOR_DIRS} directions")
    floor = compute_floor(R, vec, name, layer, alpha)
    rec["floor"] = floor
    f_over_kl0 = floor.get("F_over_KL0")
    log(f"  F = {floor.get('F'):.2f}   F/KL0 = {f_over_kl0}")

    # R4: a vector whose floor swamps its signal cannot yield a determinate
    # closure. Phase 2's criterion was F/KL0 <= 0.6. Negative controls are exempt:
    # a random direction has F/KL0 ~ 1 by construction and must still be evaluated.
    if not neg and f_over_kl0 is not None and f_over_kl0 > 0.6:
        rec["verdict"] = "UNUSABLE: F/KL0 > 0.6, closure would be indeterminate (R4)"
        log(f"  {rec['verdict']}")
        return rec

    log(f"  baseline ({mode})")
    rec["baseline"] = search_baseline(R, name, layer, alpha, mode)

    log("  evaluating T")
    rec["T"] = eval_T_on_vector(R, name, layer, alpha, floor)
    rec["baseline"].update(rec["T"].pop("baseline_heldout"))
    if neg:
        rec["verdict"] = ("NEGATIVE CONTROL PASSES (T does not beat the better baseline)"
                          if (rec["T"]["closed_vs_better"] or 0) <= 0 else
                          "NEGATIVE CONTROL FAILS: T beats the empty/generic prompt; suspect a bug before believing it")
        log(f"  {rec['verdict']}")

    rec["finished"] = datetime.now(timezone.utc).isoformat()
    return rec


def main():
    global LOGFILE, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve imports and print the plan, load nothing")
    ap.add_argument("--no-autostop", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of vector names (smoke tests)")
    args = ap.parse_args()

    stamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = RUNS / f"phase3_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    OUTDIR = outdir
    LOGFILE = outdir / "run.log"

    # Preflight the auto-stop BEFORE any long work.
    autostop = AUTOSTOP and not args.no_autostop and not args.dry_run
    pod_id = None
    if autostop:
        ok, info = preflight_autostop()
        if not ok:
            log(f"AUTOSTOP PREFLIGHT FAILED: {info}")
            log("Fix with: runpodctl config --apiKey YOUR_KEY")
            log("Refusing to start a long run that cannot stop itself.")
            log("Override with --no-autostop if you will stop it manually.")
            sys.exit(1)
        pod_id = info
        log(f"autostop armed for pod {pod_id}")

    if not LORA_CKPT.exists():
        log(f"LoRA checkpoint not found: {LORA_CKPT}")
        sys.exit(1)

    log(f"loading repo modules from {WORKSPACE}")
    R = load_repo()

    plan = HOLDOUTS + [(n, l, a, "ladder") for (n, l, a) in NEGATIVES]
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[0] in keep]
    log(f"plan: {len(plan)} vectors -> {outdir}")
    for n, l, a, m in plan:
        beh = _beh(n)
        vf = WORKSPACE / "vectors" / f"{beh}_L{l}.pt"
        status = "random dir" if beh == "random" else ("vector exists" if vf.exists() else "will BUILD vector")
        log(f"  {n:16s} L{l:<3} alpha={a if a is not None else 'sweep'}  {m:7s} behavior={beh}  [{status}]")

    if args.dry_run:
        log("dry run, exiting without evaluating")
        return

    results, failures = [], []
    try:
        for name, layer, alpha, mode in plan:
            try:
                rec = run_one(R, outdir, name, layer, alpha, mode)
            except Exception as e:
                log(f"  FAILED on {name}: {e}")
                log(traceback.format_exc())
                rec = {"vector": name, "error": str(e),
                       "traceback": traceback.format_exc()}
                failures.append(name)
            results.append(rec)
            # Write after every vector so a later crash loses nothing.
            (outdir / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False))
            (outdir / "all.json").write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))

        summarize(results, outdir, failures)

    finally:
        # Fires on success, on exception, and on sys.exit.
        if autostop and pod_id:
            stop_pod(pod_id)


def summarize(results, outdir, failures):
    lines = ["# Phase 3: holdout results", ""]
    if failures:
        lines += [f"**Failed vectors:** {', '.join(failures)}", ""]
    lines += ["| Vector | alpha | max cos (to) | F/KL0 | T vs better (%) | T vs floor (%) | T greedy vs floor (%) | "
              "Baseline (type) vs floor (%) | T - baseline (pts) | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        cosc = f"{r.get('max_cos')} ({r.get('nearest_trained', '').split('_L')[0]})" if r.get('max_cos') is not None else ""
        if "T" not in r:
            lines.append(f"| {r['vector']} | {r.get('alpha', '')} | {cosc} | {r.get('floor', {}).get('F_over_KL0', '')} | "
                         f"{r.get('verdict', r.get('error', '?'))[:60]} | | | | | |")
            continue
        T, B = r["T"], r.get("baseline", {})
        tf, bf = T.get("closed_vs_floor"), B.get("closed_vs_floor")
        delta = f"{tf - bf:+.1f}" if (tf is not None and bf is not None) else ""
        lines.append(
            f"| {r['vector']} | {r.get('alpha')} | {cosc} | {r['floor'].get('F_over_KL0')} | "
            f"{T.get('closed_vs_better')} | {tf} | {T.get('greedy_closed_vs_floor')} | "
            f"{B.get('mode', '')} {bf} | {delta} | {r.get('verdict', '')} |"
        )
    lines += ["", "## Emitted prompts (greedy)", ""]
    for r in results:
        if "T" in r:
            lines.append(f"- **{r['vector']}**: {r['T'].get('greedy_S', '')}")
    lines += [
        "", "## Reading this",
        "",
        "The question is whether T conditions on w or memorized six prompts.",
        "T succeeding on vectors it never saw is the only evidence that",
        "separates these. Note in the writeup which baselines were full rung-4",
        "searches and which were ladder-only: ladder-only is conservative",
        "against T, since Phase 2 measured rung 4 beating the ladder by 16-17",
        "points on register vectors.",
        "",
        "Both negative controls must still fail to beat the empty prompt. If",
        "either now 'succeeds', suspect a reward or evaluation bug before",
        "believing it.",
        "",
        "alpha for swept vectors was chosen by automated criteria; review the saved",
        "samples (alpha_sweep[*].samples in <vector>.json) by eye before trusting it.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines))
    log(f"summary written to {outdir / 'SUMMARY.md'}")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
