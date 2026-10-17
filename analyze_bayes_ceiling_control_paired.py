"""Permutation control, re-run under the PAIRED-WITHIN-FOLD ladder.

The original control (`analyze_bayes_ceiling_control.py`) reported "Llama recovery collapses
from +17.8% to -4.9% under shuffled labels". Its real arm quoted the OLD un-paired ceiling
(17.8%), which we retired because the two hidden-probe rungs were computed on different data.
The control's *logic* is unaffected by that fix — it tests whether the probe's signal survives
label shuffling, not whether two rungs are nested — but quoting a retired number alongside a
corrected one is exactly the drift `check_stale.sh` exists to prevent. So we re-run it through
the corrected estimator and quote a matched pair.

Design: identical block-level label permutation (each block's (accept-pattern, acc) vector is
reassigned to another block's hidden rows, preserving per-block structure and marginals), then
the SAME paired-within-fold ladder used for the real arm. Real vs shuffled therefore differ in
exactly one thing: whether the labels belong to the hidden states.

Out: results/perstep_signal/bayes_ceiling_control_paired.json
Usage: python analyze_bayes_ceiling_control_paired.py [--model llama8b]
"""
import os, json, argparse
import numpy as np
import pandas as pd

import analyze_bayes_ceiling as B
import analyze_bayes_ceiling_paired as P


def shuffled_frame(model_key, seed=123):
    pq = os.path.join(B.IN_DIR, f"hidden_full_{model_key}.parquet")
    df = pd.read_parquet(pq, engine="pyarrow")
    df["uid"] = df["workload"].astype(str) + ":" + df["gen_i"].astype(str)
    df["block"] = df.groupby(B.STEP_KEYS).ngroup()

    rng = np.random.RandomState(seed)
    blocks = df["block"].unique()
    perm_map = dict(zip(blocks, rng.permutation(blocks)))
    g = df.sort_values("position").groupby("block")
    acc_by_block = g["acc"].first().to_dict()
    accept_by_block = {b: grp["accept"].values for b, grp in g}

    df = df.sort_values(["block", "position"]).reset_index(drop=True)
    new_accept = np.empty(len(df), dtype=int)
    new_acc = np.empty(len(df), dtype=int)
    i = 0
    for b, grp in df.groupby("block", sort=False):
        src = perm_map[b]
        sa = accept_by_block[src]
        n = len(grp)
        new_accept[i:i + n] = sa[:n] if len(sa) >= n else np.resize(sa, n)
        new_acc[i:i + n] = acc_by_block[src]
        i += n
    df["accept"] = new_accept
    df["acc"] = new_acc
    return df


def paired_ladder_from_frame(df, n_splits=8):
    """The Sec-4 paired ladder, run on a pre-built frame (real or shuffled)."""
    hdim = [c for c in df.columns if c.startswith("h")]
    maxk = int(df["ndraft"].iloc[0])
    surv = df.groupby("position")["accept"].mean().to_dict()
    df["p_position"] = df["position"].map(surv)
    df["p_hidden"] = B.oof_probe_predictions(df, hdim, n_splits=n_splits)
    df = df[~df["p_hidden"].isna()].copy()

    grid = np.linspace(0.0, 1.0, 101)
    p_hid, acc_arr, step_uid = B.build_step_arrays(df, "p_hidden")
    n_steps = len(acc_arr)
    uids = sorted(set(step_uid.tolist()))
    rng = np.random.RandomState(7); rng.shuffle(uids)
    gsplits = np.array_split(uids, n_splits)

    dep, cei = [], []
    for i in range(n_splits):
        tg = set(gsplits[i].tolist())
        te = [k for k in range(n_steps) if step_uid[k] in tg]
        tr = [k for k in range(n_steps) if step_uid[k] not in tg]
        if not te or not tr:
            continue
        acc_te = acc_arr[te]
        fixed_i, _ = B.best_fixed_speedup(acc_te, maxk)
        oracle_i = B.oracle_speedup(acc_te, maxk)
        span = oracle_i - fixed_i
        if span <= 0:
            continue
        _, lam_tr = B.best_threshold_speedup([p_hid[k] for k in tr], acc_arr[tr], maxk, grid)
        d = B.speedup(B.threshold_K([p_hid[k] for k in te], lam_tr, maxk), acc_te)
        c, _ = B.best_threshold_speedup([p_hid[k] for k in te], acc_te, maxk, grid)
        dep.append(100 * (d - fixed_i) / span)
        cei.append(100 * (c - fixed_i) / span)

    def ms(v):
        v = np.array(v, dtype=float)
        return float(v.mean()), float(v.std() / max(len(v) - 1, 1) ** 0.5)
    return ms(dep), ms(cei), len(dep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    a = ap.parse_args()

    print(f"=== {a.model}: SHUFFLED arm (block-level label permutation) ===")
    sdf = shuffled_frame(a.model)
    print(f"  shuffled label mean {sdf['accept'].mean():.3f}")
    (sd, sds), (sc, scs), nf = paired_ladder_from_frame(sdf)
    print(f"  folds={nf}  deployable {sd:+.1f} +-{sds:.1f}%   oracle-threshold {sc:+.1f} +-{scs:.1f}%")

    real = json.load(open(os.path.join(B.OUT_DIR, "bayes_ceiling_paired.json")))[a.model]
    rd, rds = real["deployable_recovery_pct"]
    rc, rcs = real["ceiling_recovery_pct"]
    print(f"\n=== REAL arm (from the corrected ladder) ===")
    print(f"  deployable {rd:+.1f} +-{rds:.1f}%   oracle-threshold {rc:+.1f} +-{rcs:.1f}%")
    print(f"\nCONTROL VERDICT: deployable recovery {rd:+.1f}% (real) -> {sd:+.1f}% (shuffled); "
          f"collapse of {rd - sd:.1f}pp")

    out = {"model": a.model,
           "real": {"deployable_pct": [rd, rds], "ceiling_pct": [rc, rcs]},
           "shuffled": {"deployable_pct": [round(sd, 2), round(sds, 2)],
                        "ceiling_pct": [round(sc, 2), round(scs, 2)], "n_folds": nf},
           "collapse_pp": round(rd - sd, 2),
           "note": ("both arms scored with the paired-within-fold estimator; they differ only "
                    "in whether labels belong to their hidden states")}
    dst = os.path.join(B.OUT_DIR, "bayes_ceiling_control_paired.json")
    merged = json.load(open(dst)) if os.path.exists(dst) else {}
    merged[a.model] = out
    json.dump(merged, open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")
