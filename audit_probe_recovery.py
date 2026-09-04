"""INDEPENDENT re-derivation of the hidden-state probe RECOVERY (the +18.9% headline).

Deliberately different from analyze_bayes_ceiling.py: different fold count (5 not 8),
different seed, different PCA dim (30 not 50), and a simpler prefix-threshold policy chosen
on TRAIN. If Llama still recovers a clearly-positive fraction and DeepSeek ~0, the result is
robust to implementation choices (Yash: "have someone re-derive independently").
"""
import os, sys, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
IN   = os.path.join(BASE, "results", "eagle3_hidden_full")
C, MAXK = 0.15, 7


def load(key):
    return pd.read_parquet(os.path.join(IN, f"hidden_full_{key}.parquet"))


def speedup(K, acc):
    K = np.asarray(K, float); acc = np.asarray(acc, float)
    return (np.minimum(acc, K).mean() + 1.0) / (1.0 + C * K.mean())


def best_fixed(acc):
    return max(speedup(np.full(len(acc), k), acc) for k in range(1, MAXK + 1))


def oracle(acc):
    K = np.minimum(acc + 1, MAXK)
    return speedup(K, acc)


def blocks(df, pcol):
    """list of (acc, per-position prob array) per block, plus uid."""
    out = []
    for (w, g, s), grp in df.sort_values("position").groupby(["workload", "gen_i", "step_i"], sort=False):
        out.append((int(grp["acc"].iloc[0]), grp[pcol].values.astype(float), f"{w}:{g}"))
    return out


def policy_K(pvec, thr):
    k = 1
    for j in range(1, min(len(pvec), MAXK)):
        if pvec[j] >= thr:
            k = j + 1
        else:
            break
    return k


def recovery(key, n_splits=5, seed=7, pcadim=30):
    df = load(key)
    df["uid"] = df["workload"].astype(str) + ":" + df["gen_i"].astype(str)
    hcols = [c for c in df.columns if c.startswith("h")]
    # out-of-fold probe (fresh seed / dim / folds)
    uids = sorted(df["uid"].unique()); rng = np.random.RandomState(seed); rng.shuffle(uids)
    fold = {u: i % n_splits for i, u in enumerate(uids)}
    df["fold"] = df["uid"].map(fold)
    df["p"] = np.nan
    X = df[hcols].values.astype("float32"); y = df["accept"].values
    for k in range(n_splits):
        te = (df["fold"] == k).values; tr = ~te
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(pcadim, len(hcols)), random_state=seed).fit(sc.transform(X[tr]))
        clf = LogisticRegression(max_iter=300, C=0.1).fit(pca.transform(sc.transform(X[tr])), y[tr])
        df.loc[te, "p"] = clf.predict_proba(pca.transform(sc.transform(X[te])))[:, 1]
    df = df[~df["p"].isna()]

    blk = blocks(df, "p")
    acc_all = np.array([b[0] for b in blk], float)
    uid_all = np.array([b[2] for b in blk])
    bf = best_fixed(acc_all); orc = oracle(acc_all)

    # threshold chosen on TRAIN uids, evaluated on TEST uids (fresh 5-fold on uids)
    u2 = sorted(set(uid_all)); rng2 = np.random.RandomState(seed + 1); rng2.shuffle(u2)
    f2 = {u: i % n_splits for i, u in enumerate(u2)}
    grid = np.linspace(0.1, 0.9, 33)
    deploys = []
    for k in range(n_splits):
        test_u = {u for u in u2 if f2[u] == k}
        tr_idx = [i for i in range(len(blk)) if uid_all[i] not in test_u]
        te_idx = [i for i in range(len(blk)) if uid_all[i] in test_u]
        if not tr_idx or not te_idx:
            continue
        # best thr on train
        def sp(idx, thr):
            K = [policy_K(blk[i][1], thr) for i in idx]
            a = [blk[i][0] for i in idx]
            return speedup(K, a)
        best_thr = max(grid, key=lambda t: sp(tr_idx, t))
        deploys.append(sp(te_idx, best_thr))
    dep = float(np.mean(deploys))
    rec = 100.0 * (dep - bf) / (orc - bf) if orc > bf else 0.0
    return {"best_fixed": round(bf, 4), "oracle": round(orc, 4),
            "deploy": round(dep, 4), "recovery_pct": round(rec, 1)}


if __name__ == "__main__":
    keys = sys.argv[1:] or ["llama8b", "deepseek"]
    print(f"{'model':22s} {'best_fixed':>10s} {'oracle':>8s} {'deploy':>8s} {'recovery%':>10s}")
    for k in keys:
        r = recovery(k)
        print(f"{k:22s} {r['best_fixed']:10.4f} {r['oracle']:8.4f} {r['deploy']:8.4f} {r['recovery_pct']:10.1f}")
    print("\nReference (analyze_bayes_ceiling, 8-fold/PCA-50/seed42): llama8b +18.9%, deepseek +0.0%")
    print("If this independent 5-fold/PCA-30/seed7 run is close -> recovery is implementation-robust.")
