"""Permutation control: shuffle accept labels ACROSS blocks (keeping each block's acc
intact but breaking the hidden->accept relationship), retrain the OOF probe, and measure
recovery. If the +18.9% real-signal result is genuine, shuffled recovery must collapse to ~0.

We shuffle the per-block label *vector assignment*: reassign each block's (accept-pattern, acc)
to a RANDOM block's hidden states. That destroys h->accept while preserving the marginal
label distribution and the prefix structure exactly.
"""
import sys, os
sys.path.insert(0, r'F:\Inference Engineering\09_Pramod_Jella_Adaptive_Draft_Length\09_Pramod_Jella_Adaptive_Draft_Length')
import numpy as np, pandas as pd
import analyze_bayes_ceiling as A

IN = r'F:\Inference Engineering\09_Pramod_Jella_Adaptive_Draft_Length\09_Pramod_Jella_Adaptive_Draft_Length\results\eagle3_hidden_full\hidden_full_llama8b.parquet'

df = pd.read_parquet(IN, engine='pyarrow')
df["uid"] = df["workload"].astype(str) + ":" + df["gen_i"].astype(str)
hdim = [c for c in df.columns if c.startswith("h")]
maxk = int(df["ndraft"].iloc[0])

# Build block ids
block = df.groupby(A.STEP_KEYS).ngroup()
df["block"] = block

# For each block, capture its (acc, accept-by-position). Then permute which block's
# LABELS attach to which block's HIDDEN rows.
rng = np.random.RandomState(123)
blocks = df["block"].unique()
perm = rng.permutation(blocks)
perm_map = dict(zip(blocks, perm))

# Build a lookup: for source block -> sorted-by-position accept vector + acc
g = df.sort_values("position").groupby("block")
acc_by_block = g["acc"].first().to_dict()
accept_by_block = {b: grp["accept"].values for b, grp in g}

# Reassign labels: row in block b gets labels from block perm_map[b]
df = df.sort_values(["block", "position"]).reset_index(drop=True)
new_accept = np.empty(len(df), dtype=int)
new_acc = np.empty(len(df), dtype=int)
i = 0
for b, grp in df.groupby("block", sort=False):
    src = perm_map[b]
    src_accept = accept_by_block[src]
    n = len(grp)
    # align by position order (both length-7 sorted)
    new_accept[i:i+n] = src_accept[:n] if len(src_accept) >= n else np.resize(src_accept, n)
    new_acc[i:i+n] = acc_by_block[src]
    i += n
df["accept"] = new_accept
df["acc"] = new_acc

print("shuffled label mean:", round(df["accept"].mean(), 3), "(orig ~0.131)")

# position survival on shuffled labels (should be ~flat now, since labels scrambled across blocks)
df["p_position"] = df["position"].map(df.groupby("position")["accept"].mean().to_dict())

print("training OOF probe on SHUFFLED labels...")
df["p_hidden"] = A.oof_probe_predictions(df, hdim, n_splits=8)
df = df[~df["p_hidden"].isna()].copy()

grid = np.linspace(0.0, 1.0, 101)
p_hid_steps, acc_arr, step_uid = A.build_step_arrays(df, "p_hidden")
fixed_s, fixed_k = A.best_fixed_speedup(acc_arr, maxk)
oracle_s = A.oracle_speedup(acc_arr, maxk)
bayes_hid_s, lam = A.best_threshold_speedup(p_hid_steps, acc_arr, maxk, grid)
span = oracle_s - fixed_s
rec = 100.0 * (bayes_hid_s - fixed_s) / span if span > 1e-9 else 0.0
print(f"SHUFFLED: fixed K={fixed_k} {fixed_s:.4f} | bayes_hid {bayes_hid_s:.4f} | oracle {oracle_s:.4f}")
print(f"SHUFFLED Bayes(hidden) recovery = {rec:+.1f}%   (REAL was +17.8%)")
print(">>> If this is ~0%, the real result is genuine signal, not a pipeline leak.")
