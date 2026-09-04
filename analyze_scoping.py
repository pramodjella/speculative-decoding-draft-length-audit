"""Turn results/scoping_metrics.csv into the regime map + findings."""
import numpy as np
import pandas as pd

df = pd.read_csv("results/scoping_metrics.csv")
# Guard online_capture where there is essentially no headroom to capture.
df.loc[(df["oracle_joint"] - df["oracle_task"]) < 0.01, "online_capture_%"] = np.nan


def _md(d):
    try:
        return d.to_markdown()
    except Exception:
        return d.to_string()


lines = ["# Scoping Study — When Does Adaptive Draft-Length Control Pay?\n",
         "Simulator, 4 seeds. Metric = per-step speedup. Headroom = oracle gap over best "
         "fixed-K (Oracle-Task). r = draft/target cost ratio (0.1 = EAGLE/large-target, "
         "0.85 = measured small pair). All numbers are %.\n"]

# 1. Adaptive headroom (Oracle-Joint over Oracle-Task) by r x load regime, avg over workloads
lines.append("## Adaptive headroom (%) — avg over workloads, by draft cost r and load regime\n")
p1 = df.pivot_table(index="r", columns="load_regime", values="adaptive_headroom_%", aggfunc="mean")
p1 = p1[["static_low", "static_high", "dynamic"]]
lines.append(_md(p1.round(1)))

# 2. Where do load vs content each help (dynamic regime, by r)
lines.append("\n\n## In dynamic load: load vs content value, and their complementarity (%) by r\n")
dyn = df[df.load_regime == "dynamic"]
p2 = dyn.groupby("r")[["load_value_%", "content_value_%", "complementarity_%",
                       "adaptive_headroom_%", "online_capture_%"]].mean().round(1)
lines.append(_md(p2))

# 3. Per-workload at the favorable regime (r=0.2 dynamic)
lines.append("\n\n## Favorable regime (r=0.2, dynamic) — per workload (%)\n")
fav = df[(df.r == 0.2) & (df.load_regime == "dynamic")][
    ["workload", "load_value_%", "content_value_%", "complementarity_%",
     "adaptive_headroom_%", "online_capture_%"]].round(1)
lines.append(_md(fav.set_index("workload")))

lines.append("\n\n## Findings (the regime map)\n")
lines.append(
    "1. **Adaptive draft length only pays with a CHEAP drafter and headroom.** Headroom is "
    "8–13% at r≤0.2 under dynamic load, ~2–6% at batch 1, and **≈0% at r=0.85 (real small "
    "pairs) or under heavy static load**. This reconciles the field: 'it doesn't help' "
    "reports are simply the no-headroom regime.\n"
    "2. **At batch 1, only CONTENT matters** (load is constant): ~2–6% headroom — the "
    "SVIP/TapOut regime.\n"
    "3. **Load-awareness matters only under VARYING load**, where it is the dominant lever "
    "(load_value 4–11% vs content_value 1–6%) — the Nightjar regime.\n"
    "4. **Load+content are complementary only in dynamic load at moderate r**, peaking ~3–4% "
    "(r≈0.2). Real but modest and regime-specific — NOT a large universal win.\n"
    "5. **Online controllers capture only ~20–43%** of the achievable joint headroom in the "
    "favorable regime — the joint online-control problem is open (future work).\n")

with open("results/scoping_findings.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Heatmap figure: adaptive headroom over (r x load regime)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.makedirs("results/figures_load", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
H = p1.values
im = ax.imshow(H, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(p1.columns))); ax.set_xticklabels(p1.columns)
ax.set_yticks(range(len(p1.index))); ax.set_yticklabels([f"r={r}" for r in p1.index])
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i,j]:.1f}%", ha="center", va="center",
                color="white" if H[i, j] < H.max() * 0.6 else "black", fontsize=9)
ax.set_title("Adaptive draft-length headroom (%) over best fixed-K")
fig.colorbar(im, label="headroom %")
fig.tight_layout(); fig.savefig("results/figures_load/scoping_heatmap.png", dpi=200)
print("Wrote results/scoping_findings.md + results/figures_load/scoping_heatmap.png")
