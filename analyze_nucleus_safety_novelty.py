"""Analytical nucleus-safety pass over the incumbent lossy-SD rules (Yash round-6, item 4).

The FSD claim already broke under checking, so the rest were assumed suspect too. For each
incumbent we ask one question analytically, rather than asserting a novelty claim:

    Can this method's accept rule ever force a token with p_target(x) = 0 EXACTLY?

Setting: top-p/top-k sampling masks out-of-nucleus logits to -inf, so p(x)=0 is attainable
and common in practice -- 56.9% of rejections on Llama-3.1-8B, 90.9% on Qwen3-14B.

Out: results/nucleus_safety_novelty_pass.md
"""
import numpy as np


def js_pointmass(px):
    """JS(p, delta_x) contribution at the mass point x, in nats: p(x)=px, q(x)=1."""
    m = (px + 1.0) / 2.0
    t = 0.5 * np.log(1.0 / m)
    if px > 0:
        t += 0.5 * px * np.log(px / m)
    return float(t)


JS0 = js_pointmass(0.0)

ROWS = [
    ("FSD (2502.20704)", "divergence threshold c",
     "d_TV = 1 - p(x) so admitting p=0 needs c > 1; KL(q||p) = -log p(x) = +inf",
     "SAFE by construction", "no knob value in [0,1] can admit a p=0 token"),
    ("SPRINTER (2502.04557)", "fixed FP acceptance rate",
     "accept probability is eta, independent of p(x)",
     "UNSAFE", "forces p=0 tokens at rate ~ eta x masked-share"),
    ("Cool-SD (2601.09212)", "annealed rate schedule eta_j",
     "eta_j = f(position) only; still independent of p(x)",
     "UNSAFE", "same rate-based hazard, front-loaded by the schedule"),
    ("AdaSD (2512.11280)", "JS-distance threshold",
     f"JS(p, delta_x) at p(x)=0 = {JS0:.4f} nats -- FINITE, below ln2 = {np.log(2):.4f}",
     "CONDITIONALLY UNSAFE", f"unsafe iff the threshold exceeds {JS0:.4f} nats"),
    ("DIVERSED (2604.07622)", "per-step probability bound",
     "accept test is divergence-based, which is unbounded at p(x)=0",
     "LIKELY SAFE (verify)", "confirm the knob is p(x)-dependent and not a rate"),
]

CONCLUSION = f"""
## Conclusion for Sec 2.2 — the split is by PRIMITIVE, not by paper

**Rate-based rules** (accept with probability eta, independent of p(x)) -- SPRINTER,
Cool-SD's schedules, **and our own eta-rule** -- are UNSAFE. They can and do force
zero-probability tokens, and the hazard scales with the masked share (56.9% Llama-8B,
90.9% Qwen3-14B).

**Divergence-based rules** (accept iff D(p,q) < c) -- FSD, DIVERSED -- are safe *when D
diverges at p(x)=0*: total variation requires c > 1, KL is outright infinite.

**AdaSD is the exception that sharpens the rule.** Jensen-Shannon stays FINITE at p(x)=0
({JS0:.4f} nats, against a maximum of {np.log(2):.4f}), so a JS threshold above {JS0:.4f}
admits zero-probability tokens. Safety therefore depends on **the divergence's behaviour at
zero**, not on being "a threshold method" -- bounded divergences do not inherit the
guarantee.

### What we may claim

NOT "no prior method enforces nucleus-safety" (false for FSD, and for DIVERSED pending a
source check). The supported claim is:

1. the hazard is **intrinsic to rate-based relaxation**, which is precisely the primitive
   required for an exact per-token price and hence for the ledger;
2. divergence-based rules avoid it for free, but cannot be FP-certified (Sec 3.4);
3. safety among divergence-based rules is **not automatic** -- it holds for unbounded
   divergences (TV, KL) and fails for bounded ones (JS) at a loose threshold.

Point 3 is new and is ours: it means "use a threshold" is not itself a safety argument.
"""

if __name__ == "__main__":
    print("=" * 78)
    print("NUCLEUS-SAFETY OF INCUMBENT LOSSY-SD RULES (analytical)")
    print("=" * 78)
    for m, knob, why, verdict, note in ROWS:
        print(f"\n{m}\n  primitive : {knob}\n  analysis  : {why}\n"
              f"  VERDICT   : {verdict}\n  note      : {note}")
    print(CONCLUSION)
    md = ["# Nucleus-safety of incumbent lossy-SD rules (analytical pass)", "",
          "| method | primitive | analysis | verdict |", "|---|---|---|---|"]
    for m, knob, why, verdict, note in ROWS:
        md.append(f"| {m} | {knob} | {why} | **{verdict}** ({note}) |")
    md.append(CONCLUSION)
    open("results/nucleus_safety_novelty_pass.md", "w", encoding="utf-8").write(
        "\n".join(md) + "\n")
    print("wrote results/nucleus_safety_novelty_pass.md")
