# Nucleus-safety of incumbent lossy-SD rules (analytical pass)

| method | primitive | analysis | verdict |
|---|---|---|---|
| FSD (2502.20704) | divergence threshold c | d_TV = 1 - p(x) so admitting p=0 needs c > 1; KL(q||p) = -log p(x) = +inf | **SAFE by construction** (no knob value in [0,1] can admit a p=0 token) |
| SPRINTER (2502.04557) | fixed FP acceptance rate | accept probability is eta, independent of p(x) | **UNSAFE** (forces p=0 tokens at rate ~ eta x masked-share) |
| Cool-SD (2601.09212) | annealed rate schedule eta_j | eta_j = f(position) only; still independent of p(x) | **UNSAFE** (same rate-based hazard, front-loaded by the schedule) |
| AdaSD (2512.11280) | JS-distance threshold | JS(p, delta_x) at p(x)=0 = 0.3466 nats -- FINITE, below ln2 = 0.6931 | **CONDITIONALLY UNSAFE** (unsafe iff the threshold exceeds 0.3466 nats) |
| DIVERSED (2604.07622) | per-step probability bound | accept test is divergence-based, which is unbounded at p(x)=0 | **LIKELY SAFE (verify)** (confirm the knob is p(x)-dependent and not a rate) |

## Conclusion for Sec 2.2 — the split is by PRIMITIVE, not by paper

**Rate-based rules** (accept with probability eta, independent of p(x)) -- SPRINTER,
Cool-SD's schedules, **and our own eta-rule** -- are UNSAFE. They can and do force
zero-probability tokens, and the hazard scales with the masked share (56.9% Llama-8B,
90.9% Qwen3-14B).

**Divergence-based rules** (accept iff D(p,q) < c) -- FSD, DIVERSED -- are safe *when D
diverges at p(x)=0*: total variation requires c > 1, KL is outright infinite.

**AdaSD is the exception that sharpens the rule.** Jensen-Shannon stays FINITE at p(x)=0
(0.3466 nats, against a maximum of 0.6931), so a JS threshold above 0.3466
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

