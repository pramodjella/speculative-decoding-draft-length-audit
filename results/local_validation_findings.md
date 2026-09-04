# Step 5 — Local Real-Hardware Validation + Robustness

Real models: Qwen2.5-1.5B-Instruct (target) + Qwen2.5-0.5B-Instruct (draft), RTX 5070
8 GB, bf16, batch 1, greedy. Script: `validate_local.py`. Sensitivity: `scratch/robustness_probe.py`.

**Scope (honest):** the load/batch axis cannot be tested locally (batch-1 harness, no
Windows vLLM, 8 GB). Local hardware validates the *content* assumptions; the load axis
stays simulator-backed (cloud vLLM = future test).

## What real hardware confirmed / challenged
| Check | Result | Verdict |
|---|---|---|
| V1: draft entropy predicts acceptance | corr(entropy, accepted) = **−0.52** (153 steps) | **CONFIRMED** — content signal is real |
| V2: per-step difficulty autocorrelated | lag-1 autocorr of accepted ≈ **−0.05** | NOT confirmed (≈0) — but see robustness |
| Latency calibration | measured **r = T_draft/T_target = 0.85** (sim assumed 0.20) | mismatch — local pair is expensive-draft regime |
| V3: content controller vs best-fixed @ b=1 | best fixed = K=1 (1.015×); LinUCB 0.984× (**−3.0%**) | no win — but b=1, r=0.85 has ~no headroom |

At r=0.85 every fixed K>1 is **below 1.0×** (fixed_2=0.96, fixed_4=0.84, fixed_8=0.63):
the 0.5B draft is barely cheaper than the 1.5B target at batch 1, so there is no
headroom for any controller. This reproduces the project's long-standing small-pair
finding and Yash's "it's the regime" point.

## Robustness sweep over the two challenged assumptions (simulator, 4 seeds)
Combined vs baselines as draft-cost r and persistence rho vary:

| | vs Nightjar (load-only) | vs TapOut (content-only) | vs best-fixed |
|---|---|---|---|
| all r, all rho | **+8% to +16%** (always wins) | **+28% to +70%** (always wins) | +3–4% (low r) → −2 to −5% (high r) |
| effect of rho | negligible (0.92 vs 0.2 ~identical) | negligible | negligible |

**Two honest conclusions:**
1. **The strong claim is robust.** Combined beats the *published* single-sided methods
   (Nightjar, TapOut) at every r and rho — and the V2 autocorrelation result is moot
   because rho barely affects the gap (the edge comes from within-step entropy (V1) +
   load conditioning, not cross-step persistence).
2. **The thesis is scoped to the cheap-drafter regime (r ≲ 0.3).** Combined beats
   best-fixed only when real headroom exists (low r: EAGLE heads, large-target/small-draft).
   At high r (e.g. local 1.5B/0.5B, r=0.85) best-fixed K=1 wins and no adaptive method
   helps. This is the SAME regime the project already targeted; the local pair is simply
   the wrong testbed for the load axis.

## Net
Local hardware **confirms the content signal (V1)** and **calibrates the regime** (the
thesis needs a cheap drafter, r ≲ 0.3 — EAGLE/large-target). The combined controller's
advantage over published methods is robust; the open real-hardware test is a cloud vLLM
run on an EAGLE-3 / large-target setup (staged: `modal_vllm_eagle3.py`), where r is
genuinely small and the load axis is exercisable.
