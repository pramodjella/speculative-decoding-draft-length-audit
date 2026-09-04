# Paper framing — Adaptive Draft-Length on EAGLE-3: a signal-limited regime

## Working title
**"When Does Adaptive Draft Length Help? A Per-Step Headroom Analysis of Speculative
Decoding on EAGLE-3"**

## Abstract (draft)
Adaptive draft-length controllers promise to spend speculation where it pays, and prior
work (SVIP, BanditSpec, AdaEAGLE) reports gains by adjusting how many tokens a drafter
proposes per step. We ask a sharper question: *on a strong modern self-speculation system
(EAGLE-3), how much speedup is actually available to a draft-length controller, and what
limits it?* Using a read-only instrumentation of vLLM that recovers per-step draft entropy,
top-1/top-2 margin, target-model verification entropy, and exact accept-run lengths, we
evaluate every standard controller family — fixed-K, entropy-threshold (SVIP), bandits
(UCB/ε-greedy), acceptance-history, acceptance-persistence, and a learned per-step stopper —
on Llama-3.1-8B + EAGLE-3 across 1,295 decode steps. We find a **+21.5% per-step oracle
ceiling but no cheap controller exceeds ~+3%**, and per-*request* adaptation is worthless
(+2% ceiling; real bandits *lose*). Decomposing the gap, we show the bottleneck is the
**signal, not the policy**: draft-side confidence weakly predicts acceptance (r≈−0.23), and
while the *target's* causal bonus-token entropy is the strongest single predictor we find
(r≈−0.32, beating the draft-side signals all prior controllers use), it still explains only
~10% of acceptance variance. We conclude that adaptive draft length is **signal-limited** on
EAGLE-3, give a measurable diagnostic for when a controller is worth deploying, and identify
target-side gating as the most promising — though still insufficient — direction.

## Contributions (the novelty, stated honestly)
1. **A per-step headroom methodology.** A read-only vLLM instrumentation
   (`compute_logits` + `observe_draft` hooks) that exposes signals vLLM hides, plus an
   **oracle-ceiling decomposition** separating per-request from per-step headroom. This turns
   "does my controller help?" into a measurable quantity (`oracle − best-fixed`) computable
   before building any controller.
2. **A negative result with a mechanism.** On EAGLE-3/Llama-8B, *no* cheap controller —
   threshold, bandit, persistence, or learned predictor — beats tuned fixed-K by >3%, against
   a +21.5% per-step ceiling. The cause is signal informativeness, not policy design; we show
   this by exhausting the policy space and measuring signal–acceptance correlation directly.
3. **Draft-side vs target-side signal.** The verifier's causal bonus-token entropy
   (r≈−0.32) predicts acceptance better than the draft-side entropy/margin (r≈−0.23) that
   all prior controllers (SVIP/BanditSpec/AdaEAGLE) rely on — a redirection of where the
   signal should come from — yet even it is too weak to close the gap.
4. **A deployment rule.** For a given target/draft pair, measure the acceptance
   distribution, its lag-1 autocorrelation, and the per-step oracle ceiling cheaply; predict
   whether adaptive draft length is worth its complexity. On EAGLE-3/Llama-8B the answer is
   "no — ship fixed K=2."

## Positioning vs prior art
- **SVIP / BanditSpec / AdaEAGLE** report controller gains on EAGLE-1/2 and separate-draft
  pairs. We do not contradict them; we show their **draft-confidence lever weakens on
  EAGLE-3** (SVIP-style entropy: +13% on EAGLE-2 → +2.4% here) and explain why (signal
  informativeness), which none of them measure.
- **TapOut / Nightjar / SpecKV** (2025–26) saturate the *policy* space (bandits, learned
  predictors) and all optimize tokens/step. Our learned-stopper reproduces their recipe and
  shows it caps at +1% here — evidence the policy axis is closed on this regime.
- **Novelty is on the diagnosis + signal axis, not a new controller** — which is the honest,
  defensible claim given the saturation.

## What this is NOT (scope honesty)
- Not a new state-of-the-art controller; the data shows none exists for this regime.
- Speed-axis only. The orthogonal open direction (certified-quality / budget-constrained
  lossy SD) is future work, not claimed here.

## Reproduce
Capture: `modal_eagle3_perstep_capture.py` (draft + target hooks).
Analysis: `src/simulate_eagle_perstep.py`, `src/simulate_eagle_controllers.py`,
`src/verify_explore_perstep.py`, `src/explore_target_signal.py`.
Traces: `results/eagle3_perstep_target_llama8b.json`, `results/eagle3_multik_llama8b.json`.
Full results section: `results/eagle3_controller_findings.md`.
