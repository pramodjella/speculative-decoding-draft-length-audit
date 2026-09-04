# Adaptive Draft-Length Controllers on EAGLE-3: A Per-Step Headroom Study

**Target/draft:** Llama-3.1-8B-Instruct + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`
**Engine:** vLLM 0.23 (chain drafting, `num_speculative_tokens`=7), H100, batch 1, greedy
**Workloads:** HumanEval, GSM8K, MT-Bench. **Traces:** 24 generations / **1,295 decode steps**.

## 1. Question

Does an adaptive draft-length controller beat a *tuned fixed* draft length on top of
EAGLE-3? We separate two regimes a controller can exploit: **per-request** (one K per
prompt) and **per-step** (one K per decode step, the within-stream regime SVIP targets).

## 2. Method: per-step signal capture from vLLM

vLLM exposes no per-step draft signal through its API; its EAGLE proposer runs in a
separate process and through compiled paths. We capture signals with two **read-only**
hooks under `VLLM_ENABLE_V1_MULTIPROCESSING=0` + `enforce_eager=True`:

- `Eagle3LlamaForCausalLM.compute_logits` → per-position draft **entropy** and
  **top1–top2 margin** (fires 9,345×);
- `SpecDecodingStats.observe_draft(num_draft_tokens, num_accepted_tokens)` →
  per-step **accept-run length** (fires once per step).

Greedy verification is a Triton kernel (`rejection_greedy_sample_kernel`) and is not
hookable, but `observe_draft` records its result in Python. Because chain drafting is a
prefix, drafting K′≤7 accepts `min(K′, accept_run)`, so controllers can be replayed
**exactly** offline. We report **Mean Accepted Tokens/step (MAT)** (framework-independent)
and a cost-model net speedup `MAT/(1+c·meanK)` with draft/target cost `c=0.15` (cheap
EAGLE head). All gains are vs the **best fixed K** (the honest baseline).

## 3. Acceptance statistics

Mean accept-length **1.78** (≈1 draft token accepted/step). **61% of steps accept zero**
draft tokens; the accept-run distribution is heavily zero-inflated. Lag-1 autocorrelation
of the accept-run is a weak **0.247**.

## 4. Results

### Per-request controllers (90 prompts, measured wall-clock; `simulate_eagle_controllers.py`)
| Policy | net speedup | vs best fixed |
|---|:---:|:---:|
| Fixed K=2 (best) | 1.538× | — |
| UCB (BanditSpec) | 1.463× | **−4.9%** |
| ε-greedy | 1.432× | −6.9% |
| AcceptanceHistory | 1.394× | −9.3% |
| Oracle (per-request) | 1.569× | +2.0% |

**Per-request adaptation is futile:** the oracle ceiling is only +2%, and every reactive
bandit *loses* by paying exploration cost it cannot recoup.

### Per-step controllers (1,295 steps, cost-model speedup; `simulate_eagle_perstep.py`, `verify_explore_perstep.py`)
| Policy | best speedup~ | vs best fixed |
|---|:---:|:---:|
| Fixed K=2 (best) | 1.211 | — |
| SVIP entropy threshold (τ=6) | 1.240 | +2.4% |
| Margin threshold (τ=0.3) | 1.241 | +2.5% |
| Entropy-OR-margin | 1.239 | +2.3% |
| Persistence / EMA-of-accept-run | 1.245 | **+2.9% (best)** |
| Learned logistic stopper (held-out) | 1.161 | +1.1% |
| **Oracle (per-step)** | **1.472** | **+21.5%** |

## 5. Finding: the signal, not the policy, is the bottleneck

Per-step adaptation has a **large +21.5% oracle ceiling** (vs +2% per-request) — genuine
within-stream variance exists. Yet **no cheap controller exceeds ~+3%**: entropy, margin,
their combination, acceptance-persistence, and even a **trained logistic predictor**
(entropy+margin+position+recent-acceptance, train/test split) — the SpecKV/AdaEAGLE recipe
in miniature — all land between **+1.1% and +2.9%**. Draft-side confidence signals do not
predict per-step acceptance on this model, because acceptance depends on the *target's*
next-token distribution, which is unobserved before drafting. The 61% zero-acceptance steps
are not separable from productive steps using any draft-side feature we measured.

## 5b. Draft-side vs target-side signal

To test whether *any* cheap signal predicts acceptance, we additionally captured the
**target model's** per-position verification entropy (a read-only hook on
`LlamaForCausalLM.compute_logits`). Correlations of each signal with the accept-run
(more negative = better predictor):

| Signal | r vs accept-run | Causal? |
|---|:---:|:---:|
| Draft entropy[0] (what SVIP/BanditSpec/AdaEAGLE use) | −0.228 | yes |
| Target entropy[0] (verifier, same step) | −0.157 | no |
| **Target *bonus* entropy, previous step** | **−0.317** | **yes (usable)** |

The verifier's causal bonus-token entropy is the **strongest single predictor** — it beats
the draft-side confidence all prior controllers rely on — yet a controller gating draft
length on it still yields only **+2.5%**, because |r|=0.32 explains ~10% of acceptance
variance. *Better signal, same ceiling:* the limitation is signal informativeness, and even
the verifier's own uncertainty is too weak to close the gap.

## 5c. Generalization: a second target/draft pair (Qwen3-14B)

To check this is not a Llama-8B artifact, we repeat the full per-step analysis on
**Qwen3-14B + `AngelSlim/Qwen3-14B_eagle3`** (1,354 steps, different architecture and size).

| Metric | Llama-3.1-8B | Qwen3-14B |
|---|:---:|:---:|
| Best fixed K | 2 | **1** |
| Lag-1 autocorr of accept-run | 0.247 | 0.275 |
| Per-step oracle ceiling | +21.5% | +16.3% |
| SVIP entropy / Margin / Persistence | +2.4 / +2.5 / +2.9% | +0.5 / +1.7 / +2.0% |
| Causal target-bonus-entropy corr | −0.317 | −0.232 |
| Target-gate controller | +2.5% | −0.7% |

The mechanism replicates: a double-digit oracle ceiling, weak signal–acceptance correlation
(|r|≈0.2–0.3), and **no cheap controller beating ~+3%**. Qwen3-14B is *more* saturated (best
fixed K=1; its target-gate controller loses). The predictor of controller value is the
**signal–acceptance correlation, not the oracle ceiling** — and it forecasts "ship fixed K"
on both pairs. This yields a practical **deployment rule**: cheaply measure the acceptance
distribution, its autocorrelation, and the signal–acceptance correlation; if the latter is
weak (|r|≲0.4), adaptive draft length will not pay regardless of the oracle ceiling.

## 6. Conclusion

On EAGLE-3 (Llama-3.1-8B *and* Qwen3-14B), **adaptive draft-length control cannot beat a
tuned fixed K by more than ~3%**, despite +16–21% per-step oracle ceilings — a **negative
result** with a precise cause: **the signal, not the policy.** We establish this by exhausting the policy
space (threshold, bandit, persistence, learned) *and* directly measuring signal–acceptance
correlation, including a target-side signal that, while the best available (r≈−0.32), still
falls far short. This is regime-specific: SVIP reports +13% on EAGLE-2, where draft entropy
is more predictive; that advantage does not transfer to EAGLE-3/Llama-8B. The practical
recipe on this pairing is **fixed K=2**; closing the +21.5% ceiling requires an acceptance
predictor stronger than any draft- or target-side confidence signal we measured — the
key open problem this study localizes.

*Reproduce:* `modal_eagle3_perstep_capture.py` (draft + target capture) →
`src/simulate_eagle_perstep.py`, `src/simulate_eagle_controllers.py`,
`src/verify_explore_perstep.py`, `src/explore_target_signal.py` on
`results/eagle3_perstep_target_llama8b.json` (per-step) and
`results/eagle3_multik_llama8b.json` (per-request).
