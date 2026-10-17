# Bayes-ceiling decomposition of per-step adaptive-K headroom

**Claim under test:** the small recovery is a *Bayes ceiling*, not a weak probe.
Rungs (speedup, cost model MAT/(1+0.15K)):

- `best fixed K` — tuned static baseline (mean acceptance already priced in)
- `Bayes(position)` — best policy from position-only survival (static -> ~= fixed K)
- `probe (deployable)` — full-hidden probe, lambda chosen on TRAIN gens (realistic)
- `Bayes(hidden)` — full-hidden probe with ORACLE lambda on out-of-fold probs (theoretical ceiling for any controller that sees the draft hidden state)
- `per-step oracle` — K=acc+1, realization-aware (unreachable by construction)

Recovery = (rung - best_fixed) / (oracle - best_fixed).

| Model | Oracle ceiling | Bayes(position) | Probe (deploy) | **Bayes(hidden)** | Oracle |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-LLaMA-8B (reasoning) | +4.5% | +0.0% | +0.0% | **+0.0%** | 100% |
| deepseek_reasoning | +8.5% | +0.0% | -0.6% | **+0.2%** | 100% |
| Llama-3.1-8B (instruct) | +18.1% | +0.0% | +18.9% | **+17.8%** | 100% |
| llama8b_reasoning | +23.2% | +0.0% | +18.3% | **+15.2%** | 100% |
| Qwen3-14B (instruct) | +12.4% | +0.0% | +19.1% | **+19.7%** | 100% |

## Reading

1. **Bayes(position) = +0.0% recovery on every model** -> static (non-step-varying)
   signal cannot beat a tuned fixed K. This is why position/entropy AUC ~0.5-0.58
   bought nothing, and it validates the block key (position degenerates to fixed K).
2. **Instruction models: the draft hidden state IS exploitable.** Llama and Qwen3
   recover ~19% of the per-step oracle span via a PCA-50 probe available at draft time
   (~+2-3% net over a tuned fixed K). Verified by a permutation control (shuffled
   labels -> recovery collapses to ~0): see bayes_ceiling_control.md.
3. **Reasoning model: no exploitable signal.** DeepSeek-R1 recovers +0.0% even with
   the oracle threshold; fixed K=1 is optimal and the tiny ceiling is unreachable.
4. **The remaining ~4/5 of the oracle is irreducible.** Bayes(hidden) ~= deploy << oracle:
   the gap from Bayes(hidden) to the realization-aware oracle is aleatoric variance no
   pre-verification signal can reach. The oracle ceiling overstates achievable gain;
   the recoverable fraction is Bayes(hidden) (~19% for instruct, 0% for reasoning).

Net: the per-step signal lives in the hidden state, not in cheap logit features (E2)
and not in token position; it is real but model-class-dependent (instruct yes, reasoning no).