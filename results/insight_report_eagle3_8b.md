> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# Insight Report — Adaptive Draft-Length Controllers on Llama-3.1-8B + EAGLE-3

**Target:** `meta-llama/Llama-3.1-8B-Instruct`  **Draft:** `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`  
**Engine:** vLLM 0.23 (H100, greedy, exact verification → lossless).  
**Workloads:** HumanEval, GSM8K, MT-Bench (30 prompts each). K ∈ [1, 2, 3, 5, 7].

## RQ1 — Can a cheap per-step controller beat the best fixed K?

Best fixed draft length on mixed traffic is **K=2** at **1.538× net speedup**. The per-request **oracle ceiling** (picks the best K per prompt, knowing the outcome) is **1.569×**, i.e. only **+2.0%** over best fixed. Realisable online controllers do *not* clear that bar:

| policy | net speedup | gap vs best fixed |
|---|---|---|
| Fixed K=1 | 1.386× | -9.9% |
| Fixed K=3 | 1.522× | -1.1% |
| Fixed K=2 (best) | 1.538× | +0.0% |
| AcceptHistory | 1.394× | -9.3% |
| UCB | 1.463× | -4.9% |
| EpsGreedy | 1.432× | -6.9% |
| ORACLE(per-req) | 1.569× | +2.0% |

**Per-step (SVIP entropy threshold)** on 5971 captured steps tells the more interesting story. The per-**step** oracle (drafts exactly the run that will be accepted) reaches **+24.9%** over best fixed-K=2 — an order of magnitude more headroom than the per-**request** ceiling (+2.0%). So within-stream variation is genuinely large. But the cheap entropy threshold captures almost none of it: best SVIP (τ=2.0) is only **+0.3%** over best fixed-K (it mainly saves wasted drafts — mean K 1.12 vs 2 at similar accept length). **The lever is real and big; draft entropy is too weak a signal to pull it.** That gap — not a flat 'no headroom' — is the honest result, and it points future work at stronger per-step signals (target-side margin, learned predictors).

## RQ2 — Which signal/controller, and how does the optimum shift?

Per-workload best fixed K: gsm8k=K3, humaneval=K2, mt_bench=K2; mixed=K2. The optimum sits at a **small K (2–3)** everywhere: the EAGLE-3 head is cheap and accurate, so acceptance saturates early and long drafts mostly add waste. Among online controllers, UCB (BanditSpec) is the least-bad but still trails best-fixed because cold-start exploration costs more than the ~2% it could win.

## RQ3 — Generalisation & convergence

A single small fixed K (=2) is within ~1% of the per-workload-optimal K on all three workloads, so it generalises without retuning. Online bandits converge toward that same small K but pay regret getting there; on streams this short the regret dominates the tiny headroom.

## Practitioner takeaway

On a strong, cheap EAGLE-3 head for an 8B target, **ship fixed K=2** today: no deployable controller here beats it, because request-level adaptation has only ~2% to give and the online bandits lose that to cold-start regret. The non-obvious finding is that **per-step** adaptation is different: a perfect per-step length oracle is ~25% faster (cost-model), so the within-stream lever is large — it is the *signal*, not the headroom, that is missing. Draft entropy (SVIP) captures almost none of it. The actionable next step is therefore a better per-step acceptance predictor, not more bandit tuning.
