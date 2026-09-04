# Bayes-ceiling on a REASONING workload (competition math) — model vs. task disambiguation

Same workload (MATH-500 / AIME competition math, long CoT, the SpecDecode-Bench regime),
two model families. Question: is the instruct-vs-reasoning signal boundary about the MODEL
or about the TASK? Capture: `modal_eagle3_hidden_full_capture.py --wl reasoning`.

| Model | Best fixed K | Oracle ceiling | Bayes(position) | Probe (deploy) | Bayes(hidden) | Net gain over tuned K |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B (instruct) | K=2 | +23.2% | +0.0% | **+18.3%** | +15.2% | **+4.3%** (1.4370→1.4982) |
| DeepSeek-R1 (reasoning) | K=1 | +8.5% | +0.0% | **−0.6%** | +0.2% | **~0%** (1.1094→1.1089) |

## ⚠️ Conclusion SUPERSEDED by audit (see pre_write_audit.md §3)

The original conclusion below ("boundary is model-type") is CONFOUNDED by draft-head strength:
the DeepSeek head's base acceptance is much lower than Llama's on the same task (accept-length
1.47 vs 2.40; pos-0 acceptance 0.276 vs 0.547). The frozen claim is narrowed to: *the
DeepSeek-R1-Distill head shows no recoverable draft-side signal* — head-scoped, stated as a
limitation. Original text kept below for the record.

## Original conclusion (superseded): the boundary is MODEL-TYPE, not task-type

On identical competition-math inputs:
- The **instruction** model (Llama-3.1-8B) retains exploitable hidden-state signal — +18.3%
  of the oracle, **+4.3% net over a tuned fixed K** (even larger than the +2.9% on its default
  workloads, because reasoning-math has a bigger oracle ceiling: lower acceptance → more headroom).
- The **reasoning-distilled** model (DeepSeek-R1) has **no** exploitable signal (~0%), exactly as
  on its default workloads. Best fixed K=1 remains optimal.

So it is not that "reasoning tasks have no per-step signal" — Llama doing hard math still has it.
It is that **reasoning-distilled models produce draft hidden states whose acceptance is
unpredictable (aleatoric)**, even on the same inputs where an instruction model's is predictable.
This upgrades the model-class boundary from an observation into a controlled result: task held
fixed, only the model changes, and the signal appears/vanishes with the model.

Sanity: Bayes(position) = +0.0% for both (block key correct; static signal inert).
Data: results/eagle3_hidden_full/hidden_full_{llama8b,deepseek}_reasoning.parquet
(32,207 / 54,733 positions). Reproduce: `python analyze_bayes_ceiling.py --model llama8b_reasoning`.
