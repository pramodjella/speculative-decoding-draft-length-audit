> ⚠️ **SUPERSEDED RECOVERY NUMBERS.** The AUC values below are correct (per-row), but the
> **recovery / speedup** columns were computed with a block-key bug: draft blocks were keyed
> by `(gen_i, step_i)`, which merges blocks across the 3 workloads (gen_i repeats per workload).
> The corrected analysis is **`bayes_ceiling.md`** (key `(workload, gen_i, step_i)`,
> permutation-verified). Corrected recovery is HIGHER: instruction models recover ~19% of the
> per-step oracle (+2.4–3.4% net over tuned fixed K); the reasoning model recovers 0%.
> Use this file only for the AUC table.

# Full hidden-state probe audit

**Yash's ask:** replace the weak 16-dim random projection with a properly
trained probe on the full hidden state of the EAGLE3 draft head.

Protocol: 8 gen-splits, train on 7/8, test on 1/8. Threshold chosen on TRAIN.
Cost model: MAT/(1+0.15·K). Recovery = (probe_speedup − best_fixed) / (oracle − best_fixed).

## DeepSeek-R1-Distill-LLaMA-8B (reasoning)

Oracle ceiling: **+4.2%** over best fixed-K (1.0686→1.1130)

| Probe | AUC | Speedup | Recovery of ceiling |
|---|---:|---:|---:|
| Logistic regression (full hidden) | 0.844±0.020 | 1.0640±0.0511 | -10.4% |
| MLP (256→64, full hidden) | 0.868±0.025 | 1.0676±0.0486 | -2.3% |
| PCA-50 + logistic regression | 0.870±0.014 | 1.0704±0.0488 | +4.0% |

**Prior result (16-dim RP + GBM):** AUC ~0.484, recovery −0.7 pts.

## Llama-3.1-8B (instruct)

Oracle ceiling: **+17.5%** over best fixed-K (1.3167→1.5472)

| Probe | AUC | Speedup | Recovery of ceiling |
|---|---:|---:|---:|
| Logistic regression (full hidden) | 0.791±0.014 | 1.3241±0.0789 | +3.2% |
| MLP (256→64, full hidden) | 0.825±0.013 | 1.3280±0.0786 | +4.9% |
| PCA-50 + logistic regression | 0.842±0.011 | 1.3352±0.0805 | +8.0% |

**Prior result (16-dim RP + GBM):** AUC ~0.484, recovery −0.7 pts.

## Qwen3-14B (instruct)

Oracle ceiling: **+12.6%** over best fixed-K (1.3699→1.5430)

| Probe | AUC | Speedup | Recovery of ceiling |
|---|---:|---:|---:|
| Logistic regression (full hidden) | 0.825±0.012 | 1.3477±0.0218 | -12.9% |
| MLP (256→64, full hidden) | 0.876±0.005 | 1.3570±0.0317 | -7.4% |
| PCA-50 + logistic regression | 0.875±0.006 | 1.3730±0.0395 | +1.8% |

**Prior result (16-dim RP + GBM):** AUC ~0.484, recovery −0.7 pts.

## Interpretation

If any of the full-hidden probes substantially exceeds the 16-dim RP result
(AUC > 0.60, recovery > +5%), the original negative result was probe-limited.
If all probes remain near AUC ~0.5 and recovery < +2%, the negative result
is robust: no geometry in the draft hidden state predicts acceptance.

This is the strongest possible negative test — if a trained MLP on 4096 raw
dimensions can't find signal, no practical per-step controller can.