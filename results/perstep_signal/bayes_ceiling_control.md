# Permutation control for the Bayes-ceiling hidden-state result

Test: shuffle the per-block accept labels ACROSS draft blocks (preserving the marginal
label distribution and prefix structure exactly, but destroying the hidden-state -> accept
relationship), retrain the out-of-fold PCA-50+LR probe, and recompute Bayes(hidden) recovery.

If the real result is genuine signal, shuffled recovery must collapse to ~0.

| | Real labels | Shuffled labels |
|---|---:|---:|
| Llama-3.1-8B Bayes(hidden) recovery | **+17.8%** | **-4.9%** |

Result: shuffled recovery collapses to zero/negative -> the +17.8% (and +18.9% deployable)
is genuine exploitable signal in the EAGLE3 draft hidden state, NOT a pipeline/leakage artifact.

Reproduce: `python analyze_bayes_ceiling_control.py`
