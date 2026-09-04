# Per-step HIDDEN-STATE audit (EAGLE-3 / Llama-3.1-8B)

- clean steps 5842; labeled positions 11194 (accept rate 48.7%); hidden-norm AUC 0.484

| feature set | % of oracle ceiling recovered |
|---|---:|
| base (entropy+margin+position+history) | 6.8% ± 3.8 |
| base + EAGLE hidden state | 6.1% ± 2.9 |
| per-step oracle | 100% (= +24.8% over fixed) |

**Hidden state over cheap features: -0.7 pts of ceiling.**

## Verdict

DEAD END — even the EAGLE hidden state does not crack the per-step ceiling. Adaptive draft length has no exploitable signal. Lock the negative result; pivot to a different axis (quality/batch).
