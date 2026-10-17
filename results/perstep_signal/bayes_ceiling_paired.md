# Bayes ladder — PAIRED WITHIN FOLD (corrected comparison)

Both hidden-probe rungs scored on the SAME test rows in each fold, so the
ceiling genuinely bounds the deployable probe and their difference is a paired
per-fold statistic with a standard error.

| model | deployable recovery | ceiling recovery | threshold-selection loss | nesting control |
|---|---:|---:|---:|---|
| deepseek_reasoning | -1.3 ±0.8% | +1.9 ±0.6% | +0.0029 ±0.0005 pts (+3.1% of span) | PASS |
| llama8b | +16.8 ±2.6% | +18.1 ±2.6% | +0.0028 ±0.0015 pts (+1.2% of span) | PASS |
| qwen14b | +17.2 ±2.4% | +19.5 ±2.5% | +0.0034 ±0.0008 pts (+2.2% of span) | PASS |
| llama8b_reasoning | +9.5 ±3.7% | +12.0 ±3.8% | +0.0086 ±0.0015 pts (+2.6% of span) | PASS |
| deepseek | -0.3 ±0.1% | +0.7 ±0.5% | +0.0004 ±0.0001 pts (+0.9% of span) | PASS |
