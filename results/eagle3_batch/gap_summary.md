# EAGLE-3 batch-size sweep — gap-vs-batch read

| workload | batch | best K | best speedup | K=1 speedup | gap vs K=1 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 1 | 3 | 1.650 | 1.434 | +0.216 |
| humaneval | 1 | 2 | 1.642 | 1.442 | +0.199 |
| mt_bench | 1 | 2 | 1.361 | 1.291 | +0.070 |
| gsm8k | 8 | 2 | 1.383 | 1.306 | +0.077 |
| humaneval | 8 | 1 | 1.288 | 1.288 | +0.000 |
| mt_bench | 8 | 2 | 1.199 | 1.136 | +0.063 |
| gsm8k | 16 | 4 | 1.308 | 1.234 | +0.074 |
| humaneval | 16 | 2 | 1.376 | 1.269 | +0.107 |
| mt_bench | 16 | 2 | 1.124 | 1.077 | +0.047 |
| gsm8k | 32 | 1 | 1.205 | 1.205 | +0.000 |
| humaneval | 32 | 1 | 1.111 | 1.111 | +0.000 |
| mt_bench | 32 | 1 | 1.099 | 1.099 | +0.000 |
| gsm8k | 64 | 1 | 1.153 | 1.153 | +0.000 |
| humaneval | 64 | 1 | 1.077 | 1.077 | +0.000 |
| mt_bench | 64 | 1 | 1.103 | 1.103 | +0.000 |

## What to look for
- Best K **drops** as batch grows -> verification cost dominates -> per-step adaptive K has real headroom in the batched regime.
- Curves **flatten** at high B -> fixed K leaves money on the table precisely where a controller could pick K per step.
- If best K stays put and curves keep their shape -> Yash's hypothesis is rejected; lock in the negative result.