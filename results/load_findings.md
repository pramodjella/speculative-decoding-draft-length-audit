# Load+Content Thesis — Dynamic-Load Results

Headline metric = per-step wall-clock speedup (and accepted tokens/step), under a time-varying batch/load schedule with autocorrelated content difficulty. Seeds=8.

## Speedup by policy (mean ± 95% CI)

```
  workload  best_fixed      tapout    nightjar content_only   load_only    combined      oracle
     gsm8k 1.356±0.033 1.058±0.036 1.282±0.035  1.354±0.033 1.379±0.031 1.389±0.034 1.566±0.040
 humaneval 1.622±0.044 1.413±0.043 1.578±0.039  1.607±0.048 1.666±0.046 1.684±0.038 1.877±0.046
  mt_bench 1.336±0.025 1.040±0.024 1.253±0.026  1.338±0.031 1.353±0.025 1.360±0.024 1.533±0.030
spec_bench 1.392±0.027 1.094±0.026 1.322±0.027  1.380±0.031 1.417±0.029 1.425±0.028 1.609±0.032
```


## Does combined beat the published single-sided baselines? (paired % gain)

```
  workload vs TapOut(content-only) vs Nightjar(load-only) vs content_only(ablation) vs load_only(ablation)
     gsm8k        +31.4%±1.8 (yes)        +8.4%±0.7 (yes)           +2.6%±1.0 (yes)        +0.7%±0.7 (yes)
 humaneval        +19.2%±1.2 (yes)        +6.8%±0.5 (yes)           +4.9%±1.7 (yes)        +1.2%±1.2 (yes)
  mt_bench        +30.9%±1.1 (yes)        +8.6%±0.5 (yes)           +1.7%±1.1 (yes)        +0.6%±0.4 (yes)
spec_bench        +30.2%±1.2 (yes)        +7.8%±0.4 (yes)           +3.2%±0.7 (yes)         +0.5%±0.6 (no)
```


## Oracle-headroom capture (combined vs best single-sided, toward clairvoyant oracle)

```
  workload  best_single_sided  combined  oracle(clairvoyant)  headroom_captured_%
     gsm8k              1.379     1.389                1.566                  5.0
 humaneval              1.666     1.684                1.877                  9.0
  mt_bench              1.353     1.360                1.533                  4.0
spec_bench              1.417     1.425                1.609                  4.0
```


# Motivation — fixed-B sweep: the best fixed K shifts with batch


**gsm8k** — best fixed K by batch: B=1:K=4, B=4:K=4, B=8:K=4, B=16:K=2, B=32:K=1, B=64:K=1

**humaneval** — best fixed K by batch: B=1:K=4, B=4:K=4, B=8:K=4, B=16:K=2, B=32:K=1, B=64:K=1

**mt_bench** — best fixed K by batch: B=1:K=2, B=4:K=2, B=8:K=2, B=16:K=2, B=32:K=1, B=64:K=1

**spec_bench** — best fixed K by batch: B=1:K=4, B=4:K=4, B=8:K=4, B=16:K=2, B=32:K=1, B=64:K=1
