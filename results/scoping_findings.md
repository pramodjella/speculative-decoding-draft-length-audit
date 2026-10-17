# Scoping Study — When Does Adaptive Draft-Length Control Pay?

Simulator, 4 seeds. Metric = per-step speedup. Headroom = oracle gap over best fixed-K (Oracle-Task). r = draft/target cost ratio (0.1 = EAGLE/large-target, 0.85 = measured small pair). All numbers are %.

## Adaptive headroom (%) — avg over workloads, by draft cost r and load regime

load_regime  static_low  static_high  dynamic
r                                            
0.10                3.0          0.0     11.4
0.20                4.4          0.0      8.2
0.50                2.8          0.0      2.4
0.85                0.0          0.0      0.0


## In dynamic load: load vs content value, and their complementarity (%) by r

      load_value_%  content_value_%  complementarity_%  adaptive_headroom_%  online_capture_%
r                                                                                            
0.10           9.5              4.1                1.8                 11.4              35.8
0.20           5.2              3.0                2.8                  8.2              26.8
0.50           0.8              0.7                1.2                  2.4            -151.8
0.85           0.0              0.0                0.0                  0.0               NaN


## Favorable regime (r=0.2, dynamic) — per workload (%)

            load_value_%  content_value_%  complementarity_%  adaptive_headroom_%  online_capture_%
workload                                                                                           
humaneval            8.0              2.4                0.6                  8.6              40.0
gsm8k                4.9              3.8                3.4                  8.5              25.0
mt_bench             3.7              3.2                3.9                  7.8              21.0
spec_bench           4.1              2.4                3.5                  7.8              21.0


## Findings (the regime map)

1. **Adaptive draft length only pays with a CHEAP drafter and headroom.** Headroom is 8–13% at r≤0.2 under dynamic load, ~2–6% at batch 1, and **≈0% at r=0.85 (real small pairs) or under heavy static load**. This reconciles the field: 'it doesn't help' reports are simply the no-headroom regime.
2. **At batch 1, only CONTENT matters** (load is constant): ~2–6% headroom — the SVIP/TapOut regime.
3. **Load-awareness matters only under VARYING load**, where it is the dominant lever (load_value 4–11% vs content_value 1–6%) — the Nightjar regime.
4. **Load+content are complementary only in dynamic load at moderate r**, peaking ~3–4% (r≈0.2). Real but modest and regime-specific — NOT a large universal win.
5. **Online controllers capture only ~20–43%** of the achievable joint headroom in the favorable regime — the joint online-control problem is open (future work).
