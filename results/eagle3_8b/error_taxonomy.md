# Error Taxonomy — Llama-3.1-8B + EAGLE-3 (per-step, shipped K=2)

Scored over 5971 steps vs the per-step oracle K\*=accepted-run.

| error type | description | count | share |
|---|---|---|---|
| over_draft | drafted past the rejection point (K>K\*) → wasted compute | 4697 | 78.7% |
| under_draft | stopped too early on an easy span (K<K\*) → speedup left unused | 725 | 12.1% |
| matched | drafted exactly the accepted run (K=K\*) | 549 | 9.2% |

**Examples** (entropy of position-0 draft token; lower = more confident):

- *over_draft*: acc=0, ent0=8.4574; acc=0, ent0=0.1869; acc=0, ent0=11.0875
- *under_draft*: acc=3, ent0=5.1451; acc=4, ent0=2.184; acc=3, ent0=0.0286

**Reading:** over-drafting dominates at K=2 because most steps accept 0–1 draft tokens (per-position acceptance 0.53→0.31→0.18…). A perfect per-step controller would cut that waste — which is exactly the +25% per-step oracle headroom — but draft entropy does not separate the cases well enough to act on.