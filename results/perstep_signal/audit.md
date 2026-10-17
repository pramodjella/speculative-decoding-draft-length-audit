# Per-step signal predictability audit (EAGLE-3 / Llama-3.1-8B)

- clean steps 5842; labeled positions 11194 (accept rate 48.7%); pos-0 immediate-reject 57.8%
- protocol: GBM fit on train gens, stop-threshold chosen on train, all positions scored at sim time, reported on held-out test gens, mean±std over 8 generation splits.

## Single-feature AUC (predict per-position accept)

| feature | AUC all-pos | AUC pos-0 |
|---|---:|---:|
| draft_entropy | 0.416 | 0.394 |
| top1_margin | 0.573 | 0.587 |
| position | 0.579 | 0.500 |
| prev_acc_ema | 0.670 | nan |

## Headroom recovered on TEST (cost model c=0.15, min K=1)

| policy | % of oracle ceiling recovered |
|---|---:|
| best fixed K=2 | 0% (baseline) |
| history-only predictor | -0.8% ± 2.4 |
| full draft+history predictor | 6.8% ± 3.8 |
| per-step oracle | 100% (= +24.8% over fixed) |

**Per-position draft features over history alone: +7.5 pts of ceiling.**

## Verdict

WEAK — cheap logit-level features do NOT meaningfully beat history; the per-step ceiling is largely irreducible from logit features. The only untested lever is EAGLE hidden-state features (needs new capture).
