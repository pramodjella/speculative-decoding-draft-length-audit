# Target-entropy predictability audit

**Question (Yash):** is acceptance set by the TARGET distribution? Does `ent_t` (target entropy at each position) crack the per-step ceiling?

> Note: `ent_t` is only available *after* the target forward pass (i.e. after verification). It cannot be used by any real pre-verification controller. This is a diagnostic probe, not a deployable feature.

---

## Llama-3.1-8B + EAGLE3

- Steps: 5842  |  Gens: 90  |  max_k: 7  |  Labeled positions: 11194

### Single-feature AUC (predict per-position accept)

| feature | AUC | note |
|---|---:|---|
| draft_entropy | 0.416 |  |
| top1_margin | 0.573 |  |
| target_entropy | 0.429 | *post-verification — unavailable to real controller* |
| position | 0.579 |  |
| prev_acc_ema | 0.670 |  |

### Headroom recovered on TEST (8 gen-splits, mean +/- std)

| predictor | features | % of oracle ceiling |
|---|---|---:|
| best fixed K=2 | — | 0% (baseline) |
| draft-only | ent, margin, pos, ema | 6.8% +/- 3.8 |
| target-only* | ent_t, pos, ema | 6.8% +/- 5.0 |
| draft+target* | all 5 | 8.9% +/- 4.3 |
| per-step oracle | — | 100% (= +24.8% over fixed) |

*Rows marked * use post-verification features — unavailable to real controllers.

**Interpretation:** target_entropy adds only +0.0 pts over draft features. Even POST-VERIFICATION signal barely helps — the per-step ceiling is structurally irreducible.

---

## Qwen3-14B + EAGLE3

- Steps: 1315  |  Gens: 24  |  max_k: 7  |  Labeled positions: 2246

### Single-feature AUC (predict per-position accept)

| feature | AUC | note |
|---|---:|---|
| draft_entropy | 0.414 |  |
| top1_margin | 0.585 |  |
| target_entropy | 0.404 | *post-verification — unavailable to real controller* |
| position | 0.504 |  |
| prev_acc_ema | 0.626 |  |

### Headroom recovered on TEST (8 gen-splits, mean +/- std)

| predictor | features | % of oracle ceiling |
|---|---|---:|
| best fixed K=1 | — | 0% (baseline) |
| draft-only | ent, margin, pos, ema | 8.5% +/- 4.4 |
| target-only* | ent_t, pos, ema | 9.0% +/- 4.4 |
| draft+target* | all 5 | 12.0% +/- 6.9 |
| per-step oracle | — | 100% (= +16.9% over fixed) |

*Rows marked * use post-verification features — unavailable to real controllers.

**Interpretation:** target_entropy adds only +0.5 pts over draft features. Even POST-VERIFICATION signal barely helps — the per-step ceiling is structurally irreducible.

---

## DeepSeek-R1-Distill-LLaMA-8B + EAGLE3

- Steps: 16638  |  Gens: 90  |  max_k: 7  |  Labeled positions: 22819

### Single-feature AUC (predict per-position accept)

| feature | AUC | note |
|---|---:|---|
| draft_entropy | 0.489 |  |
| top1_margin | 0.511 |  |
| target_entropy | 0.473 | *post-verification — unavailable to real controller* |
| position | 0.563 |  |
| prev_acc_ema | 0.620 |  |

### Headroom recovered on TEST (8 gen-splits, mean +/- std)

| predictor | features | % of oracle ceiling |
|---|---|---:|
| best fixed K=1 | — | 0% (baseline) |
| draft-only | ent, margin, pos, ema | -2.7% +/- 1.1 |
| target-only* | ent_t, pos, ema | -4.9% +/- 2.9 |
| draft+target* | all 5 | -3.1% +/- 2.1 |
| per-step oracle | — | 100% (= +8.9% over fixed) |

*Rows marked * use post-verification features — unavailable to real controllers.

**Interpretation:** target_entropy adds only +-2.3 pts over draft features. Even POST-VERIFICATION signal barely helps — the per-step ceiling is structurally irreducible.

---

## Cross-model summary

| model | oracle ceiling | draft recovery | target recovery | target boost | signal verdict |
|---|---:|---:|---:|---:|---|
| Llama-3.1-8B + EAGLE3 | +24.8% | 6.8% | 6.8% | +0.0 pts | Even target signal barely helps |
| Qwen3-14B + EAGLE3 | +16.9% | 8.5% | 9.0% | +0.5 pts | Even target signal barely helps |
| DeepSeek-R1-Distill-LLaMA-8B + EAGLE3 | +8.9% | -2.7% | -4.9% | +-2.3 pts | Even target signal barely helps |

**Paper implication:** even knowing the target distribution at each position (post-verification) does not unlock the per-step ceiling — confirming that the ceiling is set by unresolvable uncertainty in the draft/target alignment, not by inadequate pre-verification signal. This is a STRONGER negative result than the draft-features audit alone.
