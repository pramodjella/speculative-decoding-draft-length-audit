# Wall-clock end-to-end test — deployable adaptive draft depth vs tuned fixed

Real tokens/sec in the SafeAILab/EAGLE-3 reference repo (plain PyTorch, eager — no CUDA-graph
penalty; vLLM 0.23 has no per-step hook so cannot be used). Llama-3.1-8B + yuhuili EAGLE3 head,
H100, greedy, n=10 prompts/workload, maxtok=128. Script: `modal_eagle_wallclock.py`.

Controller: a PCA-50 + ridge probe predicts draft depth per step from the step's SEED hidden
state (deployable, reactive), setting `ea_layer.depth` before each draft. Fixed and adaptive
run through the SAME wrapper codepath (equal per-step overhead).

## Result: adaptive does NOT beat a tuned fixed depth

| Workload | Best fixed depth | Best fixed tok/s | Adaptive tok/s | Gain |
|---|---:|---:|---:|---:|
| HumanEval | 6 | 157.0 | 156.5 | **−0.3%** |
| GSM8K     | 4 | 138.7 | 127.7 | **−8.0%** |
| MT-Bench  | 7 | 142.0 | 129.5 | **−8.8%** |

Seed→accept-length probe test correlation: **0.046** (no signal). Full fixed-depth sweep
(d=2..8) tok/s:
- HumanEval: 114.5 / 126.1 / 141.7 / 149.4 / **157.0** / 155.4 / 155.5
- GSM8K:     111.0 / 123.6 / **138.7** / 137.4 / 133.7 / 132.7 / 130.8
- MT-Bench:  115.9 / 129.2 / 137.4 / 140.2 / 138.8 / **142.0** / 132.5

## Interpretation

1. **Best fixed depth varies by workload (4–7)** → real workload-level headroom exists; a
   per-workload *tuned fixed* depth captures it (consistent with the tree-size sweep).
2. **A deployable per-step controller loses to tuned fixed** (−0.3% to −8.8%). The seed hidden
   state does not predict chain acceptance length (corr ≈ 0), and in this high-acceptance regime
   (mean accept-length 5.18/8) drafting *shorter* mostly hurts, so per-step shortening loses.

## Two controllers tested — both lose

**(A) Seed-based reactive** (`modal_eagle_wallclock.py`): probe on the step's seed hidden state
sets draft depth. *Code-audit note (2026-07-03):* the original capture paired seeds↔accept-lengths
by global flat index across prompts (one dangling unverified seed per prompt → cumulative drift),
so the first run's "corr ≈ 0.05, no signal" readout was an artifact. Fixed (per-prompt pairing)
and rerun; see `pre_write_audit.md` §5. **Corrected result:** the seed hidden state DOES carry
modest signal (corr = 0.26), yet the controller still loses to the best fixed depth:

| Workload | Best fixed depth | Best fixed tok/s | Adaptive tok/s | Gain |
|---|---:|---:|---:|---:|
| HumanEval | 6 | 184.2 | 173.8 | **−5.7%** |
| GSM8K     | 7 | 158.5 | 154.6 | **−2.4%** |
| MT-Bench  | 7 | 164.4 | 155.4 | **−5.5%** |

This makes (A) and (B) mutually reinforcing: two independent controllers, both with genuine signal
(seed corr 0.26; level AUC 0.80), both lose — the same verify-dominance economics.

**(B) In-loop within-step early-stop** (`modal_eagle_inloop.py`) — the FAITHFUL test that matches
the offline analysis. A copy of topK_genrate breaks the depth loop when a PCA-50 probe on each
level's hidden state predicts rejection (tt_eff caps the shorter tree so it never crashes; the
copy is verified to reproduce coherent generation).

| Workload | Fixed (full depth) tok/s | In-loop adaptive tok/s | Gain | mean depth |
|---|---:|---:|---:|---:|
| HumanEval | 144.5 | 137.2 | **−5.1%** | 6.6 |
| GSM8K     | 128.4 | 121.8 | **−5.1%** | 4.5 |
| MT-Bench  | 133.4 | 125.3 | **−6.1%** | 5.7 |

Within-step level probe test **AUC = 0.796** — the signal is REAL (close to the offline 0.84),
yet acting on it via early-stop LOSES 5–6% across all stop-thresholds {0.3,0.5,0.7}.

## Why the real signal still loses (the key insight)

In this EAGLE-3 regime the draft head is cheap and **verification dominates** execution, so
drafting *deeper* is almost always better: more accepted tokens per expensive verify pass.
Early-stopping saves cheap draft compute but forfeits accepted tokens worth far more. The offline
cost model MAT/(1+C·K) (C≈0.07–0.15) **overweights draft cost** relative to verify-dominated
reality, which is why its +2–3% inverts to −5–6% in wall-clock. Draft-length *shortening* is the
wrong lever when verify dominates — the same reason MagicDec argues for MORE speculation, not less.

## Bottom line for the paper

The per-step oracle gain is not merely mostly-irreducible; the *reachable* part **inverts sign in
real execution**. Adaptive draft-length shortening does not pay off, confirmed THREE ways:
(1) ~80% of the oracle is aleatoric; (2) offline, the reachable ~20% is real but small; (3) end-to-end,
even a real-signal within-step probe (AUC 0.80) loses 5–6% because verify dominates. Deployable
answer: draft deep at a tuned fixed length. This is a complete **measurement** result.

Data: `results/eagle_wallclock.json`, `results/eagle_inloop.json` (Modal volume).
Reproduce: `modal run modal_eagle_wallclock.py --n 10 --depths 2,3,4,5,6,7,8`;
`modal run modal_eagle_inloop.py --n 10 --maxtok 112`.
