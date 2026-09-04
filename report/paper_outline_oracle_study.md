> ⚠️ **SUPERSEDED — historical document.** Numbers and claims here predate the final audit
> and may be stale (e.g. pre-audit oracle estimates, retired verdicts). The current source of
> truth is `docs/CANONICAL.md`; the current paper is `report/paper_draft.md`. Kept for
> provenance only — do not cite from this file.

# Paper outline — Oracle study of the speculative-decoding adaptation ceiling

**Working title:** *Why Load-Aware Speculation Wins and Signal-Aware Control Fails:
An Oracle Study of the Adaptation Ceiling in Speculative Decoding*

**Type:** measurement / analysis paper (efficiency workshop → possibly systems venue).
**One-line thesis:** the speedup left for *adapting draft length* is small and, where it
exists, unreachable from any pre-verification signal — which is exactly why the 2025–26
frontier (Nightjar, SADDLE) adapts to **load** rather than to **signal**. We quantify
that ceiling with oracles nobody has published.

**Why it is uncontested:** Nightjar/SADDLE/AdaSpec *assert* load-level adaptation works
and disable speculation heuristically; none prove signal-level control *cannot* win,
because none compute the oracle decomposition. This paper supplies the "why" behind the
field's shift. It is complementary to those methods, not competing with them.

---

## Section plan

### 1. Introduction
- Speculative decoding is lossless and fast; the open lever everyone tried is *how many
  tokens to draft* (draft length K).
- Two families: per-request / per-step **signal-aware** controllers (BanditSpec, SpecDec++,
  AdaEAGLE, entropy/SVIP) and recent **load-aware** schemes (Nightjar, SADDLE, AdaSpec).
- Gap in the literature: nobody has measured the *ceiling* — the best any adaptive policy
  could achieve — so the community can't tell whether controllers underperform because they
  are weak or because there is nothing to win.
- Contribution: an oracle decomposition (per-request, per-step, per-batch) on a strong
  modern drafter (EAGLE-3 / Llama-3.1-8B) that bounds the headroom and a signal-predictability
  audit that shows where it is reachable. Result: signal-aware adaptation is near-dead;
  load-aware adaptation is the only structure left — explaining the 2025–26 shift.

### 2. Setup
- Target Llama-3.1-8B-Instruct + EAGLE-3 head; vLLM/H100; greedy; lossless.
- Headline metric: **accepted tokens per target step** (deterministic). Cost-model speedup
  `MAT/(1+c·meanK)` as a secondary, with wall-clock for grounding.
- Workloads: HumanEval / GSM8K / MT-Bench. 5,971 per-step traces.
- Define the three oracles precisely (per-request, per-step, per-batch best-K with hindsight).

### 3. The headroom is small (oracle decomposition)  ← core result 1
- Fixed-K curves; best fixed K per workload.
- **Per-request oracle: +2.0%** over best fixed-K → real controllers (UCB −4.9%, ε-greedy
  −6.9%, history −9.3%) lose because exploration costs more than the ceiling.
- **Per-step oracle: +24.8%** ceiling — large, but is it reachable? (→ §4)
- Error taxonomy: 78.7% over-draft / 12.1% under-draft; the win is mostly *stopping early*.

### 4. The per-step headroom is unreachable (signal-predictability audit)  ← core result 2
- Frame as: predict per-position acceptance before verification; simulate the optimal
  stop-policy; report % of the +24.8% ceiling recovered, on held-out generations.
- Cheap logit features (entropy, margin, position, history): **6.8% ± 3.8**; history alone
  ~0. Single-feature AUCs (entropy 0.42, margin 0.57, history 0.67).
- **Learning curve** plateaus by ~2k rows → not data-limited; the ceiling is the *features*.
- **EAGLE hidden state** (norm + random projection): adds **−0.7 pts**; norm AUC 0.484.
  → No pre-verification signal — logit-level or hidden-state — recovers >~7%.
- Conclusion: the bottleneck is the signal, not the headroom or the data.

### 5. At serving batch the headroom collapses (batch sweep)  ← core result 3
- B = 1, 8, 32, 64: best K converges 2/3 → 1/1/1; K≥3 goes net-negative at B≥32.
- Consistent with MagicDec's compute-bound regime for short/medium context.
- This is *why* Nightjar adapts to batch and disables speculation — we give the curve.

### 6. Why load-aware beats signal-aware (synthesis)
- Map the three results onto the design space: per-request adaptation ≈ +2% (not worth the
  exploration); per-step adaptation has headroom but no signal; batch/load adaptation is
  where the structure actually lives.
- Position against Nightjar/SADDLE: our oracle explains their design choices (variable length
  by batch, speculate-or-not gate) that they justify empirically.

### 7. (Optional, stretch) A provably-optimal speculate-or-not gate
- The oracle break-even is exact: speculate iff `E[accepted_len]·c < 1` (expected draft
  payoff exceeds draft cost). Nightjar's gate is a heuristic MAB; derive the optimal-stopping
  gate and give a regret bound. Validate against the traces. (Only if §3–6 land; not the
  backbone.)

### 8. Limitations & honesty
- Short context (≤2k); long-context high-batch (MagicDec regime) may differ — named as
  future work, not glossed.
- Hidden-state probe used norm + 16-dim random projection, not a full-vector learned probe.
- Single target/draft family; cost model c is a proxy for engine-specific verify/draft ratio.

### 9. Related work
- Signal-aware: BanditSpec, SpecDec++, AdaEAGLE, SVIP/entropy.
- Load-aware: Nightjar (2512.22420), SADDLE (HPCA'26), AdaSpec (2503.05096), PEARL (ICLR'25).
- Throughput: MagicDec (2408.11049), OWL (2510.07535), SPIRe, TurboSpec.
- Lossy/relaxed (orthogonal axis): DIVERSED (2604.07622), CoolSD (AAAI'26), LK-Losses.

---

## Assets already in hand (mapping to sections)
- §3: `results/eagle3_8b/{fixedK_by_workload,policies}.csv`, error taxonomy.
- §4: `analyze_perstep_signal_audit.py`, `analyze_perstep_learning_curve.py`,
  `analyze_perstep_hidden_audit.py` → `results/perstep_signal/{audit,hidden_audit}.md`,
  `feature_auc.csv`, `learning_curve.csv`.
- §5: `analyze_eagle3_batch.py` → `results/eagle3_batch/{curves,gap_vs_batch}.csv`,
  `figures/speedup_vs_K_by_batch.png`; 2-page note `note_for_yash.pdf`.
- §7: derivation TBD (no code yet).

## What's missing before submission
1. Wall-clock validation of the cost-model speedups (have it for fixed-K; extend to oracle).
2. A second target/draft pair to show generality (e.g., Qwen3-14B + EAGLE3 — partial data
   exists in `results/eagle3_perstep_qwen3_14b.json`).
3. Prose + figures pass; §7 derivation if pursued.
4. Decision: workshop (backbone only, §1–6+8) vs systems venue (add §7 + 2nd model + wall-clock).
