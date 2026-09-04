# Pre-write audit (per Yash's "audit first, then freeze")

## Audit status — ALL ITEMS CLOSED (2026-07-03)

| # | Yash's ask | Status | Where |
|---|---|---|---|
| A | Reproducibility + code QC | ✅ Code audit of all pipeline scripts; 1 real bug found (wall-clock (A) capture pairing), fixed, rerun — headline REINFORCED | §5 below |
| 1 | Lock & audit the decomposition | ✅ Independent re-derivation (fresh code, no shared imports): all 5 models match <0.1pp; permutation + measured-cost as standing controls | §1, §2 |
| 2 | Rule out draft-head confound | ✅ Confound CONFIRMED real (DeepSeek head much weaker); reasoning claim narrowed to the head, stated as limitation | §3 |
| 3 | Scope claims to what was tested | ✅ "irreducible from any draft-side signal"; wall-clock scoped to early-stopping; deeper-drafting named untested | §4 |
| 4 | Position precisely vs SpecDecode-Bench | ✅ Credit their verify-dominance/gap/variance; claim only our delta | §4 |

Independent checks run before committing the headline. Scripts: `audit_decomposition.py`,
`audit_probe_recovery.py` (both written from scratch, no shared imports with the analysis module).

## 1. Decomposition oracle/fixed rungs — CONFIRMED (independent)

Re-derived best-fixed-K, per-step oracle, and oracle ceiling % with fresh code, same cost model
(C=0.15) and block key. All 5 models match `bayes_ceiling.json` to <0.1pp:

| Model | best K | fixed | oracle | ceiling% | ref ceiling% |
|---|---:|---:|---:|---:|---:|
| Llama-8B | 2 | 1.2633 | 1.4915 | 18.1 | 18.1 |
| Qwen3-14B | 2 | 1.2712 | 1.4288 | 12.4 | 12.4 |
| DeepSeek | 1 | 1.0595 | 1.1073 | 4.5 | 4.5 |
| Llama-8B (reasoning) | 2 | 1.4370 | 1.7709 | 23.2 | 23.2 |
| DeepSeek (reasoning) | 1 | 1.1094 | 1.2040 | 8.5 | 8.5 |

→ The oracle/irreducible math is not buggy.

## 2. Probe recovery — SIGN robust, MAGNITUDE is a range

Independent probe (5-fold vs 8, PCA-30 vs 50, seed 7 vs 42, simpler policy):

| Model | ref recovery | independent recovery |
|---|---:|---:|
| Llama-8B | +18.9% | +12.5% |
| Qwen3-14B | +19.1% | +14.7% |
| DeepSeek | +0.0% | +0.5% |

→ Instruction models clearly positive, reasoning ~0 (robust). But magnitude is
implementation-sensitive: report **+12–19% of the oracle (≈ +2.3–3.4% net over tuned K)** as a
range, not the point +18.9%. Standing controls: permutation (shuffled → −4.9%), measured-cost
grounding (C=0.072 → +17.3%).

## 3. Draft-head confound — CONFIRMED; reasoning claim must be NARROWED

Yash's concern was correct. Base acceptance strength (mean accepted tokens/step):

| Model | accept_len | p(accept) at pos-0 |
|---|---:|---:|
| Llama-8B | 1.92 | 0.426 |
| DeepSeek | **1.33** | **0.218** |
| Llama-8B (reasoning) | 2.40 | 0.547 |
| DeepSeek (reasoning) | **1.47** | **0.276** |

The DeepSeek-R1-Distill EAGLE head is substantially weaker than Llama's (accepts its first
drafted token ~22% vs ~43%). Its small oracle ceiling (+4.5%) and 0% recovery therefore
**cannot be cleanly attributed to "reasoning model"** — a weaker head produces a small ceiling
and little detectable signal regardless. Even the same-task disambiguation is confounded
(accept_len 2.40 vs 1.47 on identical math).

**Action:** narrow the claim from "reasoning models have no exploitable signal" to
"**the DeepSeek-R1-Distill head shows no recoverable signal; we cannot separate a reasoning-model
property from a weaker draft head.**" A strong reasoning-model EAGLE head would be needed to
separate them (none currently available → stated as a limitation).

## 4. Wording fixes (per Yash)

- "irreducible" → "**irreducible from any draft-side signal**" (we test draft-side signals; a
  privileged post-verification oracle is a different quantity).
- Scope the wall-clock claim to **early-stopping (shortening)**. Our result actually points toward
  drafting *deeper* on easy positions, which we have NOT tested end-to-end → say so, don't claim
  "no adaptive draft length pays."
- Credit SpecDecode-Bench for verification-dominance, the oracle gap, and acceptance variance;
  claim only our delta: the reducible/irreducible split, hidden-state localization, wall-clock
  non-realizability.

## 5. CODE audit (pipeline QC, per Yash item A) — one real bug found

Systematic review of every pipeline script (capture hooks, label pairing, bench methodology):

**BUG FOUND — controller (A) capture pairing (`modal_eagle_wallclock.py`).** Seeds and
accept-lengths were accumulated in global flat lists across prompts and paired by index with
`min()` truncation. Each prompt leaves exactly ONE dangling seed (the final drafted tree is never
verified) — empirically confirmed: 307−289=18 = n_prompts (n=6 run); 693−663=30 = n_prompts
(n=10 run). So pairing drifted by one per preceding prompt → ~all pairs after prompt 1 mispaired
→ the probe was trained on garbage. **Consequences:** the sub-claim "the seed hidden state does
not predict chain length (corr ≈ 0.05)" is UNSUPPORTED, and controller (A)'s bench is
uninformative. FIXED (per-prompt pairing, dangling seed dropped) and RERUN.

**Rerun outcome (fix validated end-to-end):** corrected pairing → seed probe corr = **0.26**
(the "no signal" was indeed the bug), yet the controller **still loses**: −5.7% / −2.4% / −5.5%
vs best fixed depth (d=6/7/7). So (A) now agrees with (B) — two independent controllers, both
with genuine signal, both lose to a tuned deep fixed draft. The headline is reinforced.

**Why the headline is immune.** Controller (B) — the in-loop within-step probe, which carries the
wall-clock conclusion — flushes `cur_levels` at the start of each draft call and labels them at the
immediately following verification: per-step pairing, no cross-prompt drift possible. Its AUC 0.796
(vs 0.484 for a random projection) independently evidences correct pairing. The E4 offline capture
(`modal_eagle3_hidden_full_capture.py`) likewise flushes per step inside `observe_draft` and is
permutation-verified. Label semantics in (B) checked for off-by-one: loop iteration i produces the
hidden of draft level i, so `accepted = (level < accept_length)` is correct.

**Flags (not bugs, stated as limitations):**
- (B)'s per-level feature is the MEAN hidden across the top-k tree branches at that level — an
  approximation of the accepted path's hidden state.
- Bench timing is single-pass per configuration (no repeats/error bars). Run-to-run noise is
  ~1–2%; (B)'s −5–6% exceeds it, but sub-2% differences (e.g. −0.3%) should be read as ties.
- ~0.8% of E4 capture rows lacked a hidden vector and were dropped (benign, logged).
- `ent_t` capture uses a shape heuristic (`rows ≤ maxk+3`); `ent_t` is not used in any headline.

## Frozen headline (post-audit)

Against a tuned fixed draft length: ~80% of the per-step oracle is irreducible from any draft-side
signal; the reducible ~1/5 is localized to the hidden state for the two instruction models
(+2.3–3.4% net, offline) and is absent for the one weaker reasoning head (confound-limited);
and even a real-signal in-loop early-stop probe does not beat a tuned deep draft in wall-clock
(−5–6%). Deployable answer: tuned fixed (deep) draft length.
