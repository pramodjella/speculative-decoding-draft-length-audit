"""Load-aware speculative-decoding simulator (the thesis instrument).

The existing `simulator.py` is batch-1 only. This module adds the **serving-load
dimension** so that the optimal draft length depends on BOTH:
  - content  (per-step difficulty -> draft entropy -> acceptance profile), and
  - load     (batch size B -> memory-bound vs compute-bound regime).

That joint dependence is the gap the literature leaves open (see novelty_memo.md):
content-aware methods (TapOut, SpecKV, AdaEAGLE) are load-blind; load-aware methods
(Nightjar, SpecuStream, BASS) are content-blind. A controller seeing both should beat
either alone across the batch/load curve.

Latency model (roofline). A decode step over a batch of B sequences:
  draft phase  : K sequential draft forwards, each over B seqs
                 t_draft(B)   = max(M_d, C_d * B)
                 draft_phase  = K * t_draft(B)
  verify phase : ONE target forward over B seqs x (K+1) positions
                 t_verify(B,K)= max(M_t, C_t * B * (K+1))
  spec_step    = draft_phase + t_verify(B,K)

Baseline (no spec), one token over B seqs:  ar_step(B) = max(M_t, C_t * B)

Per-sequence speedup of a step that emits (accepted+1) tokens:
  speedup = (accepted + 1) * ar_step(B) / spec_step

Regime behaviour this produces (verified in self-test):
  - Low B  : memory-bound, t_verify ~ M_t regardless of K -> long K nearly free ->
             optimal K is large, big speedup.
  - High B : compute-bound, t_verify ~ C_t*B*(K+1) grows with K -> long K wasteful
             unless acceptance is high -> optimal K small, speculation can drop <1x.
"""
import math
import numpy as np

# Roofline constants (ms). Tuned so the memory/compute crossover sits near B~8 at K=4,
# matching the batch-1 floors of the legacy simulator (T_TARGET=25, T_DRAFT=5).
M_T = 25.0    # target memory floor
C_T = 0.6     # target compute slope (per seq, per verified position)
M_D = 5.0     # draft memory floor (=> batch-1 r = M_D/M_T = 0.20)
C_D = 0.12    # draft compute slope (per seq)

# Difficulty-persistence default for ContentStream (overridable for sensitivity probes).
RHO_DEFAULT = 0.92

# Per-draft-index conditional acceptance (reused from legacy simulator).
ACCEPT_PROBS = {
    "humaneval":  [0.95, 0.92, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65],
    "gsm8k":      [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15],
    "mt_bench":   [0.80, 0.70, 0.55, 0.45, 0.35, 0.20, 0.10, 0.05],
    "spec_bench": [0.85, 0.78, 0.70, 0.62, 0.52, 0.42, 0.32, 0.22],
}
MAX_LEN = 8


def t_verify(B: int, K: int) -> float:
    return max(M_T, C_T * B * (K + 1))


def t_draft_phase(B: int, K: int) -> float:
    return K * max(M_D, C_D * B)


def ar_step(B: int) -> float:
    return max(M_T, C_T * B)


def spec_step_latency(B: int, K: int) -> float:
    return t_draft_phase(B, K) + t_verify(B, K)


def step_speedup(B: int, K: int, accepted: int) -> float:
    """Per-sequence wall-clock speedup of one speculative step vs plain AR decode."""
    return (accepted + 1) * ar_step(B) / spec_step_latency(B, K)


def accepted_given_K(accept_booleans, K: int) -> int:
    """Leading run of accepted draft tokens, capped at K."""
    n = 0
    for i in range(K):
        if accept_booleans[i]:
            n += 1
        else:
            break
    return n


def _step_from_z(probs, z, rng):
    entropies, accept_booleans = [], []
    for i in range(MAX_LEN):
        h = max(0.05, -math.log2(probs[i]) * z + rng.normal(0.0, 0.05))
        entropies.append(h)
        p_accept = min(0.99, max(0.01, 2 ** (-h)))
        accept_booleans.append(rng.uniform() < p_accept)
    return entropies, accept_booleans


def draw_step(workload: str, rng: np.random.Generator):
    """Sample one step's content with i.i.d. difficulty (legacy / self-test only)."""
    z_t = rng.lognormal(mean=0.0, sigma=0.25)
    return _step_from_z(ACCEPT_PROBS[workload], z_t, rng)


class ContentStream:
    """Per-episode content generator with AUTOCORRELATED step difficulty.

    The project's premise is within-stream structure: easy ("boilerplate") spans and
    hard spans persist over consecutive steps. We model log-difficulty as an AR(1)
    process, so recent acceptance history and the pre-draft entropy become *predictive*
    of the current span -- which is exactly what a content-aware controller exploits.
    With i.i.d. difficulty (legacy draw_step) content signals carry no information and
    no content-aware method can help; autocorrelation is what makes the problem real.

    rho   : difficulty persistence (0=i.i.d., ->1 = long stable spans)
    sigma : innovation scale (within-stream difficulty spread)
    """

    def __init__(self, workload: str, rho: float = None, sigma: float = 0.40):
        self.probs = ACCEPT_PROBS[workload]
        self.rho = RHO_DEFAULT if rho is None else rho
        self.sigma = sigma
        self.logz = 0.0

    def step(self, rng: np.random.Generator):
        self.logz = self.rho * self.logz + self.sigma * rng.normal()
        z = math.exp(self.logz)
        return _step_from_z(self.probs, z, rng)


def oracle_K(B: int, accept_booleans, arms) -> int:
    """Full-knowledge per-step optimum: the arm maximizing realized step speedup."""
    best_k, best_s = arms[0], -1.0
    for k in arms:
        a = accepted_given_K(accept_booleans, k)
        s = step_speedup(B, k, a)
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def accepted_tokens_per_step(total_accepted: int, steps: int) -> float:
    """Headline metric (matches src/metrics.py): emitted tokens per target step."""
    return (total_accepted / steps + 1.0) if steps > 0 else float("nan")


if __name__ == "__main__":
    # Self-test: confirm the optimal fixed K shifts with batch size B, and that
    # speculation drops below 1x at high B with long K (the regime that motivates
    # load-aware control). Uses the easy 'humaneval' acceptance profile, accept-all.
    arms = [1, 2, 4, 8]
    print("Optimal fixed-K vs batch (humaneval, expected-acceptance):")
    for B in [1, 4, 8, 16, 32, 64]:
        # expected accepted at full acceptance profile (no early reject) per K
        row = {}
        for K in arms:
            # expected accepted = sum of cumulative accept probs
            p = ACCEPT_PROBS["humaneval"]
            exp_acc = 0.0
            cum = 1.0
            for i in range(K):
                cum *= p[i]
                exp_acc += cum
            row[K] = step_speedup(B, K, round(exp_acc))
        best = max(row, key=row.get)
        pretty = "  ".join(f"K={k}:{row[k]:.2f}x" for k in arms)
        print(f"  B={B:3d}  ->  best K={best}   {pretty}")
