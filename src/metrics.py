"""Inference metrics: latency (TTFT / TPOT), throughput, memory, accuracy.

All functions accept plain numbers / lists so they are trivially unit-testable
and reusable across every experiment in the roadmap.
"""
from __future__ import annotations

from typing import List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency_summary(per_request_ttft: List[float], per_token_latencies: List[float]) -> dict:
    """Summarise TTFT and TPOT with mean and tail percentiles."""
    return {
        "ttft_ms_mean": _mean(per_request_ttft),
        "ttft_ms_p95": percentile(per_request_ttft, 95),
        "ttft_ms_p99": percentile(per_request_ttft, 99),
        "tpot_ms_mean": _mean(per_token_latencies),
        "tpot_ms_p99": percentile(per_token_latencies, 99),
    }


def throughput_tokens_per_s(n_tokens: int, wall_seconds: float) -> float:
    return n_tokens / wall_seconds if wall_seconds > 0 else float("nan")


def accepted_tokens_per_step(total_accepted: int, steps: int) -> float:
    """Output tokens emitted per target forward pass (the headline controller metric).

    Each speculative step performs exactly one target verification forward and
    emits ``accepted + 1`` tokens (the accepted draft run plus the bonus/correction
    token). Averaged over a generation this is ``total_accepted / steps + 1``.

    Unlike wall-clock speedup this is deterministic under greedy decoding and
    hardware-independent, so it is stable across seeds/runs and is exactly the
    quantity a draft-length controller optimizes. Returns NaN when no speculative
    step ran (e.g. immediate EOS after prefill).
    """
    if steps <= 0:
        return float("nan")
    return total_accepted / steps + 1.0


def gpu_peak_mem_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return float("nan")


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")
