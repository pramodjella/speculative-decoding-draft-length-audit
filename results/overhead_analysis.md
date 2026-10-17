# Speculative-decode overhead analysis: launch vs forward vs CPU↔GPU transition

**Question:** of the per-step cost in the custom (eager PyTorch) speculative loop,
how much is (1) kernel-launch overhead, (2) forward-pass work, (3) CPU↔GPU
transitions — and would a CUDA C++ rewrite help?

## Instruments (all in repo)
- `scratch/isolate_overhead.py` — real models (Qwen2.5-0.5B), isolates per-forward
  cost, K-sequential vs 1-batched, `.item()` sync cost.
- `overhead_microbench.cu` + `modal_microbench.py` — CUDA C++ microbench on A100:
  isolates launch / compute / D2H via a pure-launch floor, sequential kernels,
  CUDA-graph replay, and a per-step D2H sync.
- `spec_decode_compare.py` — custom harness vs HF built-in assisted generation.

## Findings

### Real-model diagnostic (Qwen2.5-0.5B, RTX 5070 laptop, bf16)
- single-token forward ≈ **32.8 ms** (vs a ~3 ms memory floor) → ~10× overhead from
  launch + Python dispatch over hundreds of tiny kernels.
- **Tseq / Tbat = 3.36×** (K=4): 4 sequential draft forwards ≈ K× one forward over
  K tokens → the cost is **re-reading the draft weights once per token** (intrinsic
  to autoregressive drafting), not the compute of K tokens.
- `.item()` D2H sync ≈ **11%** of the sequential loop (removable in Python).
- torch profiler returned 0 device-time on this Win/CUDA-13 build (tooling glitch) —
  hence the CUDA microbench below.

### CUDA C++ microbench (A100-40GB, one kernel per "forward", K=4, S=22)
| | weight=1 GB (≈0.5B) | weight=14 GB (≈7B) |
|---|---|---|
| compute/memory floor (CUDA graph C) | 99.1% | 99.9% |
| kernel-launch overhead (B−C) | 11.4% | 0.5% |
| CPU↔GPU transition (D−B) | ~0 (noise) | ~0 (noise) |
| launch cost per kernel | 4.3 µs | 5.6 µs |

## Answer
1. **CPU↔GPU transitions: negligible.** Weights are resident (never transferred);
   the `.item()` D2H is a 4-byte sub-µs copy. Premise about per-step weight
   transfers does not apply.
2. **Kernel-launch: small per kernel (4–6 µs)** and a shrinking fraction as work
   grows; CUDA graphs remove it. BUT a real forward is *hundreds* of kernels, so
   raw launch + **PyTorch Python dispatch** (~10–50 µs/op, CPU-bound) accumulate —
   this is the dominant *eager-mode* overhead (the 32 ms vs 3 ms gap). CUDA graphs
   collapse both.
3. **Forward pass dominates (~99%)**, and within it the per-token weight re-read
   (memory traffic) is intrinsic.

## Conclusion — do NOT hand-write a CUDA C++ engine
- Recovering launch + Python-dispatch overhead = CUDA graphs + fused kernels →
  **already in vLLM** (which is why vLLM gave 1.44× on Qwen-7B where the eager loop
  was <1×). A custom CUDA engine just re-implements vLLM.
- The per-token weight re-read (`Tseq/Tbat≈K`) and the memory-bound floor are
  **not** beatable by better kernels — only by the algorithm: tree/batched
  drafting, a larger target/draft cost ratio, or spending draft forwards only where
  they'll be accepted (the adaptive-draft-length thesis).
- Leverage: (1) use vLLM for absolute speedup, (2) custom harness for the controller
  science (relative comparison), (3) attack intrinsic cost algorithmically.
