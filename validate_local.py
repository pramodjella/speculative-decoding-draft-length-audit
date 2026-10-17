"""Step 5 — real-hardware (batch-1) validation of the simulator's CONTENT assumptions.

Scope (honest): the load/batch dimension of the thesis cannot be validated locally
(the custom harness is batch-1, vLLM does not run on Windows, and 8 GB cannot hold
large batches). What this DOES validate on real Qwen models at batch 1 — the content
half of the thesis and the simulator's ContentStream assumptions:

  V1. Draft entropy predicts acceptance      -> negative entropy<->accepted correlation
  V2. Per-step difficulty is autocorrelated   -> positive lag-1 autocorr of accepted
                                                 (the AR(1) premise of ContentStream)
  V3. A content-aware controller (LinUCB) beats the best fixed K at batch 1 on a
      cost-aware speedup computed from the REAL per-step trace + measured latency floors.

Also calibrates the simulator's batch-1 latency floors (M_T, M_D) against measured
forward times, so r = T_draft/T_target is grounded.

Run:  python validate_local.py            (Qwen2.5-1.5B + 0.5B, cached)
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from serve.physical_runner import PhysicalSpeculativeRunner
from controllers import EntropyThreshold, LinUCBController

PROMPTS = [
    "Write a python function to compute the nth Fibonacci number.",
    "def quicksort(arr):",
    "Write a function to check if a string is a palindrome.",
    "Implement binary search in Python.",
    "What is 17 times 24? Show your work.",
    "Solve: a train travels 60 miles in 1.5 hours, what is its speed?",
    "If a shop sells 3 apples for $2, how much for 12 apples?",
    "Explain why the sky is blue in two sentences.",
    "Summarize the theory of relativity briefly.",
    "List three benefits of regular exercise.",
    "Describe the water cycle in simple terms.",
    "Write a haiku about autumn.",
    "Translate to French: Good morning, how are you?",
    "What are the primary colors?",
    "Give two tips for writing clean code.",
]


def load(model_id, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    use_cuda = torch.cuda.is_available()
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None).eval()
    return m, tok


def measure_floor(model, tok, n=15):
    """Mean single-token forward latency (ms) at batch 1, warmup excluded."""
    dev = next(model.parameters()).device
    ids = tok.encode("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(dev)
    with torch.no_grad():
        for _ in range(3):
            model(ids, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(ids, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def step_speedup_b1(accepted, K, T_t, T_d):
    """Batch-1 modeled speedup of one spec step from real trace + measured floors."""
    return (accepted + 1) * T_t / (K * T_d + T_t)


def run_policy(target, draft, tok, ctrl, mnt):
    """Return per-step list of (K, accepted, entropy) pooled over all prompts."""
    runner = PhysicalSpeculativeRunner(target, draft, tok, controller=ctrl)
    steps = []
    for p in PROMPTS:
        res = runner.generate(p, max_new_tokens=mnt)
        for s in res["step_logs"]:
            steps.append((s["K"], s["accepted"], s["entropy"]))
    return steps


def lag1_autocorr_per_prompt(target, draft, tok, ctrl, mnt):
    """Mean lag-1 autocorrelation of the accepted-count sequence within each prompt."""
    runner = PhysicalSpeculativeRunner(target, draft, tok, controller=ctrl)
    accs = []
    for p in PROMPTS:
        seq = [s["accepted"] for s in runner.generate(p, max_new_tokens=mnt)["step_logs"]]
        if len(seq) >= 5 and np.std(seq) > 1e-6:
            a = np.array(seq, float)
            accs.append(np.corrcoef(a[:-1], a[1:])[0, 1])
    return float(np.nanmean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--mnt", type=int, default=48)
    args = ap.parse_args()

    print(f"Loading {args.target} (target) + {args.draft} (draft)...")
    target, tok = load(args.target)
    draft, _ = load(args.draft)

    # ── Calibrate latency floors ──
    T_t = measure_floor(target, tok)
    T_d = measure_floor(draft, tok)
    r = T_d / T_t
    print(f"\n[CALIBRATION] T_target={T_t:.1f}ms  T_draft={T_d:.1f}ms  r=T_d/T_t={r:.2f}")
    print(f"  simulator floors: M_T=25, M_D=5 -> r=0.20. Measured r={r:.2f} "
          f"({'grounded' if 0.1 <= r <= 0.6 else 'CHECK'}).")

    # ── V1 + V2: entropy probe (fixed-8 drafting with entropy logged) ──
    probe = EntropyThreshold(tau=999.0, max_len=8)   # never stops early -> fixed_8 + entropy
    steps = run_policy(target, draft, tok, probe, args.mnt)
    K_arr = np.array([s[0] for s in steps]); acc = np.array([s[1] for s in steps])
    ent = np.array([s[2] for s in steps])
    mask = ent > 0
    corr = np.corrcoef(ent[mask], acc[mask])[0, 1] if mask.sum() > 2 else float("nan")
    print(f"\n[V1 signal validity] entropy<->accepted correlation = {corr:+.3f} "
          f"(expect NEGATIVE: higher entropy -> fewer accepted)  [{steps.__len__()} steps]")

    ac1 = lag1_autocorr_per_prompt(target, draft, tok, probe, args.mnt)
    print(f"[V2 autocorrelation] mean lag-1 autocorr of accepted = {ac1:+.3f} "
          f"(expect POSITIVE -> validates ContentStream AR(1) premise)")

    # ── V3: content-aware controller vs fixed K on real-trace batch-1 speedup ──
    print(f"\n[V3 controller vs fixed K]  cost-aware batch-1 speedup (real trace + measured floors)")
    policies = {
        "fixed_1": 1, "fixed_2": 2, "fixed_4": 4, "fixed_8": 8,
        "linucb(content)": LinUCBController(arms=(1, 4, 8), alpha=1.0),
    }
    scores = {}
    for name, ctrl in policies.items():
        st = run_policy(target, draft, tok, ctrl, args.mnt)
        spd = np.mean([step_speedup_b1(a, k, T_t, T_d) for (k, a, _e) in st])
        aps = np.mean([a for (_k, a, _e) in st]) + 1.0
        scores[name] = spd
        print(f"  {name:16s} speedup_b1={spd:.3f}   accepted_tokens/step={aps:.2f}")
    best_fixed = max(v for k, v in scores.items() if k.startswith("fixed"))
    lin = scores["linucb(content)"]
    gain = (lin / best_fixed - 1) * 100
    print(f"\n  best_fixed={best_fixed:.3f}  linucb(content)={lin:.3f}  gain={gain:+.1f}%")
    print(f"  => content-aware controller {'BEATS' if gain > 0 else 'does NOT beat'} "
          f"best fixed K at batch 1 on real models.")

    print("\n[SCOPE] Load/batch dimension not testable locally (batch-1 harness, no Windows "
          "vLLM). The combined load+content result stays simulator-backed; a cloud vLLM run "
          "is the future real-hardware test of the load axis.")


if __name__ == "__main__":
    main()
