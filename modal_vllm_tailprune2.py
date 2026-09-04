"""vLLM-NATIVE paired tail-pruning, take 2 — patching the TRUE draft loop.

Diagnosis chain (diag/diag3/diag4/diag5 + pkg dump): the live drafter is
vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.AutoRegressiveSpeculator
(EagleSpeculator subclass); everything under vllm.v1.spec_decode.* is a parallel DEAD path
for this config. The draft loop is `_multi_step_decode` (pure Python:
`for step in range(1, self.num_speculative_steps)`), and `propose` returns the PREALLOCATED
buffer self.draft_tokens[:num_reqs] — always K columns — so an early break is contract-safe.
Unwritten tail columns are overwritten with a repeat of the last drafted token (deterministic
rejection; greedy verification keeps outputs exact regardless).

Arms (all share the identical patched codepath; per-step top-prob hook runs for every arm):
  fixed-k  : break when step >= k          (k tokens drafted)
  cumprob  : break when running product of drafted-token top-probs < thr

Fail-fast gates before benchmarking: (1) patched loop executed; (2) fixed2 arm actually
truncates drafting (realized drafted-lens <= 2). Paired protocol as before: round-robin arms,
rotated order per cycle, per-cycle paired differences; interleaved odd-index prompts; B=1,
greedy, enforce_eager.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune2.py --model llama8b --n 10 --maxtok 128 --cycles 4
Out: /root/out/vllm_tailprune2_{model}.json
"""
import os, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune2")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODELS = {
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    "qwen14b": ("Qwen/Qwen3-14B", "AngelSlim/Qwen3-14B_eagle3"),
}
KMAX = 7


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model_key, n, maxtok, cycles):
    import torch, numpy as np
    import importlib, pkgutil

    TARGET, EAGLE = MODELS[model_key]
    print(f"model={model_key} target={TARGET} head={EAGLE}", flush=True)

    EL = {"arm": ("fixed", KMAX), "top_p": 1.0, "depths": [], "n_msd": 0}

    # ── hook: draft-head compute_logits -> top-1 prob (fires for step 0 too) ── #
    import vllm.model_executor.models as MM
    def wrap_logits(fn):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            try:
                lg = out if torch.is_tensor(out) else out[0]
                row = lg.view(-1, lg.shape[-1])[0].float()
                EL["top_p"] = float(torch.softmax(row, dim=-1).max())
            except Exception:
                pass
            return out
        return patched
    hooked = []
    for _, name, _ in pkgutil.iter_modules(MM.__path__):
        if "eagle" not in name.lower():
            continue
        try:
            mod = importlib.import_module(f"vllm.model_executor.models.{name}")
        except Exception:
            continue
        for cn in dir(mod):
            obj = getattr(mod, cn)
            if isinstance(obj, type) and "eagle3" in cn.lower() and hasattr(obj, "compute_logits"):
                obj.compute_logits = wrap_logits(obj.compute_logits)
                hooked.append(f"{name}.{cn}")
    print("logits-hooked:", hooked, flush=True)

    # ── patch the TRUE draft loop: AutoRegressiveSpeculator._multi_step_decode ── #
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as AS
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        build_slot_mappings_by_layer,
    )
    from vllm.config import CUDAGraphMode

    def multi_step_patched(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        EL["n_msd"] += 1
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        arm_kind, arm_val = EL["arm"]
        cum = EL["top_p"]              # top-prob of draft token 0 (set in _prefill's sample)
        tokens_drafted = 1
        stopped_early = False

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # ---- arm early-break, BEFORE paying this step's draft forward ---- #
            if arm_kind == "fixed" and step >= arm_val:
                stopped_early = True
                break
            if arm_kind == "cum" and cum < arm_val:
                stopped_early = True
                break

            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, positions, batch_desc.num_tokens)
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config)
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens)

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs, batch_desc.num_tokens, attn_metadata,
                    slot_mappings_by_layer, num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode)

            tokens_drafted = step + 1
            cum *= EL["top_p"]

        if stopped_early:
            # overwrite unwritten tail columns -> deterministic rejection at verify
            self.draft_tokens[:num_reqs, tokens_drafted:self.num_speculative_steps] = (
                self.draft_tokens[:num_reqs, tokens_drafted - 1: tokens_drafted])
        EL["depths"].append(tokens_drafted)

    AS.AutoRegressiveSpeculator._multi_step_decode = multi_step_patched
    print("AutoRegressiveSpeculator._multi_step_decode PATCHED", flush=True)

    # ── engine ───────────────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": KMAX},
              gpu_memory_utilization=0.8, max_model_len=2048,
              enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    def take_odd(x, m): return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    workloads = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                 "mt_bench": take_odd(mb, n)}

    def bench_once(prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            out = llm.generate([p], sp, use_tqdm=False)
            ntok += len(out[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    # ── fail-fast gates ──────────────────────────────────────────────────── #
    EL["arm"] = ("fixed", KMAX)
    txt = llm.generate(["What is 17*24? Show your steps."], sp,
                       use_tqdm=False)[0].outputs[0].text
    print(f"SANITY GEN: n_msd={EL['n_msd']} {repr(txt[:110])}", flush=True)
    assert EL["n_msd"] > 0, "patched _multi_step_decode did NOT execute — abort"

    EL["arm"] = ("fixed", 2); EL["depths"] = []
    llm.generate(["Warmup two"], SamplingParams(temperature=0.0, max_tokens=12),
                 use_tqdm=False)
    d2 = EL["depths"]
    print(f"fixed2 smoke: drafted-lens {d2[-6:]}", flush=True)
    assert d2 and max(d2) <= 2, f"fixed2 arm not enforced ({d2[-6:]}) — abort"
    print("GATES PASSED — arms are live", flush=True)

    # ── paired benchmark ─────────────────────────────────────────────────── #
    arms = [("fixed", 2), ("fixed", 3), ("fixed", 4), ("fixed", KMAX),
            ("cum", 0.05), ("cum", 0.2)]
    names = [f"{k}{v}" for k, v in arms]
    tps = {w: {nm: [] for nm in names} for w in workloads}
    depth_log = {nm: [] for nm in names}
    for c in range(cycles):
        order = arms[c % len(arms):] + arms[:c % len(arms)]
        for arm in order:
            nm = f"{arm[0]}{arm[1]}"
            EL["arm"] = arm
            for w, prompts in workloads.items():
                EL["depths"] = []
                t = bench_once(prompts)
                tps[w][nm].append(round(t, 2))
                if EL["depths"]:
                    depth_log[nm].append(round(float(np.mean(EL["depths"])), 2))
            print(f"cycle {c+1}/{cycles} {nm:10s} " +
                  " ".join(f"{w}:{tps[w][nm][-1]}" for w in workloads), flush=True)

    # ── paired analysis ──────────────────────────────────────────────────── #
    summary = {}
    for w in workloads:
        fixed_means = {nm: float(np.mean(tps[w][nm])) for nm in names
                       if nm.startswith("fixed")}
        per_arm = {}
        for nm in names:
            if nm.startswith("fixed"):
                continue
            vs = {}
            for fnm in fixed_means:
                diffs = [100 * (a / b - 1) for a, b in zip(tps[w][nm], tps[w][fnm])]
                vs[fnm] = {"mean_gain_pct": round(float(np.mean(diffs)), 2),
                           "se": round(float(np.std(diffs) / np.sqrt(len(diffs))), 2)}
            strongest = min(vs, key=lambda k: vs[k]["mean_gain_pct"])
            per_arm[nm] = {"vs_each_fixed": vs,
                           "vs_strongest_fixed": {"config": strongest, **vs[strongest]}}
            print(f">>> {w:10s} {nm}: vs strongest fixed ({strongest}) "
                  f"{vs[strongest]['mean_gain_pct']:+.2f}% ±{vs[strongest]['se']:.2f}",
                  flush=True)
        summary[w] = {"fixed_means": {k: round(v, 2) for k, v in fixed_means.items()},
                      "cum_arms": per_arm}
    mean_depths = {nm: (round(float(np.mean(v)), 2) if v else None)
                   for nm, v in depth_log.items()}
    print("mean drafted per arm:", mean_depths, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open(f"/root/out/vllm_tailprune2_{model_key}.json", "w") as f:
        json.dump({"model": model_key, "target": TARGET, "n": n, "maxtok": maxtok,
                   "cycles": cycles, "KMAX": KMAX, "tps": tps,
                   "mean_drafted": mean_depths, "summary": summary}, f, indent=2)
    vol.commit()
    return {w: {nm: summary[w]["cum_arms"][nm]["vs_strongest_fixed"]
                for nm in summary[w]["cum_arms"]} for w in summary}


@app.local_entrypoint()
def main(model: str = "llama8b", n: int = 10, maxtok: int = 128, cycles: int = 4):
    print(f"vllm tailprune2: model={model} n={n} maxtok={maxtok} cycles={cycles}")
    print(json.dumps(run.remote(model, n, maxtok, cycles), indent=2))
