"""FAIR-BASELINE test (Yash Q1): tail-pruning vs NATIVE fixed-K engines that keep their own
verification savings.

The tailprune2 win used an equal-verify design (fixed arms break-then-pad inside a K=7 engine,
so they verify 7 slots). A real fixed-K=2 deployment verifies only 3 positions. This script
answers: does the cum win survive against TRUE native baselines?

Design: load MULTIPLE engines in one process, each with its own num_speculative_tokens:
    E2 = native K=2 engine   (real verify savings)   -> arm "native2"
    E3 = native K=3 engine                            -> arm "native3"
    E7 = K=7 engine                                   -> arms "native7", "cum0.2", "cum0.05"
The patched _multi_step_decode (validated in tailprune2) is class-global; native arms run it
with a never-fires arm so their behavior is stock. All arms round-robin per cycle with rotated
order (paired protocol); paired per-cycle diffs of cum vs each native. Gates: patched-loop
counter, per-arm realized draft lengths (native2 must log 2, native3 log 3), arm separation.

Stability (Yash Q3): run this script on N fresh containers; JSONs are suffixed --tag.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_fairbase.py --model llama8b --tag r1
Out: /root/out/vllm_fairbase_{model}_{tag}.json
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
app = modal.App("vllm-fairbase")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODELS = {
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    "qwen14b": ("Qwen/Qwen3-14B", "AngelSlim/Qwen3-14B_eagle3"),
}
KMAX = 7


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model_key, n, maxtok, cycles, natives, tag):
    import torch, numpy as np
    import importlib, pkgutil

    TARGET, EAGLE = MODELS[model_key]
    native_ks = [int(x) for x in natives.split(",")]
    n_engines = len(native_ks) + 1
    util = round(0.78 / n_engines, 2)
    print(f"model={model_key} natives={native_ks} + K{KMAX}cum  util/engine={util}", flush=True)

    EL = {"arm": ("nofire", 0), "top_p": 1.0, "depths": [], "n_msd": 0}

    # ── hook: draft-head compute_logits -> top-1 prob ─────────────────────── #
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

    # ── patch the true draft loop (validated in tailprune2) ───────────────── #
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
        cum = EL["top_p"]
        tokens_drafted = 1
        stopped_early = False

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
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
            self.draft_tokens[:num_reqs, tokens_drafted:self.num_speculative_steps] = (
                self.draft_tokens[:num_reqs, tokens_drafted - 1: tokens_drafted])
        EL["depths"].append(tokens_drafted)

    AS.AutoRegressiveSpeculator._multi_step_decode = multi_step_patched
    print("_multi_step_decode PATCHED (nofire arms run stock behavior)", flush=True)

    # ── engines ──────────────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    from datasets import load_dataset

    def make_engine(k):
        return LLM(model=TARGET,
                   speculative_config={"method": "eagle3", "model": EAGLE,
                                       "num_speculative_tokens": k},
                   gpu_memory_utilization=util, max_model_len=2048,
                   enforce_eager=True, disable_log_stats=True)

    engines = {}
    for k in native_ks:
        engines[f"native{k}"] = make_engine(k)
        print(f"engine native{k} loaded", flush=True)
    e7 = make_engine(KMAX)
    engines[f"native{KMAX}"] = e7
    print(f"engine K{KMAX} loaded", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {100*(1-free/total):.0f}% used", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)
    def take_odd(x, m): return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    workloads = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                 "mt_bench": take_odd(mb, n)}

    def bench_once(llm, prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            out = llm.generate([p], sp, use_tqdm=False)
            ntok += len(out[0].outputs[0].token_ids)
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    # arms: (name, engine, arm_setting)
    arms = [(f"native{k}", engines[f"native{k}"], ("nofire", 0)) for k in native_ks]
    arms += [(f"native{KMAX}", e7, ("nofire", 0)),
             ("cum0.2", e7, ("cum", 0.2)),
             ("cum0.05", e7, ("cum", 0.05))]
    names = [a[0] for a in arms]

    # ── gates ────────────────────────────────────────────────────────────── #
    for nm, eng, arm in arms:
        EL["arm"] = arm; EL["depths"] = []
        eng.generate(["Warm up now"], SamplingParams(temperature=0.0, max_tokens=12),
                     use_tqdm=False)
        d = EL["depths"][-4:]
        print(f"gate {nm:9s}: drafted-lens {d}", flush=True)
        if nm.startswith("native"):
            want = int(nm.replace("native", ""))
            assert d and max(d) == want, f"{nm} drafts {d}, expected {want} — abort"
    assert EL["n_msd"] > 0, "patched loop did not execute — abort"
    print("GATES PASSED — native engines draft their own K; cum arm live", flush=True)

    # ── paired benchmark across ENGINES ──────────────────────────────────── #
    tps = {w: {nm: [] for nm in names} for w in workloads}
    depth_log = {nm: [] for nm in names}
    for c in range(cycles):
        order = arms[c % len(arms):] + arms[:c % len(arms)]
        for nm, eng, arm in order:
            EL["arm"] = arm
            for w, prompts in workloads.items():
                EL["depths"] = []
                t = bench_once(eng, prompts)
                tps[w][nm].append(round(t, 2))
                if EL["depths"]:
                    depth_log[nm].append(round(float(np.mean(EL["depths"])), 2))
            print(f"cycle {c+1}/{cycles} {nm:9s} " +
                  " ".join(f"{w}:{tps[w][nm][-1]}" for w in workloads), flush=True)

    # ── paired summary: cum arms vs each NATIVE baseline ─────────────────── #
    summary = {}
    for w in workloads:
        nat_means = {nm: float(np.mean(tps[w][nm])) for nm in names if nm.startswith("native")}
        per_arm = {}
        for nm in names:
            if nm.startswith("native"):
                continue
            vs = {}
            for fnm in nat_means:
                diffs = [100 * (a / b - 1) for a, b in zip(tps[w][nm], tps[w][fnm])]
                vs[fnm] = {"mean_gain_pct": round(float(np.mean(diffs)), 2),
                           "se": round(float(np.std(diffs) / np.sqrt(len(diffs))), 2)}
            strongest = min(vs, key=lambda k: vs[k]["mean_gain_pct"])
            per_arm[nm] = {"vs_each_native": vs,
                           "vs_strongest_native": {"config": strongest, **vs[strongest]}}
            print(f">>> {w:10s} {nm}: vs strongest NATIVE ({strongest}) "
                  f"{vs[strongest]['mean_gain_pct']:+.2f}% ±{vs[strongest]['se']:.2f}",
                  flush=True)
        summary[w] = {"native_means": {k: round(v, 2) for k, v in nat_means.items()},
                      "cum_arms": per_arm}
    mean_depths = {nm: (round(float(np.mean(v)), 2) if v else None)
                   for nm, v in depth_log.items()}
    print("mean drafted per arm:", mean_depths, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open(f"/root/out/vllm_fairbase_{model_key}_{tag}.json", "w") as f:
        json.dump({"model": model_key, "target": TARGET, "n": n, "maxtok": maxtok,
                   "cycles": cycles, "natives": native_ks, "KMAX": KMAX, "tag": tag,
                   "tps": tps, "mean_drafted": mean_depths, "summary": summary}, f, indent=2)
    vol.commit()
    return {w: {nm: summary[w]["cum_arms"][nm]["vs_strongest_native"]
                for nm in summary[w]["cum_arms"]} for w in summary}


@app.local_entrypoint()
def main(model: str = "llama8b", n: int = 10, maxtok: int = 128, cycles: int = 4,
         natives: str = "2,3", tag: str = "r1"):
    print(f"fairbase: model={model} natives={natives} cycles={cycles} tag={tag}")
    print(json.dumps(run.remote(model, n, maxtok, cycles, natives, tag), indent=2))
