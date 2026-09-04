"""FAIR-BASELINE extensions: BATCH (B>1) and TEMPERATURE (T>0) — the two reviewer gaps.

Same design as modal_vllm_fairbase.py (native K=2/K=3 engines with true verification budgets
+ patched K=7 cum engine, paired round-robin across engines, gated), extended:

  --batch B : prompts submitted in chunks of B (vLLM continuous batching). The cum policy
              becomes BATCH-LEVEL: the hook records per-request top-probs as a vector, the
              loop maintains per-request cumulative chain confidence, and stops the whole
              (synchronized) draft loop when the MEAN confidence < thr. Per-request ragged
              stopping is not possible in the batch-synchronized loop — that limitation is
              part of the finding (adaptive length degrades with batch).
  --temp T  : sampling temperature (seeded per request for variance control). T>0 switches
              verification to stochastic rejection sampling — acceptance dynamics change.

Expected per E1: the cum gain shrinks toward zero as B grows. Measuring the curve converts
the "B=1 only" objection into a scoping result.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_fairbase2.py --model llama8b --batch 4 --tag b4
     PYTHONIOENCODING=utf-8 modal run modal_vllm_fairbase2.py --model llama8b --batch 8 --tag b8
     PYTHONIOENCODING=utf-8 modal run modal_vllm_fairbase2.py --model llama8b --temp 0.8 --tag t08
Out: /root/out/vllm_fairbase2_{model}_{tag}.json
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
app = modal.App("vllm-fairbase2")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODELS = {
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
}
KMAX = 7


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model_key, n, maxtok, cycles, batch, temp, tag):
    import torch, numpy as np
    import importlib, pkgutil

    TARGET, EAGLE = MODELS[model_key]
    native_ks = [2, 3]
    util = 0.26
    print(f"model={model_key} batch={batch} temp={temp} tag={tag}", flush=True)

    EL = {"arm": ("nofire", 0), "top_p_vec": None, "depths": [], "n_msd": 0}

    # ── hook: per-REQUEST top-prob vector ─────────────────────────────────── #
    import vllm.model_executor.models as MM
    def wrap_logits(fn):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            try:
                lg = out if torch.is_tensor(out) else out[0]
                rows = lg.view(-1, lg.shape[-1]).float()
                EL["top_p_vec"] = torch.softmax(rows, dim=-1).max(dim=-1).values.cpu().numpy()
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

    # ── patched loop: batch-level cum (mean per-request chain confidence) ──── #
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as AS
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        build_slot_mappings_by_layer,
    )
    from vllm.config import CUDAGraphMode

    def multi_step_patched(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        import numpy as np
        EL["n_msd"] += 1
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        arm_kind, arm_val = EL["arm"]
        v = EL["top_p_vec"]
        cum_vec = (v[:num_reqs].copy() if v is not None and len(v) >= num_reqs
                   else np.ones(num_reqs))
        tokens_drafted = 1
        stopped_early = False

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            if arm_kind == "cum" and float(cum_vec.mean()) < arm_val:
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
            v = EL["top_p_vec"]
            if v is not None and len(v) >= num_reqs:
                cum_vec *= v[:num_reqs]

        if stopped_early:
            self.draft_tokens[:num_reqs, tokens_drafted:self.num_speculative_steps] = (
                self.draft_tokens[:num_reqs, tokens_drafted - 1: tokens_drafted])
        EL["depths"].append(tokens_drafted)

    AS.AutoRegressiveSpeculator._multi_step_decode = multi_step_patched
    print("_multi_step_decode PATCHED (batch-level cum)", flush=True)

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
    print("engines loaded", flush=True)

    def sp(seed):
        return SamplingParams(temperature=temp, max_tokens=maxtok,
                              seed=(seed if temp > 0 else None))

    def take_odd(x, m): return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    workloads = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                 "mt_bench": take_odd(mb, n)}

    def bench_once(llm, prompts):
        torch.cuda.synchronize(); t0 = time.time(); ntok = 0
        for i in range(0, len(prompts), max(batch, 1)):
            chunk = prompts[i:i + max(batch, 1)]
            outs = llm.generate(chunk, [sp(1000 + i + j) for j in range(len(chunk))],
                                use_tqdm=False)
            ntok += sum(len(o.outputs[0].token_ids) for o in outs)
        torch.cuda.synchronize()
        return ntok / (time.time() - t0)

    arms = [(f"native{k}", engines[f"native{k}"], ("nofire", 0)) for k in native_ks]
    arms += [(f"native{KMAX}", e7, ("nofire", 0)), ("cum0.2", e7, ("cum", 0.2))]
    names = [a[0] for a in arms]

    # gates
    for nm, eng, arm in arms:
        EL["arm"] = arm; EL["depths"] = []
        eng.generate(["Warm up now"] * max(batch, 1),
                     [sp(7 + j) for j in range(max(batch, 1))], use_tqdm=False)
        d = EL["depths"][-4:]
        print(f"gate {nm:9s}: drafted-lens {d}", flush=True)
        if nm.startswith("native"):
            want = int(nm.replace("native", ""))
            assert d and max(d) == want, f"{nm} drafts {d}, expected {want} — abort"
    assert EL["n_msd"] > 0, "patched loop did not execute — abort"
    print("GATES PASSED", flush=True)

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

    summary = {}
    for w in workloads:
        nat_means = {nm: float(np.mean(tps[w][nm])) for nm in names if nm.startswith("native")}
        vs = {}
        for fnm in nat_means:
            diffs = [100 * (a / b - 1) for a, b in zip(tps[w]["cum0.2"], tps[w][fnm])]
            vs[fnm] = {"mean_gain_pct": round(float(np.mean(diffs)), 2),
                       "se": round(float(np.std(diffs) / np.sqrt(len(diffs))), 2)}
        strongest = min(vs, key=lambda k: vs[k]["mean_gain_pct"])
        summary[w] = {"native_means": {k: round(v, 2) for k, v in nat_means.items()},
                      "cum_vs_each": vs,
                      "cum_vs_strongest": {"config": strongest, **vs[strongest]}}
        print(f">>> {w:10s} cum0.2 vs strongest NATIVE ({strongest}) "
              f"{vs[strongest]['mean_gain_pct']:+.2f}% ±{vs[strongest]['se']:.2f}", flush=True)
    mean_depths = {nm: (round(float(np.mean(v)), 2) if v else None)
                   for nm, v in depth_log.items()}
    print("mean drafted per arm:", mean_depths, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open(f"/root/out/vllm_fairbase2_{model_key}_{tag}.json", "w") as f:
        json.dump({"model": model_key, "batch": batch, "temp": temp, "tag": tag,
                   "n": n, "maxtok": maxtok, "cycles": cycles, "tps": tps,
                   "mean_drafted": mean_depths, "summary": summary}, f, indent=2)
    vol.commit()
    return {w: summary[w]["cum_vs_strongest"] for w in summary}


@app.local_entrypoint()
def main(model: str = "llama8b", n: int = 8, maxtok: int = 128, cycles: int = 4,
         batch: int = 1, temp: float = 0.0, tag: str = "x"):
    print(f"fairbase2: model={model} batch={batch} temp={temp} tag={tag}")
    print(json.dumps(run.remote(model, n, maxtok, cycles, batch, temp, tag), indent=2))
