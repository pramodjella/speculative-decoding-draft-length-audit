"""PER-STEP EAGLE-3 entropy/acceptance capture (draft + target signals).

Read-only hooks (under VLLM_ENABLE_V1_MULTIPROCESSING=0 + enforce_eager=True):
  - Eagle3*ForCausalLM.compute_logits -> per-position DRAFT entropy + top1-top2 margin
  - target ForCausalLM.compute_logits  -> per-position TARGET verification entropy
    (Llama/Qwen2/Qwen3 candidates; small calls only, skips prefill)
  - SpecDecodingStats.observe_draft    -> per-step accept_run + step boundary
Chain drafting (just num_speculative_tokens) is a prefix, so the offline per-step
sim is exact. Output: /root/out/eagle3_perstep_<tag>.json.

Run (Llama, default):  PYTHONIOENCODING=utf-8 modal run modal_eagle3_perstep_capture.py
Run (another pair):    ... --target Qwen/Qwen3-14B --eagle AngelSlim/Qwen3-14B_eagle3 --tag qwen3_14b
"""
import os, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("eagle3-perstep")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

D_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
D_EAGLE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=5400,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, maxk, n, maxtok, tag):
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import importlib, pkgutil
    LOG = {"cur": [], "cur_t": [], "records": [], "n_logits": 0, "n_tlogits": 0,
           "n_observe": 0, "patched": []}

    def add_entropy(logits):
        try:
            lg = logits if torch.is_tensor(logits) else logits[0]
            lg = lg.float().view(-1, lg.shape[-1])
            for row in lg:
                p = torch.softmax(row, dim=-1)
                ent = float(-(p * p.clamp_min(1e-9).log2()).sum())
                t2 = torch.topk(p, 2).values
                LOG["cur"].append((round(ent, 4), round(float(t2[0] - t2[1]), 4)))
            LOG["n_logits"] += 1
        except Exception:
            pass

    def add_target_entropy(logits):
        try:
            lg = logits if torch.is_tensor(logits) else logits[0]
            lg = lg.float().view(-1, lg.shape[-1])
            if lg.shape[0] > maxk + 3:
                return
            for row in lg:
                p = torch.softmax(row, dim=-1)
                LOG["cur_t"].append(round(float(-(p * p.clamp_min(1e-9).log2()).sum()), 4))
            LOG["n_tlogits"] += 1
        except Exception:
            pass

    def wrap(fn, cb):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            if out is not None:
                cb(out)
            return out
        return patched

    # (1) DRAFT: every Eagle3 model class
    import vllm.model_executor.models as MM
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
                obj.compute_logits = wrap(obj.compute_logits, add_entropy)
                LOG["patched"].append(f"{name}.{cn}.compute_logits(draft)")

    # (1b) TARGET: candidate base-model classes (Llama / Qwen2 / Qwen3)
    for modname, clsname in [("llama", "LlamaForCausalLM"),
                             ("qwen3", "Qwen3ForCausalLM"),
                             ("qwen2", "Qwen2ForCausalLM")]:
        try:
            m = importlib.import_module(f"vllm.model_executor.models.{modname}")
            cls = getattr(m, clsname, None)
            if cls and hasattr(cls, "compute_logits"):
                cls.compute_logits = wrap(cls.compute_logits, add_target_entropy)
                LOG["patched"].append(f"{modname}.{clsname}.compute_logits(target)")
        except Exception:
            pass

    # (2) per-step accept_run + boundary
    try:
        import vllm.v1.spec_decode.metrics as SM
        for cn in dir(SM):
            obj = getattr(SM, cn)
            if isinstance(obj, type) and "observe_draft" in dir(obj):
                def _mkod(orig):
                    def patched(self, num_draft_tokens, num_accepted_tokens, *a, **k):
                        out = orig(self, num_draft_tokens, num_accepted_tokens, *a, **k)
                        LOG["n_observe"] += 1
                        LOG["records"].append(
                            {"ent": [e for e, m in LOG["cur"]],
                             "margin": [m for e, m in LOG["cur"]],
                             "ent_t": list(LOG["cur_t"]),
                             "acc": int(num_accepted_tokens),
                             "ndraft": int(num_draft_tokens)})
                        LOG["cur"] = []; LOG["cur_t"] = []
                        return out
                    return patched
                obj.observe_draft = _mkod(obj.observe_draft)
                LOG["patched"].append(f"{cn}.observe_draft")
                break
    except Exception as e:
        print("observe_draft hook err:", repr(e)[:120], flush=True)

    print("PATCHED:", LOG["patched"], flush=True)

    def take(x, m): return list(x)[:m]
    workloads = {
        "humaneval": take(load_dataset("openai/openai_humaneval", split="test")["prompt"], n),
        "gsm8k": take(load_dataset("openai/gsm8k", "main", split="test")["question"], n),
        "mt_bench": take(load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                         .map(lambda x: {"p": x["prompt"][0]})["p"], n),
    }
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    llm = LLM(model=target,
              speculative_config={"method": "eagle3", "model": eagle,
                                  "num_speculative_tokens": maxk},
              gpu_memory_utilization=0.6, max_model_len=2048,
              enforce_eager=True, disable_log_stats=False)

    gens = []
    for w, prompts in workloads.items():
        for p in prompts:
            LOG["records"] = []; LOG["cur"] = []; LOG["cur_t"] = []
            llm.generate([p], sp, use_tqdm=False)
            gens.append({"workload": w, "steps": list(LOG["records"])})

    print(f"\nHOOK fired: draft_logits={LOG['n_logits']} target_logits={LOG['n_tlogits']} "
          f"observe_draft={LOG['n_observe']}", flush=True)
    nsteps = sum(len(g["steps"]) for g in gens)
    print(f"captured gens={len(gens)} steps={nsteps}", flush=True)
    for g in gens[:1]:
        for s in g["steps"][:4]:
            print("  step:", {"ent": s["ent"][:3], "ent_t": s["ent_t"][:3], "acc": s["acc"]}, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    fn = f"/root/out/eagle3_perstep_{tag}.json"
    with open(fn, "w") as f:
        json.dump({"target": target, "eagle_head": eagle, "max_k": maxk,
                   "n_logits": LOG["n_logits"], "n_tlogits": LOG["n_tlogits"],
                   "n_observe": LOG["n_observe"], "gens": gens}, f)
    vol.commit()
    return {"gens": len(gens), "steps": nsteps, "n_logits": LOG["n_logits"],
            "n_tlogits": LOG["n_tlogits"], "n_observe": LOG["n_observe"], "file": fn}


@app.local_entrypoint()
def main(target: str = D_TARGET, eagle: str = D_EAGLE, maxk: int = 7,
         n: int = 8, maxtok: int = 96, tag: str = "target_llama8b"):
    r = run.remote(target, eagle, maxk, n, maxtok, tag)
    print(f"\n===== {tag}: gens={r['gens']} steps={r['steps']}  "
          f"draft={r['n_logits']} target={r['n_tlogits']} observe={r['n_observe']} =====")
    print(f"  saved {r['file']}")
    if r["n_logits"] == 0:
        print("  !! draft hook never fired -> check eagle3 class name match")
    elif r["n_observe"] == 0:
        print("  !! observe_draft never fired -> stats disabled")
    else:
        print("  OK -> download and run src/simulate_eagle_perstep.py / verify_explore_perstep.py")
