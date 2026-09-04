"""PER-STEP EAGLE-3 capture WITH draft hidden-state features (Gate 2 of the signal hunt).

Extends modal_eagle3_perstep_capture.py: in addition to per-position draft entropy +
top1 margin, target entropy, and per-step accept_run, it captures cheap features of the
EAGLE head's HIDDEN STATE at each draft position — the one untested signal class.

The hidden state is the INPUT to the draft's compute_logits (per-position head state).
We store a low-dim summary so the file stays small but a classifier can still find
geometry: L2 norm + a fixed seeded 16-dim random projection. The offline audit
(analyze_perstep_signal_audit.py) then adds these to the feature set and re-checks how
much of the per-step oracle ceiling becomes recoverable.

Writes a NEW file (eagle3_perstep_hidden_<tag>.json) — does NOT touch the existing
eagle3_perstep_target_llama8b.json that the deliverable analysis reads.

Run (Llama, default):  PYTHONIOENCODING=utf-8 modal run modal_eagle3_hidden_capture.py
"""
import os, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("eagle3-hidden")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

D_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
D_EAGLE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
HDIM = 16          # random-projection dimension stored per position


@app.function(image=image, gpu="H100", timeout=5400,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, maxk, n, maxtok, tag):
    import torch
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import importlib, pkgutil
    LOG = {"cur": [], "cur_t": [], "cur_h": [], "records": [],
           "n_logits": 0, "n_tlogits": 0, "n_observe": 0, "n_hidden": 0, "patched": []}
    RP = {"mat": None}     # lazy fixed random projection [hidden_dim, HDIM]

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

    def add_hidden(hidden):
        """Per-position [L2 norm, *16-dim random projection] from the draft head state."""
        try:
            h = hidden if torch.is_tensor(hidden) else hidden[0]
            h = h.float().view(-1, h.shape[-1])           # [pos, hidden_dim]
            if RP["mat"] is None:
                g = torch.Generator(device=h.device).manual_seed(0)
                RP["mat"] = torch.randn(h.shape[-1], HDIM, generator=g,
                                        device=h.device) / (h.shape[-1] ** 0.5)
            proj = h @ RP["mat"]                            # [pos, HDIM]
            norms = h.norm(dim=-1)                          # [pos]
            for i in range(h.shape[0]):
                LOG["cur_h"].append([round(float(norms[i]), 4)] +
                                    [round(float(x), 4) for x in proj[i]])
            LOG["n_hidden"] += 1
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

    def wrap_draft(fn):
        """Draft compute_logits: capture INPUT hidden state + OUTPUT logits, aligned."""
        def patched(self, *a, **k):
            hid = a[0] if a else k.get("hidden_states")
            if hid is not None:
                add_hidden(hid)
            out = fn(self, *a, **k)
            if out is not None:
                add_entropy(out)
            return out
        return patched

    # (1) DRAFT: every Eagle3 model class (hidden + logits)
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
                obj.compute_logits = wrap_draft(obj.compute_logits)
                LOG["patched"].append(f"{name}.{cn}.compute_logits(draft+hidden)")

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

    # (2) per-step accept_run + boundary (bundles ent / margin / ent_t / hidden)
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
                             "h": list(LOG["cur_h"]),
                             "acc": int(num_accepted_tokens),
                             "ndraft": int(num_draft_tokens)})
                        LOG["cur"] = []; LOG["cur_t"] = []; LOG["cur_h"] = []
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
            LOG["records"] = []; LOG["cur"] = []; LOG["cur_t"] = []; LOG["cur_h"] = []
            llm.generate([p], sp, use_tqdm=False)
            gens.append({"workload": w, "steps": list(LOG["records"])})

    print(f"\nHOOK fired: draft_logits={LOG['n_logits']} hidden={LOG['n_hidden']} "
          f"target_logits={LOG['n_tlogits']} observe_draft={LOG['n_observe']}", flush=True)
    nsteps = sum(len(g["steps"]) for g in gens)
    print(f"captured gens={len(gens)} steps={nsteps}", flush=True)
    for g in gens[:1]:
        for s in g["steps"][:3]:
            print("  step:", {"ent": s["ent"][:2], "h0": (s["h"][0][:3] if s["h"] else None),
                              "acc": s["acc"]}, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    fn = f"/root/out/eagle3_perstep_hidden_{tag}.json"
    with open(fn, "w") as f:
        json.dump({"target": target, "eagle_head": eagle, "max_k": maxk, "hdim": HDIM,
                   "n_logits": LOG["n_logits"], "n_hidden": LOG["n_hidden"],
                   "n_tlogits": LOG["n_tlogits"], "n_observe": LOG["n_observe"],
                   "gens": gens}, f)
    vol.commit()
    return {"gens": len(gens), "steps": nsteps, "n_logits": LOG["n_logits"],
            "n_hidden": LOG["n_hidden"], "n_tlogits": LOG["n_tlogits"],
            "n_observe": LOG["n_observe"], "file": fn}


@app.local_entrypoint()
def main(target: str = D_TARGET, eagle: str = D_EAGLE, maxk: int = 7,
         n: int = 30, maxtok: int = 128, tag: str = "llama8b"):
    r = run.remote(target, eagle, maxk, n, maxtok, tag)
    print(f"\n===== {tag}: gens={r['gens']} steps={r['steps']}  draft={r['n_logits']} "
          f"hidden={r['n_hidden']} target={r['n_tlogits']} observe={r['n_observe']} =====")
    print(f"  saved {r['file']}")
    if r["n_hidden"] == 0:
        print("  !! hidden hook never fired -> compute_logits input was not the hidden state")
    elif r["n_observe"] == 0:
        print("  !! observe_draft never fired -> stats disabled")
    else:
        print("  OK -> modal volume get spec-dec-m5-results "
              f"eagle3_perstep_hidden_{tag}.json results/ ; then extend the audit")
