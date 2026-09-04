"""FULL hidden-state capture for the strongest per-step signal test.

Yash's critique of the existing probe (modal_eagle3_hidden_capture.py):
  "A random projection plus norm is a weak test ... A small trained probe on
   the full hidden state would make 'no signal' much harder to argue with."

Fix: capture the COMPLETE draft hidden state (4096-dim Llama, 2048-dim Qwen3-14B)
at every draft position, store as float16 Parquet (~30–80 MB).  A proper
logistic regression + MLP probe is then trained offline.

Models:
  llama8b   : Llama-3.1-8B-Instruct  + yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  qwen14b   : Qwen3-14B              + AngelSlim/Qwen3-14B_eagle3
  deepseek  : DeepSeek-R1-Distill-Llama-8B + yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B

Run:
  modal run modal_eagle3_hidden_full_capture.py                     # llama8b
  modal run modal_eagle3_hidden_full_capture.py --model qwen14b
  modal run modal_eagle3_hidden_full_capture.py --model deepseek

Outputs: eagle3_hidden_full/hidden_full_{model}.parquet + _meta.json
Download: modal volume get spec-dec-m5-results eagle3_hidden_full/ results/eagle3_hidden_full/
"""
import os, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "pandas", "pyarrow")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("eagle3-hidden-full")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODEL_REGISTRY = {
    "llama8b": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    ),
    "qwen14b": (
        "Qwen/Qwen3-14B",
        "AngelSlim/Qwen3-14B_eagle3",
    ),
    "deepseek": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    ),
}


@app.function(image=image, gpu="H100", timeout=7200,
              volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(target, eagle, maxk, n, maxtok, model_key, wl="default"):
    import torch
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    import importlib, pkgutil

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    import vllm; print("vllm", vllm.__version__)
    print(f">>> model={model_key}  target={target}")

    # ── hook state ──────────────────────────────────────────────────────── #
    # Same pattern as modal_eagle3_hidden_capture.py, but cur_h stores the
    # FULL float16 hidden tensor per position (not norm + 16-dim RP).
    LOG = {
        "cur_ent": [], "cur_margin": [], "cur_ent_t": [],
        "cur_h":   [],   # list of 1-D float16 CPU tensors, one per position
        "records": [],
        "n_logits": 0, "n_tlogits": 0, "n_hidden": 0, "n_observe": 0,
        "hidden_dim": None, "patched": [],
    }

    def add_draft_logits(logits):
        try:
            lg = logits if torch.is_tensor(logits) else logits[0]
            lg = lg.float().view(-1, lg.shape[-1])
            for row in lg:
                p   = torch.softmax(row, dim=-1)
                ent = float(-(p * p.clamp_min(1e-9).log2()).sum())
                t2  = torch.topk(p, 2).values
                LOG["cur_ent"].append(round(ent, 4))
                LOG["cur_margin"].append(round(float(t2[0] - t2[1]), 4))
            LOG["n_logits"] += 1
        except Exception:
            pass

    def add_hidden(hidden):
        """Store complete float16 hidden-state vector per draft position on CPU."""
        try:
            h = hidden if torch.is_tensor(hidden) else hidden[0]
            h = h.view(-1, h.shape[-1]).detach().to(torch.float16).cpu()  # [pos, dim]
            if LOG["hidden_dim"] is None:
                LOG["hidden_dim"] = h.shape[-1]
                print(f"  hidden_dim={h.shape[-1]}", flush=True)
            for i in range(h.shape[0]):
                LOG["cur_h"].append(h[i])   # 1-D tensor [hidden_dim], float16
            LOG["n_hidden"] += 1
        except Exception as e:
            print(f"  [hidden hook]: {e}", flush=True)

    def add_target_entropy(logits):
        try:
            lg = logits if torch.is_tensor(logits) else logits[0]
            lg = lg.float().view(-1, lg.shape[-1])
            if lg.shape[0] > maxk + 3:
                return
            for row in lg:
                p = torch.softmax(row, dim=-1)
                LOG["cur_ent_t"].append(
                    round(float(-(p * p.clamp_min(1e-9).log2()).sum()), 4))
            LOG["n_tlogits"] += 1
        except Exception:
            pass

    def wrap_draft(fn):
        """Intercept draft compute_logits: capture hidden INPUT + logit OUTPUT."""
        def patched(self, *a, **k):
            hid = a[0] if a else k.get("hidden_states")
            if hid is not None:
                add_hidden(hid)
            out = fn(self, *a, **k)
            if out is not None:
                add_draft_logits(out)
            return out
        return patched

    def wrap_target(fn):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            if out is not None:
                add_target_entropy(out)
            return out
        return patched

    # patch EAGLE3 draft head
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
                LOG["patched"].append(f"{name}.{cn}")

    # patch target model
    for modname, clsname in [("llama", "LlamaForCausalLM"),
                              ("qwen3", "Qwen3ForCausalLM"),
                              ("qwen2", "Qwen2ForCausalLM")]:
        try:
            m   = importlib.import_module(f"vllm.model_executor.models.{modname}")
            cls = getattr(m, clsname, None)
            if cls and hasattr(cls, "compute_logits"):
                cls.compute_logits = wrap_target(cls.compute_logits)
                LOG["patched"].append(f"{modname}.{clsname}")
        except Exception:
            pass

    # patch observe_draft for step boundary + accept label
    try:
        import vllm.v1.spec_decode.metrics as SM
        for cn in dir(SM):
            obj = getattr(SM, cn)
            if isinstance(obj, type) and "observe_draft" in dir(obj):
                def _mkod(orig):
                    def patched(self, num_draft_tokens, num_accepted_tokens, *a, **k):
                        out = orig(self, num_draft_tokens, num_accepted_tokens, *a, **k)
                        LOG["n_observe"] += 1
                        LOG["records"].append({
                            "ent":    list(LOG["cur_ent"]),
                            "margin": list(LOG["cur_margin"]),
                            "ent_t":  list(LOG["cur_ent_t"]),
                            "h":      list(LOG["cur_h"]),   # list of 1-D tensors
                            "acc":    int(num_accepted_tokens),
                            "ndraft": int(num_draft_tokens),
                        })
                        LOG["cur_ent"] = []; LOG["cur_margin"] = []
                        LOG["cur_ent_t"] = []; LOG["cur_h"] = []
                        return out
                    return patched
                obj.observe_draft = _mkod(obj.observe_draft)
                LOG["patched"].append(f"{cn}.observe_draft")
                break
    except Exception as e:
        print(f"observe_draft hook err: {repr(e)[:120]}", flush=True)

    print("PATCHED:", LOG["patched"], flush=True)

    # ── workloads ────────────────────────────────────────────────────────── #
    def take(x, m): return list(x)[:m]
    if wl == "reasoning":
        # Competition-math reasoning (elicits long CoT) — the SpecDecode-Bench regime.
        # MATH-500 + AIME; robust loaders with fallback. Chat-templated math prompts.
        def load_reasoning(nn):
            probs = []
            try:
                d = load_dataset("HuggingFaceH4/MATH-500", split="test")
                probs += [x for x in d["problem"]][:nn]
            except Exception as e:
                print(f"  [MATH-500 load failed: {e}]", flush=True)
            if len(probs) < nn:
                try:
                    d = load_dataset("AI-MO/aimo-validation-aime", split="train")
                    probs += [x for x in d["problem"]][: nn - len(probs)]
                except Exception as e:
                    print(f"  [AIME load failed: {e}]", flush=True)
            # instruct/CoT framing
            return [f"Solve the following problem step by step. Put the final answer "
                    f"in \\boxed{{}}.\n\n{p}" for p in probs[:nn]]
        workloads = {"reasoning": load_reasoning(n)}
        print(f"  reasoning workload: {len(workloads['reasoning'])} problems", flush=True)
    else:
        workloads = {
            "humaneval": take(
                load_dataset("openai/openai_humaneval", split="test")["prompt"], n),
            "gsm8k": take(
                load_dataset("openai/gsm8k", "main", split="test")["question"], n),
            "mt_bench": take(
                load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                .map(lambda x: {"p": x["prompt"][0]})["p"], n),
        }
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    # reasoning needs a longer context window for CoT
    mml = 4096 if wl == "reasoning" else 2048
    llm = LLM(
        model=target,
        speculative_config={"method": "eagle3", "model": eagle,
                            "num_speculative_tokens": maxk},
        gpu_memory_utilization=0.6,
        max_model_len=mml,
        enforce_eager=True,
        disable_log_stats=False,
    )

    # ── generate: one prompt at a time to keep LOG["records"] aligned ────── #
    all_rows = []   # per-position flat rows for parquet

    for w, prompts in workloads.items():
        for gen_i, p in enumerate(prompts):
            LOG["records"] = []
            LOG["cur_ent"] = []; LOG["cur_margin"] = []
            LOG["cur_ent_t"] = []; LOG["cur_h"] = []

            llm.generate([p], sp, use_tqdm=False)

            # Flatten records into per-position rows
            for step_i, rec in enumerate(LOG["records"]):
                mk = rec["ndraft"]
                # h: list of tensors, one per draft position (≤ mk)
                for pos_i in range(mk):
                    accept_label = 1 if pos_i < rec["acc"] else 0
                    h_vec = rec["h"][pos_i] if pos_i < len(rec["h"]) else None
                    row = {
                        "workload":  w,
                        "gen_i":     gen_i,
                        "step_i":    step_i,
                        "position":  pos_i,
                        "ent":       rec["ent"][pos_i] if pos_i < len(rec["ent"]) else 0.0,
                        "margin":    rec["margin"][pos_i] if pos_i < len(rec["margin"]) else 0.0,
                        "ent_t":     rec["ent_t"][pos_i] if pos_i < len(rec["ent_t"]) else 0.0,
                        "ndraft":    mk,
                        "acc":       rec["acc"],
                        "accept":    accept_label,
                        "h_vec":     h_vec,   # 1-D float16 tensor or None
                    }
                    all_rows.append(row)

    n_pos = len(all_rows)
    n_hid = sum(1 for r in all_rows if r["h_vec"] is not None)
    print(f"\nTotal positions captured: {n_pos}")
    print(f"Positions with hidden vec: {n_hid}")
    print(f"hidden_dim: {LOG['hidden_dim']}")

    # ── write parquet ─────────────────────────────────────────────────────── #
    out_dir = "/root/out/eagle3_hidden_full"
    os.makedirs(out_dir, exist_ok=True)

    pq_path = None
    if n_hid > 0 and LOG["hidden_dim"] is not None:
        import numpy as np
        hdim = LOG["hidden_dim"]

        # Build arrays
        meta_cols = {
            "workload": [], "gen_i": [], "step_i": [], "position": [],
            "ent": [], "margin": [], "ent_t": [],
            "ndraft": [], "acc": [], "accept": [],
        }
        H_list = []

        for r in all_rows:
            if r["h_vec"] is None:
                continue
            for col in meta_cols:
                meta_cols[col].append(r[col])
            H_list.append(r["h_vec"].numpy())   # [hdim] float16 ndarray

        H = np.stack(H_list, axis=0)  # [n_hid, hdim]
        meta_df = pd.DataFrame(meta_cols)
        h_df    = pd.DataFrame(H, columns=[f"h{i}" for i in range(hdim)]).astype("float16")
        df = pd.concat([meta_df.reset_index(drop=True),
                        h_df.reset_index(drop=True)], axis=1)

        suffix = f"_{wl}" if wl != "default" else ""
        pq_path = f"{out_dir}/hidden_full_{model_key}{suffix}.parquet"
        df.to_parquet(pq_path, index=False, compression="snappy", engine="pyarrow")
        mb = os.path.getsize(pq_path) / 1e6
        print(f"Wrote parquet: {pq_path}  ({mb:.1f} MB, {len(df)} rows × {hdim+10} cols)")
    else:
        print("WARNING: no hidden vectors captured — check PATCHED list")

    # metadata
    meta = {
        "target": target, "eagle_head": eagle, "model": model_key,
        "n_positions_total": n_pos, "n_with_hidden": n_hid,
        "hidden_dim": LOG["hidden_dim"], "patched": LOG["patched"],
        "n_observe": LOG["n_observe"], "parquet": pq_path,
    }
    suffix = f"_{wl}" if wl != "default" else ""
    with open(f"{out_dir}/hidden_full_{model_key}{suffix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    vol.commit()
    mb = os.path.getsize(pq_path) / 1e6 if pq_path else 0
    return {**meta, "parquet_mb": mb}


@app.local_entrypoint()
def main(model: str = "llama8b", maxk: int = 7, n: int = 30, maxtok: int = 128,
         wl: str = "default"):
    if model not in MODEL_REGISTRY:
        print(f"Unknown model. Choose: {list(MODEL_REGISTRY)}")
        return
    target, eagle = MODEL_REGISTRY[model]
    # reasoning needs a longer generation budget for CoT
    if wl == "reasoning" and maxtok < 256:
        maxtok = 384
    print(f"Capturing FULL hidden state: model={model} wl={wl} maxtok={maxtok}")
    r = run.remote(target, eagle, maxk, n, maxtok, model, wl)
    print(f"\n===== hidden-full [{model}] =====")
    print(f"  positions : {r['n_positions_total']}")
    print(f"  with hidden: {r['n_with_hidden']}")
    print(f"  hidden_dim : {r['hidden_dim']}")
    print(f"  parquet    : {r.get('parquet_mb', 0):.1f} MB")
    if r["n_with_hidden"] == 0:
        print("  !! hook did not fire — check PATCHED list in logs")
    else:
        print(f"\n  Download:")
        print(f"    modal volume get spec-dec-m5-results eagle3_hidden_full/ results/eagle3_hidden_full/")
        print(f"  Then analyze:")
        print(f"    python analyze_perstep_hidden_full.py --model {model}")
