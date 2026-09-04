"""Step 2 of the EAGLE-loop port: validate the SafeAILab/EAGLE substrate loads
Llama-3.1-8B + the EAGLE-3 head and generates. De-risks deps/API/model-format
before we build the q/p/accept instrumentation on it. H100.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # use the CLONED repo (not pip eagle-llm) with a late-2024 transformers that
    # still has LossKwargs and supports Llama-3.1 / EAGLE-3.
    # repo requirements.txt: torch 2.6.0, transformers >=4.53.1, accelerate 0.26.0.
    # 4.53.1 has use_kernel_forward_from_hub AND still has LossKwargs (removed in 5.x).
    .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                 "sentencepiece", "huggingface_hub", "fschat")
    .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
    .env({"PYTHONPATH": "/root/EAGLE"})
)
app = modal.App("eagle-substrate")

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch, importlib, traceback
    print("torch", torch.__version__, flush=True)
    import transformers; print("transformers", transformers.__version__, flush=True)

    # discover the EaModel entry point
    import inspect
    EaModel = None
    for modpath in ("eagle.model.ea_model", "eagle.modeling_eagle"):
        try:
            m = importlib.import_module(modpath)
            classes = [n for n in dir(m) if isinstance(getattr(m, n), type)]
            print(f"  {modpath} classes: {classes}", flush=True)
            cand = (getattr(m, "EaModel", None) or getattr(m, "EAModel", None)
                    or next((getattr(m, n) for n in classes
                             if "model" in n.lower() and hasattr(getattr(m, n), "from_pretrained")), None))
            if cand is not None and hasattr(cand, "from_pretrained"):
                EaModel = cand
                print(f"EaModel = {modpath}.{cand.__name__}", flush=True)
                try:
                    print("from_pretrained sig:", inspect.signature(EaModel.from_pretrained), flush=True)
                    print("eagenerate?", hasattr(EaModel, "eagenerate"),
                          " ea_generate?", hasattr(EaModel, "ea_generate"), flush=True)
                except Exception:
                    pass
                break
        except Exception as e:
            print(f"  import {modpath} failed: {repr(e)[:240]}", flush=True)

    if EaModel is None:
        print("!! could not import EaModel — dumping eagle package", flush=True)
        import eagle, os
        print(os.listdir(os.path.dirname(eagle.__file__)))
        return {"ok": False, "stage": "import"}

    # try to load
    try:
        model = EaModel.from_pretrained(
            base_model_path=BASE, ea_model_path=EA,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto",
            total_token=-1,            # let it auto-config the tree
        )
        model.eval()
        print("LOADED OK", flush=True)
    except Exception as e:
        print("LOAD FAILED:", repr(e)[:400], flush=True)
        traceback.print_exc()
        return {"ok": False, "stage": "load", "err": repr(e)[:400]}

    # try to generate
    try:
        tok = model.get_tokenizer()
        prompt = "Write a python function to compute factorial."
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.base_model.device)
        out = model.eagenerate(ids, temperature=0.0, max_new_tokens=48)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print("GEN OK ->", repr(text[:200]), flush=True)
        return {"ok": True, "text": text[:200]}
    except Exception as e:
        print("GEN FAILED:", repr(e)[:400], flush=True)
        traceback.print_exc()
        return {"ok": False, "stage": "generate", "err": repr(e)[:400]}


@app.local_entrypoint()
def main():
    r = run.remote()
    print("\n===== EAGLE substrate validation =====")
    print(r)
    print("  OK -> next: instrument eagenerate for per-step q/p/accept" if r.get("ok")
          else f"  FAILED at {r.get('stage')}: {r.get('err','')[:200]}")
