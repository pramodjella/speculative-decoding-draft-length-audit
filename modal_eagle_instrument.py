"""Step 3a (diagnostic): wrap EAGLE's inference functions to learn the exact data
layout (arg/return shapes) of tree_decoding (target logits p), evaluate_posterior
(accept), generate_candidates, topK_genrate (draft q). Then we build the real
per-step paired capture. Working recipe: transformers 4.53.1 / torch 2.6.0.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                 "sentencepiece", "huggingface_hub", "fschat")
    .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
    .env({"PYTHONPATH": "/root/EAGLE"})
)
app = modal.App("eagle-instrument")

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch, inspect
    import eagle.model.ea_model as EM
    from eagle.model.ea_model import EaModel

    def shp(x, d=0):
        if torch.is_tensor(x):
            return f"T{tuple(x.shape)}:{str(x.dtype).replace('torch.','')}"
        if isinstance(x, (list, tuple)):
            inner = shp(x[0], d+1) if x and d < 2 else "."
            return f"{type(x).__name__}[{len(x)}]({inner})"
        if isinstance(x, dict):
            return "dict{" + ",".join(list(x)[:6]) + "}"
        if isinstance(x, (int, float)):
            return f"{type(x).__name__}={x}"
        return type(x).__name__

    cnt = {}
    def wrap(name, fn):
        def w(*a, **k):
            out = fn(*a, **k)
            n = cnt.get(name, 0)
            if n < 3:
                print(f"\n[{name}] call#{n}", flush=True)
                print(f"   args: {[shp(x) for x in a]}", flush=True)
                if k: print(f"   kwargs: { {kk: shp(v) for kk, v in k.items()} }", flush=True)
                if isinstance(out, tuple):
                    print(f"   -> tuple({[shp(x) for x in out]})", flush=True)
                else:
                    print(f"   -> {shp(out)}", flush=True)
                cnt[name] = n + 1
            return out
        return w

    # print the source signatures we care about
    for fn_name in ("tree_decoding", "evaluate_posterior", "generate_candidates", "initialize_tree"):
        fn = getattr(EM, fn_name, None)
        if fn:
            try: print(f"SIG {fn_name}{inspect.signature(fn)}", flush=True)
            except Exception: pass
            setattr(EM, fn_name, wrap(fn_name, fn))

    # default tree config (total_token=-1 auto-configures, validated in substrate run).
    # top_k=1 broke topK_genrate's hardcoded top_k=5; we extract the accepted CHAIN
    # from best_candidate/accept_length instead of forcing a chain tree.
    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    # patch topK_genrate on the ea_layer instance's class
    ea = model.ea_layer
    if hasattr(type(ea), "topK_genrate"):
        print(f"SIG topK_genrate{inspect.signature(type(ea).topK_genrate)}", flush=True)
        type(ea).topK_genrate = wrap("topK_genrate", type(ea).topK_genrate)

    tok = model.get_tokenizer()
    ids = tok.apply_chat_template([{"role": "user", "content": "What is 17*24? Show steps."}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.base_model.device)
    print("\n=== generating (default tree) ===", flush=True)
    out = model.eagenerate(ids, temperature=0.0, max_new_tokens=32)
    print("\nGEN:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:160]), flush=True)
    return {"ok": True}


@app.local_entrypoint()
def main():
    print(run.remote())
