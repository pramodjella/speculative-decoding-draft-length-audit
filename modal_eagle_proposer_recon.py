"""Recon vLLM 0.23 EAGLE proposer internals -> locate a READ-ONLY logging hook
for per-step draft entropy/margin, so we can test the SVIP entropy controller on
EAGLE-3 offline. Free CPU container.

Dumps the eagle proposer module: the propose() source and every line that
computes/samples draft logits (logits / argmax / sample / draft_token_ids /
hidden_states), which is where a read-only patch would record entropy+margin.
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("eagle-proposer-recon")


@app.function(image=image, timeout=600)
def probe():
    import inspect, importlib, pkgutil
    import vllm
    print("vllm", vllm.__version__, flush=True)

    # 1. enumerate spec_decode submodules, find the eagle one
    import vllm.v1.spec_decode as sd
    mods = [m.name for m in pkgutil.iter_modules(sd.__path__)]
    print("spec_decode submodules:", mods, flush=True)

    eagle_mod = None
    for cand in ("eagle", "eagle_proposer", "llm_eagle_proposer"):
        if cand in mods:
            eagle_mod = importlib.import_module(f"vllm.v1.spec_decode.{cand}")
            print(f"\n=== using module vllm.v1.spec_decode.{cand} ===", flush=True)
            break
    if eagle_mod is None:
        # fall back: search all submodules for an Eagle proposer class
        for mn in mods:
            m = importlib.import_module(f"vllm.v1.spec_decode.{mn}")
            if any("Eagle" in c and "Propos" in c for c in dir(m)):
                eagle_mod = m; print(f"\n=== found eagle proposer in {mn} ===", flush=True); break

    if eagle_mod is None:
        print("!! no eagle proposer module found"); return

    # 2. dump each proposer class's propose() with line numbers; flag signal lines
    KEYS = ("logits", "argmax", "sample", "draft_token", "hidden_state",
            "entropy", "softmax", "topk", "probs", "accepted", "num_accept")
    for cls_name in dir(eagle_mod):
        obj = getattr(eagle_mod, cls_name)
        if not (inspect.isclass(obj) and "Propos" in cls_name):
            continue
        print(f"\n{'='*70}\n### {cls_name}  methods: "
              f"{[m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj,m,None))]}\n{'='*70}", flush=True)
        for meth in ("propose", "_propose", "load_model", "prepare_inputs"):
            fn = getattr(obj, meth, None)
            if fn is None or not callable(fn):
                continue
            try:
                src = inspect.getsource(fn)
            except Exception as e:
                print(f"  [{meth}: no source {e}]"); continue
            lines = src.splitlines()
            print(f"\n--- {cls_name}.{meth}  ({len(lines)} lines) ---", flush=True)
            for i, ln in enumerate(lines):
                if any(k in ln.lower() for k in KEYS):
                    print(f"  {i:3d}: {ln.strip()[:150]}", flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
