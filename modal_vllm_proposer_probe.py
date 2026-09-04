"""Reconnaissance of vLLM 0.23 proposer API — to build an adaptive-K custom proposer.

Dumps: the proposer base class + propose() signatures, the full source of
custom_class_proposer (the documented extension point), how SpeculativeConfig
selects a proposer (and whether a custom class path can be passed), and whether
proposals can be variable-length (ngram). Free CPU container.
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("vllm-proposer-probe")


@app.function(image=image, timeout=600)
def probe():
    import inspect, vllm
    print("vllm", vllm.__version__)

    def dump(obj, name, methods=None, maxlines=120):
        print(f"\n{'='*70}\n### {name}\n{'='*70}")
        try:
            if methods:
                for m in methods:
                    fn = getattr(obj, m, None)
                    if fn is None:
                        print(f"  [no method {m}]"); continue
                    try:
                        print(f"  {m}{inspect.signature(fn)}")
                    except Exception as e:
                        print(f"  {m}: <sig err {e}>")
            else:
                src = inspect.getsource(obj)
                print("\n".join(src.splitlines()[:maxlines]))
        except Exception as e:
            print(f"  <err {e}>")

    # 1. custom_class_proposer — the extension point (likely small, dump fully)
    try:
        from vllm.v1.spec_decode import custom_class_proposer as ccp
        dump(ccp, "custom_class_proposer (FULL)", maxlines=200)
    except Exception as e:
        print("custom_class_proposer import err:", e)

    # 2. base proposer interface
    try:
        from vllm.v1.spec_decode import llm_base_proposer as lbp
        print("\nllm_base_proposer members:", [n for n in dir(lbp) if not n.startswith("__")])
        for cls_name in dir(lbp):
            obj = getattr(lbp, cls_name)
            if inspect.isclass(obj) and "Propos" in cls_name:
                dump(obj, f"{cls_name} methods",
                     methods=["__init__", "propose", "load_model", "update_candidate_strategy"])
    except Exception as e:
        print("llm_base_proposer err:", e)

    # 3. ngram: confirm variable-length proposals + how k is read
    try:
        from vllm.v1.spec_decode.ngram_proposer import NgramProposer
        dump(NgramProposer, "NgramProposer.propose sig", methods=["__init__", "propose", "batch_propose"])
        src = inspect.getsource(NgramProposer.propose)
        print("\nNgramProposer.propose body (first 40 lines):")
        print("\n".join(src.splitlines()[:40]))
    except Exception as e:
        print("ngram err:", e)

    # 4. how SpeculativeConfig picks a proposer / accepts a custom class
    try:
        from vllm.config import SpeculativeConfig
        src = inspect.getsource(SpeculativeConfig)
        import re
        print("\n=== SpeculativeConfig: lines mentioning custom/proposer/method= ===")
        for ln in src.splitlines():
            if re.search(r"custom|proposer|self\.method\s*=|SpeculativeMethod|draft_model", ln, re.I):
                print("  " + ln.strip()[:140])
    except Exception as e:
        print("SpeculativeConfig err:", e)

    # 5. where the proposer is instantiated/selected in the gpu runner
    try:
        import vllm.v1.spec_decode as sd
        print("\nspec_decode modules:", [n for n in dir(sd) if not n.startswith("_")])
        # find the dispatch that maps method -> proposer class
        import vllm.v1.worker.gpu_model_runner as gmr
        gsrc = inspect.getsource(gmr)
        print("\n=== gpu_model_runner: proposer construction lines ===")
        for ln in gsrc.splitlines():
            if ("Proposer(" in ln) or ("proposer" in ln.lower() and "=" in ln and "self" in ln):
                print("  " + ln.strip()[:140])
    except Exception as e:
        print("dispatch err:", e)


@app.local_entrypoint()
def main():
    probe.remote()
