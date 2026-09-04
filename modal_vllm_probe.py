"""CPU-only probe of the installed vLLM's speculative-decoding control surface.

No GPU billed. Goal: find (a) version, (b) supported spec methods, (c) whether a
separate draft model works in the V1 engine, (d) the EAGLE/ngram proposer class
and the exact place num_speculative_tokens is read -> the per-step K patch point.
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("vllm-probe")


@app.function(image=image, timeout=1200)
def probe():
    import inspect, importlib, pkgutil
    out = []
    def p(*a):
        out.append(" ".join(str(x) for x in a)); print(*a)

    import vllm
    p("vllm.__version__ =", vllm.__version__)

    # 1) SpeculativeConfig fields + accepted methods
    try:
        from vllm.config import SpeculativeConfig as SC
        fields = getattr(SC, "model_fields", None) or getattr(SC, "__dataclass_fields__", {})
        p("\nSpeculativeConfig fields:", list(fields.keys()))
        src = inspect.getsource(SC)
        for kw in ["method", "eagle", "ngram", "draft", "num_speculative_tokens",
                   "draft_model", "eagle_dynamic", "medusa", "mlp_speculator"]:
            hits = [ln.strip() for ln in src.splitlines() if kw in ln.lower()]
            if hits:
                p(f"\n[SC source mentions '{kw}'] ({len(hits)} lines):")
                for h in hits[:6]:
                    p("   ", h[:160])
    except Exception as e:
        p("SpeculativeConfig introspection error:", repr(e))

    # 2) V1 spec_decode proposers available
    try:
        import vllm.v1.spec_decode as sd
        mods = [m.name for m in pkgutil.iter_modules(sd.__path__)]
        p("\nv1.spec_decode modules:", mods)
    except Exception as e:
        p("v1.spec_decode listing error:", repr(e))

    # 3) EAGLE proposer: where num_speculative_tokens / propose() lives (patch point)
    for modpath, clsname in [("vllm.v1.spec_decode.eagle", "EagleProposer"),
                             ("vllm.v1.spec_decode.ngram_proposer", "NgramProposer")]:
        try:
            mod = importlib.import_module(modpath)
            cls = getattr(mod, clsname, None)
            if cls is None:
                p(f"\n{modpath}: classes ->", [n for n, _ in inspect.getmembers(mod, inspect.isclass)][:10])
                continue
            methods = [n for n, _ in inspect.getmembers(cls, inspect.isfunction)]
            p(f"\n{clsname} methods:", methods)
            if hasattr(cls, "propose"):
                sig = inspect.signature(cls.propose)
                p(f"{clsname}.propose signature:", str(sig))
            src = inspect.getsource(cls)
            hits = [ln.strip() for ln in src.splitlines() if "num_speculative_tokens" in ln or "num_spec" in ln]
            p(f"{clsname} num_speculative_tokens usage ({len(hits)}):")
            for h in hits[:8]:
                p("   ", h[:160])
        except Exception as e:
            p(f"{modpath} introspection error:", repr(e))

    # 4) Does V1 still accept a separate draft model? search config validation
    try:
        from vllm.config import SpeculativeConfig as SC
        src = inspect.getsource(SC)
        draftlines = [ln.strip() for ln in src.splitlines()
                      if "draft" in ln.lower() and ("v1" in ln.lower() or "support" in ln.lower()
                                                    or "not" in ln.lower() or "error" in ln.lower())]
        p("\n[draft-model V1 support hints]:")
        for h in draftlines[:10]:
            p("   ", h[:160])
    except Exception as e:
        p("draft support probe error:", repr(e))

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
