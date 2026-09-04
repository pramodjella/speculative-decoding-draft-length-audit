"""Find vLLM 0.23's GREEDY spec-decode acceptance point (temp=0 token-match, not
rejection sampling) so we can hook accept_run per step. Free CPU.

Searches spec_decode + sample + worker modules for the function that computes
accepted-token counts / updates spec_decode_num_accepted_tokens_per_pos.
"""
import modal
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("accept-recon")


@app.function(image=image, timeout=600)
def probe():
    import inspect, importlib, pkgutil
    import vllm
    print("vllm", vllm.__version__, flush=True)

    KEYS = ("num_accepted", "accepted_tokens", "spec_decode_num_accepted",
            "def reject", "def compute_accept", "accept_length", "verify",
            "num_accepted_tokens_per_pos", "_accepted")

    def scan_module(modname):
        try:
            m = importlib.import_module(modname)
            src = inspect.getsource(m)
        except Exception:
            return
        hits = [(i, ln) for i, ln in enumerate(src.splitlines())
                if any(k in ln for k in KEYS)]
        if hits:
            print(f"\n=== {modname} ({len(hits)} hits) ===", flush=True)
            for i, ln in hits[:40]:
                print(f"  {i:4d}: {ln.strip()[:150]}", flush=True)

    # spec_decode + sample submodules
    for pkg in ("vllm.v1.spec_decode", "vllm.v1.sample"):
        p = importlib.import_module(pkg)
        scan_module(pkg)
        for mi in pkgutil.iter_modules(p.__path__):
            scan_module(f"{pkg}.{mi.name}")

    # the worker that drives verification
    for modname in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu_input_batch"):
        scan_module(modname)

    # rejection_sampler: dump function names + which is the greedy path
    try:
        import vllm.v1.sample.rejection_sampler as RS
        print("\n=== rejection_sampler members ===", flush=True)
        print([n for n in dir(RS) if not n.startswith("__")], flush=True)
        for cn in dir(RS):
            obj = getattr(RS, cn)
            if inspect.isfunction(obj) and any(k in cn.lower() for k in ("accept", "reject", "greedy", "verify")):
                print(f"\n--- function {cn}{inspect.signature(obj)} ---", flush=True)
    except Exception as e:
        print("RS err", e, flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
