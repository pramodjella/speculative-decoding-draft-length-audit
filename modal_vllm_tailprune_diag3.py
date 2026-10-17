"""Diag 3: find the ACTUAL drafter object in the live engine via gc, dump its real
propose source, and prove a patch on its true type executes.

Prior findings: hooks fire in-process, but patches on EagleProposer.propose AND
SpecDecodeBaseProposer.propose never execute (n_propose=0) while drafting clearly runs
(compute_logits fires ~K times/step). So the live drafter is some other type, or the callsite
bypasses those attrs. Stop guessing: introspect the running engine.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune_diag3.py
Out: /root/out/vllm_true_propose_src.txt (+ stdout findings)
"""
import os, modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune-diag3")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE  = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import gc, inspect, torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": 7},
              gpu_memory_utilization=0.7, max_model_len=2048,
              enforce_eager=True, disable_log_stats=True)
    sp8 = SamplingParams(temperature=0.0, max_tokens=8)
    llm.generate(["hi"], sp8, use_tqdm=False)   # warmup so everything is instantiated

    # ── find live objects that look like drafters ─────────────────────────── #
    found = {}
    for o in gc.get_objects():
        try:
            t = type(o)
            if callable(getattr(t, "propose", None)) and hasattr(o, "num_speculative_tokens"):
                found[t] = o
        except Exception:
            continue
    print(f"drafter-like live objects: {len(found)}", flush=True)
    lines = []
    import vllm.v1.spec_decode.llm_base_proposer as LBP
    for t, o in found.items():
        fn = t.propose
        mod = getattr(fn, "__module__", "?")
        qn = getattr(fn, "__qualname__", "?")
        same_as_base = fn is LBP.SpecDecodeBaseProposer.propose
        print(f"TYPE {t.__module__}.{t.__name__}  mro={[c.__name__ for c in t.__mro__[:4]]}",
              flush=True)
        print(f"  .propose -> {mod}.{qn}  is_base_fn={same_as_base}", flush=True)
        # any instance-level propose shadowing the class attr?
        inst_shadow = "propose" in getattr(o, "__dict__", {})
        print(f"  instance __dict__ shadows propose: {inst_shadow}", flush=True)
        try:
            src = inspect.getsource(fn)
            lines.append(f"===== {t.__module__}.{t.__name__}.propose "
                         f"({mod}.{qn}) =====\n{src}\n")
        except Exception as e:
            lines.append(f"===== {t.__module__}.{t.__name__}.propose: getsource failed {e}\n")

    # ── prove a patch on the TRUE type executes ───────────────────────────── #
    CNT = {"n": 0}
    for t in list(found):
        orig = t.propose
        def mk(orig):
            def counting(self, *a, **k):
                CNT["n"] += 1
                return orig(self, *a, **k)
            return counting
        t.propose = mk(orig)
        # also nuke any instance-level shadow so the class attr is used
        inst = found[t]
        if "propose" in getattr(inst, "__dict__", {}):
            del inst.__dict__["propose"]
    llm.generate(["What is 2+2?"], sp8, use_tqdm=False)
    print(f"after runtime type-patch: propose calls = {CNT['n']}", flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open("/root/out/vllm_true_propose_src.txt", "w") as f:
        f.write("".join(lines))
    vol.commit()
    return {"n_types": len(found), "counter_after_patch": CNT["n"]}


@app.local_entrypoint()
def main():
    print(run.remote())
