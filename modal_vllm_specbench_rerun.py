"""Spec-Bench rerun of the fair-baseline result (the paper's fair-baseline table).

That table used 10 prompts per workload, 128 output tokens and B=1. This script reruns the same
comparison at scale. What changes from modal_vllm_fairbase.py:

  workload  Spec-Bench, 480 prompts in 6 categories (vendored at data/spec_bench/question.jsonl;
            the 80 MT-Bench items use their first turn only). Chat template applied.
  length    max_tokens 512 (was 128).
  pairing   per prompt: every prompt runs on every arm back to back, with the arm order rotated
            from one prompt to the next (was per cycle of 10 prompts).
  identity  every arm's output token ids are saved. A separate no-spec reference run saves the
            target's own greedy tokens and its top-2 logprobs, so each arm can be checked token
            by token against non-speculative decoding (analyze_specbench_rerun.py).
  bracket   padded fixed-depth arms k=2..5 inside the K=7 engine, on a subsample, to check that
            native K=2 and K=3 bracket the best fixed depth on these workloads.
  batch     prompts in chunks of B with the batch-level mean-confidence rule, as in
            modal_vllm_fairbase2.py, paired per chunk.

Unchanged: vLLM pinned to 0.24.0 (the fair-baseline version; later releases moved the draft loop and
the patch would silently miss it), H100, enforce_eager, native K=2 and K=3 engines with their
true verification budgets, the patched K=7 engine for cum(0.2), and the same fail-fast gates.

Run, in this order (estimates at ~$3.95/h for an H100; check the Modal dashboard after each):
  modal run modal_vllm_specbench_rerun.py --mode smoke                 # ~15 min, ~$1
  modal run modal_vllm_specbench_rerun.py --mode reference             # ~30 min, ~$2
  modal run modal_vllm_specbench_rerun.py --mode main --tag c1         # ~60 min, ~$4
  modal run modal_vllm_specbench_rerun.py --mode main --tag c2 --bracket-per-cat 0   # ~$3.5
  modal run modal_vllm_specbench_rerun.py --mode batch --tag b1 --n-per-cat 40       # B=4,8, ~$1.5
Each call writes its result to results/specbench/ locally (and keeps a copy on the volume).
Then:
  python analyze_specbench_rerun.py
Every call checkpoints to the volume after each category and stops itself at --max_minutes,
so a crash or the budget guard keeps what was measured.
"""
import json
import os
import time

import modal

VLLM_VERSION = "0.24.0"
TARGET = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
KMAX = 7
NATIVE_KS = (2, 3)
BRACKET_KS = (2, 3, 4, 5)
THR = 0.2
MAX_MODEL_LEN = 4096          # longest Spec-Bench prompt is ~1.7k tokens, plus 512 out
UTIL_SPEC = 0.29              # three engines in one process; leaves KV room for B=8
UTIL_REF = 0.85
USD_PER_HOUR = 4.40           # H100 $3.95/h list price plus CPU/RAM, rounded up on purpose
BUDGET_USD = 21.0             # hard cap across every call; raised from 18 on 2026-10-19 (approved) for the third tuned container
OUT_DIR = "/root/out/specbench"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          "HF_HOME": "/root/hf"})
)
app = modal.App("vllm-specbench-rerun")
out_vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
FN_KW = dict(image=image, gpu="H100",
             volumes={"/root/out": out_vol, "/root/hf": hf_vol},
             secrets=[modal.Secret.from_name("huggingface")])


# ───────────────────────────── shared helpers ───────────────────────────── #

def _meta(mode, tag, maxtok, t_start):
    import torch
    import vllm
    return {"mode": mode, "tag": tag, "target": TARGET, "eagle": EAGLE,
            # str(): torch.__version__ is a TorchVersion object, which the caller cannot
            # unpickle without torch installed
            "vllm": str(vllm.__version__), "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0), "maxtok": maxtok, "thr": THR,
            "kmax": KMAX, "natives": list(NATIVE_KS), "max_model_len": MAX_MODEL_LEN,
            "started_unix": t_start}


def _save(path, payload, t_start):
    elapsed = time.time() - t_start
    payload["meta"]["elapsed_s"] = round(elapsed, 1)
    payload["meta"]["est_usd"] = round(elapsed / 3600 * USD_PER_HOUR, 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    out_vol.commit()


def _packed(payload):
    """Return value for the caller: meta plus the gzipped payload (well under Modal's inline
    limit), so results land locally without a separate `modal volume get`. Only plain JSON
    types cross the wire, so the caller needs neither torch nor numpy."""
    import gzip
    blob = json.dumps(payload).encode()
    return {"meta": json.loads(json.dumps(payload["meta"])), "gz": gzip.compress(blob)}


def _render(tokenizer, items):
    """Chat-template each prompt to token ids.

    Renders the template to text, then encodes it without adding special tokens, so the
    template's own <|begin_of_text|> is the only BOS. (apply_chat_template(tokenize=True)
    returns a dict in newer transformers, which put key names where token ids belong.)
    """
    rendered = []
    for it in items:
        text = tokenizer.apply_chat_template([{"role": "user", "content": it["text"]}],
                                             add_generation_prompt=True, tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = [int(t) for t in ids]
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:                      # exactly one BOS, at the front
            ids = [bos] + [t for t in ids if t != bos]
        assert len(ids) > 1, "empty prompt after templating"
        rendered.append({**it, "ids": ids})
    print(f"rendered {len(rendered)} prompts; first starts {rendered[0]['ids'][:6]}", flush=True)
    return rendered


def _by_category(items):
    cats = {}
    for it in items:
        cats.setdefault(it["cat"], []).append(it)
    return cats


def _install_patch(EL):
    """Hook the EAGLE-3 head's logits (per-request top-1 prob) and replace the V1 draft loop.

    Arm kinds: ("nofire", 0) stock behaviour; ("fixed", k) stop after k drafted tokens and pad;
    ("cum", thr) stop when the batch-mean running product of top-1 probs falls below thr.
    At B=1 the batch mean is the single request's product, i.e. the fair-baseline rule exactly.
    """
    import importlib
    import pkgutil

    import numpy as np
    import torch
    import vllm.model_executor.models as MM

    def wrap_logits(fn):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            try:
                lg = out if torch.is_tensor(out) else out[0]
                rows = lg.view(-1, lg.shape[-1]).float()
                EL["top_p_vec"] = torch.softmax(rows, dim=-1).max(dim=-1).values.cpu().numpy()
            except Exception:
                pass
            return out
        return patched

    n_wrapped = 0
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
                obj.compute_logits = wrap_logits(obj.compute_logits)
                n_wrapped += 1
    assert n_wrapped > 0, "no EAGLE-3 class found to hook — abort"

    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as AS
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        build_slot_mappings_by_layer,
    )

    def multi_step_patched(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        EL["n_msd"] += 1
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        arm_kind, arm_val = EL["arm"]
        v = EL["top_p_vec"]
        cum_vec = (v[:num_reqs].copy() if v is not None and len(v) >= num_reqs
                   else np.ones(num_reqs))
        tokens_drafted = 1
        stopped_early = False

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            if arm_kind == "fixed" and step >= arm_val:
                stopped_early = True
                break
            if arm_kind == "cum" and float(cum_vec.mean()) < arm_val:
                stopped_early = True
                break

            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, positions, batch_desc.num_tokens)
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config)
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens)

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs, batch_desc.num_tokens, attn_metadata,
                    slot_mappings_by_layer, num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode)

            tokens_drafted = step + 1
            v = EL["top_p_vec"]
            if v is not None and len(v) >= num_reqs:
                cum_vec *= v[:num_reqs]

        if stopped_early:
            self.draft_tokens[:num_reqs, tokens_drafted:self.num_speculative_steps] = (
                self.draft_tokens[:num_reqs, tokens_drafted - 1: tokens_drafted])
        EL["depths"].append(tokens_drafted)

    AS.AutoRegressiveSpeculator._multi_step_decode = multi_step_patched
    print(f"draft loop patched; {n_wrapped} EAGLE-3 logits hooks", flush=True)


# ───────────────────────── no-spec reference run ────────────────────────── #

@app.function(timeout=75 * 60, **FN_KW)
def run_reference(items, maxtok, tag, max_minutes):
    """Target alone, greedy, B=1: tokens plus top-1 minus top-2 logprob at every position."""
    import torch
    from vllm import LLM, SamplingParams

    t_start = time.time()
    llm = LLM(model=TARGET, gpu_memory_utilization=UTIL_REF, max_model_len=MAX_MODEL_LEN,
              enforce_eager=True, disable_log_stats=True)
    items = _render(llm.get_tokenizer(), items)
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok, logprobs=2)
    llm.generate([{"prompt_token_ids": items[0]["ids"]}], sp, use_tqdm=False)  # warm-up

    path = f"{OUT_DIR}/specbench_reference_{tag}.json"
    payload = {"meta": _meta("reference", tag, maxtok, t_start), "records": []}
    payload["meta"]["truncated"] = False
    for cat, its in _by_category(items).items():
        for it in its:
            if max_minutes and time.time() - t_start > max_minutes * 60:
                payload["meta"]["truncated"] = True
                break
            torch.cuda.synchronize(); t0 = time.time()
            out = llm.generate([{"prompt_token_ids": it["ids"]}], sp, use_tqdm=False)
            torch.cuda.synchronize(); dt = time.time() - t0
            o = out[0].outputs[0]
            margins = []
            for pos in (o.logprobs or []):
                lps = sorted((lp.logprob for lp in pos.values()), reverse=True)
                margins.append(round(lps[0] - lps[1], 4) if len(lps) > 1 else None)
            payload["records"].append({"qid": it["qid"], "cat": cat, "ntok": len(o.token_ids),
                                       "dt": round(dt, 4), "tokens": list(o.token_ids),
                                       "margins": margins})
        _save(path, payload, t_start)
        print(f"reference {cat}: {sum(r['cat'] == cat for r in payload['records'])} done, "
              f"{payload['meta']['elapsed_s']/60:.1f} min, ~${payload['meta']['est_usd']}",
              flush=True)
        if payload["meta"]["truncated"]:
            print("budget guard hit — stopping", flush=True)
            break
    return _packed(payload)


# ────────────────────── speculative arms (bracket/main/batch) ───────────── #

@app.function(timeout=150 * 60, **FN_KW)
def run_spec(items, phases, maxtok, bracket_per_cat, batches, tag, natives_k, extras,
             max_minutes):
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams

    t_start = time.time()
    EL = {"arm": ("nofire", 0), "top_p_vec": None, "depths": [], "n_msd": 0}
    _install_patch(EL)

    def make_engine(k):
        return LLM(model=TARGET,
                   speculative_config={"method": "eagle3", "model": EAGLE,
                                       "num_speculative_tokens": k},
                   gpu_memory_utilization=UTIL_SPEC, max_model_len=MAX_MODEL_LEN,
                   enforce_eager=True, disable_log_stats=True)

    natives_k = [int(k) for k in natives_k]
    engines = {f"native{k}": make_engine(k) for k in natives_k}
    e7 = make_engine(KMAX)
    free, total = torch.cuda.mem_get_info()
    print(f"engines loaded; VRAM {100 * (1 - free / total):.0f}% used", flush=True)

    items = _render(e7.get_tokenizer(), items)
    cats = _by_category(items)
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    ARM = {f"native{k}": (engines[f"native{k}"], ("nofire", 0)) for k in natives_k}
    ARM["native7"] = (e7, ("nofire", 0))
    ARM["cum0.2"] = (e7, ("cum", THR))
    for k in BRACKET_KS:
        ARM[f"fixed{k}"] = (e7, ("fixed", k))

    def gen(arm, chunk):
        eng, setting = ARM[arm]
        EL["arm"] = setting
        EL["depths"] = []
        torch.cuda.synchronize(); t0 = time.time()
        outs = eng.generate([{"prompt_token_ids": it["ids"]} for it in chunk],
                            [sp] * len(chunk), use_tqdm=False)
        torch.cuda.synchronize(); dt = time.time() - t0
        toks = [list(o.outputs[0].token_ids) for o in outs]
        depth = round(float(np.mean(EL["depths"])), 3) if EL["depths"] else None
        return toks, dt, depth

    # ── gates: each arm drafts what it claims; the patched loop is live ── #
    gate_item = cats[sorted(cats)[0]][0]
    gates = {}
    for arm in ARM:
        eng, setting = ARM[arm]
        EL["arm"] = setting
        EL["depths"] = []
        eng.generate([{"prompt_token_ids": gate_item["ids"]}],
                     SamplingParams(temperature=0.0, max_tokens=64), use_tqdm=False)
        d = list(EL["depths"])
        gates[arm] = {"min": min(d) if d else None, "max": max(d) if d else None,
                      "mean": round(float(np.mean(d)), 3) if d else None}
        print(f"gate {arm:8s}: drafted min/mean/max {gates[arm]}", flush=True)
        if arm.startswith("native"):
            want = int(arm.replace("native", ""))
            assert d and min(d) == max(d) == want, f"{arm} drafts {d[:8]}, expected {want}"
        if arm.startswith("fixed"):
            want = int(arm.replace("fixed", ""))
            assert d and max(d) == want, f"{arm} drafts up to {max(d) if d else None}, expected {want}"
    assert EL["n_msd"] > 0, "patched loop never ran — abort"
    assert gates["cum0.2"]["min"] is not None and gates["cum0.2"]["min"] < KMAX, \
        "cum arm never stopped early — patch is a no-op, abort"
    print("GATES PASSED", flush=True)

    path = f"{OUT_DIR}/specbench_spec_{tag}.json"
    payload = {"meta": _meta("spec", tag, maxtok, t_start), "gates": gates,
               "bracket": [], "main": [], "batch": []}
    payload["meta"].update({"phases": phases, "batches": batches, "truncated": False,
                            "natives": natives_k, "extras": list(extras),
                            "bracket_per_cat": bracket_per_cat})

    def over_budget():
        if max_minutes and time.time() - t_start > max_minutes * 60:
            payload["meta"]["truncated"] = True
            return True
        return False

    def run_paired(phase, arms, chunks_by_cat, keep_tokens, bnom=1):
        """Every chunk runs on every arm back to back; arm order rotates chunk by chunk."""
        i = 0
        for cat, chunks in chunks_by_cat.items():
            for chunk in chunks:
                if over_budget():
                    return
                order = arms[i % len(arms):] + arms[:i % len(arms)]
                for pos, arm in enumerate(order):
                    toks, dt, depth = gen(arm, chunk)
                    rec = {"cat": cat, "qids": [it["qid"] for it in chunk], "arm": arm,
                           "order_pos": pos, "B": len(chunk), "Bnom": bnom, "chunk": i,
                           "ntok": sum(len(t) for t in toks), "dt": round(dt, 4),
                           "depth": depth}
                    if keep_tokens:
                        rec["tokens"] = toks
                    payload[phase].append(rec)
                i += 1
            _save(path, payload, t_start)
            print(f"{phase} B={bnom} {cat}: done, {payload['meta']['elapsed_s']/60:.1f} min, "
                  f"~${payload['meta']['est_usd']}", flush=True)

    natives = [f"native{k}" for k in natives_k] + list(extras)   # extras: padded arms
    if "bracket" in phases and not payload["meta"]["truncated"]:
        sub = {c: [[it] for it in its[:bracket_per_cat]] for c, its in cats.items()}
        run_paired("bracket", [f"native{k}" for k in natives_k]
                   + [f"fixed{k}" for k in BRACKET_KS] + ["cum0.2"],
                   sub, keep_tokens=False)
    if "main" in phases and not payload["meta"]["truncated"]:
        run_paired("main", natives + ["cum0.2"],
                   {c: [[it] for it in its] for c, its in cats.items()}, keep_tokens=True)
    if "batch" in phases and not payload["meta"]["truncated"]:
        for B in batches:
            if payload["meta"]["truncated"]:
                break
            run_paired("batch", natives + ["cum0.2"],
                       {c: [its[j:j + B] for j in range(0, len(its), B)]
                        for c, its in cats.items()}, keep_tokens=True, bnom=B)
    _save(path, payload, t_start)
    if payload["meta"]["truncated"]:
        print("budget guard hit — partial results saved", flush=True)
    print(f"done: {payload['meta']['elapsed_s']/60:.1f} min, ~${payload['meta']['est_usd']}",
          flush=True)
    return _packed(payload)


# ───────────────────────────────── entry ────────────────────────────────── #

def _load_items(n_per_cat):
    here = os.path.dirname(os.path.abspath(__file__))
    items = []
    with open(os.path.join(here, "data", "spec_bench", "question.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            cat = r["category"]
            if cat not in ("translation", "summarization", "qa", "math_reasoning", "rag"):
                cat = "mt_bench"          # the 8 MT-Bench sub-categories, 10 items each
            items.append({"qid": r["question_id"], "cat": cat, "text": r["turns"][0]})
    if n_per_cat:
        kept, seen = [], {}
        for it in items:
            if seen.get(it["cat"], 0) < n_per_cat:
                kept.append(it)
                seen[it["cat"]] = seen.get(it["cat"], 0) + 1
        items = kept
    return items


def _results_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "specbench")


def _spent_usd():
    """Spend so far, from the local ledger of remote-call wall times (which bound GPU time,
    including container start), plus any result file written before the ledger existed."""
    out = _results_dir()
    ledger_path = os.path.join(out, "ledger.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else []
    seen = {e["file"] for e in ledger}
    usd = sum(e["seconds"] for e in ledger) / 3600 * USD_PER_HOUR
    if os.path.isdir(out):
        for f in os.listdir(out):
            if f.startswith("specbench_") and f.endswith(".json") and f not in seen:
                meta = json.load(open(os.path.join(out, f))).get("meta", {})
                usd += meta.get("elapsed_s", 0) / 3600 * USD_PER_HOUR + 0.25
    return usd


def _guarded(fn, name, default_minutes, *args):
    """Run fn.remote(*args, max_minutes) inside what is left of BUDGET_USD; ledger the call."""
    out = _results_dir()
    os.makedirs(out, exist_ok=True)
    left = BUDGET_USD - _spent_usd()
    minutes = min(default_minutes, int(left / USD_PER_HOUR * 60) - 4)
    print(f"budget: ${BUDGET_USD - left:.2f} spent, ${left:.2f} left -> this call capped at "
          f"{minutes} min")
    if minutes < 8:
        print(f"SKIPPED {name}: not enough budget left for a useful run")
        return None
    t0 = time.time()
    try:
        return _write_local(fn.remote(*args, minutes), name)
    finally:
        ledger_path = os.path.join(out, "ledger.json")
        ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else []
        ledger.append({"file": name, "seconds": round(time.time() - t0, 1),
                       "finished_unix": round(time.time())})
        with open(ledger_path, "w") as f:
            json.dump(ledger, f, indent=1)
        print(f"budget after {name}: ${_spent_usd():.2f} of ${BUDGET_USD:.2f}")


def _write_local(res, name):
    """Write the returned payload to results/specbench/ and return its meta for printing."""
    import gzip
    path = os.path.join(_results_dir(), name)
    with open(path, "w") as f:
        f.write(gzip.decompress(res["gz"]).decode())
    print(f"wrote {path}")
    return res["meta"]


@app.local_entrypoint()
def main(mode: str = "smoke", tag: str = "", n_per_cat: int = 0, maxtok: int = 512,
         bracket_per_cat: int = 8, batches: str = "4,8", max_minutes: int = 0,
         natives: str = "2,3", extras: str = ""):
    """mode: smoke | reference | main | batch. Every call stays inside BUDGET_USD overall.

    --natives: K values that get their own native engine (true verification budget).
    --extras: padded fixed-depth arms (e.g. fixed2) run inside the K=7 engine in the
              main/batch phases too, for depths that do not get a native engine.
    """
    bs = [int(b) for b in batches.split(",") if b]
    nk = [int(k) for k in natives.split(",") if k]
    ex = [e for e in extras.split(",") if e]
    assert len(nk) == 2, "two native engines fit beside the K=7 engine on one H100"
    assert all(e in [f"fixed{k}" for k in BRACKET_KS] for e in ex), f"unknown extra arm {ex}"
    if mode == "smoke":
        items = _load_items(2)
        print(json.dumps(_guarded(run_reference, "specbench_reference_smoke.json", 20,
                                  items, 64, "smoke"), indent=2))
        print(json.dumps(_guarded(run_spec, "specbench_spec_smoke.json", 25,
                                  items, ["bracket", "main", "batch"], 64, 1, [2], "smoke",
                                  nk, ex),
                         indent=2))
        print("SMOKE OK — both code paths ran end to end. Next: --mode reference")
        return
    items = _load_items(n_per_cat)
    print(f"{mode}: {len(items)} prompts, max_tokens {maxtok}")
    if mode == "reference":
        t = tag or "full"
        res = _guarded(run_reference, f"specbench_reference_{t}.json", max_minutes or 60,
                       items, maxtok, t)
    elif mode == "main":
        t = tag or "c1"
        res = _guarded(run_spec, f"specbench_spec_{t}.json", max_minutes or 120,
                       items, ["bracket", "main"], maxtok, bracket_per_cat, bs, t, nk, ex)
    elif mode == "batch":
        t = tag or "b1"
        res = _guarded(run_spec, f"specbench_spec_{t}.json", max_minutes or 75,
                       items, ["batch"], maxtok, 0, bs, t, nk, ex)
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(json.dumps(res, indent=2))
