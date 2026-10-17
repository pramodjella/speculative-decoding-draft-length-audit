"""vLLM-NATIVE paired tail-pruning test (chain regime) — Llama-3.1-8B and Qwen3-14B.

Answers two open questions in the production engine where all offline experiments ran:
  (1) does the saturation tail-pruning survivor validate vLLM-native (chain drafting)?
  (2) the Qwen3-14B replication (its AngelSlim head is vLLM-format; works here, not in the
      reference repo).

Design (controlled drafting-cost comparison):
  - Engine built once with num_speculative_tokens = KMAX = 7, enforce_eager, greedy, B=1.
  - EagleProposer.propose is replaced by a verbatim copy (dumped from this exact vLLM 0.23
    install: results/vllm_proposer_src.txt) with two additions inside the draft loop:
      * fixed-k arm : force-break after k tokens drafted
      * cumprob arm : break when the running product of drafted-token top-probs < thr
    After any break the proposal is PADDED to KMAX columns (pad tokens get rejected at
    verification), preserving the engine's [B, KMAX] contract. Every arm therefore pays
    IDENTICAL verification cost; differences are pure draft forwards — the same economics
    isolated in the reference-repo tree experiment.
  - Top-prob per drafted token comes from a compute_logits hook (runs in ALL arms; equal
    overhead).
  - Paired protocol: all arms round-robin per cycle, order rotated per cycle, per-cycle
    paired differences (drift-cancelling). Interleaved odd-index prompts.

Run: PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune.py --model llama8b --n 10 --maxtok 128 --cycles 4
     PYTHONIOENCODING=utf-8 modal run modal_vllm_tailprune.py --model qwen14b ...
Out: /root/out/vllm_tailprune_{model}.json
"""
import os, time, json
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
)
app = modal.App("vllm-tailprune")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

MODELS = {
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    "qwen14b": ("Qwen/Qwen3-14B", "AngelSlim/Qwen3-14B_eagle3"),
}
KMAX = 7


@app.function(image=image, gpu="H100", timeout=10800, volumes={"/root/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model_key, n, maxtok, cycles):
    import torch, numpy as np
    import importlib, pkgutil

    TARGET, EAGLE = MODELS[model_key]
    print(f"model={model_key} target={TARGET} head={EAGLE}", flush=True)

    # ── control state ─────────────────────────────────────────────────────── #
    EL = {"arm": ("fixed", KMAX), "top_p": 1.0, "depths": [], "n_propose": 0}

    # hook compute_logits on all eagle3 draft classes -> stash top-1 prob (B=1)
    import vllm.model_executor.models as MM
    def wrap_logits(fn):
        def patched(self, *a, **k):
            out = fn(self, *a, **k)
            try:
                lg = out if torch.is_tensor(out) else out[0]
                row = lg.view(-1, lg.shape[-1])[0].float()
                EL["top_p"] = float(torch.softmax(row, dim=-1).max())
            except Exception:
                pass
            return out
        return patched
    patched_classes = []
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
                patched_classes.append(f"{name}.{cn}")
    print("logits-hooked:", patched_classes, flush=True)

    # ── patched propose: verbatim copy of vLLM 0.23 EagleProposer.propose ──── #
    import vllm.v1.spec_decode.eagle as E
    from vllm.forward_context import set_forward_context

    @torch.no_grad()
    def propose_patched(self, target_token_ids, target_positions, target_hidden_states,
                        next_token_ids, token_indices_to_sample, common_attn_metadata,
                        sampling_metadata, mm_embed_inputs=None,
                        num_rejected_tokens_gpu=None, slot_mappings=None):
        EL["n_propose"] += 1
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids, next_token_ids=next_token_ids,
                target_positions=target_positions, target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample, cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu))

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata))

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens))

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs)
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)

        with set_forward_context(
                per_layer_attn_metadata, self.vllm_config, num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    slot_mapping_size, common_attn_metadata.slot_mapping)):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(True)

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata)
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            self.positions[:batch_size] = positions

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata)
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError("Unsupported attention metadata type")

        draft_token_ids_list = [draft_token_ids]

        # ---- arm control: token 0 drafted; cum from its top prob ---- #
        arm_kind, arm_val = EL["arm"]
        cum = EL["top_p"]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size))

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]).clone()

        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0
        for token_index in range(self.num_speculative_tokens - 1):
            # ---- early-break decision BEFORE spending the next draft forward ---- #
            n_drafted = len(draft_token_ids_list)
            if arm_kind == "fixed" and n_drafted >= arm_val:
                break
            if arm_kind == "cum" and cum < arm_val:
                break

            input_ids = draft_token_ids_list[-1].int()

            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions, common_attn_metadata, batch_size, input_batch_size,
                    block_size)

            if not self.constant_draft_positions or token_index == 0:
                _, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(
                        common_attn_metadata, draft_index=token_index + 1))

            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_kwargs = {"input_ids": input_ids,
                            "positions": self._get_positions(input_batch_size),
                            "inputs_embeds": inputs_embeds}
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                    per_layer_attn_metadata, self.vllm_config,
                    num_tokens=input_batch_size,
                    num_tokens_across_dp=batch_size_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(input_batch_size)):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata)
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)
            cum *= EL["top_p"]

        EL["depths"].append(len(draft_token_ids_list))

        # ---- pad to the engine contract [B, KMAX]: pad tokens get rejected ---- #
        while len(draft_token_ids_list) < self.num_speculative_tokens:
            draft_token_ids_list.append(draft_token_ids_list[-1])
            if draft_probs_list is not None:
                draft_probs_list.append(draft_probs_list[-1])

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    # DIAG FINDING (modal_vllm_tailprune_diag.py): EagleProposer does NOT override propose —
    # all proposer subclasses inherit ONE implementation from SpecDecodeBaseProposer, and the
    # engine's drafter instance is not an EagleProposer, so patching the subclass intercepted
    # nothing (n_propose stayed 0; fixed2/fixed7 separation 1.001). Patch the BASE class, and
    # every eagle-family subclass for good measure.
    import vllm.v1.spec_decode.llm_base_proposer as LBP
    LBP.SpecDecodeBaseProposer.propose = propose_patched
    for cls in (getattr(E, "EagleProposer", None),):
        if cls is not None:
            cls.propose = propose_patched
    print("SpecDecodeBaseProposer.propose PATCHED (base + eagle subclass)", flush=True)

    # ── engine ───────────────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    llm = LLM(model=TARGET,
              speculative_config={"method": "eagle3", "model": EAGLE,
                                  "num_speculative_tokens": KMAX},
              gpu_memory_utilization=0.8, max_model_len=2048,
              enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=maxtok)

    def take_odd(x, m): return list(x)[1:2 * m:2]
    he = load_dataset("openai/openai_humaneval", split="test")["prompt"]
    gk = load_dataset("openai/gsm8k", "main", split="test")["question"]
    mb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").map(
        lambda x: {"p": x["prompt"][0]})["p"]
    workloads = {"humaneval": take_odd(he, n), "gsm8k": take_odd(gk, n),
                 "mt_bench": take_odd(mb, n)}

    def bench_once(prompts):
        import torch as T
        T.cuda.synchronize(); t0 = time.time(); ntok = 0
        for p in prompts:
            out = llm.generate([p], sp, use_tqdm=False)
            ntok += len(out[0].outputs[0].token_ids)
        T.cuda.synchronize()
        return ntok / (time.time() - t0)

    # sanity: patched propose must EXECUTE and generate coherently — fail fast otherwise
    EL["arm"] = ("fixed", KMAX)
    txt = llm.generate(["What is 17*24? Show your steps."], sp, use_tqdm=False)[0].outputs[0].text
    print(f"SANITY GEN (patched, full K): n_propose={EL['n_propose']} {repr(txt[:120])}",
          flush=True)
    assert EL["n_propose"] > 0, "patched propose did NOT execute — abort (no-op benchmark)"
    # arm-separation smoke check: fixed2 must be measurably slower than fixed7
    EL["arm"] = ("fixed", 2)
    llm.generate(["Warm"], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
    d2 = EL["depths"][-3:]
    print(f"fixed2 smoke: recent drafted-lens {d2}", flush=True)
    assert d2 and max(d2) <= 2, f"fixed2 arm not enforced (drafted {d2}) — abort"

    arms = [("fixed", 2), ("fixed", 3), ("fixed", 4), ("fixed", KMAX),
            ("cum", 0.05), ("cum", 0.2)]
    names = [f"{k}{v}" for k, v in arms]

    tps = {w: {nm: [] for nm in names} for w in workloads}
    depth_log = {nm: [] for nm in names}
    for c in range(cycles):
        order = arms[c % len(arms):] + arms[:c % len(arms)]
        for arm in order:
            nm = f"{arm[0]}{arm[1]}"
            EL["arm"] = arm
            for w, prompts in workloads.items():
                EL["depths"] = []
                t = bench_once(prompts)
                tps[w][nm].append(round(t, 2))
                if EL["depths"]:
                    depth_log[nm].append(round(float(np.mean(EL["depths"])), 2))
            print(f"cycle {c+1}/{cycles} {nm:10s} " +
                  " ".join(f"{w}:{tps[w][nm][-1]}" for w in workloads), flush=True)

    # paired analysis: cum arms vs best fixed arm (strongest per workload by mean)
    summary = {}
    for w in workloads:
        fixed_means = {nm: float(np.mean(tps[w][nm])) for nm in names if nm.startswith("fixed")}
        per_arm = {}
        for nm in names:
            if nm.startswith("fixed"):
                continue
            best = {}
            for fnm in fixed_means:
                diffs = [100 * (a / b - 1) for a, b in zip(tps[w][nm], tps[w][fnm])]
                best[fnm] = {"mean_gain_pct": round(float(np.mean(diffs)), 2),
                             "se": round(float(np.std(diffs) / np.sqrt(len(diffs))), 2)}
            strongest = min(best, key=lambda k: best[k]["mean_gain_pct"])
            per_arm[nm] = {"vs_each_fixed": best,
                           "vs_strongest_fixed": {"config": strongest, **best[strongest]}}
            print(f">>> {w:10s} {nm}: vs strongest fixed ({strongest}) "
                  f"{best[strongest]['mean_gain_pct']:+.2f}% ±{best[strongest]['se']:.2f}",
                  flush=True)
        summary[w] = {"fixed_means": {k: round(v, 2) for k, v in fixed_means.items()},
                      "cum_arms": per_arm}
    mean_depths = {nm: (round(float(np.mean(v)), 2) if v else None)
                   for nm, v in depth_log.items()}
    print("mean drafted tokens per arm:", mean_depths, flush=True)

    os.makedirs("/root/out", exist_ok=True)
    with open(f"/root/out/vllm_tailprune_{model_key}.json", "w") as f:
        json.dump({"model": model_key, "target": TARGET, "n": n, "maxtok": maxtok,
                   "cycles": cycles, "KMAX": KMAX, "tps": tps,
                   "mean_drafted": mean_depths, "summary": summary}, f, indent=2)
    vol.commit()
    return {w: {nm: summary[w]["cum_arms"][nm]["vs_strongest_fixed"]
                for nm in summary[w]["cum_arms"]} for w in summary}


@app.local_entrypoint()
def main(model: str = "llama8b", n: int = 10, maxtok: int = 128, cycles: int = 4):
    print(f"vllm tailprune: model={model} n={n} maxtok={maxtok} cycles={cycles}")
    print(json.dumps(run.remote(model, n, maxtok, cycles), indent=2))
