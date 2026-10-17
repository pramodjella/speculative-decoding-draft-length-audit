"""Option B (diagnostic): re-derive draft distributions by a clean ea_layer forward
(teacher-forced), bypassing tree node-ordering. First nail the ea_layer API and
validate a depth-1 re-derivation, then build the per-step chain version.

Steps: dump ea_layer forward sig + components; run base_model.model -> fused hidden
[1,T,12288]; call ea_layer(hidden, input_ids) -> out_hidden -> norm+lm_head ->
draft logits; validate d2t-argmax(draft[i]) vs token[i+1] across input alignments.
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.53.1", "accelerate==0.26.0",
                 "sentencepiece", "huggingface_hub", "fschat")
    .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
    .env({"PYTHONPATH": "/root/EAGLE"})
)
app = modal.App("eagle-rederive")

BASE = "meta-llama/Llama-3.1-8B-Instruct"
EA = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


@app.function(image=image, gpu="H100", timeout=2400,
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import torch, inspect
    from eagle.model.ea_model import EaModel

    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=BASE, ea_model_path=EA,
                                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                    device_map="auto", total_token=-1)
    model.eval()
    tok = model.get_tokenizer()
    ea = model.ea_layer
    dev = model.base_model.device

    print("ea_layer type:", type(ea).__name__, flush=True)
    try: print("ea forward sig:", inspect.signature(ea.forward), flush=True)
    except Exception as e: print("sig err", e, flush=True)
    print("ea components:", [a for a in ("norm", "lm_head", "embed_tokens", "fc", "midlayer", "d2t", "t2d") if hasattr(ea, a)], flush=True)
    d2t = ea.d2t.detach().to(dev) if hasattr(ea, "d2t") else None

    # build a real sequence via greedy base decoding
    ids = tok.apply_chat_template([{"role": "user", "content": "What is 248*17? Show steps."}],
                                  add_generation_prompt=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        gen = model.eagenerate(ids, temperature=0.0, max_new_tokens=40)
    S = gen if torch.is_tensor(gen) else gen[0]
    if S.dim() == 1:
        S = S[None]
    T = S.shape[1]
    print(f"seq len {T}", flush=True)

    # fused hidden states from base model (EAGLE-3 base returns the 3-layer fused dim)
    with torch.no_grad():
        outs = model.base_model.model(input_ids=S)
        hidden = outs[0] if isinstance(outs, (tuple, list)) else outs.last_hidden_state
    print("base_model.model hidden shape:", tuple(hidden.shape),
          "(expect [...,12288] if EAGLE-3 fused)", flush=True)
    from itertools import combinations
    with torch.no_grad():
        o2 = model.base_model.model(input_ids=S, output_hidden_states=True)
        hs = list(o2.hidden_states if hasattr(o2, "hidden_states") else o2[-1])
    print(f"  output_hidden_states n={len(hs)} each {tuple(hs[0].shape)}", flush=True)
    cand = {combo: torch.cat([hs[i] for i in combo], dim=-1)
            for combo in combinations(range(len(hs)), 3)}
    if hidden.shape[-1] == 12288:
        cand[("base",)] = hidden

    am = torch.ones_like(S); pos = torch.arange(T, device=dev)[None]
    def draft_logits(h, inp):
        with torch.no_grad():
            try:
                out = ea(h, input_ids=inp, attention_mask=am, position_ids=pos, use_cache=False)
            except TypeError:
                out = ea(h, input_ids=inp)
            oh = out[0] if isinstance(out, (tuple, list)) else out
            return ea.lm_head(ea.norm(oh)).float()

    print("  fusion search (draft argmax == actual next token; want ~60-80%):", flush=True)
    best = (None, 0)
    for combo, h in cand.items():
        try:
            lg = draft_logits(h, S)
            m = t = 0
            for i in range(lg.shape[1] - 1):
                red = int(lg[0, i].argmax())
                full = red + int(d2t[red]) if d2t is not None else red
                t += 1; m += int(full == int(S[0, i + 1]))
            r = 100 * m / max(1, t)
            print(f"    fused={combo}: {m}/{t} = {r:.0f}%", flush=True)
            if r > best[1]:
                best = (combo, r)
        except Exception as e:
            print(f"    fused={combo} FAILED: {repr(e)[:110]}", flush=True)
    print(f"  -> BEST fusion {best[0]} at {best[1]:.0f}%", flush=True)
    return {"best_combo": str(best[0]), "best_pct": round(best[1], 1)}


@app.local_entrypoint()
def main():
    print(run.remote())
