"""
DFlash smoke test — RTX 5070 8GB local setup.

Target : Qwen/Qwen3-4B  (4-bit quantised via bitsandbytes, ~2.5 GB)
Draft  : z-lab/Qwen3-4B-DFlash-b16  (~0.5B, bf16, ~1 GB)
Total  : ~3.5 GB — fits on 8 GB with room for KV cache.

Calls dflash_generate() directly with return_stats=True so we can
inspect per-block acceptance_lengths from the start.
"""

import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent / "dflash_repo"))
from dflash.model import dflash_generate

# ── config ────────────────────────────────────────────────────────────────────
TARGET_ID  = str(Path(__file__).parent / "models" / "Qwen3-4B")
DRAFT_ID   = str(Path(__file__).parent / "dflash_repo" / "models" / "Qwen3-4B-DFlash-b16")
DEVICE     = "cuda:0"
MAX_NEW    = 256
BLOCK_SIZE = 16        # DFlash default for this checkpoint
TEMPERATURE = 0.0      # greedy — cleanest for acceptance analysis

PROMPTS = [
    "Solve step by step: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the remainder at $2 each. How much does she make per day?",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France and why is it historically significant?",
]

# ── 4-bit quantisation config ──────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

def vram_mb():
    return torch.cuda.memory_allocated(DEVICE) / 1024**2

def load_models():
    print(f"\n[1/3] Loading tokenizer from {TARGET_ID}")
    tok = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"[2/3] Loading target model (4-bit)  — VRAM before: {vram_mb():.0f} MB")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        quantization_config=bnb_cfg,
        device_map=DEVICE,
        output_hidden_states=True,  # needed by DFlash draft conditioning
    )
    target.eval()
    print(f"       Target loaded                 — VRAM after:  {vram_mb():.0f} MB")

    print(f"[3/3] Loading DFlash draft model    — VRAM before: {vram_mb():.0f} MB")
    draft = AutoModel.from_pretrained(
        DRAFT_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    draft.eval()
    print(f"       Draft loaded                  — VRAM after:  {vram_mb():.0f} MB\n")

    return tok, target, draft


def run_prompt(tok, target, draft, prompt: str, idx: int):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    print(f"── Prompt {idx+1} ({input_ids.shape[1]} input tokens) ──")
    print(f"   Q: {prompt[:80]}...")

    stop_ids = [tok.eos_token_id]
    if hasattr(tok, "convert_tokens_to_ids"):
        end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if end_id: stop_ids.append(end_id)

    stats = dflash_generate(
        model=draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW,
        stop_token_ids=stop_ids,
        temperature=TEMPERATURE,
        block_size=BLOCK_SIZE,
        return_stats=True,
    )

    output_text = tok.decode(
        stats.output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    acc_lens = stats.acceptance_lengths          # list of ints, one per draft step
    mean_acc = sum(acc_lens) / len(acc_lens) if acc_lens else 0
    # token-level acceptance rate: accepted / (block_size) per step
    token_acc_rate = (mean_acc - 1) / (BLOCK_SIZE - 1) if BLOCK_SIZE > 1 else 0

    print(f"   A: {output_text[:120]}...")
    print(f"   Steps          : {len(acc_lens)}")
    print(f"   Output tokens  : {stats.num_output_tokens}")
    print(f"   Mean acc/step  : {mean_acc:.2f}  (block_size={BLOCK_SIZE})")
    print(f"   Token acc rate : {token_acc_rate:.1%}  (accepted draft tokens / available slots)")
    print(f"   TTFT           : {stats.time_to_first_token*1000:.1f} ms")
    print(f"   ms/output tok  : {stats.time_per_output_token*1000:.2f} ms")
    print(f"   Tok/s          : {1/stats.time_per_output_token:.1f}")
    print(f"   Acceptance dist: {dict(zip(*[list(x) for x in zip(*[(v, acc_lens.count(v)) for v in sorted(set(acc_lens))])]))}")
    print()

    return {
        "prompt_idx": idx,
        "prompt": prompt,
        "output": output_text,
        "num_output_tokens": stats.num_output_tokens,
        "num_steps": len(acc_lens),
        "mean_acceptance_per_step": mean_acc,
        "token_acceptance_rate": token_acc_rate,
        "time_to_first_token_ms": stats.time_to_first_token * 1000,
        "ms_per_output_token": stats.time_per_output_token * 1000,
        "tokens_per_sec": 1 / stats.time_per_output_token,
        "acceptance_lengths": acc_lens,
    }


def main():
    print("=" * 60)
    print("DFlash Local Smoke Test — RTX 5070 8GB")
    print("=" * 60)

    tok, target, draft = load_models()

    results = []
    for i, prompt in enumerate(PROMPTS):
        r = run_prompt(tok, target, draft, prompt, i)
        results.append(r)

    # summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_tps   = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_acc   = sum(r["mean_acceptance_per_step"] for r in results) / len(results)
    avg_trate = sum(r["token_acceptance_rate"] for r in results) / len(results)
    print(f"Avg tokens/sec        : {avg_tps:.1f}")
    print(f"Avg acceptance/step   : {avg_acc:.2f} / {BLOCK_SIZE}")
    print(f"Avg token accept rate : {avg_trate:.1%}")
    print()
    print("Token accept rate is the KEY baseline for our BSP research.")
    print("BSP goal: raise this rate by accepting semantically-equivalent")
    print("draft tokens that the lexical check currently rejects.")

    out_path = Path(__file__).parent / "results" / "dflash_smoke_test.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
