"""
DFlash Acceptance Logger — characterises token-level over-rejection.

This is the first research instrument for the BSP (Block Semantic Probe) paper.
It runs DFlash on GSM8K examples and captures per-block acceptance statistics,
rejected-but-correct-meaning patterns, and positional acceptance decay — the
core evidence for the "semantic over-rejection" phenomenon we propose to fix.

Output:
  results/dflash_acceptance_log.json   — raw per-step data
  results/dflash_acceptance_summary.json — aggregate stats for the paper

Key metrics:
  - mean_acc_per_step      : avg tokens accepted per draft step (τ)
  - token_accept_rate      : (τ-1)/(block_size-1) — fraction of draft slots used
  - positional_accept_rate : acceptance rate at each position within the block
    (position 0 always accepted = the greedy token; position k = draft token k)
  - first_reject_pos_dist  : distribution of "where does the first rejection happen"
    → if rejections cluster at pos 1-2, semantic-level accept may help enormously
    → if they're spread uniformly, the problem is harder

Run:  python dflash_acceptance_logger.py --n 50 --split train
"""

import json, sys, argparse, torch, numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent / "dflash_repo"))
from dflash.model import dflash_generate

# ── config ────────────────────────────────────────────────────────────────────
TARGET_ID  = str(Path(__file__).parent / "models" / "Qwen3-4B")
DRAFT_ID   = str(Path(__file__).parent / "dflash_repo" / "models" / "Qwen3-4B-DFlash-b16")
DEVICE     = "cuda:0"
BLOCK_SIZE = 16
MAX_NEW    = 512
TEMP       = 0.0   # greedy — cleanest acceptance signal

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ── instrumented generate ─────────────────────────────────────────────────────

def dflash_generate_with_positions(model, target, input_ids, max_new_tokens,
                                    stop_token_ids, temperature, block_size):
    """
    Wraps dflash_generate and additionally captures the per-position
    accept/reject pattern within each block. Returns stats + position data.

    We re-implement the inner loop minimally rather than monkey-patching,
    so this stays readable for the paper supplementary.
    """
    import time
    from transformers import DynamicCache
    from dflash.model import extract_context_feature, sample, _cuda_time

    num_input_tokens = input_ids.shape[1]
    max_length       = num_input_tokens + max_new_tokens
    mask_token_id    = model.mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id,
        dtype=torch.long, device=target.device,
    )
    position_ids            = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target  = DynamicCache()
    past_key_values_draft   = DynamicCache()

    # prefill
    t0 = _cuda_time()
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens]  = sample(out.logits, temperature)
    target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)
    ttft = _cuda_time() - t0

    # decode loop
    acceptance_lengths    = []
    position_accept_mask  = []   # per step: list of bool per position in block
    first_reject_positions = []  # position of first rejection per step

    t1 = _cuda_time()
    start = num_input_tokens

    while start < max_length:
        block_ids      = output_ids[:, start : start + block_size].clone()
        block_pos_ids  = position_ids[:, start : start + block_size]

        noise_emb   = target.model.embed_tokens(block_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size :, :])
        past_key_values_draft.crop(start)
        block_ids[:, 1:] = sample(draft_logits)

        out = target(
            block_ids,
            position_ids=block_pos_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(out.logits, temperature)

        # ── per-position accept/reject (this is the key logging) ──────────
        match = (block_ids[:, 1:] == posterior[:, :-1])[0]  # shape: (block_size-1,)
        cumok = match.cumprod(dim=0)
        acc_len = cumok.sum().item()

        position_accept_mask.append(match.cpu().tolist())    # True/False per slot

        # first rejection = first False in cumulative product
        if acc_len < block_size - 1:
            first_reject_positions.append(int(acc_len))      # 0-indexed draft slot
        else:
            first_reject_positions.append(block_size - 1)    # all accepted

        acceptance_lengths.append(int(acc_len) + 1)

        output_ids[:, start : start + acc_len + 1] = block_ids[:, : acc_len + 1]
        output_ids[:, start + acc_len + 1] = posterior[:, acc_len]
        start += acc_len + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            out.hidden_states, model.target_layer_ids
        )[:, : acc_len + 1, :]

        if stop_token_ids and any(
            sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    total_time = _cuda_time() - t1
    num_out    = output_ids.shape[1] - num_input_tokens

    return dict(
        output_ids            = output_ids,
        num_input_tokens      = num_input_tokens,
        num_output_tokens     = num_out,
        time_to_first_token   = ttft,
        time_per_output_token = total_time / max(num_out, 1),
        acceptance_lengths    = acceptance_lengths,
        position_accept_mask  = position_accept_mask,   # NEW
        first_reject_positions = first_reject_positions, # NEW
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",     type=int,  default=50,    help="Number of GSM8K examples")
    p.add_argument("--split", type=str,  default="train")
    p.add_argument("--seed",  type=int,  default=42)
    return p.parse_args()


def load_models():
    print(f"Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"Loading target (4-bit)...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        quantization_config=bnb_cfg,
        device_map=DEVICE,
        output_hidden_states=True,
    )
    target.eval()

    print(f"Loading DFlash draft...")
    draft = AutoModel.from_pretrained(
        DRAFT_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    draft.eval()

    vram = torch.cuda.memory_allocated(DEVICE) / 1024**2
    print(f"Models loaded — VRAM: {vram:.0f} MB / {torch.cuda.get_device_properties(0).total_memory//1024**2} MB\n")
    return tok, target, draft


def run_gsm8k(tok, target, draft, n, split, seed):
    ds  = load_dataset("gsm8k", "main", split=split)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()

    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    stop_ids = [s for s in stop_ids if s is not None]

    all_results = []
    for i, idx in enumerate(idxs):
        q = ds[idx]["question"]
        messages = [{"role": "user", "content": q}]
        input_ids = tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(DEVICE)

        with torch.inference_mode():
            r = dflash_generate_with_positions(
                draft, target, input_ids,
                max_new_tokens=MAX_NEW,
                stop_token_ids=stop_ids,
                temperature=TEMP,
                block_size=BLOCK_SIZE,
            )

        acc_lens     = r["acceptance_lengths"]
        pos_masks    = r["position_accept_mask"]
        frp          = r["first_reject_positions"]
        mean_acc     = np.mean(acc_lens)
        token_rate   = (mean_acc - 1) / (BLOCK_SIZE - 1)

        # per-position acceptance rate across all steps in this example
        if pos_masks:
            pos_rates = np.mean(pos_masks, axis=0).tolist()
        else:
            pos_rates = []

        output_text = tok.decode(
            r["output_ids"][0][r["num_input_tokens"]:], skip_special_tokens=True
        )

        result = {
            "example_idx"          : int(idx),
            "question"             : q,
            "output"               : output_text,
            "num_output_tokens"    : r["num_output_tokens"],
            "num_steps"            : len(acc_lens),
            "mean_acc_per_step"    : float(mean_acc),
            "token_accept_rate"    : float(token_rate),
            "ms_per_output_token"  : r["time_per_output_token"] * 1000,
            "tokens_per_sec"       : 1 / r["time_per_output_token"],
            "acceptance_lengths"   : acc_lens,
            "positional_accept_rates" : pos_rates,  # rate at each block position
            "first_reject_pos_dist": frp,
        }
        all_results.append(result)

        bar = "█" * int(token_rate * 20) + "░" * (20 - int(token_rate * 20))
        print(f"[{i+1:3d}/{n}] τ={mean_acc:.2f}  accept={token_rate:.0%}  [{bar}]  "
              f"{r['num_output_tokens']} tok  {1/r['time_per_output_token']:.0f} tok/s")

    return all_results


def summarise(results, block_size):
    all_acc_lens = [a for r in results for a in r["acceptance_lengths"]]
    all_frp      = [f for r in results for f in r["first_reject_pos_dist"]]

    # positional acceptance rate across all examples
    all_pos_masks = []
    for r in results:
        n = len(r["acceptance_lengths"])
        for step in range(n):
            if step < len(r["positional_accept_rates"]):
                pass  # already aggregated per example
    # recompute raw from first_reject_pos_dist
    pos_counts = Counter(all_frp)

    # per-position cumulative accept rate
    # position k is accepted if first_reject > k
    total_steps = len(all_frp)
    pos_acc_rates = []
    for pos in range(block_size - 1):
        accepted_at_pos = sum(1 for f in all_frp if f > pos)
        pos_acc_rates.append(accepted_at_pos / total_steps)

    mean_tau   = np.mean(all_acc_lens)
    mean_rate  = (mean_tau - 1) / (block_size - 1)

    summary = {
        "num_examples"            : len(results),
        "block_size"              : block_size,
        "mean_tau"                : float(mean_tau),
        "mean_token_accept_rate"  : float(mean_rate),
        "acceptance_length_dist"  : dict(Counter(all_acc_lens)),
        "first_reject_pos_dist"   : dict(sorted(Counter(all_frp).items())),
        "positional_accept_rates" : pos_acc_rates,  # rate[k] = P(accept at position k)
        "avg_tok_per_sec"         : float(np.mean([r["tokens_per_sec"] for r in results])),
        "bsp_opportunity"         : {
            "description": (
                "Fraction of rejection events at position 0-2 of the draft block. "
                "High value = early rejection dominates = BSP has largest headroom."
            ),
            "early_reject_frac": sum(
                1 for f in all_frp if f < 3
            ) / len(all_frp) if all_frp else 0,
        },
    }

    print("\n" + "=" * 60)
    print("ACCEPTANCE SUMMARY")
    print("=" * 60)
    print(f"Mean τ (accepted/step)  : {mean_tau:.3f}")
    print(f"Token acceptance rate   : {mean_rate:.1%}")
    print(f"Avg throughput          : {summary['avg_tok_per_sec']:.1f} tok/s")
    print(f"\nFirst-rejection position distribution (0=first draft slot):")
    for pos, cnt in sorted(Counter(all_frp).items()):
        bar = "█" * int(40 * cnt / total_steps)
        print(f"  pos {pos:2d}: {cnt:5d} ({cnt/total_steps:.1%}) {bar}")
    print(f"\nPositional accept rates (what BSP must beat at each position):")
    for k, rate in enumerate(pos_acc_rates):
        bar = "█" * int(rate * 30)
        print(f"  draft pos {k:2d}: {rate:.1%}  {bar}")
    print(f"\nBSP Opportunity — early-rejection fraction: "
          f"{summary['bsp_opportunity']['early_reject_frac']:.1%}")
    print("  (if high: semantic accept at block-level can capture most gains)")

    return summary


def main():
    args = parse_args()
    tok, target, draft = load_models()

    print(f"Running on {args.n} GSM8K examples (split={args.split})...\n")
    results = run_gsm8k(tok, target, draft, args.n, args.split, args.seed)

    summary = summarise(results, BLOCK_SIZE)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "dflash_acceptance_log.json").write_text(json.dumps(results, indent=2))
    (out_dir / "dflash_acceptance_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → results/dflash_acceptance_log.json")
    print(f"Saved → results/dflash_acceptance_summary.json")


if __name__ == "__main__":
    main()
