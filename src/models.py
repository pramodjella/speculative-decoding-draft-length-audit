"""Model + tokenizer loading helpers (Hugging Face Transformers / vLLM).

Kept thin on purpose: swap ``MODEL_ID`` in the config and everything downstream
(quantization, speculation, benchmarking) reuses these loaders.
"""
from __future__ import annotations

from typing import Any


def load_hf_model(model_id: str, dtype: str = "float16", device: str = "auto") -> Any:
    """Load a causal LM with Hugging Face Transformers.

    Returns (model, tokenizer). Import is local so ``--smoke`` runs without torch.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype, torch.float16),
        device_map=device,
    )
    model.eval()
    return model, tok


def load_vllm(model_id: str, **kwargs) -> Any:
    """Load a model under vLLM for serving-style benchmarks (optional)."""
    from vllm import LLM
    return LLM(model=model_id, **kwargs)


class DummyModel:
    """A tiny CPU stand-in used by ``--smoke`` so the pipeline runs anywhere."""

    def __init__(self, vocab: int = 256):
        self.vocab = vocab

    def generate(self, prompt: str, max_new_tokens: int = 16):
        # deterministic pseudo-output; replace with a real model in your runs
        return prompt + " " + " ".join(f"tok{i}" for i in range(max_new_tokens))
