"""Physical speculative decoding runner with proper KV-cache reuse.

Design:
  - Both draft and target maintain persistent KV caches across decoding steps
  - Only NEW tokens are fed to each model (no re-processing)
  - On rejection, KV caches are trimmed via _slice_kv()
  - Correction token is fed through both models to keep caches in sync

Note on small models (< 3B params):
  HuggingFace .generate() uses highly optimized C++/CUDA internals that are
  hard to match from pure Python. For small model pairs (0.5B/1.5B), the Python
  loop overhead dominates, so absolute speedup may be < 1.0x. The relative
  comparison between controllers is still valid and meaningful.
"""
import time
import torch
import torch.nn.functional as F
from controllers.linucb import LinUCBController


def _get_cache_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
        return past_key_values.key_cache[0].shape[2]
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "get_seq_len"):
        return past_key_values.get_seq_len()
    # tuple of tuples
    return past_key_values[0][0].shape[2]


def _get_attention_mask(input_tensor, past_key_values, device) -> torch.Tensor:
    past_len = _get_cache_len(past_key_values)
    seq_len = input_tensor.shape[1]
    return torch.ones((1, past_len + seq_len), dtype=torch.long, device=device)


class PhysicalSpeculativeRunner:
    def __init__(self, target_model, draft_model, tokenizer, controller=None):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.controller = controller

        self.target_model.eval()
        self.draft_model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict:
        device = next(iter(self.target_model.parameters())).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        n_generated = 0
        steps = 0
        total_accepted = 0
        total_wasted = 0
        step_logs = []
        self._last_entropy = 0.0  # Track for contextual controllers
        generated_ids = []

        # Reset LinUCB episode state for this new generation
        if isinstance(self.controller, LinUCBController):
            self.controller.set_max_steps(max_new_tokens)
            self.controller.reset_episode()

        # Which controllers actually consume the draft entropy signal? The
        # threshold (tau) controller checks it on every draft token; LinUCB uses
        # the first token's entropy as context. Everything else ignores it, so we
        # can skip the per-token full-vocab softmax + D2H sync for those.
        is_linucb = isinstance(self.controller, LinUCBController)
        has_tau = hasattr(self.controller, "tau")
        compute_entropy = is_linucb or has_tau

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # ── Prefill both models ──
        attention_mask_init = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        draft_out = self.draft_model(input_ids, attention_mask=attention_mask_init, use_cache=True)
        draft_kv = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]
        del draft_out

        target_out = self.target_model(input_ids, attention_mask=attention_mask_init, use_cache=True)
        target_kv = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]
        del target_out

        # Sample the first token from target prefill output
        x_next = torch.argmax(target_next_logits, dim=-1)[0].item()
        generated_ids.append(x_next)
        n_generated = 1

        if x_next == self.tokenizer.eos_token_id:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.time() - start_time
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return {
                "generated_tokens": 1,
                "steps": 0,
                "latency_s": latency,
                "tokens_per_sec": 1.0 / latency if latency > 0 else 0,
                "avg_accepted": 0.0,
                "accepted_tokens_per_step": float("nan"),
                "total_wasted": 0,
                "step_logs": [],
                "text": text,
                "generated_ids": generated_ids,
            }

        # Initialize draft cache by feeding x_next
        x_next_tensor = torch.tensor([[x_next]], device=device)
        attention_mask_d = _get_attention_mask(x_next_tensor, draft_kv, device)
        d_out = self.draft_model(x_next_tensor, past_key_values=draft_kv, attention_mask=attention_mask_d, use_cache=True)
        draft_kv = d_out.past_key_values
        draft_next_logits = d_out.logits[:, -1, :]
        del d_out

        while n_generated < max_new_tokens:
            K = self._choose_K()

            # ── Draft Phase ──
            draft_tok_tensors = []
            first_entropy = 0.0
            logits = draft_next_logits

            for j in range(K):
                tok = torch.argmax(logits, dim=-1)  # [1], stays on GPU
                draft_tok_tensors.append(tok)

                # Only compute entropy when a controller consumes it. tau needs it
                # every token (early-stop check); LinUCB needs only the first.
                if compute_entropy and (has_tau or j == 0):
                    probs = F.softmax(logits[0], dim=-1)
                    entropy = -(probs * torch.log2(probs + 1e-9)).sum().item()
                    if j == 0:
                        first_entropy = entropy
                    if has_tau and entropy > self.controller.tau:
                        break

                tok_tensor = tok.unsqueeze(0)
                attention_mask_d = _get_attention_mask(tok_tensor, draft_kv, device)
                out = self.draft_model(tok_tensor, past_key_values=draft_kv, attention_mask=attention_mask_d, use_cache=True)
                draft_kv = out.past_key_values
                logits = out.logits[:, -1, :]
                del out

            K_actual = len(draft_tok_tensors)
            draft_stack = torch.cat(draft_tok_tensors)  # [K_actual] on-device, no per-token sync

            # ── Target Verification (single batched forward pass) ──
            # Prepend x_next to the draft tokens (all on-device; no per-token sync).
            x_next_t = torch.tensor([x_next], device=device, dtype=draft_stack.dtype)
            target_input = torch.cat([x_next_t, draft_stack]).unsqueeze(0)
            attention_mask_t = _get_attention_mask(target_input, target_kv, device)
            target_out = self.target_model(target_input, past_key_values=target_kv, attention_mask=attention_mask_t, use_cache=True)
            target_kv = target_out.past_key_values

            # Verify in one shot: position i predicts the token that should follow
            # draft_stack[i-1]; accept the leading run where target agrees with draft.
            target_preds = torch.argmax(target_out.logits[0, : K_actual + 1, :], dim=-1)  # [K_actual+1]
            mismatches = (target_preds[:K_actual] != draft_stack).nonzero()
            if mismatches.numel() > 0:
                accepted = int(mismatches[0].item())
            else:
                accepted = K_actual
            # target_preds[accepted] is the mismatch token, or the bonus token if all accepted
            correction = int(target_preds[accepted].item())

            del target_out

            draft_ids = draft_stack.tolist()  # single D2H sync for logging / output

            # ── Determine correction and fix caches ──
            discard = K_actual - accepted
            if discard > 0:
                target_kv = _slice_kv(target_kv, -discard)
                draft_kv = _slice_kv(draft_kv, -discard)

            # ── Track generated tokens ──
            if accepted > 0:
                generated_ids.extend(draft_ids[:accepted])
            generated_ids.append(correction)

            # ── Feed correction through draft model only ──
            # (No target model run for correction! We prepend it to target_input in the next step)
            x_next = correction
            n_tokens = accepted + 1
            n_generated += n_tokens
            steps += 1
            total_accepted += accepted
            total_wasted += discard

            self._last_entropy = first_entropy

            step_logs.append({
                "K": K_actual,
                "accepted": accepted,
                "entropy": self._last_entropy,
            })

            self._update_controller(K_actual, accepted)

            if correction == self.tokenizer.eos_token_id:
                break

            # Initialize next draft logits by feeding x_next
            x_next_tensor = torch.tensor([[x_next]], device=device)
            attention_mask_d = _get_attention_mask(x_next_tensor, draft_kv, device)
            d_out = self.draft_model(x_next_tensor, past_key_values=draft_kv, attention_mask=attention_mask_d, use_cache=True)
            draft_kv = d_out.past_key_values
            draft_next_logits = d_out.logits[:, -1, :]
            del d_out

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - start_time
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "generated_tokens": n_generated,
            "steps": steps,
            "latency_s": latency,
            "tokens_per_sec": n_generated / latency if latency > 0 else 0,
            "avg_accepted": total_accepted / max(1, steps),
            "accepted_tokens_per_step": (total_accepted / steps + 1.0) if steps > 0 else float("nan"),
            "total_wasted": total_wasted,
            "step_logs": step_logs,
            "text": text,
            "generated_ids": generated_ids,
        }

    def _choose_K(self) -> int:
        if isinstance(self.controller, int):
            return max(1, self.controller)
        if hasattr(self.controller, "tau"):
            return self.controller.max_len if hasattr(self.controller, "max_len") else 8
        if isinstance(self.controller, LinUCBController):
            # Pass last observed entropy as context signal
            return max(1, self.controller.choose(entropy=self._last_entropy))
        if hasattr(self.controller, "choose"):
            return max(1, self.controller.choose())
        return 4

    def _update_controller(self, K_actual: int, accepted: int):
        if not hasattr(self.controller, "update"):
            return
        import inspect
        sig = inspect.signature(self.controller.update)
        if "reward" in sig.parameters:
            reward = (accepted + 1) / (K_actual + 1)
            self.controller.update(K_actual, reward)
        else:
            self.controller.update(K_actual, accepted)


class PythonAutoregressiveBaseline:
    """Honest, apples-to-apples target-only baseline (E0.1).

    Uses the *exact same* pure-Python forward-call + KV-cache code path as
    PhysicalSpeculativeRunner — NOT HuggingFace `.generate()`. This makes the
    net-speedup comparison fair: any speedup the speculative runner shows is a
    real algorithmic win, not an artifact of comparing Python against compiled
    C++/CUDA generate() internals.

    Greedy argmax decoding, identical to how the speculative runner verifies,
    so the two produce bit-identical token sequences (see verify_equivalence).
    """

    def __init__(self, target_model, tokenizer):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.target_model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> dict:
        device = next(iter(self.target_model.parameters())).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # ── Prefill ──
        attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
        out = self.target_model(input_ids, attention_mask=attn, use_cache=True)
        kv = out.past_key_values
        next_logits = out.logits[:, -1, :]
        del out

        x_next = torch.argmax(next_logits, dim=-1)[0].item()
        generated_ids.append(x_next)

        # ── Decode loop (one forward per token, same call shape as spec runner) ──
        while len(generated_ids) < max_new_tokens:
            if x_next == self.tokenizer.eos_token_id:
                break
            tok = torch.tensor([[x_next]], device=device)
            attn = _get_attention_mask(tok, kv, device)
            out = self.target_model(tok, past_key_values=kv, attention_mask=attn, use_cache=True)
            kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            del out
            x_next = torch.argmax(next_logits, dim=-1)[0].item()
            generated_ids.append(x_next)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - start_time
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "generated_tokens": len(generated_ids),
            "latency_s": latency,
            "tokens_per_sec": len(generated_ids) / latency if latency > 0 else 0,
            "generated_ids": generated_ids,
            "text": text,
        }


def _slice_kv(past_key_values, trim: int):
    """Remove `abs(trim)` tokens from the END of each layer's KV cache.

    past_key_values: tuple of (key, value) per layer OR Cache object
    trim: negative int, e.g. -3 means remove last 3 positions
    """
    if past_key_values is None or trim == 0:
        return past_key_values

    # 1. Use the official Hugging Face crop method if available (modern transformers Cache)
    if hasattr(past_key_values, "crop"):
        cur_len = _get_cache_len(past_key_values)
        past_key_values.crop(cur_len - abs(trim))
        return past_key_values

    # 2. Support DynamicCache / Cache objects that might not have crop (older Cache classes)
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_key_values, Cache):
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :trim, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :trim, :]
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = max(0, past_key_values._seen_tokens + trim)
            return past_key_values
    except ImportError:
        pass

    # 3. Legacy tuple format (handles 2-tuples and newer 3-tuples containing metadata)
    sliced = []
    for layer_kv in past_key_values:
        k = layer_kv[0]
        v = layer_kv[1]
        if len(layer_kv) > 2:
            metadata = layer_kv[2:]
            sliced.append((k[:, :, :trim, :], v[:, :, :trim, :]) + metadata)
        else:
            sliced.append((k[:, :, :trim, :], v[:, :, :trim, :]))
    return tuple(sliced)
