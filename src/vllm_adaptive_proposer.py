"""Adaptive-length ngram proposer for vLLM 0.23 (custom_class extension point).

vLLM calls a custom_class proposer like its NgramProposer:
    propose(sampled_token_ids: list[list[int]],
            num_tokens_no_spec: np.ndarray,
            token_ids_cpu: np.ndarray,
            slot_mappings=None) -> list[list[int]]   # variable length per request

We exploit that `num_speculative_tokens` is the MAX: each step a controller picks
K_i in [1, max_k] per request from the acceptance it can infer (how many of last
step's drafts were committed), then we do an n-gram lookup for K_i tokens. The
draft/verify forwards stay in vLLM's optimized (CUDA-graphed) path; only the tiny
K-decision is custom. Set:
    speculative_config = {"model": "vllm_adaptive_proposer.AdaptiveNgramProposer",
                          "num_speculative_tokens": MAX_K}

Controller is selected by env VS_CTRL in {fixed, history, ucb, eps}. (Entropy/margin
controllers need a neural draft, which this ngram hook does not have — those are
evaluated in the custom harness.)
"""
import os
import math
import numpy as np


class _AcceptanceController:
    """Picks K per step from running acceptance (the signal an ngram hook has).

    kind=fixed   : always max_k
    kind=history : raise K when recent acceptance high, lower when low
    kind=ucb     : UCB1 over arms, reward = (accepted+1)/(K+1)
    kind=eps     : epsilon-greedy over arms
    """
    def __init__(self, max_k, kind="history", arms=None, seed=0):
        self.max_k = max_k
        self.kind = kind
        self.arms = list(arms or sorted(set([1, 2, max_k // 2 or 1, max_k])))
        self.kind = kind
        self._val = {a: 0.0 for a in self.arms}
        self._n = {a: 0 for a in self.arms}
        self._t = 0
        self._hist = []
        self._idx = len(self.arms) // 2
        self._rng = np.random.default_rng(seed)
        self._last_choice = self.arms[self._idx]

    def choose(self):
        self._t += 1
        if self.kind == "fixed":
            c = self.max_k
        elif self.kind == "history":
            if len(self._hist) >= 5:
                r = sum(self._hist[-10:]) / len(self._hist[-10:])
                if r > 0.7:
                    self._idx = min(len(self.arms) - 1, self._idx + 1)
                elif r < 0.4:
                    self._idx = max(0, self._idx - 1)
            c = self.arms[self._idx]
        elif self.kind == "ucb":
            for a in self.arms:
                if self._n[a] == 0:
                    c = a; break
            else:
                c = max(self.arms, key=lambda a: self._val[a] + 2.0 * math.sqrt(math.log(self._t) / self._n[a]))
        elif self.kind == "eps":
            c = (int(self._rng.choice(self.arms)) if self._rng.random() < 0.1
                 else max(self.arms, key=lambda a: self._val[a]))
        else:
            c = self.max_k
        self._last_choice = int(c)
        return self._last_choice

    def update(self, k, accepted):
        rate = accepted / max(1, k)
        self._hist.append(rate)
        if k in self._n:
            self._n[k] += 1
            reward = (accepted + 1) / (k + 1)
            self._val[k] += (reward - self._val[k]) / self._n[k]


class AdaptiveNgramProposer:
    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.max_k = int(getattr(spec, "num_speculative_tokens", 4) or 4)
        self.n_match = int(os.environ.get("VS_NGRAM_N", "2"))    # suffix length to match
        self.kind = os.environ.get("VS_CTRL", "history")
        # per-slot state for batch=1 eval (one request at a time)
        self._ctrl = {}        # slot -> controller
        self._last_prop = {}   # slot -> list[int] proposed last step
        self._last_hist_len = {}

    def _get_ctrl(self, slot):
        if slot not in self._ctrl:
            self._ctrl[slot] = _AcceptanceController(self.max_k, self.kind)
        return self._ctrl[slot]

    def _ngram_lookup(self, hist, k):
        """Find the most recent earlier occurrence of the last n_match tokens;
        propose the k tokens that followed it."""
        n = self.n_match
        if len(hist) <= n:
            return []
        suffix = hist[-n:]
        # search backwards for an earlier match
        for start in range(len(hist) - n - 1, -1, -1):
            if hist[start:start + n] == suffix:
                nxt = hist[start + n: start + n + k]
                return [int(t) for t in nxt]
        return []

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        out = []
        for i, sampled in enumerate(sampled_token_ids):
            if not len(sampled):
                out.append([]); continue
            ntok = int(num_tokens_no_spec[i])
            hist = list(token_ids_cpu[i][:ntok])

            # reset per-slot state on a new prompt (history shrank)
            if self._last_hist_len.get(i, 0) > ntok:
                self._ctrl.pop(i, None); self._last_prop.pop(i, None)
            self._last_hist_len[i] = ntok
            ctrl = self._get_ctrl(i)

            # infer acceptance of last step's proposal: committed = len(sampled);
            # accepted drafts = committed - 1 (the bonus token), clipped to last K
            if i in self._last_prop and self._last_prop[i]:
                k_prev = len(self._last_prop[i])
                accepted = max(0, min(len(sampled) - 1, k_prev))
                ctrl.update(k_prev, accepted)

            k = ctrl.choose()
            draft = self._ngram_lookup(hist, k)
            self._last_prop[i] = draft
            out.append(draft)
        return out
