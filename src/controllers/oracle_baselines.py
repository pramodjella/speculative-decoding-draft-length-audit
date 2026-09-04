"""Idealized single-sided oracles — strawman-proof baselines for the thesis.

The head-to-head's strongest claim avoids any dependence on the quality of our
reimplemented bandits (TapOut/Nightjar). We instead compare the combined *learner*
against the strongest POSSIBLE single-sided policies, each clairvoyant on its own axis:

  OracleLoad    : knows the best fixed K for every batch level (content-blind).
                  = the ceiling of any load-only method (Nightjar, DSD, TETRIS).
  OracleContent : knows the best fixed K for every draft-entropy bin (batch-blind).
                  = the ceiling of any content-only method (TapOut, SVIP, AdaEDL).

If the online combined controller beats BOTH of these oracles, then perfect knowledge
of either axis alone is provably insufficient — the axes must be combined. This is the
unimpeachable version of the contribution (immune to "your baselines are weak").

Both are fit by a calibration pass over the same content/load distribution.
"""
import numpy as np
from serve.load_simulator import ContentStream, accepted_given_K, step_speedup
# (BATCHES/ARMS live in the runner; passed in explicitly to avoid coupling.)


def _expected_best_K(per_sample_acc, batches, arms, weight=None):
    """argmax_K mean over samples & batches of step_speedup(b,K,accepted_K)."""
    best_K, best_v = arms[0], -1e9
    n = len(per_sample_acc)
    w = np.ones(n) if weight is None else np.asarray(weight, float)
    for K in arms:
        accs = np.array([s[K] for s in per_sample_acc], float)
        v = 0.0
        for b in batches:
            v += np.sum(w * np.array([step_speedup(b, K, a) for a in accs]))
        if v > best_v:
            best_v, best_K = v, K
    return best_K


def calibrate(workload, batches, arms, rng, n=8000, nbins=6):
    """Return (OracleLoad table, OracleContent edges+table) for `workload`."""
    stream = ContentStream(workload)
    e0s, per_sample_acc = [], []
    for _ in range(n):
        ent, acc = stream.step(rng)
        e0s.append(ent[0])
        per_sample_acc.append({K: accepted_given_K(acc, K) for K in arms})

    # OracleLoad: best K per batch (averaged over all content samples).
    load_tbl = {}
    for b in batches:
        best_K, best_v = arms[0], -1e9
        for K in arms:
            v = np.mean([step_speedup(b, K, s[K]) for s in per_sample_acc])
            if v > best_v:
                best_v, best_K = v, K
        load_tbl[b] = best_K

    # OracleContent: best K per pre-draft-entropy bin (averaged over batch levels).
    edges = np.quantile(e0s, np.linspace(0, 1, nbins + 1)[1:-1])
    bins = np.digitize(e0s, edges)
    content_tbl = {}
    for bi in range(nbins):
        idx = np.where(bins == bi)[0]
        if len(idx) == 0:
            content_tbl[bi] = arms[len(arms) // 2]
            continue
        sub = [per_sample_acc[i] for i in idx]
        content_tbl[bi] = _expected_best_K(sub, batches, arms)

    # OracleLoadContent: best K per (batch x entropy bin), EXPECTED speedup. This is the
    # ACHIEVABLE joint ceiling (no realization-clairvoyance) — its gap over OracleLoad /
    # OracleContent is the clean measure of load+content COMPLEMENTARITY.
    joint_tbl = {}
    for b in batches:
        for bi in range(nbins):
            idx = np.where(bins == bi)[0]
            if len(idx) == 0:
                joint_tbl[(b, bi)] = arms[len(arms) // 2]
                continue
            best_K, best_v = arms[0], -1e9
            for K in arms:
                v = np.mean([step_speedup(b, K, per_sample_acc[i][K]) for i in idx])
                if v > best_v:
                    best_v, best_K = v, K
            joint_tbl[(b, bi)] = best_K

    return (OracleLoad(load_tbl, batches), OracleContent(edges, content_tbl),
            OracleLoadContent(load_tbl, batches, edges, joint_tbl))


def _bucket(batch, batches):
    # nearest batch level present in the table
    return min(batches, key=lambda b: abs(b - batch))


class OracleLoad:
    """Best fixed K per batch level; content-blind (load-only ceiling)."""
    def __init__(self, table, batches):
        self.table = table
        self.batches = batches
        self.arms = tuple(sorted(set(table.values())))

    def set_max_steps(self, m): pass
    def reset_episode(self): pass

    def choose_k(self, entropy=0.0, batch=1, load=None):
        return self.table[_bucket(batch, self.batches)]

    def update(self, *a, **k): pass


class OracleContent:
    """Best fixed K per pre-draft entropy bin; batch-blind (content-only ceiling)."""
    def __init__(self, edges, table):
        self.edges = edges
        self.table = table
        self.arms = tuple(sorted(set(table.values())))

    def set_max_steps(self, m): pass
    def reset_episode(self): pass

    def choose_k(self, entropy=0.0, batch=1, load=None):
        bi = int(np.digitize([entropy], self.edges)[0])
        return self.table.get(bi, self.table[0])

    def update(self, *a, **k): pass


class OracleLoadContent:
    """Best fixed K per (batch, entropy bin), EXPECTED. Achievable joint ceiling: its
    gap over OracleLoad/OracleContent is the clean load+content complementarity."""
    def __init__(self, load_table, batches, edges, joint_table):
        self.batches = batches
        self.edges = edges
        self.table = joint_table
        self.arms = tuple(sorted(set(joint_table.values())))

    def set_max_steps(self, m): pass
    def reset_episode(self): pass

    def choose_k(self, entropy=0.0, batch=1, load=None):
        b = _bucket(batch, self.batches)
        bi = int(np.digitize([entropy], self.edges)[0])
        return self.table.get((b, bi), self.arms[len(self.arms) // 2])

    def update(self, *a, **k): pass
