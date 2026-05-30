#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


# ---------- fvecs writer ----------
def write_fvecs(path: str, vecs: np.ndarray) -> None:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError(f"vecs must be 2D, got shape={vecs.shape}")
    n, d = vecs.shape

    dt = np.dtype([("d", "<i4"), ("x", "<f4", (d,))])
    out = np.empty(n, dtype=dt)
    out["d"] = d
    out["x"] = vecs
    out.tofile(path)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------- pattern generators ----------
@dataclass(frozen=True)
class Config:
    dim: int = 128
    n_base: int = 100_000
    n_query: int = 1_000


def gen_G(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Baseline Gaussian: base N(0,1), query N(0,1)"""
    rng = np.random.default_rng(42)

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        for d in range(cfg.dim):
            base[i, d] = rng.normal(0.0, 1.0)

    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = rng.normal(0.0, 1.0)

    meta = {
        "seed": 42,
        "type": "baseline_gaussian",
        "base": "N(0,1)",
        "query": "N(0,1)",
    }
    return base, query, meta


def gen_A1(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Directional mild shift: base N(0,1), query dim0 shifts 0->5"""
    rng = np.random.default_rng(42)

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        for d in range(cfg.dim):
            base[i, d] = rng.normal(0.0, 1.0)

    shift_max = 5.0
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        mu = shift_max * i / (cfg.n_query - 1) if cfg.n_query > 1 else shift_max
        for d in range(cfg.dim):
            if d == 0:
                query[i, d] = rng.normal(0.0, 1.0) + mu
            else:
                query[i, d] = rng.normal(0.0, 1.0)

    meta = {
        "seed": 42,
        "type": "mean_shift_dim0_linear",
        "base": "N(0,1)",
        "query": "N(0,1) + dim0_mu(t), mu: 0->5",
    }
    return base, query, meta


def gen_A2(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Local mild shift: base N(0,0.5), query N(0,0.7)"""
    rng = np.random.default_rng(43)

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        for d in range(cfg.dim):
            base[i, d] = rng.normal(0.0, 0.5)

    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = rng.normal(0.0, 0.7)

    meta = {
        "seed": 43,
        "type": "variance_shift",
        "base": "N(0,0.5)",
        "query": "N(0,0.7)",
    }
    return base, query, meta


def gen_B1(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Abrupt severe shift (fixed direction).
    base: segments of 5000, center shifts by delta each segment
    query: around last segment center
    """
    rng = np.random.default_rng(44)

    # First center from N(0,1)
    base_center = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        base_center[d] = rng.normal(0.0, 1.0)

    # delta: random +-1 * shift_size per dimension
    shift_size = 10.0
    delta = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        delta[d] = (1.0 if rng.normal(0.0, 1.0) >= 0 else -1.0) * shift_size

    segment_size = 5000
    num_segments = (cfg.n_base + segment_size - 1) // segment_size

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        seg = i // segment_size
        center = base_center + seg * delta
        for d in range(cfg.dim):
            base[i, d] = center[d] + rng.normal(0.0, 3.0)

    # query around last segment center
    last_center = base_center + (num_segments - 1) * delta
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = last_center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 44,
        "type": "abrupt_segmented_shift_fixed_dir",
        "segment_size": segment_size,
        "shift_size": shift_size,
        "base_std": 3.0,
        "query_std": 1.0,
    }
    return base, query, meta


def gen_B2(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Abrupt severe shift (random direction).
    base: segments of 5000, center shifts by random unit vector * 10.0
    query: around last segment center
    """
    rng = np.random.default_rng(47)

    # First center from N(0,1)
    current_center = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        current_center[d] = rng.normal(0.0, 1.0)

    shift_magnitude = 10.0
    segment_size = 5000

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        # Shift center at segment boundary
        if i > 0 and i % segment_size == 0:
            direction = np.empty(cfg.dim, dtype=np.float32)
            norm_sq = 0.0
            for d in range(cfg.dim):
                direction[d] = rng.normal(0.0, 1.0)
                norm_sq += direction[d] * direction[d]
            norm = np.sqrt(norm_sq)
            if norm == 0:
                norm = 1.0
            for d in range(cfg.dim):
                current_center[d] += (direction[d] / norm) * shift_magnitude

        for d in range(cfg.dim):
            base[i, d] = current_center[d] + rng.normal(0.0, 3.0)

    # query around last center
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = current_center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 47,
        "type": "abrupt_segmented_shift_rand_dir",
        "segment_size": segment_size,
        "shift_magnitude": shift_magnitude,
        "base_std": 3.0,
        "query_std": 1.0,
        "query_center": "last",
    }
    return base, query, meta


def gen_B3(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Abrupt severe shift (random direction, large segments).
    base: segments of 30000, center shifts by random unit vector * 10.0
    query: around last segment center
    """
    rng = np.random.default_rng(47)

    current_center = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        current_center[d] = rng.normal(0.0, 1.0)

    shift_magnitude = 10.0
    segment_size = 30000

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        if i > 0 and i % segment_size == 0:
            direction = np.empty(cfg.dim, dtype=np.float32)
            norm_sq = 0.0
            for d in range(cfg.dim):
                direction[d] = rng.normal(0.0, 1.0)
                norm_sq += direction[d] * direction[d]
            norm = np.sqrt(norm_sq)
            if norm == 0:
                norm = 1.0
            for d in range(cfg.dim):
                current_center[d] += (direction[d] / norm) * shift_magnitude

        for d in range(cfg.dim):
            base[i, d] = current_center[d] + rng.normal(0.0, 3.0)

    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = current_center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 47,
        "type": "abrupt_segmented_shift_rand_dir",
        "segment_size": segment_size,
        "shift_magnitude": shift_magnitude,
        "base_std": 3.0,
        "query_std": 1.0,
        "query_center": "last",
    }
    return base, query, meta


def gen_B4(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Abrupt severe shift (random direction, large segments).
    base: same as B3
    query: around first segment center
    """
    rng = np.random.default_rng(47)

    current_center = np.empty(cfg.dim, dtype=np.float32)
    first_center = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        val = rng.normal(0.0, 1.0)
        current_center[d] = val
        first_center[d] = val

    shift_magnitude = 10.0
    segment_size = 30000

    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        if i > 0 and i % segment_size == 0:
            direction = np.empty(cfg.dim, dtype=np.float32)
            norm_sq = 0.0
            for d in range(cfg.dim):
                direction[d] = rng.normal(0.0, 1.0)
                norm_sq += direction[d] * direction[d]
            norm = np.sqrt(norm_sq)
            if norm == 0:
                norm = 1.0
            for d in range(cfg.dim):
                current_center[d] += (direction[d] / norm) * shift_magnitude

        for d in range(cfg.dim):
            base[i, d] = current_center[d] + rng.normal(0.0, 3.0)

    # query around first center
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        for d in range(cfg.dim):
            query[i, d] = first_center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 47,
        "type": "abrupt_segmented_shift_rand_dir",
        "segment_size": segment_size,
        "shift_magnitude": shift_magnitude,
        "base_std": 3.0,
        "query_std": 1.0,
        "query_center": "first",
    }
    return base, query, meta


def gen_C1(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Recurring two-state drift (small delta).
    base: first half around cA, second half around cB, cB = cA + 5.0
    query: toggle every 100 between A and B
    """
    rng = np.random.default_rng(45)

    # cA from N(0,1), cB = cA + delta
    cA = np.empty(cfg.dim, dtype=np.float32)
    cB = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        cA[d] = rng.normal(0.0, 1.0)
        cB[d] = cA[d] + 5.0

    n1 = cfg.n_base // 2
    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        center = cA if i < n1 else cB
        for d in range(cfg.dim):
            base[i, d] = center[d] + rng.normal(0.0, 1.0)

    block = 100
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        center = cA if ((i // block) % 2 == 0) else cB
        for d in range(cfg.dim):
            query[i, d] = center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 45,
        "type": "recurring_two_state_drift",
        "delta": 5.0,
        "base": "half A, half B",
        "query": "toggle every 100 between A and B",
    }
    return base, query, meta


def gen_C2(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Recurring two-state drift (large delta).
    base: first half around cA, second half around cB, cB = cA + 50.0
    query: toggle every 100 between A and B
    """
    rng = np.random.default_rng(45)

    cA = np.empty(cfg.dim, dtype=np.float32)
    cB = np.empty(cfg.dim, dtype=np.float32)
    for d in range(cfg.dim):
        cA[d] = rng.normal(0.0, 1.0)
        cB[d] = cA[d] + 50.0

    n1 = cfg.n_base // 2
    base = np.empty((cfg.n_base, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_base):
        center = cA if i < n1 else cB
        for d in range(cfg.dim):
            base[i, d] = center[d] + rng.normal(0.0, 1.0)

    block = 100
    query = np.empty((cfg.n_query, cfg.dim), dtype=np.float32)
    for i in range(cfg.n_query):
        center = cA if ((i // block) % 2 == 0) else cB
        for d in range(cfg.dim):
            query[i, d] = center[d] + rng.normal(0.0, 1.0)

    meta = {
        "seed": 45,
        "type": "recurring_two_state_drift",
        "delta": 50.0,
        "base": "half A, half B",
        "query": "toggle every 100 between A and B",
    }
    return base, query, meta


PATTERNS: Dict[str, Callable[[Config], Tuple[np.ndarray, np.ndarray, Dict]]] = {
    "G": gen_G,
    "A1": gen_A1,
    "A2": gen_A2,
    "B1": gen_B1,
    "B2": gen_B2,
    "B3": gen_B3,
    "B4": gen_B4,
    "C1": gen_C1,
    "C2": gen_C2,
}


def main(
    out_root: str = "FreewayML",
    dim: int = 128,
    n_base: int = 100_000,
    n_query: int = 1_000,
) -> None:
    cfg = Config(dim=dim, n_base=n_base, n_query=n_query)
    ensure_dir(out_root)

    summary = {
        "out_root": out_root,
        "dim": dim,
        "n_base": n_base,
        "n_query": n_query,
        "patterns": list(PATTERNS.keys()),
    }
    with open(os.path.join(out_root, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name, fn in PATTERNS.items():
        sub = os.path.join(out_root, name)
        ensure_dir(sub)

        base, query, meta = fn(cfg)

        write_fvecs(os.path.join(sub, "base.fvecs"), base)
        write_fvecs(os.path.join(sub, "query.fvecs"), query)

        with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] {name}: base={base.shape}, query={query.shape} -> {sub}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="FreewayML")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_base", type=int, default=100_000)
    ap.add_argument("--n_query", type=int, default=1_000)
    args = ap.parse_args()

    main(out_root=args.out, dim=args.dim, n_base=args.n_base, n_query=args.n_query)
