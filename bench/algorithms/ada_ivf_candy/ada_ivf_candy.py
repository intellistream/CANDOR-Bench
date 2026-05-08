"""AdaIVF baseline using the same candy bindings as GammaFresh.

Provides apples-to-apples IVF comparison alongside the FaissHNSW baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseStreamingANN


ROOT = Path(__file__).resolve().parents[3]
GAMMAFRESH_ROOT = ROOT / "algorithms_impl" / "GammaFresh"
GAMMAFRESH_BINDINGS = GAMMAFRESH_ROOT / "build" / "src" / "bindings" / "python"
GAMMAFRESH_PYTHON = GAMMAFRESH_ROOT / "python"
for _path in (GAMMAFRESH_BINDINGS, GAMMAFRESH_PYTHON):
    path_str = str(_path)
    if _path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from candy import Index  # type: ignore  # noqa: E402


# Defaults mirror algorithms_impl/GammaFresh/config/defaults.json:indexes.ada_ivf
DEFAULT_ADA_IVF: dict[str, Any] = {
    "target_posting_size": 128,
    "probe_ratio": 0.01,
    "nprobe_multiplier": 2,
    "min_nprobe": 10,
    "max_nprobe": 1024,
    "recluster_min_vectors": 20000,
    "brute_force_threshold": 5000,
    "kmeans_iterations": 12,
    "kmeans_max_iterations": 100,
    "kmeans_min_iterations": 1,
    "max_training_points": 100000,
    "candidate_fanout": 4,
    "centroid_small_multiplier": 2,
    "default_seed": 17,
    "min_clusters": 64,
    "max_clusters": 65536,
}


class AdaIvfCandy(BaseStreamingANN):
    def __init__(self, metric: str, index_params: dict[str, Any]):
        if metric not in {"euclidean", "angular"}:
            raise ValueError(f"AdaIvfCandy only supports euclidean/angular, got {metric}")
        self.metric = metric
        self.index_params = index_params or {}
        self.name = "ada_ivf_candy"
        self.index = None
        self.res = None

    def setup(self, dtype, max_pts, ndim):
        if str(dtype) != "float32":
            raise ValueError(f"AdaIvfCandy only supports float32, got {dtype}")
        cfg = dict(DEFAULT_ADA_IVF)
        cfg.update(self.index_params)
        payload = {"global": {"dim": int(ndim)}, "indexes": {"ada_ivf": cfg}}
        self.index = Index("AdaIVF", int(ndim), config=payload)

    def insert(self, X, ids):
        assert self.index is not None
        self.index.add(
            np.ascontiguousarray(ids, dtype=np.uint64),
            np.ascontiguousarray(X, dtype=np.float32),
        )

    def initial_load(self, X, ids):
        assert self.index is not None
        bootstrap = getattr(self.index, "initial_load", None)
        if callable(bootstrap):
            bootstrap(
                np.ascontiguousarray(ids, dtype=np.uint64),
                np.ascontiguousarray(X, dtype=np.float32),
            )
        else:
            self.insert(X, ids)

    def delete(self, ids):
        assert self.index is not None
        self.index.delete(np.ascontiguousarray(ids, dtype=np.uint64))

    def query(self, X, k):
        assert self.index is not None
        ids, dists = self.index.search(np.ascontiguousarray(X, dtype=np.float32), k)
        self.res = ids.astype(np.uint32, copy=False)
        return self.res, dists

    def get_results(self):
        return self.res

    def maintain(self, vector_budget: int = 0, time_budget_ms: int = 0, force: bool = False):
        """No-op for AdaIVF (it self-tunes online); kept for bench parity."""
        return {"executed": 0.0}
