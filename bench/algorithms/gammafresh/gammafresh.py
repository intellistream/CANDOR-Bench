"""
GammaFresh Algorithm Implementation

Uses the GammaFresh Python bindings from the local submodule and exposes the
same streaming interface as the rest of the root benchmark stack.
"""

from __future__ import annotations

import copy
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


DEFAULT_HNSW = {
    "m": 32,
    "ef_construction": 120,
    "ef_search": 80,
    "tombstone_rebuild_threshold": 512,
}

DEFAULT_FAISS_HNSW = {
    "m": 32,
    "ef_construction": 120,
    "ef_search": 80,
    "tombstone_rebuild_threshold": 512,
}

DEFAULT_FRESH_VAMANA = {
    "alpha": 1.1,
    "L": 60,
    "R": 18,
    "batch_size": 1500,
    "max_iterations": 100,
    "num_lock_shards": 64,
}

DEFAULT_GAMMAFRESH = {
    "backend": "FaissHNSW",
    "log_interval": 0,
    "split_factor": 4.0,
    "gamma_split_threshold": 0.4,
    "maintenance_interval": 32,
    "maintenance_query_interval": 32,
    "maintenance_query_threshold_scale": 0.5,
    "min_size_for_split": 32,
    "candidate_reserve_size": 64,
    "ema_beta": 0.05,
    "radius_beta": 0.1,
    "bic_min_variance": 1e-6,
    "bic_gaussian_log_coeff": -0.5,
    "bic_log_likelihood_weight": 1.0,
    "bic_min_improvement": 2.0,
    "bic_min_points": 4,
    "slab_size": 2097152,
    "internal_logs_dir": str(ROOT / "results" / "gammafresh" / "internal_logs"),
    "placement_controller": {
        "ema_beta": 0.05,
        "newcomer_epsilon_ms": 0.0,
        "migration_epsilon_ms": 0.0,
        "default_expected_lifetime_ops": 1000.0,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _build_config(
    *,
    dim: int,
    backend: str,
    gammafresh: dict[str, Any] | None,
    backend_config: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "global": {"dim": int(dim)},
        "indexes": {
            "gamma_fresh": copy.deepcopy(DEFAULT_GAMMAFRESH),
            "hnsw": copy.deepcopy(DEFAULT_HNSW),
            "faiss_hnsw": copy.deepcopy(DEFAULT_FAISS_HNSW),
            "fresh_vamana": copy.deepcopy(DEFAULT_FRESH_VAMANA),
        },
    }
    payload["indexes"]["gamma_fresh"]["backend"] = backend
    if gammafresh:
        _deep_update(payload["indexes"]["gamma_fresh"], gammafresh)
    if backend == "HNSW" and backend_config:
        _deep_update(payload["indexes"]["hnsw"], backend_config)
    if backend == "FaissHNSW" and backend_config:
        _deep_update(payload["indexes"]["faiss_hnsw"], backend_config)
    if backend == "FreshVamana" and backend_config:
        _deep_update(payload["indexes"]["fresh_vamana"], backend_config)
    return payload


class Gammafresh(BaseStreamingANN):
    def __init__(self, metric: str, index_params: dict[str, Any]):
        if metric not in {"euclidean", "angular"}:
            raise ValueError(f"GammaFresh only supports euclidean/angular metrics, got {metric}")
        self.metric = metric
        self.index_params = index_params or {}
        self.name = "gammafresh"
        self.index = None
        self.res = None

    def setup(self, dtype, max_pts, ndim):
        if str(dtype) != "float32":
            raise ValueError(f"GammaFresh only supports float32 vectors, got {dtype}")
        backend = self.index_params.get("backend", "FaissHNSW")
        gammafresh_cfg = dict(self.index_params.get("gamma_fresh", {}))
        gammafresh_cfg.setdefault("internal_logs_dir", str(ROOT / "results" / "gammafresh" / "internal_logs"))
        backend_cfg = dict(self.index_params.get("backend_config", {}))
        payload = _build_config(
            dim=int(ndim),
            backend=backend,
            gammafresh=gammafresh_cfg,
            backend_config=backend_cfg,
        )
        self.index = Index("GammaFresh", int(ndim), config=payload)

    def insert(self, X, ids):
        assert self.index is not None
        self.index.add(
            np.ascontiguousarray(ids, dtype=np.uint64),
            np.ascontiguousarray(X, dtype=np.float32),
        )

    def initial_load(self, X, ids):
        assert self.index is not None
        self.index.initial_load(
            np.ascontiguousarray(ids, dtype=np.uint64),
            np.ascontiguousarray(X, dtype=np.float32),
        )

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

    def maintain(self, vector_budget: int = 0, time_budget_ms: int = 0, force: bool = True):
        """Bench-driven incremental maintenance: drain partition buffers into the graph,
        run BIC splits / merges, etc. With ``vector_budget=0`` the call is unbounded.
        ``force=True`` bypasses the internal counter-due check so callers always
        get the work they ask for (the counter-due check is for the deprecated
        auto-trigger codepath)."""
        assert self.index is not None
        return self.index.maintain(int(vector_budget), int(time_budget_ms), bool(force))
