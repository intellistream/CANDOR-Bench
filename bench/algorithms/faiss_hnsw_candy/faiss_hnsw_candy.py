"""FaissHNSW baseline using the same candy bindings as GammaFresh.

This wrapper exists for apples-to-apples comparison: same C++ compute engine,
same FaissHNSW backend code, same bench harness — only the GammaFresh partition
layer is removed.
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


class FaissHnswCandy(BaseStreamingANN):
    def __init__(self, metric: str, index_params: dict[str, Any]):
        if metric not in {"euclidean", "angular"}:
            raise ValueError(f"FaissHnswCandy only supports euclidean/angular, got {metric}")
        self.metric = metric
        self.index_params = index_params or {}
        self.name = "faiss_hnsw_candy"
        self.index = None
        self.res = None

    def setup(self, dtype, max_pts, ndim):
        if str(dtype) != "float32":
            raise ValueError(f"FaissHnswCandy only supports float32, got {dtype}")
        cfg = {
            "m": int(self.index_params.get("m", 32)),
            "ef_construction": int(self.index_params.get("ef_construction", 120)),
            "ef_search": int(self.index_params.get("ef_search", 80)),
            "tombstone_rebuild_threshold": int(self.index_params.get("tombstone_rebuild_threshold", 512)),
        }
        payload = {"global": {"dim": int(ndim)}, "indexes": {"faiss_hnsw": cfg}}
        self.index = Index("FaissHNSW", int(ndim), config=payload)

    def insert(self, X, ids):
        assert self.index is not None
        self.index.add(
            np.ascontiguousarray(ids, dtype=np.uint64),
            np.ascontiguousarray(X, dtype=np.float32),
        )

    def initial_load(self, X, ids):
        # candy's FaissHNSW exposes initial_load too — use it for symmetry with GammaFresh.
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
        """No-op for plain FaissHNSW; provided for parity with maintenance-aware
        algorithms so the bench can call maintain() uniformly."""
        return {"executed": 0.0}
