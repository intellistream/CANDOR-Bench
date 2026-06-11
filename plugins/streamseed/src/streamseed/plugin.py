from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .backends.base import StreamSeedBackend, StreamSeedConfig


class StreamSeedPlugin:
    """Backend-agnostic StreamSeed plugin entry.

    The plugin owns StreamSeed configuration and delegates concrete ANN calls to
    a selected backend implementation.
    """

    def __init__(self, backend: StreamSeedBackend):
        self.backend = backend
        self.config = StreamSeedConfig()

    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        self.backend.setup(max_pts=max_pts, ndim=ndim, verbose=verbose)

    def configure(self, **kwargs) -> None:
        self.config = StreamSeedConfig(**kwargs)
        self.backend.configure(self.config)

    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        self.backend.insert(x, ids)

    def delete(self, ids: np.ndarray) -> None:
        self.backend.delete(ids)

    def query(self, x: np.ndarray, k: int, query_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if query_ids is not None:
            try:
                return self.backend.query(x, k, query_ids=query_ids)
            except TypeError:
                pass
        return self.backend.query(x, k)

    def get_results(self):
        return self.backend.get_results()
