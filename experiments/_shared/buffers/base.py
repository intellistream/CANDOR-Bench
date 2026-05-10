"""Buffer interface: holds newly-inserted vectors not yet in the backend.

Designed to be minimal so any of {FlatBuffer, ClusterBuffer, future
hierarchical buffer} can plug into the same ModularRouter.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Buffer(ABC):
    """Abstract buffer holding (id, vec) pairs that haven't reached the
    backend yet. Supports kill-in-place ("absorb") on delete."""

    dim: int

    @abstractmethod
    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        """Append ids/vecs to the buffer. ids: int64; vecs: float32 (N, dim)."""

    @abstractmethod
    def delete(self, id_int: int) -> bool:
        """Try to kill `id_int` in the buffer.

        Returns True if the id was alive in the buffer (and is now
        marked dead — *absorbed*), False if not present (caller is
        expected to mark_deleted on the backend instead)."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search alive vectors in the buffer for top-k per query.

        Returns (ids, dists), shape (Q, k). Pad with -1 / +inf if fewer
        than k alive."""

    @abstractmethod
    def drain_alive(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, vecs) of all alive vectors and reset the buffer."""

    @property
    @abstractmethod
    def n_alive(self) -> int:
        """Number of alive vectors currently held."""

    @property
    @abstractmethod
    def n_dead(self) -> int:
        """Number of dead (absorbed) vectors still occupying slots."""
