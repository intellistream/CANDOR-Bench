from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class StreamSeedConfig:
    ef_search: int = 120
    streamseed_mode: int | str = 1
    hint_hops: int = 1
    hint_max_candidates: int = 256
    hint_gate: float = -1.0
    hint_table_slots: int = 1024


class StreamSeedBackend(ABC):
    @abstractmethod
    def setup(self, max_pts: int, ndim: int, verbose: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def configure(self, config: StreamSeedConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, x: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> Optional[np.ndarray]:
        raise NotImplementedError
