from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class StreamSeedConfig:
    ef_search: int = 120
    streamseed_mode: int | str = 1
    hint_level1_only: int | bool = 0
    hint_adaptive_gate_mode: int = 0
    hint_hops: int = 1
    hint_max_candidates: int = 256
    hint_gate: float = -1.0
    hint_qual_gate: float = -1.0
    hint_cons_gate: float = -1.0
    hint_gate_m_quantile: float = 0.25
    hint_gate_o_quantile: float = 0.30
    hint_gate_min_samples: int = 128
    hint_table_slots: int = 1024
    hint_slot_capacity: int = 2


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
    def query(self, x: np.ndarray, k: int, query_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> Optional[np.ndarray]:
        raise NotImplementedError
