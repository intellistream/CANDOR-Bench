"""Compatibility shims: ANN implementations moved to sage.libs.annimplementations.

This package now proxies imports to `sage.libs.annimplementations`.
"""
from __future__ import annotations

import importlib
import sys
from typing import Iterable

_MIGRATED = [
    "base",
    "registry",
    "candy_lshapg",
    "candy_mnru",
    "candy_sptag",
    "cufe",
    "diskann",
    "faiss_HNSW",
    "faiss_HNSW_Optimized",
    "faiss_IVFPQ",
    "faiss_NSW",
    "faiss_fast_scan",
    "faiss_lsh",
    "faiss_onlinepq",
    "faiss_pq",
    "gti",
    "ipdiskann",
    "plsh",
    "puck",
    "pyanns",
    "vsag_hnsw",
]

_BASE = "sage.libs.annimplementations"


def _load(name: str):
    module = importlib.import_module(f"{_BASE}.{name}")
    sys.modules[f"{__name__}.{name}"] = module
    return module


for _name in _MIGRATED:
    _load(_name)


def __getattr__(name: str):  # pragma: no cover - compatibility path
    if name in _MIGRATED:
        return sys.modules[f"{__name__}.{name}"]
    raise AttributeError(name)


def __dir__() -> Iterable[str]:  # pragma: no cover - introspection
    return sorted(list(globals().keys()) + _MIGRATED)
