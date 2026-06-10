"""Binary dataset formats (DiskANN layouts, little-endian)."""

from __future__ import annotations

import os
import struct

import numpy as np


def load_aligned_bin(path: str) -> np.ndarray:
    """Aligned bin: int32 npts, int32 dim, float32 data — (npts, dim)."""
    with open(path, "rb") as f:
        npts, dim = struct.unpack("<ii", f.read(8))
        expected = npts * dim * 4 + 8
        actual = os.fstat(f.fileno()).st_size
        if actual != expected:
            raise ValueError(
                f"{path}: size mismatch, got {actual}, want {expected} "
                f"(npts={npts}, dim={dim})"
            )
        data = np.fromfile(f, dtype="<f4", count=npts * dim)
    return data.reshape(npts, dim)


def write_aligned_bin(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", arr.shape[0], arr.shape[1]))
        np.ascontiguousarray(arr, dtype="<f4").tofile(f)


def load_groundtruth_bin(path: str) -> np.ndarray:
    """Ground truth bin: int32 npts, int32 k, uint32 ids[npts*k],
    optionally followed by float32 distances. Returns (npts, k) ids."""
    with open(path, "rb") as f:
        npts, k = struct.unpack("<ii", f.read(8))
        ids_bytes = npts * k * 4
        size = os.fstat(f.fileno()).st_size - 8
        if size not in (ids_bytes, ids_bytes * 2):
            raise ValueError(
                f"{path}: unexpected ground truth size {size} "
                f"for npts={npts}, k={k}"
            )
        ids = np.fromfile(f, dtype="<u4", count=npts * k)
    return ids.reshape(npts, k)
