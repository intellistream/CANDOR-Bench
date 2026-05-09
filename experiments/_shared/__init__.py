"""Shared helpers for all GammaFresh experiments."""
import os, sys

# Always set single-thread before any numpy/faiss import so OMP/BLAS pick it up.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Make `candy` and the dataset registry importable.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "algorithms_impl/GammaFresh/build/src/bindings/python"))
sys.path.insert(0, os.path.join(_REPO, "algorithms_impl/GammaFresh/python"))

from .data import load_sift, compute_gt, gt_for_surviving  # noqa: E402,F401
from .builders import build_gamma, build_faiss, build_ivf, build  # noqa: E402,F401
from .metrics import recall_at_k, percentile_ms, format_row  # noqa: E402,F401
