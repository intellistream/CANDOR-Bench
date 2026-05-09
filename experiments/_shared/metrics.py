"""Common metric helpers."""
import numpy as np


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int = 10) -> float:
    return float(np.mean([len(set(p) & set(g)) / k for p, g in zip(pred, gt)]))


def percentile_ms(latencies_s: list[float], p: float) -> float:
    if not latencies_s:
        return 0.0
    return float(np.percentile(np.asarray(latencies_s) * 1000.0, p))


def format_row(row: dict, cols: list[str], widths: dict[str, int] | None = None) -> str:
    widths = widths or {}
    out = []
    for c in cols:
        v = row.get(c, "")
        w = widths.get(c, 10)
        if isinstance(v, float):
            out.append(f"{v:>{w}.4f}")
        elif isinstance(v, int):
            out.append(f"{v:>{w}d}")
        else:
            out.append(f"{str(v):>{w}}")
    return "  ".join(out)


def maintain(idx, name: str, hnsw_force_rebuild: bool = False, vector_budget: int = 0):
    """Unified maint for all algos. Only gamma has explicit maint. Optionally
    force HNSW rebuild via the same maintain API."""
    if name == "gamma":
        idx.maintain(int(vector_budget), 0, False)
    elif name == "faiss" and hnsw_force_rebuild:
        idx.maintain(0, 0, True)
