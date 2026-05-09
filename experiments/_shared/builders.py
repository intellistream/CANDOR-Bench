"""Index builders shared by all experiments. ONE source of truth for configs."""

# `candy` is the C++ binding for native GammaFresh and is not needed by the
# Python-POC experiments (e15+). Import it lazily so e15-e19 work without
# the C++ build. Native experiments (e01-e09) still use build_*; those will
# trigger the import on demand.
def _candy_index():
    from candy import Index
    return Index


# ---- per-algorithm config defaults (1M-tuned) ----

GAMMA_CFG = {
    "backend": "FaissHNSW",
    "split_factor": 16.0,
    "gamma_split_threshold": 1.0,
    "min_size_for_split": 512,
    "candidate_reserve_size": 8,
    "maintenance_interval": 4,
    "maintenance_query_interval": 4,
    "maintenance_query_threshold_scale": 0.5,
}

FAISS_HNSW_CFG = {
    "m": 32,
    "ef_construction": 120,
    "ef_search": 80,
    "tombstone_rebuild_threshold": 512,
}

ADA_IVF_CFG = {
    "target_posting_size": 4096,
    "min_nprobe": 16, "max_nprobe": 256,
    "min_clusters": 64, "brute_force_threshold": 5000,
    "recluster_min_vectors": 100000, "max_training_points": 100000,
    "probe_ratio": 0.02, "nprobe_multiplier": 2,
    "kmeans_iterations": 12, "kmeans_max_iterations": 100,
    "kmeans_min_iterations": 1, "candidate_fanout": 4,
    "centroid_small_multiplier": 2, "default_seed": 17,
    "max_clusters": 65536,
}


def build_gamma(dim: int, **gamma_overrides):
    cfg = dict(GAMMA_CFG)
    cfg.update(gamma_overrides)
    return _candy_index()("GammaFresh", dim,
                 config={"global": {"dim": dim},
                         "indexes": {"gamma_fresh": cfg, "faiss_hnsw": FAISS_HNSW_CFG}})


def build_faiss(dim: int, **hnsw_overrides):
    cfg = dict(FAISS_HNSW_CFG)
    cfg.update(hnsw_overrides)
    return _candy_index()("FaissHNSW", dim,
                 config={"global": {"dim": dim},
                         "indexes": {"faiss_hnsw": cfg}})


def build_ivf(dim: int, **ivf_overrides):
    cfg = dict(ADA_IVF_CFG)
    cfg.update(ivf_overrides)
    return _candy_index()("AdaIVF", dim,
                 config={"global": {"dim": dim}, "indexes": {"ada_ivf": cfg}})


def build(name: str, dim: int, **overrides):
    if name == "gamma":
        return build_gamma(dim, **overrides)
    if name == "faiss":
        return build_faiss(dim, **overrides)
    if name == "ivf":
        return build_ivf(dim, **overrides)
    raise ValueError(f"unknown algo: {name}")
