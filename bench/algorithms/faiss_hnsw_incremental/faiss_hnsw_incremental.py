"""
Faiss HNSW Incremental Algorithm Implementation.

This is a baseline clone of faiss_HNSW, ready for incremental optimizations.
"""

import numpy as np
import time
from ..base import BaseStreamingANN

try:
    import PyCANDYAlgo

    PYCANDY_AVAILABLE = True
except ImportError:
    PYCANDY_AVAILABLE = False


class FaissHnswIncremental(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.indexkey = index_params.get("indexkey", "HNSWIncremental32")
        self.efConstruction = index_params.get("efConstruction", 40)
        self.verbose = index_params.get("verbose", False)
        self.metric = metric
        self.name = "faiss_hnsw_incremental"
        self.ef = 16
        self.warm_start_levels = 0
        self.direct_reuse = False
        self.trained = False
        self.index = None
        self.ntotal = 0

        # ID mapping: faiss internal ID <-> external ID
        self.my_index = None  # my_index[internal_id] = external_id
        self.my_inverse_index = None  # my_inverse_index[external_id] = internal_id

    def setup(self, dtype, max_pts, ndim):
        if not PYCANDY_AVAILABLE:
            raise RuntimeError("PyCANDYAlgo not available")

        # Create faiss index via PyCANDYAlgo
        if self.metric == "euclidean":
            self.index = PyCANDYAlgo.index_factory_l2(ndim, self.indexkey)
        else:
            self.index = PyCANDYAlgo.index_factory_ip(ndim, self.indexkey)

        # Initialize ID maps
        self.my_index = -1 * np.ones(max_pts, dtype=int)
        self.my_inverse_index = -1 * np.ones(max_pts, dtype=int)

        self.ntotal = 0
        self.trained = False
        self.index.verbose = self.verbose

    def insert(self, X, ids):
        # Filter existing IDs to avoid duplicate insertions
        mask = self.my_inverse_index[ids] == -1
        new_ids = ids[mask]
        new_data = X[mask]

        if new_data.shape[0] == 0:
            print("Not inserting duplicate data")
            return

        if not self.trained:
            self.index.train(new_data.shape[0], new_data.flatten())
            self.trained = True

        self.index.add(new_data.shape[0], new_data.flatten())

        indices = np.arange(self.ntotal, self.ntotal + new_data.shape[0])
        self.my_index[indices] = new_ids
        self.my_inverse_index[new_ids] = indices
        self.ntotal += new_data.shape[0]

    def delete(self, ids):
        # Faiss HNSW does not support deletion; only update ID maps.
        for ext_id in ids:
            ext_id = int(ext_id)
            if ext_id < len(self.my_inverse_index):
                internal_id = self.my_inverse_index[ext_id]
                if internal_id != -1:
                    self.my_inverse_index[ext_id] = -1
                    if internal_id < len(self.my_index):
                        self.my_index[internal_id] = -1

    def query(self, X, k):
        query_size = X.shape[0]
        x_contig = np.ascontiguousarray(X, dtype=np.float32).ravel()
        query_t0 = time.time()

        if self.warm_start_levels > 0 or self.direct_reuse:
        
            results = np.array(
                self.index.search_warm(
                    query_size,
                    x_contig,
                    k,
                    self.ef,
                    self.warm_start_levels,
                    self.direct_reuse,
                )
            )
        else:
            results = np.array(self.index.search(query_size, x_contig, k, self.ef))

        query_t1 = time.time()
        print("faiss_hnsw_incremental query elapsed {:.3f} ms".format(
            (query_t1 - query_t0) * 1e3))

        ids = self.my_index[results]
        self.res = ids.reshape(X.shape[0], k)
        return self.res, None

    def set_query_arguments(self, query_args):
        self.ef = query_args.get("ef", 16)
        self.warm_start_levels = query_args.get("warm_start_levels", 0)
        self.direct_reuse = query_args.get("direct_reuse", False)

    def get_results(self):
        return self.res
