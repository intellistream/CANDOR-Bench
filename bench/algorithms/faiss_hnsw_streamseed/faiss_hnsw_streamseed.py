"""Benchmark adapter for the standalone StreamSeed plugin.

Core query-hint logic is implemented in `plugins/streamseed` and this file
intentionally remains a thin wrapper for benchmark integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..base import BaseStreamingANN

try:
    from streamseed import FaissHnswStreamSeedBackend, StreamSeedPlugin
except ImportError as exc:
    repo_root = Path(__file__).resolve().parents[3]
    local_plugin_src = repo_root / "plugins" / "streamseed" / "src"
    if local_plugin_src.exists():
        sys.path.insert(0, str(local_plugin_src))
        try:
            from streamseed import FaissHnswStreamSeedBackend, StreamSeedPlugin
        except ImportError:
            StreamSeedPlugin = None
            FaissHnswStreamSeedBackend = None
            _STREAMSEED_IMPORT_ERROR = exc
        else:
            _STREAMSEED_IMPORT_ERROR = None
    else:
        StreamSeedPlugin = None
        FaissHnswStreamSeedBackend = None
        _STREAMSEED_IMPORT_ERROR = exc
else:
    _STREAMSEED_IMPORT_ERROR = None


class FaissHnswStreamSeed(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.name = "faiss_hnsw_streamseed"
        self.indexkey = index_params.get("indexkey", "HNSWIncremental32")
        self.efConstruction = index_params.get("efConstruction", 40)
        self.verbose = index_params.get("verbose", False)
        self.plugin = None

        self.ef = 16
        self.streamseed_mode = "streamseed_core"
        self.hint_hops = 1
        self.hint_max_candidates = 256
        self.hint_gate = -1.0
        self.hint_table_slots = 1024
        self.res = None

    def setup(self, dtype, max_pts, ndim):
        if StreamSeedPlugin is None or FaissHnswStreamSeedBackend is None:
            raise RuntimeError(
                "streamseed package is required for faiss_hnsw_streamseed adapter. "
                "Install via plugins/streamseed (pip install -e plugins/streamseed)."
            ) from _STREAMSEED_IMPORT_ERROR

        backend = FaissHnswStreamSeedBackend(metric=self.metric, indexkey=self.indexkey)
        self.plugin = StreamSeedPlugin(backend=backend)
        self.plugin.setup(max_pts=max_pts, ndim=ndim, verbose=self.verbose)

    def insert(self, X, ids):
        self.plugin.insert(X, ids)

    def delete(self, ids):
        self.plugin.delete(ids)

    def query(self, X, k):
        self.plugin.configure(
            ef_search=self.ef,
            streamseed_mode=self.streamseed_mode,
            hint_hops=self.hint_hops,
            hint_max_candidates=self.hint_max_candidates,
            hint_gate=self.hint_gate,
            hint_table_slots=self.hint_table_slots,
        )
        self.res, distances = self.plugin.query(X, k)
        return self.res, distances

    def set_query_arguments(self, query_args):
        self.ef = int(query_args.get("ef", 16))
        self.streamseed_mode = query_args.get("streamseed_mode", "streamseed_core")
        self.hint_hops = int(query_args.get("hint_hops", 1))
        self.hint_max_candidates = int(query_args.get("hint_max_candidates", 256))
        self.hint_gate = float(query_args.get("hint_gate", -1.0))
        self.hint_table_slots = int(query_args.get("hint_table_slots", 1024))

    def get_results(self):
        return self.res
