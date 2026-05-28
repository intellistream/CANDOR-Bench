"""Benchmark adapter for the standalone StreamSeed plugin.

Core query-hint logic is implemented in `plugins/streamseed` and this file
intentionally remains a thin wrapper for benchmark integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..base import BaseStreamingANN

try:
	from streamseed import FaissHnswStreamSeedBackend, StreamSeedPlugin, SymphonyQGBackend
except ImportError as exc:
	repo_root = Path(__file__).resolve().parents[3]
	local_plugin_src = repo_root / "plugins" / "streamseed" / "src"
	if local_plugin_src.exists():
		sys.path.insert(0, str(local_plugin_src))
		try:
			from streamseed import FaissHnswStreamSeedBackend, StreamSeedPlugin, SymphonyQGBackend
		except ImportError:
			StreamSeedPlugin = None
			FaissHnswStreamSeedBackend = None
			SymphonyQGBackend = None
			_STREAMSEED_IMPORT_ERROR = exc
		else:
			_STREAMSEED_IMPORT_ERROR = None
	else:
		StreamSeedPlugin = None
		FaissHnswStreamSeedBackend = None
		SymphonyQGBackend = None
		_STREAMSEED_IMPORT_ERROR = exc
else:
	_STREAMSEED_IMPORT_ERROR = None


class StreamSeedHybrid(BaseStreamingANN):
	def __init__(self, metric, index_params):
		self.metric = metric
		self.name = "streamseed_hybrid"
		self.backend_name = str(index_params.get("backend", "faiss")).lower()
		self.indexkey = index_params.get("indexkey", "HNSWIncremental32")
		self.efConstruction = index_params.get("efConstruction", 40)
		self.verbose = index_params.get("verbose", False)

		# SymphonyQG backend params (effective when backend=symphonyqg)
		self.symphony_index_type = index_params.get("index_type", "QG")
		self.symphony_degree_bound = int(index_params.get("degree_bound", 32))
		self.symphony_ef_construction = int(index_params.get("efConstruction", 200))
		self.symphony_num_iter = int(index_params.get("num_iter", 3))
		self.symphony_num_threads = int(index_params.get("num_threads", 0))
		self.symphony_enable_batch_search = bool(index_params.get("enable_batch_search", True))
		self.symphony_rebuild_every_n_updates = int(
			index_params.get("rebuild_every_n_updates", 1)
		)
		self.symphony_rebuild_on_query_if_dirty = bool(
			index_params.get("rebuild_on_query_if_dirty", True)
		)

		self.plugin = None

		self.ef = 16
		self.streamseed_mode = "streamseed_core"
		self.hint_level1_only = 0
		self.hint_adaptive_gate_mode = 0
		self.hint_hops = 1
		self.hint_max_candidates = 256
		self.hint_gate = -1.0
		self.hint_qual_gate = -1.0
		self.hint_cons_gate = -1.0
		self.hint_gate_m_quantile = 0.25
		self.hint_gate_o_quantile = 0.30
		self.hint_gate_min_samples = 128
		self.hint_table_slots = 1024
		self.hint_slot_capacity = 2
		self.res = None

	def setup(self, dtype, max_pts, ndim):
		if StreamSeedPlugin is None:
			raise RuntimeError(
				"streamseed package is required for streamseed_hybrid adapter. "
				"Install via plugins/streamseed (pip install -e plugins/streamseed)."
			) from _STREAMSEED_IMPORT_ERROR

		if self.backend_name == "faiss":
			if FaissHnswStreamSeedBackend is None:
				raise RuntimeError("FaissHnswStreamSeedBackend is unavailable in streamseed package")
			backend = FaissHnswStreamSeedBackend(metric=self.metric, indexkey=self.indexkey)
		elif self.backend_name == "symphonyqg":
			if SymphonyQGBackend is None:
				raise RuntimeError("SymphonyQGBackend is unavailable in streamseed package")
			backend = SymphonyQGBackend(
				metric=self.metric,
				index_type=self.symphony_index_type,
				degree_bound=self.symphony_degree_bound,
				ef_construction=self.symphony_ef_construction,
				num_iter=self.symphony_num_iter,
				num_threads=self.symphony_num_threads,
				enable_batch_search=self.symphony_enable_batch_search,
				rebuild_every_n_updates=self.symphony_rebuild_every_n_updates,
				rebuild_on_query_if_dirty=self.symphony_rebuild_on_query_if_dirty,
			)
		else:
			raise ValueError(
				f"Unsupported backend '{self.backend_name}'. "
				"Use one of: faiss, symphonyqg"
			)

		self.plugin = StreamSeedPlugin(backend=backend)
		self.plugin.setup(max_pts=max_pts, ndim=ndim, verbose=self.verbose)

	def insert(self, x, ids):
		self.plugin.insert(x, ids)

	def delete(self, ids):
		self.plugin.delete(ids)

	def query(self, x, k):
		self.plugin.configure(
			ef_search=self.ef,
			streamseed_mode=self.streamseed_mode,
			hint_level1_only=self.hint_level1_only,
			hint_adaptive_gate_mode=self.hint_adaptive_gate_mode,
			hint_hops=self.hint_hops,
			hint_max_candidates=self.hint_max_candidates,
			hint_gate=self.hint_gate,
			hint_qual_gate=self.hint_qual_gate,
			hint_cons_gate=self.hint_cons_gate,
			hint_gate_m_quantile=self.hint_gate_m_quantile,
			hint_gate_o_quantile=self.hint_gate_o_quantile,
			hint_gate_min_samples=self.hint_gate_min_samples,
			hint_table_slots=self.hint_table_slots,
			hint_slot_capacity=self.hint_slot_capacity,
		)
		self.res, distances = self.plugin.query(x, k)
		return self.res, distances

	def set_query_arguments(self, query_args):
		self.ef = int(query_args.get("ef", 16))
		self.streamseed_mode = query_args.get("streamseed_mode", "streamseed_core")
		self.hint_level1_only = int(query_args.get("hint_level1_only", 0))
		self.hint_adaptive_gate_mode = int(query_args.get("hint_adaptive_gate_mode", 0))
		self.hint_hops = int(query_args.get("hint_hops", 1))
		self.hint_max_candidates = int(query_args.get("hint_max_candidates", 256))
		self.hint_gate = float(query_args.get("hint_gate", -1.0))
		self.hint_qual_gate = float(query_args.get("hint_qual_gate", -1.0))
		self.hint_cons_gate = float(query_args.get("hint_cons_gate", -1.0))
		self.hint_gate_m_quantile = float(query_args.get("hint_gate_m_quantile", 0.25))
		self.hint_gate_o_quantile = float(query_args.get("hint_gate_o_quantile", 0.30))
		self.hint_gate_min_samples = int(query_args.get("hint_gate_min_samples", 128))
		self.hint_table_slots = int(query_args.get("hint_table_slots", 1024))
		self.hint_slot_capacity = int(query_args.get("hint_slot_capacity", 2))

	def get_results(self):
		return self.res


__all__ = ["StreamSeedHybrid"]
