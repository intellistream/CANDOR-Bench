"""Benchmark adapter for the standalone StreamSeed plugin.

Core query-hint logic is implemented in `plugins/streamseed` and this file
intentionally remains a thin wrapper for benchmark integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..base import BaseStreamingANN

try:
	from streamseed import FaissHnswStreamSeedBackend, FreshDiskANNStreamSeedBackend, HnswlibStreamSeedBackend, StreamSeedPlugin, SymphonyQGBackend, WolverineStreamSeedBackend
except ImportError as exc:
	repo_root = Path(__file__).resolve().parents[3]
	local_plugin_src = repo_root / "plugins" / "streamseed" / "src"
	if local_plugin_src.exists():
		sys.path.insert(0, str(local_plugin_src))
		try:
			from streamseed import FaissHnswStreamSeedBackend, FreshDiskANNStreamSeedBackend, HnswlibStreamSeedBackend, StreamSeedPlugin, SymphonyQGBackend, WolverineStreamSeedBackend
		except ImportError:
			StreamSeedPlugin = None
			FaissHnswStreamSeedBackend = None
			FreshDiskANNStreamSeedBackend = None
			HnswlibStreamSeedBackend = None
			SymphonyQGBackend = None
			WolverineStreamSeedBackend = None
			_STREAMSEED_IMPORT_ERROR = exc
		else:
			_STREAMSEED_IMPORT_ERROR = None
	else:
		StreamSeedPlugin = None
		FaissHnswStreamSeedBackend = None
		FreshDiskANNStreamSeedBackend = None
		HnswlibStreamSeedBackend = None
		SymphonyQGBackend = None
		WolverineStreamSeedBackend = None
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

		# Wolverine backend params (effective when backend=wolverine)
		self.wolverine_M = int(index_params.get("M", 16))
		self.wolverine_ef_construction = int(index_params.get("efConstruction", 120))
		self.wolverine_random_seed = int(index_params.get("random_seed", 100))
		self.wolverine_allow_replace_deleted = bool(index_params.get("allow_replace_deleted", True))
		self.wolverine_delete_model = int(index_params.get("delete_model", 4))
		self.wolverine_new_link_size = int(index_params.get("new_link_size", self.wolverine_M))
		self.wolverine_num_threads = int(index_params.get("num_threads", 8))

		# hnswlib backend params (effective when backend=hnswlib)
		self.hnswlib_M = int(index_params.get("M", 16))
		self.hnswlib_ef_construction = int(index_params.get("efConstruction", 120))
		self.hnswlib_random_seed = int(index_params.get("random_seed", 100))
		self.hnswlib_allow_replace_deleted = bool(index_params.get("allow_replace_deleted", True))
		self.hnswlib_replace_deleted = bool(index_params.get("replace_deleted", False))
		self.hnswlib_num_threads = int(index_params.get("num_threads", 8))

		# FreshDiskANN backend params (effective when backend=freshdiskann)
		self.freshdiskann_R = int(index_params.get("R", index_params.get("graph_degree", 32)))
		self.freshdiskann_L = int(index_params.get("L", index_params.get("complexity", 100)))
		self.freshdiskann_alpha = float(index_params.get("alpha", 1.2))
		self.freshdiskann_num_threads = int(index_params.get("num_threads", 16))
		self.freshdiskann_insert_threads = int(index_params.get("insert_threads", self.freshdiskann_num_threads))
		self.freshdiskann_search_threads = int(index_params.get("search_threads", index_params.get("T", self.freshdiskann_num_threads)))
		self.freshdiskann_initial_search_complexity = int(index_params.get("initial_search_complexity", max(self.freshdiskann_L, 100)))
		self.freshdiskann_filter_complexity = int(index_params.get("filter_complexity", 0))
		self.freshdiskann_num_frozen_points = int(index_params.get("num_frozen_points", 1))
		self.freshdiskann_max_occlusion_size = int(index_params.get("max_occlusion_size", 750))
		self.freshdiskann_saturate_graph = bool(index_params.get("saturate_graph", False))
		self.freshdiskann_concurrent_consolidation = bool(index_params.get("concurrent_consolidation", True))
		self.freshdiskann_consolidate_every = int(index_params.get("consolidate_every", 0))

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
		elif self.backend_name == "wolverine":
			if WolverineStreamSeedBackend is None:
				raise RuntimeError("WolverineStreamSeedBackend is unavailable in streamseed package")
			backend = WolverineStreamSeedBackend(
				metric=self.metric,
				M=self.wolverine_M,
				ef_construction=self.wolverine_ef_construction,
				random_seed=self.wolverine_random_seed,
				allow_replace_deleted=self.wolverine_allow_replace_deleted,
				delete_model=self.wolverine_delete_model,
				new_link_size=self.wolverine_new_link_size,
				num_threads=self.wolverine_num_threads,
			)
		elif self.backend_name == "hnswlib":
			if HnswlibStreamSeedBackend is None:
				raise RuntimeError("HnswlibStreamSeedBackend is unavailable in streamseed package")
			backend = HnswlibStreamSeedBackend(
				metric=self.metric,
				M=self.hnswlib_M,
				ef_construction=self.hnswlib_ef_construction,
				random_seed=self.hnswlib_random_seed,
				allow_replace_deleted=self.hnswlib_allow_replace_deleted,
				replace_deleted=self.hnswlib_replace_deleted,
				num_threads=self.hnswlib_num_threads,
			)
		elif self.backend_name == "freshdiskann":
			if FreshDiskANNStreamSeedBackend is None:
				raise RuntimeError("FreshDiskANNStreamSeedBackend is unavailable in streamseed package")
			backend = FreshDiskANNStreamSeedBackend(
				metric=self.metric,
				R=self.freshdiskann_R,
				L=self.freshdiskann_L,
				alpha=self.freshdiskann_alpha,
				num_threads=self.freshdiskann_num_threads,
				insert_threads=self.freshdiskann_insert_threads,
				search_threads=self.freshdiskann_search_threads,
				initial_search_complexity=self.freshdiskann_initial_search_complexity,
				filter_complexity=self.freshdiskann_filter_complexity,
				num_frozen_points=self.freshdiskann_num_frozen_points,
				max_occlusion_size=self.freshdiskann_max_occlusion_size,
				saturate_graph=self.freshdiskann_saturate_graph,
				concurrent_consolidation=self.freshdiskann_concurrent_consolidation,
				consolidate_every=self.freshdiskann_consolidate_every,
			)
		else:
			raise ValueError(
				f"Unsupported backend '{self.backend_name}'. "
				"Use one of: faiss, symphonyqg, wolverine, hnswlib, freshdiskann"
			)

		self.plugin = StreamSeedPlugin(backend=backend)
		self.plugin.setup(max_pts=max_pts, ndim=ndim, verbose=self.verbose)

	def insert(self, x, ids):
		self.plugin.insert(x, ids)

	def delete(self, ids):
		self.plugin.delete(ids)

	def query(self, x, k, query_ids=None):
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
		self.res, distances = self.plugin.query(x, k, query_ids=query_ids)
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
