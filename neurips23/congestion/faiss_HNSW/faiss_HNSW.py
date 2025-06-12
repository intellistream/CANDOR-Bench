from neurips23.congestion.base import BaseCongestionDropANN
from neurips23.streaming.faiss_HNSW.faiss_HNSW import faiss_HNSW as faiss_HNSW_streaming

class faiss_HNSW(BaseCongestionDropANN):
    def __init__(self, metric, index_params):
        super().__init__([faiss_HNSW_streaming(metric, index_params)], metric, index_params)
        self.metric = metric
        self.indexkey=self.workers[0].my_index_algo.indexkey
        self.name = self.workers[0].my_index_algo.name

    def set_query_arguments(self, query_args):
        self.workers[0].my_index_algo.set_query_arguments(query_args)


