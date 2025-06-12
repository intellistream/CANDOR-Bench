from neurips23.concurrent.base import BaseConcurrentANN

class hnswlib_NSW(BaseConcurrentANN):
    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)
        self.name = "hnswlib_NSW"
        self.cm.edit("concurrentAlgoTag", "HNSWlib") 
        self.cm.edit("isNSW", 1)
        self.cm.edit("maxConnection", index_params["maxConnection"])
        self.cm.edit("efConstruction", index_params["efConstruction"])
        
    def set_query_arguments(self, query_args):
        # TODO:
        pass 