import PyCANDYAlgo

import numpy as np
from numpy import typing as npt

from neurips23.streaming.base import BaseStreamingANN
import torch

class candy_hnsw(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.indexkey= index_params['indexkey']
        self.metric = metric
        self.name = "candy_HNSW"
        self.ef=16
        self.trained = False

    def setup(self, dtype, max_pts, ndim):
        index = PyCANDYAlgo.createIndex(self.indexkey, ndim)

        cm = PyCANDYAlgo.ConfigMap()
        if self.metric == 'euclidean':
            cm.edit("metricType", "L2")
        else:
            cm.edit("metricType", "IP")
        cm.edit("indexTag", self.indexkey)
        cm.edit("vecDim", ndim)
        index.setConfig(cm)
        self.index = index

        return

    def insert(self, X,ids):
        subA = torch.from_numpy(X.copy())
        if(self.trained):
            self.index.insertTensorWithIds(ids,subA)
        else:
            self.index.loadInitialTensorWithIds(ids,subA)
            self.trained = True


    def delete(self, ids):
        self.index.deleteIndex(ids)
        return

    def query(self, X, k):


        queryTensor = torch.from_numpy(X.copy())
        results = self.index.searchIndex(queryTensor, k)
        res = np.array(results).reshape(X.shape[0], k)

        self.res = res

    def set_query_arguments(self, query_args):
        if "ef" in query_args:
            self.ef = query_args['ef']
        else:
            self.ef = 16

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"