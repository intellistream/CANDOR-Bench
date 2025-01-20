import PyCANDYAlgo

import numpy as np
from numpy import typing as npt

from neurips23.streaming.base import BaseStreamingANN
import torch
import traceback

class linear(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.indexkey= index_params['indexkey']
        self.metric = metric
        self.name = "linear"
        self.trained = False

    def setup(self, dtype, max_pts, ndim):
        index = None
        if self.metric == 'euclidean':
            index = PyCANDYAlgo.index_factory_l2(ndim, self.indexkey)
        else:
            index = PyCANDYAlgo.index_factory_ip(ndim, self.indexkey)

        self.index = index

    def insert(self, X,ids):
        if(self.trained):
            self.index.add(X.shape[0],X.flatten())
        else:
            self.index.train(X.shape[0],X.flatten())
            self.index.add(X.shape[0], X.flatten())
            self.trained=True


    def delete(self, ids):
        return


    def query(self, X, k):


        querySize = X.shape[0]

        results = self.index.search(querySize, X.flatten(), k, self.ef)
        res = np.array(results).reshape(X.shape[0], k)
        self.res = res

    def set_query_arguments(self, query_args):
        if "ef" in query_args:
            self.ef = query_args['ef']
        else:
            self.ef = 16

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"