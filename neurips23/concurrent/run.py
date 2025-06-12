import numpy as np
import time
import json

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS
from benchmark.results import get_cc_result_filename
    
class ConcurrentRunner(BaseRunner):
    def build(algo, dataset, max_pts, cc_config):
        t0 = time.time()
        ds = DATASETS[dataset]()
        ndims = ds.d
        algo.setup(ds.dtype, max_pts, cc_config, ndims)
        print('Algorithm set up')
        return time.time() - t0
        
    def run_task(algo, ds, distance, count, run_count, search_type, private_query, runbook, definition, query_arguments, runbook_path, dataset):
        all_results = []

        Q = ds.get_queries() if not private_query else ds.get_private_queries()
        print(fr"Got {Q.shape[0]} queries")  
        
        cc_config = algo.get_cc_config()
        # cc_res_file = get_cc_result_filename(dataset, count, definition, query_arguments, neurips23track="concurrent", runbook_path=runbook_path, cc_config=cc_config)
        # cc_res_file += ".cc.hdf5"
        
        ccQ = Q[:cc_config["cc_query_size"]]

        result_map = {}
        num_searches = 0
        for step, entry in enumerate(runbook):
            start_time = time.time()
            match entry['operation']:
                case 'initial':
                    start = entry['start']
                    end = entry['end']
                    algo.initial(ds.get_data_in_range(start, end))
                case 'insert_and_search':
                    start = entry['start']
                    end = entry['end']
                    algo.cc_insert_and_query(ds.get_data_in_range(start, end), ccQ, count)
                case 'search':
                    algo.query(Q, count)
                    all_results.append(algo.get_results())
                    result_map[num_searches] = step + 1
                    num_searches += 1
                case _:
                    raise NotImplementedError('Invalid runbook operation.')
            step_time = (time.time() - start_time)
            print(f"Step {step+1} took {step_time}s.")
        
        cc_res_filename = "stepwise/" + str(algo) + "_" + dataset + "_" + str(cc_config["write_ratio"])
        recall_filename = cc_res_filename + ".csv"
        cc_filename = cc_res_filename + ".txt"
        cc_res = algo.save_and_get_cc_results(recall_filename)
        with open(cc_filename, 'w') as f:
            json.dump(cc_res, f, indent=4)
        
        attrs = {
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "count": int(count),
            "private_queries": private_query,
            "num_searches": num_searches,
            "batch_size": cc_config["batch_size"],
            "write_ratio": cc_config["write_ratio"],
            "num_threads": cc_config["num_threads"],
            "insert_throughput": cc_res["insertThroughput"],
            "search_throughput": cc_res["searchThroughput"],
            "insert_latency_avg": cc_res["insertLatencyAvg"],
            "search_latency_avg": cc_res["searchLatencyAvg"],
            "insert_latency_95": cc_res["insertLatency95"],
            "search_latency_95": cc_res["searchLatency95"],
            "cc_result_filename": "",
        }
        
        for k, v in result_map.items():
            attrs['step_' + str(k)] = v

        additional = algo.get_additional()
        for k in additional:
            attrs[k] = additional[k]
            
        return (attrs, all_results)
    
    
    
