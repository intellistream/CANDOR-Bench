import argparse
import os
import numpy as np
import PyCANDYAlgo

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.concurrent.load_runbook import load_runbook_concurrent

        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)


def get_next_set(id_list: list, entry):
    match entry['operation']:
        case 'initial' | 'insert_and_search':
            id_list.extend(range(entry['start'], entry['end']))
            return id_list
        case 'search':
            return id_list
        case _:
            raise ValueError('Undefined entry in runbook')


def output_gt(ds, id_list, step, args):
    ids = np.array(id_list, dtype=np.uint32)
    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[ids]  

    dir = gt_dir(ds, args.runbook_file)
    prefix = os.path.join(dir, f'step{step}')
    os.makedirs(dir, exist_ok=True)

    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  
        f.write(ds.d.to_bytes(4, byteorder='little'))  
        data_slice.tofile(f)

    print("Executing GT")
    PyCANDYAlgo.compute_vec_gt(data_file, args.query_file, gt_file, args.k, args.dist_fn)

    print(f"Removing data file: {data_file}")
    os.system(f"rm {data_file}")


def output_stepwise_gt(ds, id_list, step, max_pts, cc_config, args):
    ids = np.array(id_list, dtype=np.uint32)
    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[ids]  

    dir = gt_dir(ds, args.runbook_file)
    prefix = os.path.join(dir, f'step{step}')
    os.makedirs(dir, exist_ok=True)

    data_file = prefix + '.data'
    gt_file = prefix + '.cc.gt.hdf5'

    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little'))  
        f.write(ds.d.to_bytes(4, byteorder='little'))  
        data_slice.tofile(f)
  
    print("Executing stepwise GT")
    PyCANDYAlgo.compute_stepwise_gt(data_file, args.query_file, gt_file, args.k, args.dist_fn, 
        cc_config["batch_size"], cc_config["initial_count"], cc_config["cc_query_size"])
    print("Stepwise GT file: ", gt_file)

    print(f"Removing data file: {data_file}")
    os.system(f"rm {data_file}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path'
    )
    parser.add_argument(
        '--private_query',
        action='store_true'
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    
    max_pts, cc_config, runbook  = load_runbook_concurrent(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    match ds.distance():
        case 'euclidean': setattr(args, 'dist_fn', 'l2')
        case 'ip': setattr(args, 'dist_fn', 'mips')
        case _: raise RuntimeError('Invalid metric')

    match ds.dtype:
        case 'float32': setattr(args, 'data_type', 'float')
        case 'int8': setattr(args, 'data_type', 'int8')
        case 'uint8': setattr(args, 'data_type', 'uint8')
        case _: raise RuntimeError('Invalid datatype')
         
    setattr(args, 'k', 10)
    setattr(args, 'query_file', os.path.join(ds.basedir, query_file))
    
    step = 1
    id_list = []
    
    for entry in runbook:
        id_list = get_next_set(id_list, entry)
        
        if entry['operation'] == 'search':
            output_gt(ds, id_list, step, args)
            
        elif entry['operation'] == 'insert_and_search':
            output_stepwise_gt(ds, id_list, step, max_pts, cc_config, args)

        step += 1


if __name__ == '__main__':
    main()