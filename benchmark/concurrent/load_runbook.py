import os
import yaml
from collections import namedtuple

def load_runbook_concurrent(dataset_name, max_pts, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset_name] 

    run_list = []
    initial_count = 0
    i = 1
    
    while i in runbook:  
        entry = runbook[i]
        print(entry)
        
        if entry['operation'] not in {'initial', 'insert_and_search', 'search'}:
            raise Exception('Undefined runbook operation')
        
        if entry['operation'] in {'initial', 'insert_and_search'}:
            if 'start' not in entry or 'end' not in entry:
                raise Exception(f'Start/End missing in runbook at entry {i}')
            
            if entry['start'] < 0 or entry['start'] >= max_pts:
                raise Exception(f'Start out of range at entry {i}')
            if entry['end'] < 0 or entry['end'] > max_pts:
                print(entry['end'], max_pts)
                raise Exception(f'End out of range at entry {i}')
            
            if entry['operation'] == 'initial':
                initial_count = entry['end'] - entry['start']

        run_list.append(entry)
        i += 1
        
    print("run_List : ", run_list)
        
    max_pts = runbook.get('max_pts')
    if max_pts == None:
        raise Exception('max points not listed for dataset in runbook')
    
    write_ratio = runbook.get('write_ratio')
    if write_ratio == None:
        raise Exception('write threads ratio not listed in runbook')

    batch_size = runbook.get('batch_size')
    if batch_size == None:
        raise Exception('batch size not listed in runbook')

    cc_query_size = runbook.get('cc_query_size')
    if cc_query_size == None:
        cc_query_size = 100

    num_threads = runbook.get('num_threads')
    if num_threads == None:
        num_threads = os.cpu_count()
        print(f"number of threads not listed in runbook, use default threads {num_threads}")
        
    random_mode = runbook.get('random_mode')
    if random_mode == None:
        raise Exception('random mode not listed in runbook')
    
    cc_config = {
        'write_ratio': write_ratio,
        'batch_size': batch_size,
        'num_threads': num_threads,
        'random_mode': random_mode,
        'initial_count': initial_count,
        'cc_query_size': cc_query_size
    }

    return max_pts, cc_config, run_list

