from __future__ import absolute_import

import h5py
import json
import os
import re
import csv
import torch
import traceback

def get_result_filename(dataset=None, count=None, definition=None,
                        query_arguments=None, neurips23track=None, runbook_path=None):
    d = ['results']
    if neurips23track and neurips23track != 'none':
        d.append('neurips23')
        d.append(neurips23track)
        if neurips23track in ['streaming', 'congestion', 'concurrent']:
            if runbook_path == None:
                raise RuntimeError('Need runbook_path to store results')
            else:
                d.append(os.path.split(runbook_path)[1])
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm)
        build_args = definition.arguments
        try:
            for args in build_args:
                if type(args) == dict and 'indexkey' in args:
                    build_args = [args['indexkey']]
        except:
                pass
        data = build_args + query_arguments
        data = re.sub(r'\W+', '_', json.dumps(data, sort_keys=True)).strip('_')
        if len(data) > 150:
            data = data[-149:]
        d.append(data)

    return os.path.join(*d)


def get_cc_result_filename(dataset=None, count=None, definition=None,
                            query_arguments=None, neurips23track=None, 
                            runbook_path=None, cc_config={}):
    d = ['results']
    if neurips23track and neurips23track != 'none':
        d.append('neurips23')
        d.append(neurips23track)
        if neurips23track in ['streaming', 'congestion', 'concurrent']:
            if runbook_path == None:
                raise RuntimeError('Need runbook_path to store results')
            else:
                d.append(os.path.split(runbook_path)[1])
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm)
        build_args = definition.arguments
        try:
            for args in build_args:
                if type(args) == dict and 'indexkey' in args:
                    build_args = [args['indexkey']]
        except:
                pass
        data = build_args + query_arguments
        data = re.sub(r'\W+', '_', json.dumps(data, sort_keys=True)).strip('_')
        
        data += f"_b{cc_config['batch_size']}"
        data += f"_t{cc_config['num_threads']}"
            
        write_ratio_suffix = int(cc_config['write_ratio'] * 100)
        read_ratio_suffix = int((1 - cc_config['write_ratio']) * 100)
        data += f"_w{write_ratio_suffix}r{read_ratio_suffix}"
        
        if len(data) > 150:
            data = data[-149:]
        d.append(data)

    return os.path.join(*d)


def add_cc_results_to_h5py(f, search_type, results, count, suffix=''):
    if search_type in ["knn", "knn_filtered"]:
        if isinstance(results, list) and all(isinstance(x, torch.Tensor) for x in results):
            results = torch.stack(results)  
        results = results.numpy()
        results = results.T  
        f.create_dataset('neighbors' + suffix, results.shape, 'i', data=results)
    elif search_type == "range":
        lims, D, I = results
        f.create_dataset('neighbors' + suffix, data=I)
        f.create_dataset('lims' + suffix, data=lims)
        f.create_dataset('distances' + suffix, data=D)
    else:
        raise NotImplementedError(f"Search type '{search_type}' not supported.")


def add_results_to_h5py(f, search_type, results, count, suffix = ''):
    if search_type in ["knn", "knn_filtered"]:
        neighbors = f.create_dataset('neighbors' + suffix, (len(results), count), 'i', data = results)
    elif search_type == "range":
        lims, D, I= results
        f.create_dataset('neighbors' + suffix, data=I)
        f.create_dataset('lims' + suffix, data=lims)
        f.create_dataset('distances' + suffix, data=D)
    else:
        raise NotImplementedError()


def store_results(dataset, count, definition, query_arguments,
        attrs, results, search_type, neurips23track=None, runbook_path=None, cc_config={}):
    
    fn = ""
    fn_attr = ""
    
    if neurips23track != "concurrent":
        fn = get_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path) + '.hdf5'
        fn_attr = get_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path) + '.csv'
    else:
        fn = get_cc_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path, cc_config) + '.hdf5'
        fn_attr = get_cc_result_filename(
            dataset, count, definition, query_arguments, neurips23track, runbook_path, cc_config) + '.csv'
        
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(name=fn, mode='w', libver='latest')
    import pandas as pd
    df = pd.DataFrame([attrs])

    # Write the DataFrame to a CSV file
    df.to_csv(fn_attr, index=False)

    for k, v in attrs.items():
        # TODO: here is one potential bug
        f.attrs[k] = v
    
    if neurips23track in ['streaming', 'congestion']:
        for i, step_results in enumerate(results):
            step = attrs['step_' + str(i)]
            # store labels
            add_results_to_h5py(f, search_type, step_results, count, '_step' + str(step))
    elif neurips23track == 'concurrent':
        for i, step_results in enumerate(results):
            step = attrs['step_' + str(i)]
            # store tensors
            add_cc_results_to_h5py(f, search_type, step_results, count, '_step' + str(step))
    else:
        add_results_to_h5py(f, search_type, results, count)

    f.close()


def load_all_results(dataset=None, count=None, neurips23track="congestion", runbook_path=None):
    """
    A generator for all result files.
    """
    for root, _, files in os.walk(get_result_filename(dataset, count, \
                                                      neurips23track=neurips23track, \
                                                      runbook_path=runbook_path)):
        for fn in files:
            if os.path.splitext(fn)[-1] != '.hdf5':
                continue
            if fn.endswith('.cc.hdf5'):
                continue
            full_path = os.path.join(root, fn)
            print(f"Found HDF5 file: {full_path}")
            try:
                f = h5py.File(name=os.path.join(root, fn), mode='r+', libver='latest')
                properties = dict(f.attrs)
                yield properties, f
                f.close()
            except:
                print('Was unable to read', fn)
                traceback.print_exc()


def load_all_attrs(dataset=None, count=None, neurips23track="concurrent", runbook_path=None):
    for root, _, files in os.walk(get_result_filename(dataset, count, \
                                                      neurips23track=neurips23track, \
                                                      runbook_path=runbook_path)):
        for fn in files:
            if os.path.splitext(fn)[-1].lower() != '.csv':
                continue
            full_path = os.path.join(root, fn)
            print(f"Found CSV file: {full_path}")
            try:
                with open(full_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)  
                    properties = {"filename": fn}  
                    yield properties, reader  
            except Exception as e:
                print(f'Was unable to read {fn}')
                traceback.print_exc()
                

def get_unique_algorithms():
    return set(properties['algo'] for properties, _ in load_all_results())
