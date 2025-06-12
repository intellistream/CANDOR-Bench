#!/bin/bash

# Define the algorithms and datasets
GENERAL_DATASETS=("glove" "msong" "sun" "trevi" "dpr" "reddit" "random-experiment" "random-plus-experiment" "sift" "msturing-30M-clustered" "wte-0.05" "wte-0.2" "wte-0.4" "wte-0.6" "wte-0.8" "cirr" "coco")
GENERAL_DATASETS=("sift")
# Iterate through each combination of algorithm and dataset
for DS in "${GENERAL_DATASETS[@]}"; do
  python3 benchmark/congestion/compute_gt.py --runbook neurips23/runbooks/congestion/algo_optimizations/algo_optimizations.yaml --dataset "$DS" --gt_cmdline_tool ~/DiskANN/build/apps/utils/compute_groundtruth
done