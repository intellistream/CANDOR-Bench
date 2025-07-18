#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("candy_lshapg" "candy_mnru" "candy_sptag" "faiss_fast_scan" "faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq" "ipdiskann" "gti")
DATASETS=("glove" "msong" "sift" )

# Iterate through each combination of algorithm and dataset
for DS in "${DATASETS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    echo "Running with algorithm: $ALGO and dataset: $DS"
    python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/congestion/general_experiment/general_experiment.yaml --dataset "$DS"
  done
done

echo "All experiments completed."
