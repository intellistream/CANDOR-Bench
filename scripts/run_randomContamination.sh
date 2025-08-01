#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("candy_lshapg" "candy_mnru" "candy_sptag" "cufe" "diskann" "faiss_fast_scan" "faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq" "puck" "pyanns" "ipdiskann" "gti")
DATASETS=("sift")
RUNBOOKS=("0.05.yaml" "0.10.yaml" "0.15.yaml" "0.20.yaml" "0.25.yaml")
# Iterate through each combination of algorithm and dataset

for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running with algorithm: $ALGO and dataset: $DS Using $DS"
      python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/congestion/randomContamination/randomContamination"$RUN" --dataset "$DS"
    done
  done
done

echo "All experiments completed."