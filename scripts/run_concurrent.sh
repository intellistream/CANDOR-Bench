#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("diskann" "hnswlib_HNSW" "hnswlib_NSW")
# ALGORITHMS=("cufe")
# ALGORITHMS=("pyanns")
DATASETS=("reddit" "sift" "glove" "msong")
RUNBOOKS=("batch100_w05r95.yaml" "batch100_w10r90.yaml" "batch100_w20r80.yaml"
          "batch100_w50r50.yaml" "batch100_w80r20.yaml" "batch100_w90r10.yaml"
          )

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running with algorithm: $ALGO and dataset: $DS Using $DS"
      python3 run.py --neurips23track concurrent --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/concurrent/"$RUN" --dataset "$DS"
    done
  done
done

echo "All experiments completed."