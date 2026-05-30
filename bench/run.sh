#!/bin/bash
export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

configs=()
datasets=("sift" "msong" "gist" "deep1M")

for dataset in "${datasets[@]}"; do
    for batch in 20 100 200; do
        # configs+=("config/hnsw/${dataset}/ch_b${batch}.yaml")
        # configs+=("config/hnsw/${dataset}/pk_b${batch}.yaml")

        # configs+=("config/annchor/${dataset}/ch_b${batch}.yaml")
        configs+=("config/annchor/${dataset}/ch_compare_b${batch}.yaml")
        configs+=("config/hnsw/${dataset}/ch_compare_b${batch}.yaml")
        # configs+=("config/hnsw/${dataset}/pk_compare_b${batch}.yaml")

        # configs+=("config/annchor/${dataset}/pk_b${batch}.yaml")
        # configs+=("config/annchor/${dataset}/pk_compare_b${batch}.yaml")

    done
done

go build -gcflags "all=-N -l" -o bench .

if [ $? -eq 0 ]; then
    for config in "${configs[@]}"; do
        echo "Running: $config"
        ./bench -config "$config"
    done
else
    echo "Build failed."
    exit 1
fi
