#!/bin/bash
set -euo pipefail

if [ ! -d "$(pwd)/build/lib" ]; then
  echo "Native libs not found, building with CMake..."
  cmake -S . -B build
  cmake --build build -j
fi

export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH:-}"

go build -o bench .

configs=(
  # "config/annchor/sift/overall_only.yaml"
  # "config/annchor/msong/overall_only.yaml"
  "config/annchor/gist/overall_only.yaml"
  "config/annchor/deep1M/overall_only.yaml"
)

for cfg in "${configs[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "Skipping missing config: $cfg"
    continue
  fi
  echo "Running overall-only benchmark: $cfg"
  ./bench -config "$cfg"
done
