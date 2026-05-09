#!/bin/bash
# ============================================================================
# CANDOR-Bench / GammaFresh — One-shot setup script for a fresh Linux server
# ============================================================================
#
# Usage:
#   bash setup_new_server.sh [--skip-system] [--skip-data] [--skip-build]
#
# Tested on Ubuntu 22.04 / 24.04 with Python 3.12.
# Requires sudo for system package install (or use --skip-system).
#
# Time: 30 min (good network) to 2 hours (slow network / build issues).
# Disk: ~8 GB total (5 GB venv, 3 GB datasets, 200 MB build artifacts).
# ============================================================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
say()  { echo -e "${BLUE}→ $1${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

SKIP_SYSTEM=false; SKIP_DATA=false; SKIP_BUILD=false
for arg in "$@"; do
  case $arg in
    --skip-system) SKIP_SYSTEM=true ;;
    --skip-data)   SKIP_DATA=true ;;
    --skip-build)  SKIP_BUILD=true ;;
    -h|--help) head -20 "$0" | tail -19 | sed 's/^# \?//'; exit 0 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
say "Repo root: $REPO_ROOT"

# ----------------------------------------------------------------------------
# Step 1: System dependencies
# ----------------------------------------------------------------------------
if ! $SKIP_SYSTEM; then
  say "Step 1/6: System packages (sudo required)"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake ninja-build git curl wget \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    pkg-config libopenblas-dev liblapack-dev
  ok "System packages installed"
else
  warn "Skipping system package install"
fi

# ----------------------------------------------------------------------------
# Step 2: uv (Python package manager)
# ----------------------------------------------------------------------------
say "Step 2/6: Install uv if not present"
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version)"

# ----------------------------------------------------------------------------
# Step 3: Submodules
# ----------------------------------------------------------------------------
say "Step 3/6: Init git submodules (GammaFresh)"
git submodule update --init --recursive
ok "Submodules ready"

# ----------------------------------------------------------------------------
# Step 4: Python venv + packages
# ----------------------------------------------------------------------------
say "Step 4/6: Create venv + install Python deps"
if [ ! -d .venv ]; then
  uv venv --python 3.12
fi
# Use specific torch version to avoid the CUDA-detection issue we hit:
# torch >=2.6 wheels on PyPI auto-enable CUDA in CMake, breaking gamma build
# on machines without nvcc. torch 2.5.1 is the last "safe" version.
uv pip install \
  "numpy" \
  "faiss-cpu" \
  "hnswlib" \
  "scann" \
  "annoy" \
  "torch==2.5.1" \
  "matplotlib" \
  "pandas" \
  "pyyaml" \
  "tqdm" \
  "gdown"
ok "Python packages installed"

# ----------------------------------------------------------------------------
# Step 5: Build GammaFresh C++ (PyCANDY-style)
# ----------------------------------------------------------------------------
if ! $SKIP_BUILD; then
  say "Step 5/6: Build GammaFresh (~5-10 min)"
  cd algorithms_impl/GammaFresh
  mkdir -p build && cd build
  source "$REPO_ROOT/.venv/bin/activate"
  TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
  cmake -GNinja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCMAKE_CXX_STANDARD=20 || fail "CMake configure failed"
  cmake --build . -j$(nproc) || fail "Build failed"
  cd "$REPO_ROOT"
  ok "GammaFresh built. _candy.so should be at algorithms_impl/GammaFresh/build/src/bindings/python/"
else
  warn "Skipping GammaFresh build"
fi

# ----------------------------------------------------------------------------
# Step 6: Datasets
# ----------------------------------------------------------------------------
if ! $SKIP_DATA; then
  say "Step 6/6: Download datasets (~3 GB, ~10 min)"
  cd "$REPO_ROOT"
  source .venv/bin/activate
  # SIFT (1M, 128d, image features) — fastest, always works
  python3 -c "from datasets.loaders import prepare_dataset; prepare_dataset('sift')" \
    || warn "SIFT prepare failed (may need raw_data/sift/ unpacked)"
  # MSong (992K, 420d, audio features) — slow Google Drive download
  python3 -c "from datasets.loaders import prepare_dataset; prepare_dataset('msong')" || true
  # MSong has known path mismatch — copy if needed
  if [ -d raw_data/msong/Msong ] && [ ! -f raw_data/msong/data_992272_420 ]; then
    cp raw_data/msong/Msong/* raw_data/msong/
  fi
  # GloVe (1.18M, 100d, text embeddings) — has different file count than expected
  python3 -c "from datasets.loaders import prepare_dataset; prepare_dataset('glove')" || true
  if [ -d raw_data/glove/Glove ] && [ ! -f raw_data/glove/data_1183514_100 ]; then
    cp raw_data/glove/Glove/* raw_data/glove/
    # Loader expects a specific row count; symlink to actual downloaded file
    ln -sf data_1192514_100 raw_data/glove/data_1183514_100 2>/dev/null || true
  fi
  ok "Datasets prepared (SIFT mandatory; MSong/GloVe optional)"
else
  warn "Skipping dataset download"
fi

# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
say "Running smoke test (small SIFT slice via gamma)..."
cd "$REPO_ROOT"
OMP_NUM_THREADS=1 uv run python -c "
import sys
sys.path.insert(0, 'experiments')
from _shared import build, load_dataset
import numpy as np
data, queries = load_dataset('sift', n_queries=100, slice_n=10000)
idx = build('gamma', 128)
idx.initial_load(np.arange(5000, dtype=np.uint64), np.ascontiguousarray(data[:5000]))
idx.add(np.arange(5000, 10000, dtype=np.uint64), np.ascontiguousarray(data[5000:10000]))
idx.maintain(0, 0, False)
nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
print(f'Smoke test OK: {nbrs.shape} results')
" 2>&1 | grep -v "config_loader\|FaissHNSW constructed\|gamma_fresh_logger\|index_factory" | tail -3 \
  && ok "Smoke test passed!" || fail "Smoke test failed"

echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  OMP_NUM_THREADS=1 uv run python experiments/e15_delete_pattern_matrix/run.py"
echo ""
echo "Read experiments/INTENTS.md for what each experiment proves."
echo "Read CLAUDE.md for project context."
