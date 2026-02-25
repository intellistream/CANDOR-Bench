#!/bin/bash
# ============================================================================
# Build + install + verify PyCANDYAlgo only
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}→ $1${NC}"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/sage-db-bench"

# Ensure venv
if [ ! -d "$VENV_DIR" ]; then
    print_error "Virtualenv not found: $VENV_DIR"
    print_info "Run deploy.sh once to create it."
    exit 1
fi

# Activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
print_success "Activated venv: $VIRTUAL_ENV"

# Optional MKL env
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
fi
if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
    export MKLROOT="/opt/intel/oneapi/mkl/latest"
    export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
    export CPATH="$MKLROOT/include:$CPATH"
fi

# Build PyCANDYAlgo
print_info "Building PyCANDYAlgo..."
cd "$SCRIPT_DIR/algorithms_impl"
if [ -f "build.sh" ]; then
    bash build.sh
    print_success "PyCANDYAlgo build finished"
else
    print_error "build.sh not found in algorithms_impl"
    exit 1
fi

# Install .so to site-packages
SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    print_error "PyCANDYAlgo.so not found in algorithms_impl"
    exit 1
fi

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
print_info "Installing $SO_FILE to $SITE_PACKAGES"
cp "$SO_FILE" "$SITE_PACKAGES/"
print_success "Installed PyCANDYAlgo"

# Verify import
print_info "Verifying PyCANDYAlgo import..."
if python3 -c "import PyCANDYAlgo; print('VERSION:', PyCANDYAlgo.__version__)" 2>&1; then
    print_success "PyCANDYAlgo import OK"
else
    print_error "PyCANDYAlgo import failed"
    exit 1
fi

print_success "Done"
