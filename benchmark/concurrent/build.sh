#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CC_SRC_DIR="$(cd "$SCRIPT_DIR/../../src/CANDY/ConcurrentIndex" && pwd)"
INCLUDE_CC_DIR="$(cd "$SCRIPT_DIR/../../include/CANDY/ConcurrentIndex" && pwd)"
BUILD_DIR="$CC_SRC_DIR/build"

# Use common local Go install path when go is present but not in PATH.
if ! command -v go >/dev/null 2>&1; then
  if [ -x "/usr/local/go/bin/go" ]; then
    export PATH="/usr/local/go/bin:$PATH"
  fi
fi

if ! command -v go >/dev/null 2>&1; then
  echo "Go toolchain not found. Install Go or add it to PATH."
  exit 127
fi
GO_BIN="$(command -v go)"

if [ -f "$BUILD_DIR/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$CC_SRC_DIR" "$BUILD_DIR/CMakeCache.txt"; then
  echo "Detected stale CMake cache at $BUILD_DIR; recreating build directory."
  rm -rf "$BUILD_DIR"
fi

echo "=== Step 1: Build C++ shared library (libindex.so) ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$CC_SRC_DIR" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "=== Step 2: Build Go benchmark runner ==="
cd "$SCRIPT_DIR"
GO_CACHE_DIR="$SCRIPT_DIR/.cache/go-build"
GO_TMP_DIR="$SCRIPT_DIR/.cache/go-tmp"
mkdir -p "$GO_CACHE_DIR" "$GO_TMP_DIR"
CGO_LDFLAGS="-L$BUILD_DIR/lib -lindex -Wl,-rpath,$BUILD_DIR/lib" \
CGO_CFLAGS="-I$CC_SRC_DIR -I$INCLUDE_CC_DIR" \
GOCACHE="$GO_CACHE_DIR" \
GOTMPDIR="$GO_TMP_DIR" \
"$GO_BIN" build -o cc-bench .

echo ""
echo "=== Build complete ==="
echo "Run: ./cc-bench -config ../configs/concurrent/<config>.yaml"
