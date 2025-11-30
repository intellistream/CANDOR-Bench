#!/bin/bash
# 构建 PyCANDYAlgo 模块的脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "Building PyCANDYAlgo Module"
echo "========================================="
echo ""

# 检查依赖
echo "Checking dependencies..."
python3 -c "import torch" 2>/dev/null || { echo "Error: PyTorch not installed. Run: pip install torch"; exit 1; }
echo "  ✓ PyTorch found"

which cmake >/dev/null 2>&1 || { echo "Error: CMake not installed"; exit 1; }
echo "  ✓ CMake found"

pkg-config --exists gflags 2>/dev/null || echo "  ⚠ Warning: gflags not found (may cause build errors)"

echo ""

# 清理旧构建
if [ -d "build" ]; then
    echo "Cleaning old build directory..."
    rm -rf build
fi

# 创建构建目录
mkdir -p build
cd build

echo ""
echo "Configuring with CMake..."
echo "----------------------------------------"

# 设置 MKL 环境变量（Puck 需要）
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "  Loading Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
fi

if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
    export MKLROOT="/opt/intel/oneapi/mkl/latest"
    export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
    export CPATH="$MKLROOT/include:$CPATH"
    export CMAKE_PREFIX_PATH="$MKLROOT:$CMAKE_PREFIX_PATH"
    echo "  MKL found: $MKLROOT"
elif [ -d "/opt/intel/mkl" ]; then
    export MKLROOT="/opt/intel/mkl"
    export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
    export CPATH="$MKLROOT/include:$CPATH"
    export CMAKE_PREFIX_PATH="$MKLROOT:$CMAKE_PREFIX_PATH"
    echo "  MKL found: $MKLROOT"
else
    echo "  ⚠ Warning: MKL not found"
    echo "  Puck may fail to build"
fi

# 获取 PyTorch CMake 路径（如果存在）
TORCH_CMAKE_PATH=""
if python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null; then
    TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
    echo "  PyTorch CMake path: $TORCH_CMAKE_PATH"
else
    echo "  ⚠ Warning: Could not get torch.utils.cmake_prefix_path"
fi

# 构建 CMake 参数
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DPYTHON_EXECUTABLE=$(which python3)
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
)

# 添加 torch 路径
if [ -n "$TORCH_CMAKE_PATH" ]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH")
fi

# 运行 CMake 配置
echo "  Running: cmake ${CMAKE_ARGS[@]} .."
cmake "${CMAKE_ARGS[@]}" .. \
    2>&1 | tee cmake_config.log \
    || { echo ""; echo "❌ CMake configuration failed"; cat cmake_config.log | tail -100; exit 1; }

echo ""
echo "========================================="
echo "Building Thirdparty Libraries"
echo "========================================="
echo ""

# 计算并行编译数 (每个编译进程约需 1-2GB)
NPROC=$(nproc)
MAX_JOBS=$(($(free -g | awk '/^Mem:/{print $7}') / 2))  # 可用内存 / 2GB
JOBS=$((MAX_JOBS < NPROC ? MAX_JOBS : NPROC))
JOBS=$((JOBS < 1 ? 1 : JOBS))  # 至少 1 个
JOBS=$((JOBS > 4 ? 4 : JOBS))  # 最多 4 个(安全起见)

echo "Using -j${JOBS} for compilation (available cores: ${NPROC})"
echo ""

# 返回到 algorithms_impl 目录
cd ..

# 安装所有缺失的系统依赖库（在构建前统一安装）
echo "Checking and installing system dependencies..."
echo "----------------------------------------"
MISSING_DEPS=()

# 检查各个库
pkg-config --exists fmt 2>/dev/null || MISSING_DEPS+=("libfmt-dev")
pkg-config --exists pybind11 2>/dev/null || MISSING_DEPS+=("pybind11-dev")
dpkg -l | grep -q libnuma-dev 2>/dev/null || MISSING_DEPS+=("libnuma-dev")
dpkg -l | grep -q libunwind-dev 2>/dev/null || MISSING_DEPS+=("libunwind-dev")
pkg-config --exists libglog 2>/dev/null || MISSING_DEPS+=("libgoogle-glog-dev")
pkg-config --exists spdlog 2>/dev/null || MISSING_DEPS+=("libspdlog-dev")

# 如果有缺失的依赖，则安装
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "  Installing missing dependencies: ${MISSING_DEPS[*]}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y "${MISSING_DEPS[@]}"
    elif command -v yum &> /dev/null; then
        # 转换包名为 yum 格式
        YUM_DEPS=()
        for dep in "${MISSING_DEPS[@]}"; do
            case $dep in
                libfmt-dev) YUM_DEPS+=("fmt-devel") ;;
                pybind11-dev) YUM_DEPS+=("pybind11-devel") ;;
                libnuma-dev) YUM_DEPS+=("numactl-devel") ;;
                libunwind-dev) YUM_DEPS+=("libunwind-devel") ;;
                libgoogle-glog-dev) YUM_DEPS+=("glog-devel") ;;
                libspdlog-dev) YUM_DEPS+=("spdlog-devel") ;;
            esac
        done
        sudo yum install -y "${YUM_DEPS[@]}"
    elif command -v brew &> /dev/null; then
        # 转换包名为 brew 格式
        BREW_DEPS=()
        for dep in "${MISSING_DEPS[@]}"; do
            case $dep in
                libfmt-dev) BREW_DEPS+=("fmt") ;;
                pybind11-dev) BREW_DEPS+=("pybind11") ;;
                libnuma-dev) BREW_DEPS+=("numactl") ;;
                libunwind-dev) BREW_DEPS+=("libunwind") ;;
                libgoogle-glog-dev) BREW_DEPS+=("glog") ;;
                libspdlog-dev) BREW_DEPS+=("spdlog") ;;
            esac
        done
        brew install "${BREW_DEPS[@]}"
    else
        echo "  ⚠ Warning: Could not install dependencies automatically"
    fi
    echo "  ✓ Dependencies installed"
else
    echo "  ✓ All dependencies already installed"
fi
echo ""

# === 1. Build GTI ===
if [ -d "gti/GTI" ]; then
    echo "Building GTI..."
    echo "----------------------------------------"
    
    # 先构建 GTI 的依赖库 n2
    if [ -d "gti/GTI/extern_libraries/n2" ]; then
        echo "  Building GTI dependency: n2 library..."
        cd gti/GTI/extern_libraries/n2
        if [ -d build ]; then
            rm -rf build
        fi
        mkdir -p build
        make shared_lib || { echo "❌ Failed to build n2 library"; exit 1; }
        echo "  ✓ n2 library built"
        cd ../../../..
    fi
    
    # 构建 GTI
    cd gti/GTI
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir -p bin build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. || { echo "❌ GTI cmake failed"; exit 1; }
    make -j${JOBS} || { echo "❌ GTI build failed"; exit 1; }
    make install || echo "  ⚠ GTI install failed (not critical)"
    cd ../../..
    echo "✓ GTI built successfully"
    echo ""
else
    echo "⚠ GTI not found (git submodule may not be initialized)"
    echo ""
fi

# === 2. Build IP-DiskANN ===
if [ -d "ipdiskann" ]; then
    echo "Building IP-DiskANN..."
    echo "----------------------------------------"
    cd ipdiskann
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir -p build
    cd build
    cmake .. || { echo "❌ IP-DiskANN cmake failed"; exit 1; }
    make -j${JOBS} || { echo "❌ IP-DiskANN build failed"; exit 1; }
    make install || echo "  ⚠ IP-DiskANN install failed (not critical)"
    cd ../..
    echo "✓ IP-DiskANN built successfully"
    echo ""
else
    echo "⚠ IP-DiskANN not found (git submodule may not be initialized)"
    echo ""
fi

# === 3. Build PLSH ===
if [ -d "plsh" ]; then
    echo "Building PLSH..."
    echo "----------------------------------------"
    cd plsh
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir -p build
    cd build
    cmake .. || { echo "❌ PLSH cmake failed"; exit 1; }
    make -j${JOBS} || { echo "❌ PLSH build failed"; exit 1; }
    make install || echo "  ⚠ PLSH install failed (not critical)"
    cd ../..
    echo "✓ PLSH built successfully"
    echo ""
else
    echo "⚠ PLSH not found (git submodule may not be initialized)"
    echo ""
fi

# 返回到 build 目录
cd build

echo ""
echo "========================================="
echo "Building PyCANDYAlgo Main Module"
echo "========================================="
echo ""
echo "Compiling..."
echo "----------------------------------------"


# 编译 PyCANDYAlgo (JOBS 已经在前面计算好了)
make -j${JOBS} || { echo ""; echo "❌ Build failed"; exit 1; }

# 返回上级目录
cd ..

echo ""
echo "========================================="
echo "✅ Build Complete!"
echo "========================================="
echo ""

# 检查生成的文件
SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)

if [ -n "$SO_FILE" ]; then
    echo "PyCANDYAlgo module generated:"
    ls -lh "$SO_FILE"
    echo ""
    
    # 测试本地导入
    echo "Testing local import..."
    python3 -c "import sys; sys.path.insert(0, '.'); import PyCANDYAlgo; print('✅ Local import successful')" || {
        echo "⚠ Local import test failed"
        exit 1
    }
    
    # 询问是否安装到 site-packages
    echo ""
    read -p "Install PyCANDYAlgo to site-packages for global use? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SITE_PACKAGES=$(python3 -c "import site; print(site.USER_SITE)")
        mkdir -p "$SITE_PACKAGES"
        cp "$SO_FILE" "$SITE_PACKAGES/"
        echo "✅ Installed to $SITE_PACKAGES"
        echo ""
        echo "Testing global import..."
        cd /tmp
        python3 -c "import PyCANDYAlgo; print('✅ Global import successful')" || echo "⚠ Global import failed"
        cd - > /dev/null
    fi
else
    echo "⚠ PyCANDYAlgo.so not found in current directory"
    echo "Check build/ directory:"
    find build -name "PyCANDYAlgo*.so" 2>/dev/null || echo "No .so file found"
    exit 1
fi

echo ""
echo "To use PyCANDYAlgo:"
echo "  python3 -c 'import PyCANDYAlgo'"
echo ""
