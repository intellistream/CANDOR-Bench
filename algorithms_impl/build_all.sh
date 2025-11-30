#!/bin/bash
# ============================================================================
# 统一构建脚本：构建所有算法并生成 Python 包
# ============================================================================
# 
# 本脚本用于构建 algorithms_impl 文件夹中的所有算法实现：
# 1. PyCANDY 算法 (通过 CMake + pybind11)
# 2. 第三方库：GTI, IP-DiskANN, PLSH (标准 CMake 构建)
# 3. VSAG (通过 Makefile + Python wheel)
#
# 使用方法:
#   ./build_all.sh [--skip-pycandy] [--skip-third-party] [--skip-vsag] [--install]
#
# 选项:
#   --skip-pycandy       跳过 PyCANDY 构建
#   --skip-third-party   跳过第三方库构建
#   --skip-vsag          跳过 VSAG 构建
#   --install            构建后自动安装 Python 包
#   --help              显示帮助信息
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 颜色定义
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# ============================================================================
# 解析命令行参数
# ============================================================================
BUILD_PYCANDY=true
BUILD_THIRD_PARTY=true
BUILD_VSAG=true
AUTO_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pycandy)
            BUILD_PYCANDY=false
            shift
            ;;
        --skip-third-party)
            BUILD_THIRD_PARTY=false
            shift
            ;;
        --skip-vsag)
            BUILD_VSAG=false
            shift
            ;;
        --install)
            AUTO_INSTALL=true
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# 环境检查
# ============================================================================
print_header "Environment Check"

# 获取脚本所在目录 (algorithms_impl/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "Working directory: $SCRIPT_DIR"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3."
    exit 1
fi
print_success "Python: $(python3 --version)"

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    print_error "cmake not found. Please install CMake."
    exit 1
fi
print_success "CMake: $(cmake --version | head -n1)"

# 检查 make
if ! command -v make &> /dev/null; then
    print_error "make not found. Please install make."
    exit 1
fi
print_success "Make: $(make --version | head -n1)"

# 计算编译并行数
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAX_JOBS=$((NPROC > 8 ? 8 : NPROC))
print_info "Using -j${MAX_JOBS} for parallel compilation"

# ============================================================================
# 1. 构建 PyCANDY 算法
# ============================================================================
if [ "$BUILD_PYCANDY" = true ]; then
    print_header "Building PyCANDY Algorithms"
    
    if [ -f "build.sh" ]; then
        print_info "Running build.sh..."
        if bash build.sh; then
            # 检查生成的 .so 文件
            SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)
            if [ -n "$SO_FILE" ]; then
                print_success "PyCANDY built: $SO_FILE"
                
                if [ "$AUTO_INSTALL" = true ]; then
                    print_info "Installing PyCANDYAlgo..."
                    pip install -e . --no-build-isolation
                    print_success "PyCANDYAlgo installed"
                fi
            else
                print_error "PyCANDYAlgo.so not found after build"
                exit 1
            fi
        else
            print_error "build.sh failed with exit code $?"
            exit 1
        fi
    else
        print_warning "build.sh not found, skipping PyCANDY"
    fi
else
    print_info "Skipping PyCANDY build (--skip-pycandy)"
fi

# ============================================================================
# 2. 构建第三方库 (GTI, IP-DiskANN, PLSH)
# ============================================================================
if [ "$BUILD_THIRD_PARTY" = true ]; then
    print_header "Building Third-Party Libraries"
    
    # === GTI ===
    if [ -d "gti" ]; then
        print_info "Building GTI..."
        
        # 构建 n2 依赖
        if [ -d "gti/GTI/extern_libraries/n2" ]; then
            print_info "  Building n2 dependency..."
            cd gti/GTI/extern_libraries/n2
            [ -d build ] && rm -rf build
            make shared_lib -j${MAX_JOBS}
            print_success "  n2 library built"
            cd "$SCRIPT_DIR"
        fi
        
        # 构建 GTI
        cd gti/GTI
        [ -d build ] && rm -rf build
        mkdir -p bin build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j${MAX_JOBS}
        make install || print_warning "GTI install failed (not critical)"
        cd "$SCRIPT_DIR"
        print_success "GTI built successfully"
    else
        print_warning "GTI not found (submodule may not be initialized)"
    fi
    
    # === IP-DiskANN ===
    if [ -d "ipdiskann" ]; then
        print_info "Building IP-DiskANN..."
        cd ipdiskann
        [ -d build ] && rm -rf build
        mkdir -p build
        cd build
        cmake ..
        make -j${MAX_JOBS}
        make install || print_warning "IP-DiskANN install failed (not critical)"
        cd "$SCRIPT_DIR"
        print_success "IP-DiskANN built successfully"
    else
        print_warning "IP-DiskANN not found (submodule may not be initialized)"
    fi
    
    # === PLSH ===
    if [ -d "plsh" ]; then
        print_info "Building PLSH..."
        cd plsh
        [ -d build ] && rm -rf build
        mkdir -p build
        cd build
        cmake ..
        make -j${MAX_JOBS}
        make install || print_warning "PLSH install failed (not critical)"
        cd "$SCRIPT_DIR"
        print_success "PLSH built successfully"
    else
        print_warning "PLSH not found (submodule may not be initialized)"
    fi
else
    print_info "Skipping third-party libraries (--skip-third-party)"
fi

# ============================================================================
# 3. 构建 VSAG
# ============================================================================
if [ "$BUILD_VSAG" = true ]; then
    print_header "Building VSAG"
    
    if [ -d "vsag" ]; then
        cd vsag
        
        # 检测 Python 版本
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        print_info "Detected Python version: $PYTHON_VERSION"
        
        print_info "Building VSAG Python wheel for Python $PYTHON_VERSION..."
        
        # 先构建 release 版本
        if [ ! -d "build-release" ] || [ ! -f "build-release/CMakeCache.txt" ]; then
            print_info "Building VSAG release version..."
            make release COMPILE_JOBS=${MAX_JOBS}
        else
            print_info "VSAG release build already exists, skipping..."
        fi
        
        # 构建 Python wheel
        print_info "Building Python wheel..."
        make pyvsag PY_VERSION=${PYTHON_VERSION} COMPILE_JOBS=${MAX_JOBS}
        
        # 检查生成的 wheel 文件
        WHEEL_FILE=$(ls wheelhouse/pyvsag*.whl 2>/dev/null | head -1)
        if [ -n "$WHEEL_FILE" ]; then
            print_success "VSAG wheel built: $WHEEL_FILE"
            
            if [ "$AUTO_INSTALL" = true ]; then
                print_info "Installing pyvsag..."
                pip install "$WHEEL_FILE" --force-reinstall
                print_success "pyvsag installed"
            fi
        else
            print_error "VSAG wheel not found after build"
            cd "$SCRIPT_DIR"
            exit 1
        fi
        
        cd "$SCRIPT_DIR"
    else
        print_warning "VSAG not found (submodule may not be initialized)"
    fi
else
    print_info "Skipping VSAG build (--skip-vsag)"
fi

# ============================================================================
# 构建完成总结
# ============================================================================
print_header "Build Summary"

echo "Built components:"
[ "$BUILD_PYCANDY" = true ] && echo "  ✓ PyCANDY algorithms"
[ "$BUILD_THIRD_PARTY" = true ] && echo "  ✓ Third-party libraries (GTI, IP-DiskANN, PLSH)"
[ "$BUILD_VSAG" = true ] && echo "  ✓ VSAG"

if [ "$AUTO_INSTALL" = true ]; then
    echo ""
    print_success "All packages installed automatically"
else
    echo ""
    print_info "To install the packages manually:"
    [ "$BUILD_PYCANDY" = true ] && echo "  cd $SCRIPT_DIR && pip install -e ."
    [ "$BUILD_VSAG" = true ] && echo "  pip install $SCRIPT_DIR/vsag/wheelhouse/pyvsag*.whl"
fi

echo ""
print_success "Build completed successfully!"
