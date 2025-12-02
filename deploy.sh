#!/bin/bash
# ============================================================================
# SAGE-DB-Bench 部署脚本 (仅 PyCANDYAlgo)
# ============================================================================
#
# 本脚本会：
# 1. 检查并安装系统依赖
# 2. 创建 Python 虚拟环境
# 3. 安装 Python 依赖包
# 4. 初始化 Git submodules
# 5. 构建 PyCANDYAlgo
# 6. 安装 PyCANDYAlgo 到虚拟环境
# 7. 验证 PyCANDYAlgo 导入
#
# 使用方法:
#   ./deploy.sh
#
# 选项:
#   --skip-system-deps    跳过系统依赖安装
#   --skip-build          跳过构建（仅设置环境）
#   --help                显示帮助
#
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 颜色定义
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================
print_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}$1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

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

print_step() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')]${NC} $1"
}

# ============================================================================
# 解析命令行参数
# ============================================================================
SKIP_SYSTEM_DEPS=false
SKIP_BUILD=false
PYTHON_CMD="python3.10"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            head -n 22 "$0" | tail -n +2 | sed 's/^# //'
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
# 检查 Python 3.10
# ============================================================================
if ! command -v python3.10 &> /dev/null; then
    print_error "Python 3.10 未找到"
    print_info "安装方法: sudo apt-get install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

print_info "Python 版本: $(python3.10 --version)"

# ============================================================================
# 开始部署
# ============================================================================
print_banner "SAGE-DB-Bench 部署 (PyCANDYAlgo)"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "项目目录: $SCRIPT_DIR"
echo ""

# ============================================================================
# 步骤 1: 系统依赖
# ============================================================================
if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    print_header "步骤 1/7: 安装系统依赖"
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        print_info "操作系统: $PRETTY_NAME"
    fi
    
    if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
        print_step "安装构建依赖..."
        sudo apt-get update -qq || true
        sudo apt-get install -y \
            build-essential cmake git pkg-config \
            libgflags-dev libgoogle-glog-dev libfmt-dev \
            libboost-all-dev libomp-dev libnuma-dev libaio-dev \
            python3.10 python3.10-venv python3.10-dev python3-pip \
            || print_warning "部分包可能未安装"
        
        # 安装 Intel MKL
        print_step "安装 Intel MKL..."
        if [ ! -d "/opt/intel/oneapi/mkl" ]; then
            wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB 2>/dev/null | sudo apt-key add - 2>/dev/null || true
            echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
            sudo apt-get update -qq || true
            sudo apt-get install -y intel-oneapi-mkl-devel || print_warning "Intel MKL 安装失败"
        else
            print_info "Intel MKL 已安装"
        fi
        
        print_success "系统依赖安装完成"
    else
        print_warning "非 Ubuntu/Debian 系统，请手动安装依赖"
    fi
else
    print_header "步骤 1/7: 跳过系统依赖安装"
fi

# ============================================================================
# 步骤 2: 创建虚拟环境
# ============================================================================
print_header "步骤 2/7: 创建 Python 虚拟环境"

VENV_DIR="$SCRIPT_DIR/sage-db-bench"

if [ ! -d "$VENV_DIR" ]; then
    print_step "创建虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "虚拟环境创建完成"
else
    print_info "虚拟环境已存在"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
print_success "虚拟环境已激活: $VIRTUAL_ENV"

# 升级 pip
pip install --upgrade pip setuptools wheel -q

# ============================================================================
# 步骤 3: 安装 Python 依赖
# ============================================================================
print_header "步骤 3/7: 安装 Python 依赖"

print_step "安装 PyTorch (CPU 版本)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
print_success "PyTorch 安装完成"

print_step "安装其他依赖..."
pip install numpy pybind11 PyYAML pandas -q
print_success "Python 依赖安装完成"

# ============================================================================
# 步骤 4: 初始化 Git Submodules
# ============================================================================
print_header "步骤 4/7: 初始化 Git Submodules"

if [ -f ".gitmodules" ]; then
    print_step "初始化 submodules..."
    git submodule update --init --recursive
    print_success "Submodules 初始化完成"
else
    print_warning ".gitmodules 不存在"
fi

# ============================================================================
# 步骤 5: 构建 PyCANDYAlgo
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    print_header "步骤 5/7: 构建 PyCANDYAlgo"
    
    cd "$SCRIPT_DIR/algorithms_impl"
    
    # 设置 MKL 环境
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    fi
    if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
        export MKLROOT="/opt/intel/oneapi/mkl/latest"
        export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
        export CPATH="$MKLROOT/include:$CPATH"
    fi
    
    # 运行构建脚本
    if [ -f "build.sh" ]; then
        print_step "运行 build.sh..."
        if bash build.sh; then
            print_success "PyCANDYAlgo 构建完成"
        else
            print_error "PyCANDYAlgo 构建失败"
            exit 1
        fi
    else
        print_error "build.sh 不存在"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
else
    print_header "步骤 5/7: 跳过构建"
fi

# ============================================================================
# 步骤 6: 安装 PyCANDYAlgo
# ============================================================================
print_header "步骤 6/7: 安装 PyCANDYAlgo"

cd "$SCRIPT_DIR/algorithms_impl"

# 设置库路径
if [ -d "/opt/intel/oneapi/mkl/latest/lib/intel64" ]; then
    export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH"
fi
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)
if [ -n "$SO_FILE" ]; then
    print_info "找到: $SO_FILE"
    
    if [ -f "setup.py" ]; then
        pip install -e . --no-build-isolation -q
        print_success "PyCANDYAlgo 已安装到虚拟环境"
    else
        SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        cp "$SO_FILE" "$SITE_PACKAGES/"
        print_success "PyCANDYAlgo 已复制到 $SITE_PACKAGES"
    fi
else
    print_error "PyCANDYAlgo.so 未找到"
    exit 1
fi

cd "$SCRIPT_DIR"

# ============================================================================
# 步骤 7: 验证 PyCANDYAlgo 导入
# ============================================================================
print_header "步骤 7/7: 验证 PyCANDYAlgo"

print_step "测试导入..."

# 测试 PyCANDYAlgo 导入
if python3 -c "import PyCANDYAlgo; print('VERSION:', PyCANDYAlgo.__version__)" 2>&1; then
    print_success "PyCANDYAlgo 导入成功"
else
    print_error "PyCANDYAlgo 导入失败"
    # 显示详细错误
    python3 -c "import PyCANDYAlgo" 2>&1 || true
    exit 1
fi

# 测试核心依赖
print_step "测试核心依赖..."
python3 -c "import numpy; print('numpy:', numpy.__version__)" || print_warning "numpy 不可用"
python3 -c "import torch; print('torch:', torch.__version__)" || print_warning "torch 不可用"

# ============================================================================
# 完成
# ============================================================================
print_banner "部署完成！"

echo "✅ PyCANDYAlgo 已成功构建并安装"
echo ""
echo "使用方法:"
echo "  source sage-db-bench/bin/activate"
echo "  python3 -c 'import PyCANDYAlgo; print(PyCANDYAlgo.__version__)'"
echo ""
print_info "部署用时: $SECONDS 秒"
echo ""
