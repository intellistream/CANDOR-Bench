#!/bin/bash
# ============================================================================
# SAGE-DB-Bench 一键部署脚本
# ============================================================================
#
# 本脚本会：
# 1. 检查并安装系统依赖
# 2. 创建 Python 虚拟环境
# 3. 安装 Python 依赖包
# 4. 初始化 Git submodules
# 5. 构建所有算法（PyCANDY, 第三方库, VSAG）
# 6. 安装算法 Python 包到虚拟环境
# 7. 验证安装
#
# 使用方法:
#   ./deploy.sh
#
# 选项:
#   --skip-system-deps    跳过系统依赖安装（如果已安装）
#   --skip-build          跳过算法构建（仅设置环境）
#   --python-version VER  指定 Python 版本（默认：python3）
#   --help               显示帮助
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
        --python-version)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --help)
            head -n 25 "$0" | tail -n +2 | sed 's/^# //'
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
    print_info "SAGE-DB-Bench 需要 Python 3.10"
    print_info "安装方法:"
    echo "  Ubuntu/Debian:"
    echo "    sudo apt-get install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

PYTHON_VERSION=$(python3.10 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    print_error "需要 Python 3.10，当前版本: $PYTHON_VERSION"
    exit 1
fi

print_info "Python 版本: $(python3.10 --version)"

# ============================================================================
# 开始部署
# ============================================================================
print_banner "SAGE-DB-Bench 一键部署"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "项目目录: $SCRIPT_DIR"
print_info "开始时间: $(date)"
echo ""

# ============================================================================
# 步骤 1: 系统依赖检查和安装
# ============================================================================
if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    print_header "步骤 1/7: 安装系统依赖"
    
    # 检测操作系统
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        print_info "检测到操作系统: $PRETTY_NAME"
    else
        print_error "无法检测操作系统"
        exit 1
    fi
    
    case $OS in
        ubuntu|debian)
            print_step "更新软件包列表..."
            print_info "如果卡住超过30秒，请按 Ctrl+C 然后用 --skip-system-deps 重新运行"
            sudo timeout 60 apt-get update || print_warning "更新超时或失败，继续安装"
            echo ""
            
            print_step "安装构建工具和依赖..."
            print_info "正在安装以下包:"
            echo "  - build-essential, cmake, git"
            echo "  - libgflags-dev, libboost-all-dev, libomp-dev"
            echo "  - libgoogle-glog-dev, libfmt-dev, libnuma-dev"
            echo "  - python3.10, python3.10-venv, python3.10-dev"
            echo "  - wget, curl, linux-tools"
            echo "  - Intel MKL (for Puck)"
            echo ""
            
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                pkg-config \
                libgflags-dev \
                libgoogle-glog-dev \
                libfmt-dev \
                libboost-all-dev \
                libomp-dev \
                libnuma-dev \
                python3.10 \
                python3.10-venv \
                python3.10-dev \
                python3-pip \
                wget \
                curl \
                linux-tools-common \
                linux-tools-generic \
                linux-tools-$(uname -r) || print_warning "部分 perf 工具可能未安装"
            
            # 安装 Intel MKL (Puck 需要)
            print_step "安装 Intel MKL..."
            wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add - 2>/dev/null || print_warning "添加 Intel GPG key 失败"
            echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
            sudo apt-get update -qq || print_warning "更新 Intel 源失败"
            sudo apt-get install -y intel-oneapi-mkl-devel || print_warning "Intel MKL 安装失败，Puck 可能无法构建"
            
            # 设置 MKL 环境变量
            if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
                print_step "设置 MKL 环境变量..."
                source /opt/intel/oneapi/setvars.sh --force
                print_success "MKL 环境已配置"
            fi
            
            echo ""
            print_success "系统依赖安装完成"
            ;;
            
        *)
            print_warning "未知操作系统: $OS"
            print_info "请手动安装以下依赖:"
            echo "  - build-essential, cmake, git"
            echo "  - libgflags-dev, libboost-all-dev, libomp-dev"
            echo "  - python3.10, python3.10-venv, python3.10-dev, python3-pip"
            echo "  - linux-tools (for perf)"
            read -p "是否继续? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
else
    print_header "步骤 1/7: 跳过系统依赖安装"
    print_warning "假设所有系统依赖已安装"
fi

# ============================================================================
# 步骤 2: 创建 Python 虚拟环境
# ============================================================================
print_header "步骤 2/7: 创建 Python 虚拟环境"

VENV_DIR="$SCRIPT_DIR/sage-db-bench"

if [ -d "$VENV_DIR" ]; then
    print_warning "虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "删除旧的虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        print_info "使用现有虚拟环境"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_step "创建虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "虚拟环境创建完成"
else
    print_info "虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
print_step "激活虚拟环境..."
source "$VENV_DIR/bin/activate"
print_success "虚拟环境已激活: $VIRTUAL_ENV"

# 设置 LD_LIBRARY_PATH 优先使用系统库（避免 torch 的 libgomp 冲突）
export LD_LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:$LD_LIBRARY_PATH"

# 升级 pip
print_step "升级 pip..."
pip install --upgrade pip setuptools wheel

# ============================================================================
# 步骤 3: 安装 Python 依赖
# ============================================================================
print_header "步骤 3/7: 安装 Python 依赖"

# 先安装 PyTorch CPU 版本（避免默认安装 CUDA 版本）
print_step "安装 PyTorch (CPU 版本)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
print_success "PyTorch CPU 版本安装完成"

if [ -f "requirements.txt" ]; then
    print_step "从 requirements.txt 安装其他依赖..."
    # 跳过 torch（已安装）
    grep -v "^torch" requirements.txt > /tmp/requirements_no_torch.txt || true
    pip install -r /tmp/requirements_no_torch.txt
    rm -f /tmp/requirements_no_torch.txt
    print_success "Python 依赖安装完成"
else
    print_warning "requirements.txt 不存在，安装核心依赖..."
    pip install numpy pybind11 PyYAML pandas scipy h5py matplotlib psutil
    print_success "核心 Python 依赖安装完成"
fi

# ============================================================================
# 步骤 4: 初始化 Git Submodules
# ============================================================================
print_header "步骤 4/7: 初始化 Git Submodules"

if [ -f ".gitmodules" ]; then
    print_step "初始化 submodules..."
    git submodule update --init --recursive
    print_success "Submodules 初始化完成"
    
    # 显示 submodule 状态
    print_info "Submodule 状态:"
    git submodule status | while read line; do
        echo "  $line"
    done
else
    print_warning ".gitmodules 文件不存在，跳过"
fi

# ============================================================================
# 步骤 5: 构建算法
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    print_header "步骤 5/7: 构建所有算法"
    
    cd "$SCRIPT_DIR/algorithms_impl"
    
    if [ -f "build_all.sh" ]; then
        print_step "运行 build_all.sh..."
        print_info "这可能需要 15-40 分钟，请耐心等待..."
        echo ""
        
        # 显示进度
        bash build_all.sh 2>&1 | while IFS= read -r line; do
            echo "$line"
        done
        
        if [ $? -eq 0 ]; then
            print_success "算法构建完成"
        else
            print_error "算法构建失败"
            exit 1
        fi
    else
        print_error "build_all.sh 不存在"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
else
    print_header "步骤 5/7: 跳过算法构建"
    print_warning "假设算法已构建完成"
fi

# ============================================================================
# 步骤 6: 安装算法 Python 包
# ============================================================================
print_header "步骤 6/7: 安装算法 Python 包"

cd "$SCRIPT_DIR/algorithms_impl"

# 安装 PyCANDY
print_step "安装 PyCANDYAlgo..."
SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)
if [ -n "$SO_FILE" ] && [ -f "setup.py" ]; then
    pip install -e . --no-build-isolation
    print_success "PyCANDYAlgo 已安装"
else
    print_warning "PyCANDYAlgo.so 未找到，跳过安装"
fi

# 安装 VSAG
print_step "安装 pyvsag..."
if [ -d "vsag/wheelhouse" ]; then
    WHEEL_FILE=$(ls vsag/wheelhouse/pyvsag*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL_FILE" ]; then
        pip install "$WHEEL_FILE" --force-reinstall
        print_success "pyvsag 已安装"
    else
        print_warning "pyvsag wheel 未找到，跳过安装"
    fi
else
    print_warning "vsag/wheelhouse 目录不存在，跳过安装"
fi

cd "$SCRIPT_DIR"

# ============================================================================
# 步骤 7: 验证安装
# ============================================================================
print_header "步骤 7/7: 验证安装"

print_step "测试 Python 包导入..."

# 测试 PyCANDYAlgo - 使用更详细的错误信息
echo "正在测试 PyCANDYAlgo 导入..."
PYCANDY_TEST=$(python3 << 'PYEOF'
import sys
try:
    import PyCANDYAlgo
    print("SUCCESS")
    sys.exit(0)
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"OTHER_ERROR: {e}")
    sys.exit(2)
PYEOF
)

if echo "$PYCANDY_TEST" | grep -q "SUCCESS"; then
    print_success "PyCANDYAlgo 导入成功"
else
    print_warning "PyCANDYAlgo 导入失败"
    echo "$PYCANDY_TEST" | head -3
    echo ""
    print_info "可能的解决方案："
    echo "  1. 检查 .so 文件是否存在: find algorithms_impl -name '*.so'"
    echo "  2. 手动测试: python3 -c 'import PyCANDYAlgo'"
    echo "  3. 查看详细错误: python3 -c 'import PyCANDYAlgo' 2>&1"
fi

# 测试 pyvsag
if python3 -c "import pyvsag" 2>/dev/null; then
    print_success "pyvsag 导入成功"
else
    print_warning "pyvsag 导入失败（可能未构建）"
fi

# 测试其他核心包
print_step "测试核心依赖..."
for pkg in numpy torch yaml pandas; do
    if python3 -c "import $pkg" 2>/dev/null; then
        print_success "$pkg 可用"
    else
        print_error "$pkg 不可用"
    fi
done

# 测试 perf 工具
print_step "检查 perf 工具..."
if command -v perf &> /dev/null; then
    print_success "perf 工具已安装"
    perf --version 2>/dev/null || print_warning "perf 可能需要 sudo 权限"
else
    print_warning "perf 工具未找到"
    print_info "如需性能分析，请安装: sudo apt-get install linux-tools-$(uname -r)"
fi

# ============================================================================
# 部署完成
# ============================================================================
print_banner "部署完成！"

cat << 'EOF'

┌─────────────────────────────────────────────────────────────┐
│                     🎉 部署成功！                           │
└─────────────────────────────────────────────────────────────┘

📦 虚拟环境位置:
EOF

echo "   $VENV_DIR"
echo ""
echo "🚀 使用方法:"
echo ""
echo "1. 激活虚拟环境:"
echo "   source sage-db-bench/bin/activate"
echo ""
echo "2. 运行 benchmark:"
echo "   python run_benchmark.py --dataset sift --algorithm vsag_hnsw"
echo ""
echo "3. 运行测试:"
echo "   pytest tests/ -v"
echo ""
echo "4. 退出虚拟环境:"
echo "   deactivate"
echo ""
echo "📝 配置文件:"
echo "   - runbooks/baseline.yaml       # 基准测试配置"
echo "   - runbooks/algo_optimizations/ # 算法优化配置"
echo ""
echo "📊 结果输出:"
echo "   - results/                     # 实验结果"
echo ""

# 创建快速激活脚本
print_step "创建快速启动脚本..."
cat > "$SCRIPT_DIR/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# 快速激活 SAGE-DB-Bench 虚拟环境

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/sage-db-bench/bin/activate"

echo "✓ SAGE-DB-Bench 虚拟环境已激活"
echo ""
echo "可用命令:"
echo "  python run_benchmark.py --help"
echo "  pytest tests/ -v"
echo ""
ACTIVATE_EOF

chmod +x "$SCRIPT_DIR/activate.sh"
print_success "快速启动脚本已创建: ./activate.sh"

echo ""
print_info "部署用时: $SECONDS 秒"
echo ""
print_success "现在可以开始使用 SAGE-DB-Bench！"
echo ""
