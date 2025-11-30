#!/bin/bash
# ============================================================================
# 自动化测试脚本：在隔离环境中测试算法构建
# ============================================================================
# 
# 本脚本会：
# 1. 测试 PyCANDY 构建
# 2. 测试第三方库构建
# 3. 测试 VSAG 构建
# 4. 验证 Python 导入
# 5. 生成测试报告
#
# 使用方法:
#   cd tests && ./test_build_algorithms.sh
#   或者: ./tests/test_build_algorithms.sh
#
# ============================================================================

set -e  # 遇到错误继续执行，收集所有错误

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
# 测试结果跟踪
# ============================================================================
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

record_pass() {
    ((TESTS_PASSED++))
    print_success "$1"
}

record_fail() {
    ((TESTS_FAILED++))
    FAILED_TESTS+=("$1")
    print_error "$1"
}

# ============================================================================
# 开始测试
# ============================================================================
print_header "SAGE-DB-Bench Algorithm Build Test"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ALGO_DIR="$SCRIPT_DIR/../algorithms_impl"

# 检查 algorithms_impl 目录是否存在
if [ ! -d "$ALGO_DIR" ]; then
    print_error "algorithms_impl directory not found at: $ALGO_DIR"
    exit 1
fi

cd "$ALGO_DIR"

print_info "Test script: $SCRIPT_DIR"
print_info "Working directory: $ALGO_DIR"
print_info "Start time: $(date)"
echo ""

# ============================================================================
# 环境检查
# ============================================================================
print_header "Environment Check"

# 检查 Python
if command -v python3 &> /dev/null; then
    record_pass "Python: $(python3 --version)"
else
    record_fail "Python not found"
fi

# 检查 CMake
if command -v cmake &> /dev/null; then
    record_pass "CMake: $(cmake --version | head -n1)"
else
    record_fail "CMake not found"
fi

# 检查 make
if command -v make &> /dev/null; then
    record_pass "Make: $(make --version | head -n1)"
else
    record_fail "Make not found"
fi

# 检查 GCC
if command -v gcc &> /dev/null; then
    record_pass "GCC: $(gcc --version | head -n1)"
else
    record_fail "GCC not found"
fi

# 检查 Python 包
echo ""
print_info "Checking Python packages..."
for pkg in torch numpy pybind11; do
    if python3 -c "import $pkg" 2>/dev/null; then
        record_pass "Python package: $pkg"
    else
        record_fail "Python package missing: $pkg"
    fi
done

# ============================================================================
# 测试 1: PyCANDY 构建
# ============================================================================
print_header "Test 1: Building PyCANDY"

if [ -f "build.sh" ]; then
    print_info "Running build.sh..."
    
    # 创建日志文件
    BUILD_LOG="test_pycandy_build.log"
    
    if bash build.sh > "$BUILD_LOG" 2>&1; then
        record_pass "PyCANDY build script executed successfully"
        
        # 检查 .so 文件
        SO_FILE=$(ls PyCANDYAlgo*.so 2>/dev/null | head -1)
        if [ -n "$SO_FILE" ]; then
            record_pass "PyCANDY .so file generated: $SO_FILE"
            
            # 测试导入
            if python3 -c "import sys; sys.path.insert(0, '.'); import PyCANDYAlgo" 2>/dev/null; then
                record_pass "PyCANDYAlgo import test passed"
            else
                record_fail "PyCANDYAlgo import test failed"
            fi
        else
            record_fail "PyCANDY .so file not found"
        fi
    else
        record_fail "PyCANDY build failed (see $BUILD_LOG)"
        tail -n 50 "$BUILD_LOG"
    fi
else
    record_fail "build.sh not found"
fi

# ============================================================================
# 测试 2: 第三方库构建（快速测试）
# ============================================================================
print_header "Test 2: Building Third-Party Libraries (Sample)"

test_third_party_lib() {
    local lib_name=$1
    local lib_path=$2
    
    if [ -d "$lib_path" ]; then
        print_info "Testing $lib_name build..."
        
        cd "$lib_path"
        
        # 清理
        [ -d build ] && rm -rf build
        mkdir -p build
        cd build
        
        # 测试 CMake 配置
        if cmake .. > /dev/null 2>&1; then
            record_pass "$lib_name: CMake configuration successful"
            
            # 不实际编译，仅测试配置
            # 如果需要完整测试，取消下面的注释
            # if make -j2 > /dev/null 2>&1; then
            #     record_pass "$lib_name: Build successful"
            # else
            #     record_fail "$lib_name: Build failed"
            # fi
        else
            record_fail "$lib_name: CMake configuration failed"
        fi
        
        cd "$ALGO_DIR"
    else
        print_warning "$lib_name not found (skipped)"
    fi
}

# 测试 GTI（如果存在）
if [ -d "gti/GTI" ]; then
    test_third_party_lib "GTI" "gti/GTI"
fi

# 测试 IP-DiskANN（如果存在）
if [ -d "ipdiskann" ]; then
    test_third_party_lib "IP-DiskANN" "ipdiskann"
fi

# 测试 PLSH（如果存在）
if [ -d "plsh" ]; then
    test_third_party_lib "PLSH" "plsh"
fi

# ============================================================================
# 测试 3: VSAG 构建（快速测试）
# ============================================================================
print_header "Test 3: Testing VSAG Build Configuration"

if [ -d "vsag" ]; then
    cd vsag
    
    # 检查 Makefile
    if [ -f "Makefile" ]; then
        record_pass "VSAG: Makefile found"
        
        # 检查构建脚本
        if [ -f "scripts/python/local_build_wheel.sh" ]; then
            record_pass "VSAG: Python build script found"
        else
            record_fail "VSAG: Python build script not found"
        fi
        
        # 测试 CMake 配置（不实际构建）
        if [ ! -d "build-release" ]; then
            print_info "Testing VSAG CMake configuration..."
            if cmake -B build-test -S . -DCMAKE_BUILD_TYPE=Release \
                -DENABLE_TESTS=OFF -DENABLE_PYBINDS=ON > /dev/null 2>&1; then
                record_pass "VSAG: CMake configuration successful"
                rm -rf build-test
            else
                record_fail "VSAG: CMake configuration failed"
            fi
        else
            record_pass "VSAG: Already configured (build-release exists)"
        fi
        
        # 检查已有的 wheel
        if [ -d "wheelhouse" ] && [ -n "$(ls wheelhouse/*.whl 2>/dev/null)" ]; then
            WHEEL_FILE=$(ls wheelhouse/*.whl | head -1)
            record_pass "VSAG: Wheel file found: $(basename $WHEEL_FILE)"
        else
            print_warning "VSAG: No wheel file found (not yet built)"
        fi
    else
        record_fail "VSAG: Makefile not found"
    fi
    
    cd "$ALGO_DIR"
else
    print_warning "VSAG not found (submodule not initialized)"
fi

# ============================================================================
# 测试 4: 完整构建测试（可选）
# ============================================================================
print_header "Test 4: Full Build Test (Optional)"

read -p "Do you want to run a full build test? This may take 15-40 minutes. (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting full build test..."
    
    FULL_BUILD_LOG="test_full_build.log"
    
    if ./build_all.sh > "$FULL_BUILD_LOG" 2>&1; then
        record_pass "Full build completed successfully"
        
        # 测试安装
        if ./install_packages.sh > /dev/null 2>&1; then
            record_pass "Package installation successful"
            
            # 最终导入测试
            print_info "Testing final imports..."
            if python3 -c "import PyCANDYAlgo" 2>/dev/null; then
                record_pass "Final test: PyCANDYAlgo import OK"
            else
                record_fail "Final test: PyCANDYAlgo import failed"
            fi
            
            if python3 -c "import pyvsag" 2>/dev/null; then
                record_pass "Final test: pyvsag import OK"
            else
                record_fail "Final test: pyvsag import failed"
            fi
        else
            record_fail "Package installation failed"
        fi
    else
        record_fail "Full build failed (see $FULL_BUILD_LOG)"
        tail -n 100 "$FULL_BUILD_LOG"
    fi
else
    print_info "Skipping full build test"
fi

# ============================================================================
# 生成测试报告
# ============================================================================
print_header "Test Summary"

echo "Total tests run: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# 保存报告到 tests 目录
cd "$SCRIPT_DIR"
REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "SAGE-DB-Bench Algorithm Build Test Report"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Directory: $SCRIPT_DIR"
    echo ""
    echo "Results:"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    echo ""
    if [ $TESTS_FAILED -gt 0 ]; then
        echo "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
    fi
} > "$REPORT_FILE"

print_info "Test report saved to: $REPORT_FILE"
echo ""

# 返回状态
if [ $TESTS_FAILED -eq 0 ]; then
    print_success "All tests passed! ✨"
    exit 0
else
    print_error "Some tests failed. Please check the logs."
    exit 1
fi
