# CI/CD 构建修复说明

## 问题总结

在 CI/CD 环境中构建时遇到了两个主要问题：

### 1. VSAG (pyvsag) - Intel MKL 依赖问题

**错误信息:**
```
ImportError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory
```

**原因:**
- VSAG 依赖 Intel MKL 库进行数学计算
- CI 环境中虽然安装了 MKL，但运行时链接器找不到库文件

**解决方案:**
1. 在 `deploy.sh` 步骤 1 中确保 Intel MKL 被正确安装
2. 在 `deploy.sh` 步骤 2 创建虚拟环境后，修改 `activate` 脚本：
   - 在虚拟环境的 `activate` 脚本中添加 MKL 库路径配置
   - 确保每次激活虚拟环境时自动设置 `LD_LIBRARY_PATH`
   - 这样无论是否退出虚拟环境，再次进入都能找到 MKL 库
3. 在 `deploy.sh` 步骤 6 构建 VSAG 之前，设置 MKL 环境变量：
   - 加载 Intel oneAPI 环境（如果可用）
   - 设置 `LD_LIBRARY_PATH` 包含 MKL 库路径
   - 设置 `LIBRARY_PATH` 和 `CPATH` 用于编译
   - 在 CMake 配置中传递 `MKLROOT` 参数
4. 在 `deploy.sh` 步骤 9 测试模块导入前，再次确认 MKL 环境变量

### 2. GTI - tcmalloc 依赖问题

**错误信息:**
```
/usr/bin/ld: cannot find -ltcmalloc_minimal: No such file or directory
```

**原因:**
- GTI 主可执行文件链接了 `tcmalloc_minimal` 库（Google Performance Tools）
- CI 环境中可能未安装 gperftools 包
- 对于 Python bindings (gti_wrapper)，不需要 tcmalloc

**解决方案:**
1. 在 `deploy.sh` 步骤 1 中添加 `libgoogle-perftools-dev` 包的安装
2. 修改 `algorithms_impl/gti/GTI/src/CMakeLists.txt`：
   - 使 tcmalloc 成为可选依赖
   - 使用 `find_library()` 查找 tcmalloc
   - 如果找不到，发出警告但继续构建
3. 在构建脚本中只构建 Python bindings：
   - `deploy.sh`: 使用 `make gti_wrapper` 代替 `make`
   - `build_all.sh`: 同样只构建 `gti_wrapper` 目标

## 修改的文件

### 1. `/home/mingqi/SAGE-DB-Bench/deploy.sh`

**修改内容:**

#### a) 步骤 1: 添加 gperftools 包
```bash
sudo apt-get install -y \
    ... \
    libgoogle-perftools-dev \  # 新增：提供 tcmalloc
    ...
```

#### b) 步骤 2: 配置虚拟环境，持久化 MKL 路径
```bash
# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 配置虚拟环境的 activate 脚本，自动设置 MKL 路径
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if ! grep -q "# MKL Library Path" "$ACTIVATE_SCRIPT"; then
    cat >> "$ACTIVATE_SCRIPT" << 'EOF'

# MKL Library Path (added by deploy.sh)
if [ -d "/opt/intel/oneapi/mkl/latest/lib/intel64" ]; then
    export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH"
elif [ -d "/opt/intel/mkl/lib/intel64" ]; then
    export LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH"
fi
EOF
fi

# 重新加载 activate 脚本以应用 MKL 路径
source "$ACTIVATE_SCRIPT"
```

**关键改进**: 将 MKL 路径写入虚拟环境的 `activate` 脚本，确保每次激活虚拟环境时都能找到 MKL 库，解决了退出虚拟环境再进入后无法导入 pyvsag 的问题。

#### c) 步骤 6: 配置 MKL 环境
```bash
# 设置 MKL 环境变量（VSAG 依赖 Intel MKL）
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH"
elif [ -d "/opt/intel/oneapi/mkl/latest" ]; then
    export MKLROOT="/opt/intel/oneapi/mkl/latest"
    export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$MKLROOT/lib/intel64:$LIBRARY_PATH"
    export CPATH="$MKLROOT/include:$CPATH"
fi
```

#### d) 步骤 6: VSAG CMake 配置添加 MKL 参数
```bash
CMAKE_ARGS=(
    ...
    -DPython3_EXECUTABLE=$(which python3)
)

if [ -n "$MKLROOT" ]; then
    CMAKE_ARGS+=(
        -DMKLROOT="$MKLROOT"
        -DCMAKE_PREFIX_PATH="$MKLROOT"
    )
fi

cmake "${CMAKE_ARGS[@]}" ...
```

#### e) 步骤 7: GTI 只构建 Python bindings
```bash
# 只构建 gti_wrapper（Python bindings），不构建主可执行文件（需要 tcmalloc）
make gti_wrapper -j${JOBS} 2>&1 | tail -10
```

#### f) 步骤 9: 测试前确保 MKL 环境已配置
```bash
# 确保 MKL 环境变量已配置（用于 VSAG）
if [ -d "/opt/intel/oneapi/mkl/latest/lib/intel64" ]; then
    export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH"
elif [ -d "/opt/intel/mkl/lib/intel64" ]; then
    export LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH"
fi
```

### 2. `/home/mingqi/SAGE-DB-Bench/algorithms_impl/gti/GTI/src/CMakeLists.txt`

**修改内容:**
```cmake
# 尝试查找 tcmalloc（主可执行文件的可选依赖）
find_library(TCMALLOC_LIB NAMES tcmalloc_minimal tcmalloc)

target_link_libraries(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/extern_libraries/n2/build/lib/libn2.so
    fmt::fmt
    OpenMP::OpenMP_CXX
)

# 如果找到 tcmalloc，则链接
if(TCMALLOC_LIB)
    target_link_libraries(${PROJECT_NAME} ${TCMALLOC_LIB})
    message(STATUS "Found tcmalloc: ${TCMALLOC_LIB}")
else()
    message(WARNING "tcmalloc not found - GTI executable will use system malloc")
endif()
```

### 3. `/home/mingqi/SAGE-DB-Bench/algorithms_impl/build_all.sh`

**修改内容:**
```bash
# 构建 GTI Python bindings (只构建 gti_wrapper，不构建主可执行文件)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3) \
      $PYBIND11_ARG ..

# 只构建 Python bindings
make gti_wrapper -j${MAX_JOBS}

# 尝试构建主可执行文件（可选）
make GTI -j${MAX_JOBS} 2>/dev/null || print_warning "GTI executable build skipped"
```

## 验证

修复后，CI/CD 构建应该能够：

1. ✅ 成功编译 VSAG (pyvsag)
2. ✅ 成功导入 `import pyvsag`（不再出现 MKL 库缺失错误）
3. ✅ 成功编译 GTI (gti_wrapper)
4. ✅ 成功导入 `import gti_wrapper`（不再出现链接器错误）

## 测试命令

在修复后，可以使用以下命令验证：

```bash
# 激活虚拟环境
source sage-db-bench/bin/activate

# 测试 pyvsag
python3 -c "import pyvsag; print('pyvsag version:', pyvsag.__version__)"

# 测试 gti_wrapper
python3 -c "import gti_wrapper; print('gti_wrapper imported successfully')"
```

## 注意事项

1. **MKL 安装**: 如果 CI 环境中 MKL 安装失败，VSAG 将无法运行。可以考虑使用 OpenBLAS 作为替代方案。

2. **tcmalloc 可选**: GTI Python bindings (gti_wrapper) 不需要 tcmalloc，只有主可执行文件需要。如果不需要运行 GTI 命令行工具，可以跳过 tcmalloc 安装。

3. **旧 ABI**: GTI 和 n2 库使用 `-D_GLIBCXX_USE_CXX11_ABI=0` 编译，确保二进制兼容性。

4. **增量构建**: 修改后的脚本支持增量构建，如果 CMake 缓存存在，将跳过重新配置。
