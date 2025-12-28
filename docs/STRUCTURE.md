# SAGE-DB-Bench 目录结构说明

## 目录结构

```
benchmark_db/
├── algorithms_impl/           # 第三方算法实现（作为依赖）
│   ├── diskann-ms/            # Microsoft DiskANN 改版
│   ├── ipdiskann/             # IntelliStream IP-DiskANN (submodule)
│   ├── faiss/                 # Facebook FAISS
│   ├── vsag/                  # Vector Search Algorithm Gateway (submodule)
│   ├── gti/                   # Graph-based Tree Index (submodule)
│   ├── plsh/                  # PLSH (submodule)
│   └── ...
├── bench/                     # SAGE 的 benchmark 代码
├── datasets/                  # 数据集管理
├── runbooks/                  # 实验配置
├── tests/                     # SAGE-DB-Bench 自己的测试
├── docs/                      # 文档（README、INSTALL、STRUCTURE 等）
├── scripts/                   # 部署/安装/激活/本地测试脚本
├── docker/                    # Dockerfile 与 docker-compose
├── config/                    # pytest.ini、pre-commit、setup.cfg
└── pyproject.toml / requirements.txt
```

## ⚠️ 重要说明

### algorithms_impl/ 目录

- **用途**: 存放第三方 ANNS 算法库的实现代码
- **测试**: 这些目录下的测试文件**不会被 pytest 自动运行**
- **原因**: 
  - 这些测试依赖外部库（如 `diskannpy`, `faiss.contrib`）
  - 需要特殊的编译环境和依赖
  - 不是 SAGE-DB-Bench 的核心测试

### DiskANN 重命名说明

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `DiskANN/` (根目录) | **已移除** | Microsoft 官方版本，已移除避免混淆 |
| `algorithms_impl/DiskANN/` | `algorithms_impl/diskann-ms/` | Microsoft 改版（非 submodule） |
| `algorithms_impl/ipdiskann/` | 保持不变 | IntelliStream IP-DiskANN (submodule) |

### 如何运行算法测试

如果需要运行第三方算法的测试，需要：

1. **安装依赖**:
   ```bash
   # DiskANN
   pip install diskannpy
   
   # Faiss (with contrib)
   pip install faiss-gpu  # 或 faiss-cpu
   ```

2. **显式运行测试**:
   ```bash
   # 运行特定算法的测试
   pytest algorithms_impl/diskann-ms/ -v
   pytest algorithms_impl/faiss/ -v
   ```

3. **或者使用 pytest marker**:
   ```bash
   # 运行被标记为 third_party 的测试
   pytest -m third_party
   ```

### pytest 行为

- **默认行为**: `pytest` 只运行 `tests/` 目录下的测试
- **自动跳过**: `algorithms_impl/` 下的所有测试会被自动标记为 `skip`
- **配置文件位置**: pytest 配置位于 [config/pytest.ini](../config/pytest.ini)，`conftest.py` 位于 tests/ 自动发现；推荐命令：`pytest -c config/pytest.ini`

## 开发建议

1. **添加新算法**: 放在 `algorithms_impl/` 下
2. **Benchmark 代码**: 放在 `bench/` 下
3. **单元测试**: 放在 `tests/` 下
4. **集成测试**: 也放在 `tests/` 下，使用 `@pytest.mark.integration` 标记

## 重构记录

- **2024-12-28**: 
  - 移除根目录的 `DiskANN/` submodule
  - 重命名 `algorithms_impl/DiskANN/` → `algorithms_impl/diskann-ms/`
  - 添加 `conftest.py` 自动跳过第三方测试
  - 添加 `pytest.ini` 配置文件
  - 更新目录结构文档
