# SAGE-DB-Bench

整理后的目录导航：

- 快速使用与完整说明：见 [docs/README.md](docs/README.md)
- 部署与安装指南：见 [docs/INSTALL.md](docs/INSTALL.md)
- 目录结构与测试约定：见 [docs/STRUCTURE.md](docs/STRUCTURE.md)
- 算法发布与 CI 备注：见 [docs/ALGORITHM_DEPLOYMENT.md](docs/ALGORITHM_DEPLOYMENT.md) 和 [docs/CICD_FIXES.md](docs/CICD_FIXES.md)
- 脚本：位于 [scripts/](scripts/)（如 deploy、install、activate、test_local）
- 容器配置：位于 [docker/](docker/)（Dockerfile、docker-compose.yml）
- 测试与工具配置：位于 [config/](config/)（pytest.ini、pre-commit、setup.cfg）；`tests/conftest.py` 自动生效

核心代码与数据：

- 基准框架与算法接口：bench/
- 数据集管理：datasets/
- ANNS 算法实现：已迁移至 `sage-libs/src/sage/libs/anns/implementations/`
- 实验配置：runbooks/
- 结果导出与工具：compute_gt.py、run_benchmark.py、export_results.py

**注意**: 所有 ANNS 算法的具体实现（包括 DiskANN, FAISS, VSAG, GTI, IP-DiskANN, PLSH 等）已迁移到 SAGE 主仓库的 `sage-libs` 包中。本仓库仅包含基准测试框架和算法接口。

运行 pytest 或 pre-commit 请参考 docs/STRUCTURE.md 中的最新路径与命令。

