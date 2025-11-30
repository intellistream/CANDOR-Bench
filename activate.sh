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
