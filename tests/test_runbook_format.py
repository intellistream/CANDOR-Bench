#!/usr/bin/env python3
"""
测试 runbook 格式解析

验证 runner.py 和 run_benchmark.py 是否正确处理实际的 runbook 格式
"""

import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_benchmark import load_runbook


def test_runbook_format():
    """测试 runbook 格式解析"""
    
    runbook_dir = Path(__file__).resolve().parent.parent / 'runbooks'
    
    test_cases = [
        'simple.yaml',
        'baseline.yaml',
        'random_drop/randomDrop0.05.yaml',
    ]
    
    print("=" * 80)
    print("测试 Runbook 格式解析")
    print("=" * 80)
    
    for runbook_name in test_cases:
        runbook_path = runbook_dir / runbook_name
        
        if not runbook_path.exists():
            print(f"\n⚠️  跳过不存在的文件: {runbook_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"测试: {runbook_name}")
        print(f"{'='*80}")
        
        try:
            # 加载 runbook
            runbook, dataset_name = load_runbook(runbook_path)
            
            print(f"\n✓ 成功加载 runbook")
            print(f"  - 数据集: {dataset_name}")
            
            # 检查数据集配置
            if dataset_name in runbook:
                dataset_config = runbook[dataset_name]
                
                # 统计操作数
                operations = []
                i = 1
                while i in dataset_config:
                    operations.append(dataset_config[i])
                    i += 1
                
                print(f"  - 操作数: {len(operations)}")
                
                # 显示操作列表
                print(f"\n操作列表:")
                for idx, op in enumerate(operations, 1):
                    op_type = op.get('operation', 'unknown')
                    print(f"  {idx}. {op_type}")
                    
                    # 显示重要参数
                    if op_type == 'initial':
                        print(f"     - start: {op.get('start')}, end: {op.get('end')}")
                    elif op_type == 'batch_insert':
                        print(f"     - start: {op.get('start')}, end: {op.get('end')}")
                        print(f"     - batchSize: {op.get('batchSize')}, eventRate: {op.get('eventRate')}")
                        if 'continuousQueryInterval' in op:
                            print(f"     - continuousQueryInterval: {op.get('continuousQueryInterval')}")
                    elif op_type == 'enableScenario':
                        if 'randomDrop' in op:
                            print(f"     - randomDrop: {op.get('randomDrop')}, randomDropProb: {op.get('randomDropProb')}")
                        if 'randomContamination' in op:
                            print(f"     - randomContamination: {op.get('randomContamination')}")
                
                # 显示其他配置
                if 'max_pts' in dataset_config:
                    print(f"\n其他配置:")
                    print(f"  - max_pts: {dataset_config['max_pts']}")
                if 'gt_url' in dataset_config:
                    print(f"  - gt_url: {dataset_config['gt_url']}")
            
            print(f"\n✓ {runbook_name} 解析成功")
            
        except Exception as e:
            print(f"\n✗ {runbook_name} 解析失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}\n")


def test_runbook_structure():
    """测试实际 runbook 文件的结构"""
    
    runbook_dir = Path(__file__).resolve().parent.parent / 'runbooks'
    
    print("\n" + "=" * 80)
    print("检查 Runbook 文件结构")
    print("=" * 80)
    
    # 检查 simple.yaml
    simple_path = runbook_dir / 'simple.yaml'
    if simple_path.exists():
        print(f"\n文件: simple.yaml")
        with open(simple_path, 'r') as f:
            content = yaml.safe_load(f)
        
        print(f"顶层键: {list(content.keys())}")
        
        for key, value in content.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                sub_keys = list(value.keys())
                print(f"  - 子键: {sub_keys}")
                
                # 检查是否有数字键（操作序列）
                int_keys = [k for k in sub_keys if isinstance(k, int)]
                if int_keys:
                    print(f"  - 操作序号: {sorted(int_keys)}")
                
                # 显示第一个操作
                if 1 in value:
                    print(f"  - 第一个操作: {value[1]}")


if __name__ == '__main__':
    test_runbook_structure()
    test_runbook_format()
