#!/usr/bin/env python3
"""
绘制 QPS 随 Batch 动态变化的图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
insert_qps = pd.read_csv('/home/mingqi/SAGE-DB-Bench/results/sift/faiss_HNSW/ef-40/ef-40_batch_insert_qps.csv')

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 主图：QPS 随 Batch 变化
ax1 = axes[0, 0]
ax1.plot(insert_qps['batch_idx'], insert_qps['insert_qps'], 'b-', linewidth=0.8, alpha=0.7)
ax1.scatter(insert_qps['batch_idx'], insert_qps['insert_qps'], c='blue', s=10, alpha=0.5)

# 添加移动平均线
window = 10
rolling_mean = insert_qps['insert_qps'].rolling(window=window, center=True).mean()
ax1.plot(insert_qps['batch_idx'], rolling_mean, 'r-', linewidth=2, label=f'Moving Avg (window={window})')

ax1.axhline(y=insert_qps['insert_qps'].mean(), color='green', linestyle='--', linewidth=1.5, label=f'Mean: {insert_qps["insert_qps"].mean():,.0f}')
ax1.set_xlabel('Batch Index', fontsize=12)
ax1.set_ylabel('Insert QPS', fontsize=12)
ax1.set_title('Insert QPS vs Batch Index (HNSW)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, insert_qps['insert_qps'].max() * 1.1)

# 标注异常点
outliers = insert_qps[insert_qps['insert_qps'] < 3000]
for _, row in outliers.iterrows():
    ax1.annotate(f'Batch {int(row["batch_idx"])}\n{row["insert_qps"]:.0f}', 
                 xy=(row['batch_idx'], row['insert_qps']),
                 xytext=(row['batch_idx'] + 5, row['insert_qps'] + 2000),
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

# 2. 分段统计
ax2 = axes[0, 1]
n = len(insert_qps)
segment_size = n // 6
segments = []
segment_labels = []
for i in range(6):
    start = i * segment_size
    end = (i + 1) * segment_size if i < 5 else n
    seg_data = insert_qps.iloc[start:end]['insert_qps']
    segments.append(seg_data.values)
    segment_labels.append(f'{start}-{end-1}')

bp = ax2.boxplot(segments, labels=segment_labels, patch_artist=True)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, 6))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_xlabel('Batch Range', fontsize=12)
ax2.set_ylabel('Insert QPS', fontsize=12)
ax2.set_title('QPS Distribution by Batch Range', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. QPS 变化率
ax3 = axes[1, 0]
qps_changes = np.diff(insert_qps['insert_qps'].values) / insert_qps['insert_qps'].values[:-1] * 100
ax3.bar(range(len(qps_changes)), qps_changes, color=['red' if x < 0 else 'green' for x in qps_changes], alpha=0.6, width=1.0)
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.axhline(y=30, color='orange', linestyle='--', linewidth=1, label='+30%')
ax3.axhline(y=-30, color='orange', linestyle='--', linewidth=1, label='-30%')
ax3.set_xlabel('Batch Index', fontsize=12)
ax3.set_ylabel('QPS Change (%)', fontsize=12)
ax3.set_title('Batch-to-Batch QPS Change Rate', fontsize=14, fontweight='bold')
ax3.set_ylim(-100, 200)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. QPS 直方图
ax4 = axes[1, 1]
ax4.hist(insert_qps['insert_qps'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax4.axvline(x=insert_qps['insert_qps'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {insert_qps["insert_qps"].mean():,.0f}')
ax4.axvline(x=insert_qps['insert_qps'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {insert_qps["insert_qps"].median():,.0f}')
ax4.set_xlabel('Insert QPS', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('QPS Distribution Histogram', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 添加统计信息
stats_text = f"""Statistics:
Total Batches: {len(insert_qps)}
Mean QPS: {insert_qps['insert_qps'].mean():,.0f}
Std Dev: {insert_qps['insert_qps'].std():,.0f}
CV: {insert_qps['insert_qps'].std()/insert_qps['insert_qps'].mean()*100:.1f}%
Min: {insert_qps['insert_qps'].min():,.0f}
Max: {insert_qps['insert_qps'].max():,.0f}
Max/Min: {insert_qps['insert_qps'].max()/insert_qps['insert_qps'].min():.1f}x"""

fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='bottom')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# 保存图片
output_path = '/home/mingqi/SAGE-DB-Bench/results/sift/faiss_HNSW/ef-40/insert_qps_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_path}")

# 也保存一个单独的主图
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(insert_qps['batch_idx'], insert_qps['insert_qps'], 'b-', linewidth=0.8, alpha=0.7)
ax.scatter(insert_qps['batch_idx'], insert_qps['insert_qps'], c='blue', s=15, alpha=0.5)
ax.plot(insert_qps['batch_idx'], rolling_mean, 'r-', linewidth=2.5, label=f'Moving Average (window={window})')
ax.axhline(y=insert_qps['insert_qps'].mean(), color='green', linestyle='--', linewidth=1.5, label=f'Mean: {insert_qps["insert_qps"].mean():,.0f}')
ax.fill_between(insert_qps['batch_idx'], 0, insert_qps['insert_qps'], alpha=0.1, color='blue')

ax.set_xlabel('Batch Index', fontsize=14)
ax.set_ylabel('Insert QPS (ops/s)', fontsize=14)
ax.set_title('HNSW Insert QPS Over Time', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, insert_qps['insert_qps'].max() * 1.1)
ax.set_xlim(0, len(insert_qps))

output_path2 = '/home/mingqi/SAGE-DB-Bench/results/sift/faiss_HNSW/ef-40/insert_qps_main.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"主图已保存到: {output_path2}")

plt.show()
