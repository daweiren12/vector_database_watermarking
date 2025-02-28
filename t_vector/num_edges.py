import h5py
import numpy as np
import faiss
import matplotlib.pyplot as plt
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找系统中的中文字体，这里以Windows为例，Linux或Mac可能不同
zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 使用微软雅黑字体


# 加载 MNIST 数据集
file_path = 'sift-128-euclidean.hdf5'
with h5py.File(file_path, 'r') as f:
    train_data = f['train'][:10000]

# 参数设置
d = train_data.shape[1]  # 数据维度
nb = train_data.shape[0]  # 数据数量
M = 8  # HNSW 图的邻居数量
efConstruction = 20  # HNSW 图的 efConstruction 参数

# 构建索引
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = efConstruction
index.add(train_data)

# 将 offsets 和 neighbors 转换为 NumPy 数组
offsets = faiss.vector_to_array(index.hnsw.offsets)
neighbors = faiss.vector_to_array(index.hnsw.neighbors)

# 记录每个向量被查询到的次数
query_counts = np.zeros(nb, dtype=int)

# 使用所有训练数据进行查询，并记录每个向量被查询到的次数
for i in range(nb):
    distances, indices = index.search(np.array([train_data[i]]), 10)
    for idx in indices[0]:
        query_counts[idx] += 1

# 记录双向边的数量
double_edges_count = np.zeros(nb, dtype=int)

for node in range(nb):
    start_idx = int(offsets[node])
    end_idx = int(offsets[node + 1])
    node_neighbors = neighbors[start_idx:start_idx + 2 * M]  # 只考虑最底层的边

    for neighbor in node_neighbors:
        if neighbor == -1:
            continue
        
        neighbor_start_idx = int(offsets[neighbor])
        neighbor_end_idx = int(offsets[neighbor + 1])
        neighbor_neighbors = neighbors[neighbor_start_idx:neighbor_start_idx + 2 * M]
        
        if node in neighbor_neighbors:
            double_edges_count[node] += 1

from scipy.stats import pearsonr, spearmanr
pearson_corr, _ = pearsonr(double_edges_count, query_counts)
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
# 统计不同双向边数量下的点的平均查询次数
double_edge_to_query_count = defaultdict(list)

for node in range(nb):
    double_edge_to_query_count[double_edges_count[node]].append(query_counts[node])

# 计算不同双向边数量下的平均查询次数
double_edges = sorted(double_edge_to_query_count.keys())
avg_query_counts = [np.mean(double_edge_to_query_count[de]) for de in double_edges]

# 设置字体大小
plt.rcParams.update({
    'font.size': 25,
    #'axes.titlesize': 26,  # 设置标题字体大小
    'axes.labelsize': 22,  # 设置轴标签字体大小
    'xtick.labelsize': 20,  # 设置x轴刻度字体大小
    'ytick.labelsize': 20,  # 设置y轴刻度字体大小
    #'legend.fontsize': 14   # 设置图例字体大小
})

# 设置图表大小
fig, ax = plt.subplots(figsize=(5, 5))

# 绘制折线图
ax.plot(double_edges, avg_query_counts, marker='o', linestyle='-')

# 添加网格线
ax.grid(True)

# 设置 x 轴刻度标记
xticks = np.arange(min(double_edges), max(double_edges) + 1, 4)  # 每个整数位置都有一个刻度
ax.set_xticks(xticks)
ax.set_xticklabels([str(i) for i in xticks])

# 设置 y 轴刻度标记
yticks = np.arange(0, max(avg_query_counts) + 1, 5)  # 根据实际最大值动态设置 y 轴刻度
ax.set_yticks(yticks)
ax.set_yticklabels([int(i) for i in yticks])

# 设置轴标签
ax.set_xlabel('Number of Edges')
#ax.set_ylabel('Average Query Counts')

# 设置图表标题
#ax.set_title('Average Query Counts vs Number of Bidirectional Edges', fontsize=26)

# 添加图例
#ax.legend(fontsize=18)



# 自动调整布局
plt.tight_layout()

# 保存图像为 PDF 文件
plt.savefig('edge_query_count.pdf')

# 显示图表
plt.show()


# # plt.rcParams['font.family'] = zh_font.get_name()  # 设置字体
# # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# #fig, ax = plt.subplots(figsize=(5, 5))
# fig, ax = plt.subplots(figsize=(12, 8))


# xticks = np.arange(min(double_edges), max(double_edges)+1, 2)

# yticks = np.arange(0,20,5)

# ax.plot(double_edges, avg_query_counts, marker='o', linestyle='-')

# # 添加网格线
# ax.grid(True)

# ax.set_xlabel('Edge Counts', fontsize=24)
# # 设置 y 轴标签
# ax.set_ylabel('Average Query Counts', fontsize=24)
# # 设置标题
# # 设置 x 轴刻度标记
# ax.set_xticks(xticks)
# ax.set_xticklabels([str(i) for i in xticks], fontsize=24)

# # 设置 y 轴刻度标记
# ax.set_yticks(yticks)
# ax.set_yticklabels([str(i) for i in yticks], fontsize=24)

# plt.tight_layout()

# # 保存图像
# #plt.savefig('double_edge_query_count_chine.png')
# plt.savefig('double_edge_query_count.pdf')
# plt.show()
