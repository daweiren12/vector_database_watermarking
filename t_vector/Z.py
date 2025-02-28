import h5py
import numpy as np
import faiss
import matplotlib.pyplot as plt
from collections import defaultdict
import math

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
efConstruction = 50  # HNSW 图的 efConstruction 参数

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

# 辅助函数：获取邻居信息
def get_neighbors(node, M, offsets, neighbors):
    """获取指定节点的邻居列表"""
    start = int(offsets[node])
    end = int(offsets[node + 1])
    n = neighbors[start:start + 2 * M]
    neighbors_info = n[n != -1]
    return neighbors_info

def count_bidirectional_edges(query_index, M, offsets, neighbors):
    """计算指定节点的双向边数量"""
    node_neighbors = get_neighbors(query_index, M, offsets, neighbors)
    bidirectional_edges = 0
    for neighbor in node_neighbors:
        neighbor_neighbors = get_neighbors(neighbor, M, offsets, neighbors)
        if query_index in neighbor_neighbors:
            bidirectional_edges += 1
    return bidirectional_edges

#5.3根据边长选择待嵌入水印的节点
def calculate_average_neighbor_distance(node, M, offsets, neighbors, data):
    start_idx = int(offsets[node])
    node_neighbors = neighbors[start_idx:start_idx + 2 * M]  # 只考虑最底层的边

    edge_lengths = []
    for neighbor in node_neighbors:
        if neighbor == -1:
            continue

        # 计算边的长度
        edge_length = np.linalg.norm(data[node] - data[neighbor])
        edge_lengths.append(edge_length)
    #print(np.mean(edge_lengths))
    return float(np.mean(edge_lengths))

def calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors):
    bidirectional_edge_counts = []
    average_neighbor_distances = []

    for node in range(len(train_data)):
        bidirectional_edges = count_bidirectional_edges(node, M, offsets, neighbors)
        avg_distance = calculate_average_neighbor_distance(node, M, offsets, neighbors, train_data)
        
        bidirectional_edge_counts.append(bidirectional_edges)
        average_neighbor_distances.append(avg_distance)

    mean_edges = np.nanmean(bidirectional_edge_counts)
    std_edges = np.nanstd(bidirectional_edge_counts)

    mean_distances = np.nanmean(average_neighbor_distances)
    std_distances = np.nanstd(average_neighbor_distances)

    z_scores_edges = [(d - mean_edges) / std_edges for d in bidirectional_edge_counts]
    z_scores_distances = [(d - mean_distances) / std_distances for d in average_neighbor_distances]

    # 创建一个NumPy数组来存储可达性指数
    accessibility_indices = np.array([0.5 * zd - 0.5 * ze for zd, ze in zip(z_scores_distances, z_scores_edges)])

    return accessibility_indices

accessibility_indices = calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors)

from scipy.stats import pearsonr, spearmanr   
pearson_corr, _ = pearsonr(accessibility_indices, query_counts)
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")


# 分段计算平均查询次数
num_bins = 8
bins = np.linspace(min(accessibility_indices), max(accessibility_indices), num_bins + 1)
#print((max(average_edge_length) - min(average_edge_length)) / 15)
bin_indices = np.digitize(accessibility_indices, bins)
#print(bin_indices)

bin_avg_query_counts = []
bin_avg_z = []

for i in range(1, num_bins + 1):
    indices_in_bin = np.where(bin_indices == i)[0]
    if len(indices_in_bin) > 0:
        bin_avg_query_counts.append(np.mean(query_counts[indices_in_bin]))
        bin_avg_z.append(np.mean(accessibility_indices[indices_in_bin]))

# plt.rcParams['font.family'] = zh_font.get_name()  # 设置字体
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置字体大小
plt.rcParams.update({
    'font.size': 25,
    #'axes.titlesize': 26,  # 设置标题字体大小
    'axes.labelsize': 22,  # 设置轴标签字体大小
    'xtick.labelsize': 20,  # 设置x轴刻度字体大小
    'ytick.labelsize': 20,  # 设置y轴刻度字体大小
    #'legend.fontsize': 14   # 设置图例字体大小
})
# 创建折线图
fig, ax = plt.subplots(figsize=(5, 5))
#fig, ax = plt.subplots(figsize=(12, 8))

# 添加网格线
ax.grid(True)

# 设置 xticks 为 z 分数的合理间隔
xticks = np.arange(-4, 4, 1)  # 根据需要调整数量
ax.set_xticks(xticks)
ax.set_xticklabels([f'{x}' for x in xticks])

# 设置 yticks 为平均查询次数的合理间隔
yticks = np.arange(0, max(bin_avg_query_counts) + 2, 5) # 根据需要调整数量
ax.set_yticks(yticks)
ax.set_yticklabels([f'{int(y)}' for y in yticks])

# 设置 y 轴刻度标记
#ax.tick_params(axis='both', which='major', labelsize=14)

# 设置轴标签
ax.set_xlabel('Transparency Score(ts)')
#ax.set_ylabel('Average Query Counts')

# 绘制折线图
ax.plot(bin_avg_z, bin_avg_query_counts, marker='o', linestyle='-')

plt.tight_layout()
# 保存图像
plt.savefig('ts_query_count.pdf')
#plt.savefig('average_edge_length_query_count_line.pdf')
plt.show()
