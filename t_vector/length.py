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

# 记录平均边长
average_edge_length = np.zeros(nb, dtype=float)

for node in range(nb):
    start_idx = int(offsets[node])
    end_idx = int(offsets[node + 1])
    node_neighbors = neighbors[start_idx:start_idx + 2 * M]  # 只考虑最底层的边

    edge_lengths = []
    for neighbor in node_neighbors:
        if neighbor == -1:
            continue

        # 计算边的长度
        edge_length = np.linalg.norm(train_data[node] - train_data[neighbor])
        edge_lengths.append(edge_length)

    # 计算平均边长
    if edge_lengths:
        average_edge_length[node] = np.mean(edge_lengths)

from scipy.stats import pearsonr, spearmanr   
pearson_corr, _ = pearsonr(average_edge_length, query_counts)
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")

# 分段计算平均查询次数
num_bins = 8
bins = np.linspace(min(average_edge_length), max(average_edge_length), num_bins + 1)
#print((max(average_edge_length) - min(average_edge_length)) / 15)
bin_indices = np.digitize(average_edge_length, bins)
#print(bin_indices)

bin_avg_query_counts = []
bin_avg_edge_lengths = []

for i in range(1, num_bins + 1):
    indices_in_bin = np.where(bin_indices == i)[0]
    if len(indices_in_bin) > 0:
        bin_avg_query_counts.append(np.mean(query_counts[indices_in_bin]))
        bin_avg_edge_lengths.append(np.mean(average_edge_length[indices_in_bin]))

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

# 计算最接近的整百点
rounded_xticks = [int(round(x, -2)) for x in bin_avg_edge_lengths]

# 设置 xticks 为最接近的整百点
xticks = np.arange(min(rounded_xticks), max(rounded_xticks)+1, 100)

yticks = np.arange(0,27,5)
ax.plot(bin_avg_edge_lengths, bin_avg_query_counts, marker='o', linestyle='-')

# 添加网格线
ax.grid(True)

#设置 x 轴标签
ax.set_xlabel('Average Edge Length')
# 设置 y 轴标签
#ax.set_ylabel('Average Query Counts')
# 设置标题
#ax.set_title('Effect of Average Edge Length on Query Count', fontsize=16)

# ax.set_xlabel('平均边长')
# # 设置 y 轴标签
# ax.set_ylabel('被查询次数')

# 设置 x 轴刻度标记
ax.set_xticks(xticks)
ax.set_xticklabels([str(i) for i in xticks])

# 设置 y 轴刻度标记
ax.set_yticks(yticks)
ax.set_yticklabels([str(i) for i in yticks])

# 设置 y 轴刻度标记
#ax.tick_params(axis='both', which='major', labelsize=14)

# 添加图例
#ax.legend(fontsize=14)



plt.tight_layout()
# 保存图像
plt.savefig('average_edge_length_query_count.pdf')
#plt.savefig('average_edge_length_query_count_line.pdf')
plt.show()
