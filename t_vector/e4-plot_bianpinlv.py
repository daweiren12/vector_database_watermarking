
import h5py
import faiss
import numpy as np
import os
import struct
from collections import defaultdict
import math
import random
import hashlib

# 设置随机数种子以确保结果可重复
random_seed = 20
np.random.seed(random_seed)
watermark = '10101'

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"

# 1. 导入数据
def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data

# 辅助函数：获取邻居信息
def get_neighbors(node, M, offsets, neighbors):
    """获取指定节点的邻居列表"""
    start = int(offsets[node])
    end = int(offsets[node + 1])
    n = neighbors[start:start + 2 * M]
    neighbors_info = n[n != -1]
    return neighbors_info

# 辅助函数：计算指定节点的双向边数量
def count_bidirectional_edges(query_index, M, offsets, neighbors):
    """计算指定节点的双向边数量"""
    node_neighbors = get_neighbors(query_index, M, offsets, neighbors)
    bidirectional_edges = 0
    for neighbor in node_neighbors:
        neighbor_neighbors = get_neighbors(neighbor, M, offsets, neighbors)
        if query_index in neighbor_neighbors:
            bidirectional_edges += 1
    return bidirectional_edges

def calculate_all_bidirectional_edges(M, offsets, neighbors):
    """计算所有节点的双向边数量"""
    bidirectional_edge_counts = []
    for query_index in range(len(offsets) - 1):  # 假设offsets的长度比neighbors多1
        bidirectional_edges = count_bidirectional_edges(query_index, M, offsets, neighbors)
        bidirectional_edge_counts.append(bidirectional_edges)
    return bidirectional_edge_counts

from collections import Counter

def count_edge_frequencies(bidirectional_edge_counts):
    """统计每个边数出现的频率"""
    frequency_counter = Counter(bidirectional_edge_counts)
    total_nodes = len(bidirectional_edge_counts)
    frequencies = {edges: count / total_nodes for edges, count in frequency_counter.items()}
    return frequencies

import matplotlib.pyplot as plt

def plot_edge_frequencies(frequencies, title="Bidirectional Edge Frequencies"):
    """绘制不同边数的占比图"""
    edges, freqs = zip(*sorted(frequencies.items()))
    
    plt.figure(figsize=(10, 6))
    plt.bar(edges, freqs, color='skyblue')
    plt.xlabel('Number of Bidirectional Edges')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(edges)
    plt.show()

# 假设你已经有了M, offsets, neighbors的数据
# M, offsets, neighbors = ...  # 根据实际情况加载或定义

train_data = load_data(file_path, 10000)

M = 8   # HNSW参数
efConstruction = 50
index = faiss.IndexHNSWFlat(128, M)
index.hnsw.efConstruction = efConstruction
index.add(train_data)
index.hnsw.efSearch = 100

offsets = faiss.vector_to_array(index.hnsw.offsets)
neighbors = faiss.vector_to_array(index.hnsw.neighbors)

# 计算所有节点的双向边数量
bidirectional_edge_counts = calculate_all_bidirectional_edges(M, offsets, neighbors)

# 统计每个边数出现的频率
frequencies = count_edge_frequencies(bidirectional_edge_counts)

# 绘制图表
plot_edge_frequencies(frequencies)