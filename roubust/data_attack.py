import numpy as np
import faiss
import h5py
import matplotlib.pyplot as plt
import hashlib
import math
from scipy.stats import norm
import sys
from pathlib import Path
# 将上一级目录添加到系统路径

# 获取当前文件的路径，并解析出其父级目录的父级目录
parent_directory = Path(__file__).resolve().parent.parent

sys.path.append(str(parent_directory))

import watermarking
# 假设我们已经定义了前面提到的函数和全局变量
#num_samples = 50000
M = 8 
file_path = parent_directory / "sift-128-euclidean.hdf5"
seed = 122
num_samples = 10000


# 1. 导入数据
def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data
    
def build_hnsw_index(data, d, M, ef_construction):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index

# 统计各个向量的查询结果
def compute_query_counts(index, data, k):
    query_results = []
    i = 0
    for query_vector in data:
        distances, indices = index.search(query_vector.reshape(1, -1), k)
        query_results.append(indices.flatten())
        # if i < 10:
        #     print(query_vector[:5], indices)
        i += 1
        #print(i)
    return query_results

def compute_query_counts_with_map(index, train_data, remain_data, k):
    query_results = []
    
    # 创建一个映射：remain_data的索引 -> train_data的原始索引
    remain_to_train_mapping = {}
    for i, vec in enumerate(remain_data):
        # 查找原始数据中的索引
        original_idx = np.where(np.all(train_data == vec, axis=1))[0][0]
        remain_to_train_mapping[i] = original_idx
    
    i = 0
    # 对每个查询向量进行搜索
    for query_vector in train_data:
        distances, indices = index.search(query_vector.reshape(1, -1), k)
        #print(indices.flatten())
        # 将indices（删除后的索引）映射回原始数据的索引
        original_indices = [remain_to_train_mapping[idx] for idx in indices.flatten()  if idx != -1]
        query_results.append(original_indices)
        # if i < 10:
        #     print(query_vector[:5], original_indices)
        # i += 1
    
    return query_results

def compute_query_counts_with_map_modi(index, train_data, remain_data, k):
    query_results = []
    i = 0
    # 对每个查询向量进行搜索
    for query_vector in train_data:
        distances, indices = index.search(query_vector.reshape(1, -1), k)
        result = [idx for idx in indices.flatten()]
        query_results.append(result)
    
    return query_results


# 删除后的查询结果和删除之前的命中率
def hit_rate(query_result_baseline, query_result_p):
    # 计算删除前后的命中率
    total_queries = len(query_result_baseline)
    total_hits = 0
    intersection_sizes = []
    for baseline, after in zip(query_result_baseline, query_result_p):
        x = len(set(baseline).intersection(set(after)))
        total_hits += x
        intersection_sizes.append(x)
    
    print(total_hits, total_queries)
     # 统计交集大小的分布
    unique_sizes, counts = np.unique(intersection_sizes, return_counts=True)
    size_distribution = dict(zip(unique_sizes, counts))
    #print("Size distribution:", size_distribution)
    rate = total_hits / (total_queries * k)  # 每次查询返回 k 个结果
    print(rate)
    return rate

# 删除后的查询结果和删除之前的命中率
def ca_miss_and_false(query_result_baseline, query_result_p):
    if not query_result_baseline or not query_result_p:
        return 0.0, 0.0
    
    # 检查输入长度是否一致
    if len(query_result_baseline) != len(query_result_p):
        raise ValueError("The length of the baseline and modified query results must be equal.")
    
    total_misses = 0
    total_falses = 0
    
    for baseline, modified in zip(query_result_baseline, query_result_p):
        set_baseline = set(baseline)
        set_modified = set(modified)
        
        # 计算遗漏查询的数量
        misses = len(set_baseline - set_modified)
        total_misses += misses
        
        # 计算错误查询的数量
        falses = len(set_modified - set_baseline)
        total_falses += falses
    
    # 计算平均遗漏查询次数和平均错误查询次数
    avg_misses = total_misses / len(query_result_baseline)
    avg_falses = total_falses / len(query_result_p)
    
    return avg_misses, avg_falses


# 创建初始索引并计算基线查询结果
train_data = load_data(file_path, num_samples=10000)
d = train_data.shape[1]
index = build_hnsw_index(train_data, d, M, 100)
k = 100
query_result_baseline = compute_query_counts(index, train_data, k)
offsets = faiss.vector_to_array(index.hnsw.offsets)
neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    
# 计算可达性指数
accessibility_indices = watermarking.calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors)
    
# 按照可达性指数降序排列索引
sorted_indices = np.argsort(accessibility_indices)[::-1]

# 记录不同删除比例下的命中率
#pt = [0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5,0.6]
pt = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7,0.8,0.9]
hit_rates = []

#dele
def dele_impact():
    miss = []
    false = []
    for p in pt:
        num_to_remove = int(len(train_data) * p)
        remaining_data = [indice for indice in sorted_indices[num_to_remove:]]
        remain_data = train_data[remaining_data]
        index_p = build_hnsw_index(remain_data, d, M, 100)
        #print("done1")
        query_result_p = compute_query_counts_with_map(index_p, train_data, remain_data, k)
        #print(len(query_result_p), len(query_result_p[0]))
        #print(query_result_baseline[0][:10], query_result_p[:3][:10])
        # print(train_data[0][:5], remain_data[0][:5])
        avg_misses, avg_falses = ca_miss_and_false(query_result_baseline, query_result_p)
        miss.append(avg_misses)
        false.append(avg_falses)

    return miss, false

#modify
pt = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7,0.8,0.9]

def modi_impact():
    miss = []
    false = []
    for p in pt:
        num_to_modify = int(len(train_data) * p)
        # 按照可达性指数降序排列索引
        sorted_indices = np.argsort(accessibility_indices)[::-1]
        
        # 获取前 p 比例的索引
        indices_to_modify = sorted_indices[:num_to_modify]
        modified_data = np.copy(train_data)

        dim_to_modify = np.random.randint(0, train_data.shape[1])

        min_val = np.min(train_data[:, dim_to_modify])
        max_val = np.max(train_data[:, dim_to_modify])
        
        # 对每个选中的向量在指定维度上进行修改
        for idx in indices_to_modify:
            # 在目标维度的取值范围内随机选择一个新值
            new_value = np.random.uniform(min_val, max_val)
            dim_to_modify = np.random.randint(0, train_data.shape[1])
            for i in range(24):
                #modified_data[idx][dim_to_modify + i] = new_value
                modified_data[idx][(dim_to_modify + i) % train_data.shape[1]] = new_value  # 可以根据具体需求调整修改方式
            #print(data[idx][dim_to_modify], modified_data[idx][dim_to_modify])
            #print(np.linalg.norm(data[idx] - modified_data[idx]))
        
        index_p = build_hnsw_index(modified_data, d, M, 100)
        #print("done1")
        query_result_p = compute_query_counts_with_map_modi(index_p, train_data, modified_data, k)
        #print(len(query_result_p), len(query_result_p[0]))
        #print(query_result_baseline[0][:10], query_result_p[:3][:10])
        # print(train_data[0][:5], remain_data[0][:5])
        #rate = hit_rate(query_result_baseline, query_result_p)
        avg_misses, avg_falses = ca_miss_and_false(query_result_baseline, query_result_p)
        miss.append(avg_misses)
        false.append(avg_falses)

    return miss, false

def plot_impact(miss, false, Attack):
    # 设置字体大小
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 18,  # 设置标题字体大小
        'axes.labelsize': 18,  # 设置轴标签字体大小
        'xtick.labelsize': 18,  # 设置x轴刻度字体大小
        'ytick.labelsize': 18,  # 设置y轴刻度字体大小
        'legend.fontsize': 18   # 设置图例字体大小
    })

    # 创建图表
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pt, miss, label='Flase Queries', marker='o')
    #ax.plot(pt, false, label='False Queries', marker='s')

    ax.set_xlabel(f'{Attack} Proportion (p)')
    ax.set_ylabel('Count')
    #ax.set_title(f'Impact of Data {Attack}')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(Attack+".pdf")
    plt.show()

Attack = "Modification"
Attack = "Deletion"
miss, false = dele_impact()
#miss, false = modi_impact()
#print(miss, false)
plot_impact(miss, false, Attack)

#dele
#hit_rates  = [0.93454, 0.8827, 0.82071, 0.74838, 0.66704, 0.57797, 0.47376, 0.35445, 0.20778]

# 绘制命中率随删除比例变化的曲线
# plt.figure(figsize=(10, 7.5))
# plt.rcParams.update({
#         'font.size': 25,
#         'axes.titlesize': 26,  # 设置标题字体大小
#         'axes.labelsize': 25,  # 设置轴标签字体大小
#         'xtick.labelsize': 26,  # 设置x轴刻度字体大小
#         'ytick.labelsize': 26,  # 设置y轴刻度字体大小
#         'legend.fontsize': 24   # 设置图例字体大小
#     })
# plt.plot(pt, hit_rates, marker='o')
# plt.title('Hit Rate vs. Deletion Proportion')
# plt.xlabel('Modification Proportion (p)')
# plt.ylabel('Hit Rate')
# plt.grid(True)
# #plt.savefig('Deletion_Attacks_vs_Query_Accuracy100.pdf')
# plt.savefig('Modification_Attacks_vs_Query_Accuracy100.pdf')
# plt.show()
