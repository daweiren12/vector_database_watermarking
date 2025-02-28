import numpy as np
import faiss
import h5py
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle


def save_results(results, filename):
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

def load_results(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# 设置随机种子以确保结果可重复
random_seed = 42
np.random.seed(random_seed)

# 导入数据集
file_path = "sift-128-euclidean.hdf5"

with h5py.File(file_path, 'r') as f:
    train_data = f['train'][:60000]  # 使用前60000个数据点
d = train_data.shape[1]

# 构建HNSW索引并计算隐身节点
def build_hnsw_and_get_hidden_nodes(train_data, M=8, ef_construction=40, ef_search=50, k=10, hidden_node_ratio=0.1):
    num_points = train_data.shape[0]

    # 构建HNSW索引
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(train_data)

    # 记录每个节点被搜索到的次数
    search_counts = defaultdict(int)

    # 使用train_data中的每个向量作为查询节点
    for i in range(num_points):
        query_vector = train_data[i].reshape(1, -1)
        _, I = index.search(query_vector, k)
        for idx in I.flatten():
            search_counts[idx] += 1

    # 确定隐身节点（被搜索次数最少的10%的节点）
    sorted_counts = sorted(search_counts.items(), key=lambda item: item[1])
    num_hidden_nodes = int(hidden_node_ratio * num_points)
    hidden_nodes = {node for node, _ in sorted_counts[:num_hidden_nodes]}

    return hidden_nodes


filename = "overlap.pkl"
M_values = [8, 12, 16, 24]
efConstruction_values = [50, 75, 100, 150, 200]

def cal():
    # 基准参数
    M_base = 12
    efconstruct_base = 100

    # 构建基准隐身节点集合
    hidden_nodes_base = build_hnsw_and_get_hidden_nodes(train_data, M=M_base, ef_construction=efconstruct_base)

    # 测试不同M和efConstruction参数
    
    # M_values = [8, 12]
    # efConstruction_values = [50, 75]
    hidden_nodes_dict = {}

    # 构建HNSW索引并计算隐身节点
    for M in M_values:
        for efconstruct in efConstruction_values:
            hidden_nodes = build_hnsw_and_get_hidden_nodes(train_data, M, efconstruct)
            hidden_nodes_dict[(M, efconstruct)] = hidden_nodes

    # 比较不同参数条件下隐身节点集合与基准集合的重合度
    overlap_results = defaultdict(list)

    for M in M_values:
        for efconstruct in efConstruction_values:
            hidden_nodes = hidden_nodes_dict[(M, efconstruct)]
            overlap = hidden_nodes.intersection(hidden_nodes_base)
            #union = hidden_nodes.union(hidden_nodes_base)
            overlap_ratio = len(overlap) / 6000
            overlap_results[M].append(overlap_ratio)
            print(f"M = {M}, ef = {efconstruct}, overlap = {overlap_ratio}")

    save_results(overlap_results, filename)

#cal()

overlap_results = load_results(filename)

# 可视化结果
#fig, ax = plt.subplots(figsize=(12, 8))
#fig, ax = plt.subplots()
# 设置字体大小
#设置字体大小
plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 26,  # 设置标题字体大小
        'axes.labelsize': 25,  # 设置轴标签字体大小
        'xtick.labelsize': 26,  # 设置x轴刻度字体大小
        'ytick.labelsize': 26,  # 设置y轴刻度字体大小
        'legend.fontsize': 24   # 设置图例字体大小
})
fig, ax = plt.subplots(figsize=(12, 8))


# 添加网格线
ax.grid(True)

for M in M_values:
    ax.plot(efConstruction_values, overlap_results[M], marker='o', label=f'M={M}')

ax.set_xlabel('efConstruct')
ax.set_ylabel('Overlap Ratio')

# 添加图例
ax.legend()


plt.tight_layout()
plt.savefig('hidden_nodes_overlap_curves.pdf')
plt.show()






