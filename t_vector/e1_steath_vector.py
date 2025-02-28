import faiss
import numpy as np
from collections import defaultdict, deque
import h5py
import math

seed1 = 12
np.random.seed(seed1)

# 加载MNIST数据集
with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
    train_data = f['train'][-60001:-1]

# 构造HNSW索引
d = train_data.shape[1]  # 数据维度
M = 8   # HNSW参数
efConstruction = 50
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = efConstruction
index.add(train_data)
index.hnsw.efSearch = 100

# 记录每个向量被查询到的次数
def get_neighbor_count(index, queries, k_neighbors=5):
    neighbor_count = defaultdict(int)
    i = 0
    for query in queries:
        distances, neighbors = index.search(np.array([query]), k_neighbors)
        for neighbor in neighbors[0]:
            neighbor_count[neighbor] += 1
        #print(i)
        i += 1
    return neighbor_count

queries = train_data
neighbor_count = get_neighbor_count(index, queries)

# 找出那些从没被查询到的向量
never_searched_nodes = [node for node in range(len(train_data)) if neighbor_count[node] == 0]

def bfs(start_node, visited, neighbors, offsets, M):
    queue = deque([start_node])
    connected_component = set()
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            connected_component.add(node)
            start_idx = int(offsets[node])

            for neighbor in neighbors[start_idx:start_idx + 2 * M]:
                if neighbor != -1:
                    queue.append(neighbor)
    return connected_component


def get_neighbors(node, M, offsets, neighbors):
    """获取指定节点的邻居列表"""
    start = int(offsets[node])
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

def analyze_small_clusters(clusters, index, offsets, neighbors_edge, max_cluster_size=10):
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        data = f['train'][-60001:-1]
    average_distances = list([calculate_average_neighbor_distance(i, M, offsets, neighbors, data) for i in range(min(60000, len(data)))])
    # 计算基准值 b，这里使用平均邻居距离的中位数
    b = np.mean(average_distances)
    v = math.sqrt(np.var(average_distances))
    ave_dis = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            for node in cluster:
                node_neighbors = get_neighbors(node, M, offsets, neighbors)
                bidirectional_edges = 0
                sum_len = 0
                for neighbor in node_neighbors:
                    len_node_nei = np.linalg.norm(index.reconstruct(int(node)) - index.reconstruct(int(neighbor)))
                    neighbor_neighbors = get_neighbors(neighbor, M, offsets, neighbors)
                    dis = []
                    for nei in neighbor_neighbors:
                        dis.append(np.linalg.norm(index.reconstruct(int(nei)) - index.reconstruct(int(neighbor))))
                    if node in neighbor_neighbors:
                        bidirectional_edges += 1
                    sum_len += len_node_nei
                    #print(node,max(dis), len_node_nei, bidirectional_edges, len(node_neighbors))
                ave_dis.append(sum_len / len(node_neighbors))
    no_bid_len = np.mean(ave_dis)
    var_no = math.sqrt(np.var(ave_dis))
    print("The length of edges in average is {}, {}, The no bid is {}, {},{},{}".format(b,v, no_bid_len, var_no, min(ave_dis), max(ave_dis)))


def analyze_nodes_in_largest_cluster_never_searched(nodes_in_largest_cluster_never_searched, index, offsets, neighbors_edge, max_cluster_size=10):
    sum = 0
    dis = []
    for node in nodes_in_largest_cluster_never_searched:
        a = count_bidirectional_edges(node, M, offsets, neighbors)
        #print(a)
        sum += a
        node_neighbors = get_neighbors(node, M, offsets, neighbors)
        len_node_nei = np.linalg.norm(index.reconstruct(int(node)) - index.reconstruct(int(node_neighbors[0])))
        dis.append(len_node_nei)
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        data = f['train'][-60001:-1]
    num_nei = list([count_bidirectional_edges(i, M, offsets, neighbors) for i in range(min(60000, len(data)))])
    ave_dis = np.mean(dis)
    average_num_nei = np.mean(num_nei)
    print("average edegs count = {}, all vector average edegs count = {}, average edges length = {}, min = {}, max = {}".format(sum / len(nodes_in_largest_cluster_never_searched), average_num_nei, ave_dis, min(dis), max(dis)))
        
            
offsets = faiss.vector_to_array(index.hnsw.offsets)
neighbors = faiss.vector_to_array(index.hnsw.neighbors)
visited = set()
isolated_clusters = []

for node in never_searched_nodes:
    if node not in visited:
        #print(node)
        component = bfs(node, visited, neighbors, offsets, M)
        isolated_clusters.append(component)


print(f"构造条件如下, M = {M}, efconstruct = {efConstruction}, efsearch = {index.hnsw.efSearch}")
print(f"从没被搜到的点有:{len(never_searched_nodes)},部分点如下：{never_searched_nodes[:5]}")



print(f"Total isolated clusters found: {len(isolated_clusters)}")
print(f"Sizes of isolated clusters: {[len(cluster) for cluster in isolated_clusters]}")
analyze_small_clusters(isolated_clusters, index, offsets, neighbors)

# 找出在最大簇中的未被搜索到的点
largest_cluster = max(isolated_clusters, key=len)
nodes_in_largest_cluster_never_searched = [node for node in never_searched_nodes if node in largest_cluster]

def get_neighbors_info(node, neighbors, offsets, M):
    start_idx = int(offsets[node])
    end_idx = start_idx + 2 * M
    return neighbors[start_idx:end_idx]

print(f"大簇中从没被搜到的点有:{len(nodes_in_largest_cluster_never_searched)},部分点如下：{nodes_in_largest_cluster_never_searched[:5]}")
#查询这些点的邻居信息
analyze_nodes_in_largest_cluster_never_searched(nodes_in_largest_cluster_never_searched, index, offsets, neighbors)
# 随机选择一些节点
num_nodes_to_check = 100
random_nodes = np.random.choice(60000, num_nodes_to_check, replace=False)