import h5py
import faiss
import numpy as np
from collections import defaultdict
import math
import random
import hashlib
from collections import Counter
import matplotlib.pyplot as plt
import assistance

# 设置随机数种子以确保结果可重复
random_seed = 20
np.random.seed(random_seed)
watermark = '10101'

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"

def hash_re(clusterid):
    id = str(clusterid)
    hid = hashlib.md5()
    #hid.update(ks.encode('utf-8'))
    hid.update(id.encode('utf-8'))
    hash_result = hid.hexdigest()
    hash_re_int = int(hash_result, 16)
    return hash_re_int

# 1. 导入数据
def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data

def generate_id_from_selected_dims(vector, selected_dims):
    id_parts = []
    for dim in selected_dims[:-1]:
        value = vector[dim]
        if value < 0:
            value = abs(value)  # 处理负数，取绝对值
        highest_digit = str(int(value))[0]  # 提取十进制最高位
        id_parts.append(highest_digit)
    
    # 拼接为唯一标识符
    unique_id = "".join(id_parts)
    return unique_id

# 辅助函数：在向量中嵌入LSB水印
def embed_watermark_single_vector(vector, watermark_bit, embedding_dims, position, id, min_pos_fraction=0.5):
    hash_value = hash_re(id)
    position = hash_value % len(vector)
    #print(f"p1 = {position}")
    while position in embedding_dims:
        position = (position + 1) % len(vector) 
    #print(f"position = {position}")
    bin_rep = assistance.float_to_bin(vector[position])
    #print(bin_rep)
    len_bin = len(bin_rep.replace('.', ''))  
        # 去掉小数点后的二进制总长度

        # 定义允许修改的位置范围（低位）
    min_pos = int(len_bin * min_pos_fraction)
    max_pos = len_bin
    
    bit_position = hash_value % (max_pos - min_pos) + min_pos

    # 确保不会修改小数点
    bin_list = list(bin_rep)
    if bin_list[bit_position] == '.':
        bit_position = bit_position - 1  # 避免修改小数点
    

    dim_lowest_bit = position % 2 # 维度最低位
    #print(watermark_bit, dim_lowest_bit)
    bit = int(watermark_bit) ^ dim_lowest_bit
        # 修改对应的位
    bin_list[bit_position] = str(bit)
    modified_bin = "".join(bin_list)
    modified_data = assistance.bin_to_float(modified_bin)
    vector[position] = modified_data
    #print(modified_data, vector[position])
    return vector

def extract_watermark_single_vector(vector, embedding_dims, position, id, min_pos_fraction=0.5):
    # 获取当前维度的数据的二进制表示
    # while vector[position] == 0 or position in embedding_dims[:-1]:
    #     position += 1
    hash_value = hash_re(id)
    position = hash_value % len(vector)
    while position in embedding_dims:
        position = (position + 1) % len(vector) 

    bin_rep = assistance.float_to_bin(vector[position])
    len_bin = len(bin_rep.replace('.', ''))  # 二进制总长度

        # 定义允许修改的位置范围（低位）
    min_pos = int(len_bin * min_pos_fraction)
    max_pos = len_bin
    
    bit_position = hash_value % (max_pos - min_pos) + min_pos
        # 重新计算哈希值，确定修改位置
    
        # 确保位置不为小数点
    bin_list = list(bin_rep)
    if bin_list[bit_position] == '.':
        bit_position = bit_position - 1  # 避免小数点

    # 计算水印位：与维度的最低位进行异或
    bit = int(bin_list[bit_position])  # 提取原始的水印位
    dim_lowest_bit = position % 2  # 维度最低位
        
    extracted_bit = str(bit ^ dim_lowest_bit)  # 异或操作   
    return extracted_bit

def group_vectors_and_classify_bits(train_data, embedding_dims, L, min_pos_fraction=0.5):
    # 初始化分组容器
    groups = {i: {"express_1": [], "express_0": []} for i in range(L)}

    for i in range(len(train_data)):  # 使用enumerate来获取每个vector的index
        # 根据向量的ID确定分组
        vector = train_data[i]
        vector_id = generate_id_from_selected_dims(vector, embedding_dims)  # 生成向量ID
        group_id = hash_re(vector_id) % L  # 哈希分组

        # 提取向量隐藏的信息
        hidden_bit = extract_watermark_single_vector(
            vector, embedding_dims, embedding_dims[-1], vector_id, min_pos_fraction=min_pos_fraction
        )

        # 按表达的隐藏信息分类，并且保存向量及其在train_data中的下标
        if hidden_bit == "1":
            groups[group_id]["express_1"].append(i)  # 保存vector和它的index
        else:
            groups[group_id]["express_0"].append(i)  # 保存vector和它的index
        # 查看分组信息
    # for group_id, data in groups.items():
    #     print(f"Group {group_id}:")
    #     express_1_count = len(data['express_1'])
    #     express_0_count = len(data['express_0'])
    #     print(f"  Express 1: {express_1_count} vectors")
    #     print(f"  Express 0: {express_0_count} vectors")
    #     total = express_0_count + express_1_count
    #     ratio = express_0_count / total if total > 0 else 0
    #     print(f"  ratio 0: {ratio}, {total}")

    return groups

def random_select_vectors(num_need_modi, vectors, M, index):
    offsets = faiss.vector_to_array(index.hnsw.offsets)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    if num_need_modi > len(vectors):
        raise ValueError("num_need_modi exceeds the number of available vectors.")

    # 使用随机采样从列表中选择向量
    selected_vectors = random.sample(vectors, num_need_modi)
    e = []
    for ve in selected_vectors:
        edges_count = count_bidirectional_edges(ve, M, offsets, neighbors)
        e.append(edges_count)
    #print(np.mean(e))
    return selected_vectors

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

def select_by_ai(num_need_modi, available_vectors, rank_ratio, accessibility_indices, query_counts):

    # # 提取可用向量及其对应的AI分数
    # ai_scores = [accessibility_indices[idx] for idx in available_vectors]
    
    # # 对AI分数进行排序并找到排名位置
    # sorted_ais = sorted(ai_scores, reverse=True)
    ai_scores = [(idx, accessibility_indices[idx]) for idx in available_vectors]

# 对AI分数进行排序（按AI值降序），并保持索引信息
    sorted_ais = sorted(ai_scores, key=lambda x: x[1], reverse=True)
    threshold_index = int(len(sorted_ais) * rank_ratio) - 1
    threshold_ai = sorted_ais[threshold_index][1] if threshold_index >= 0 else float('-inf')
    
    #print(f"Threshold AI score for top {rank_ratio * 100}%: {threshold_ai}")

    selected_neighbors = set()
    selected_neighbors_list = []
    count = []
    for query_index in available_vectors:
        if len(selected_neighbors_list) >= num_need_modi:
            break
        ai = accessibility_indices[query_index]
        #print(edges_count)
        if query_index not in selected_neighbors and ai >= threshold_ai :
            selected_neighbors.add(query_index)
            selected_neighbors_list.append(query_index)
            count.append(query_counts[query_index])
    #print(f"mean = {np.mean(count)}")

    # 如果初次选择的数量不够，则继续选择剩下的向量
    if len(selected_neighbors_list) < num_need_modi:
        #print(len(selected_neighbors_list), num_need_modi)
        remaining_ais = [item for item in sorted_ais if item[0] not in selected_neighbors]
        
        for idx, _ in remaining_ais:
            if len(selected_neighbors_list) >= num_need_modi:
                break
            if idx not in selected_neighbors:
                selected_neighbors.add(idx)
                selected_neighbors_list.append(idx)
                count.append(query_counts[idx])

    return selected_neighbors_list


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

# 主函数：嵌入水印的整体流程
def watermark_embedding_by_ai(file_path, strength, th, num_samples=None, watermark="0000000000", random_seed = 20):
    random.seed(random_seed)
    L = len(watermark)

    train_data = load_data(file_path, num_samples)
    d = train_data.shape[1]
    
    embedding_dims = random.sample(range(d), 10)
    #embedding_dims = [1,2,3,4,5,6]

    M = 8   # HNSW参数
    efConstruction = 50
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.add(train_data)
    index.hnsw.efSearch = 100

    #print(f"Embedding dimensions: {embedding_dims}")

    offsets = faiss.vector_to_array(index.hnsw.offsets)
    neighbors = faiss.vector_to_array(index.hnsw.neighbors)
    accessibility_indices = calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors)

    grouped_data = group_vectors_and_classify_bits(train_data, embedding_dims, L, min_pos_fraction=0.5)

    
    nb = len(train_data)
    query_counts = np.zeros(nb, dtype=int)

    for i in range(nb):
        distances, indices = index.search(np.array([train_data[i]]).astype('float32'), 10)
        for idx in indices[0]:
            if idx != i:  # 不包括自身
                query_counts[idx] += 1

    # 用于存储修改后的向量
    all_carrier_vectors = []
    modified_vectors = []

    watermarked_data = train_data.copy()
    for i in range(L):
        sum_vectors = len(grouped_data[i]['express_1']) + len(grouped_data[i]['express_0'])

        if watermark[i] == '1':
            num_need_modi = max(0, math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_1']))
            #num_need_modi = math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_1'])
            available_vectors = grouped_data[i]['express_0']
        else:
            num_need_modi = max(0, math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_0']))
            #num_need_modi = math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_0'])
            available_vectors = grouped_data[i]['express_1']

        # 确保不会超出可用数量
        num_need_modi = min(num_need_modi, len(available_vectors))
        carrier_vectors_id = select_by_ai(num_need_modi, available_vectors, th, accessibility_indices, query_counts)
        #print(num_need_modi, len(carrier_vectors_id), len(available_vectors))

        for id in carrier_vectors_id:
            all_carrier_vectors.append(id)
            vector = train_data[id]
            vector_id = generate_id_from_selected_dims(vector, embedding_dims)
            modified_vector = embed_watermark_single_vector(
                vector, watermark[i], embedding_dims, embedding_dims[-1], vector_id, min_pos_fraction=0.5
            )
            watermarked_data[id] = modified_vector
            modified_vectors.append(modified_vector)
    #print(f"Final data shape: {watermarked_data.shape}")
    return watermarked_data, len(watermark), all_carrier_vectors, train_data, accessibility_indices

# 主函数：嵌入水印的整体流程
def watermark_embedding(file_path, strength, num_samples=None, watermark="0000000000", random_seed = 20):
    train_data = load_data(file_path, num_samples)
    #print(f"Final data shape: {train_data.shape}")
    d = train_data.shape[1]
    L = len(watermark)
    random.seed(random_seed)
    embedding_dims = random.sample(range(d), 10)
    #print(embedding_dims)
    #embedding_dims = [1,2,3,4,5,6]

    M = 8   # HNSW参数
    efConstruction = 50
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.add(train_data)
    index.hnsw.efSearch = 100

    #print(f"Embedding dimensions: {embedding_dims}")
    grouped_data = group_vectors_and_classify_bits(train_data, embedding_dims, L, min_pos_fraction=0.5)
    

    # 用于存储修改后的向量
    all_carrier_vectors = []
    modified_vectors = []
    
    watermarked_data = train_data.copy()
    for i in range(L):
        sum_vectors = len(grouped_data[i]['express_1']) + len(grouped_data[i]['express_0'])

        if watermark[i] == '1':
            num_need_modi = max(0, math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_1']))
            #num_need_modi = math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_1'])
            available_vectors = grouped_data[i]['express_0']
        else:
            num_need_modi = max(0, math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_0']))
            #num_need_modi = math.ceil(strength * sum_vectors) - len(grouped_data[i]['express_0'])
            available_vectors = grouped_data[i]['express_1']

        
        # 确保不会超出可用数量
        num_need_modi = min(num_need_modi, len(available_vectors))
        carrier_vectors_id = random_select_vectors(num_need_modi, available_vectors, M, index)
        #print(num_need_modi, len(available_vectors), len(carrier_vectors_id))

        for id in carrier_vectors_id:
            all_carrier_vectors.append(id)
            vector = train_data[id]
            vector_id = generate_id_from_selected_dims(vector, embedding_dims)
            modified_vector = embed_watermark_single_vector(
                vector, watermark[i], embedding_dims, embedding_dims[-1], vector_id, min_pos_fraction=0.5
            )
            watermarked_data[id] = modified_vector
            modified_vectors.append(modified_vector)

    #print(f"Final data shape: {watermarked_data.shape}")
    return watermarked_data, len(watermark), all_carrier_vectors, train_data

def watermark_extraction(watermarked_data, watermark_length, random_seed = 20):
    # 加载数据
    d = watermarked_data.shape[1]
    L = watermark_length
    random.seed(random_seed)
    embedding_dims = random.sample(range(d), 10)
    #print(embedding_dims)
    # 初始化分组容器
    grouped_data = group_vectors_and_classify_bits(watermarked_data, embedding_dims, L, min_pos_fraction=0.5)

    # 存储提取的水印
    extracted_watermark = []

    for group_id, data in grouped_data.items():
        num_express_1 = len(data["express_1"])
        num_express_0 = len(data["express_0"])
        #print(num_express_0,num_express_1)
        # 按多数决提取该组的水印位
        if num_express_1 > num_express_0:
            extracted_bit = "1"
        else:
            extracted_bit = "0"

        extracted_watermark.append(extracted_bit)

    return "".join(extracted_watermark)


def adaptive_dele(data, p, accessibility_indices):
    
    # 确定要删除的元素数量
    num_to_remove = int(np.ceil(len(data) * p))
    # 按照可达性指数降序排列索引
    sorted_indices = np.argsort(accessibility_indices)[::-1]
    
    # 获取要保留的索引
    indices_to_keep = sorted_indices[num_to_remove:]
    
    # 创建新的数据集，不包含要删除的元素
    filtered_data = data[indices_to_keep]
    
    return filtered_data

def BER(wm, ex_wm):
    sum = 0
    for i in range(len(wm)):
        if wm[i] != ex_wm[i]:
            sum += 1
    #print(wm, ex_wm, sum / len(wm))
    return sum/ len(wm)

# train_data = load_data(file_path, 10000)
# d = train_data.shape[1]
# M = 8   # HNSW参数
# efConstruction = 50
# index = faiss.IndexHNSWFlat(d, M)
# index.hnsw.efConstruction = efConstruction
# index.add(train_data)
# index.hnsw.efSearch = 100
# offsets = faiss.vector_to_array(index.hnsw.offsets)
# neighbors = faiss.vector_to_array(index.hnsw.neighbors)
# accessibility_indices = calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors)

# watermark="1010110101101011010110101101"

# watermarked_data, watermark_length, all_carrier_vectors, train_data = watermark_embedding(
#         file_path, 0.55, 10000, watermark, random_seed = 20
#         )


# de_data = adaptive_dele(watermarked_data, 0.9, accessibility_indices)
# ex_watermark = watermark_extraction(de_data, watermark_length, random_seed = 20)
# #print(watermark, ex_wm, len(watermark), len(ex_wm))
# ber = BER(watermark, ex_watermark)
# print(ber)
