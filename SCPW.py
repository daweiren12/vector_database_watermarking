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
import watermarking

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

def build_hnsw_index(data, d, M, ef_construction):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index

def found_e(train_data, position, Standard_e = 1):
    # 确保 train_data 是 NumPy 数组
    if not isinstance(train_data, np.ndarray):
        train_data = np.array(train_data)
    
    # 检查 position 是否有效
    if position < 0 or position >= train_data.shape[1]:
        raise ValueError("position 超出了数据的维度范围")

    # 提取指定列的数据
    column_data = train_data[:, position]

    #print(column_data[:50])
    # 计算均值和方差
    mean = np.mean(column_data)
    variance = np.var(column_data)
    Standard_x = math.sqrt(variance)

    kx = - Standard_e ** 2 / (2 * variance)
    k1 = math.sqrt(Standard_e ** 2 - kx ** 2 * variance)
    k2 = - kx * mean
    E = []
    #print(mean, variance, kx, k1, k2)
    for i in range(len(column_data)):
        u = random.gauss(0,1)
        e = kx * column_data[i] + k1 * u + k2
        E.append(e)
    # print(E[:50])
    # mean = np.mean(E)
    # variance = np.var(E)
    # print(mean, variance)
    return E

def hash_re(clusterid):
    id = str(clusterid)
    hid = hashlib.md5()
    #hid.update(ks.encode('utf-8'))
    hid.update(id.encode('utf-8'))
    hash_result = hid.hexdigest()
    hash_re_int = int(hash_result, 16)
    return hash_re_int

def generate_id_from_selected_dims(vector, top_k):
    id_parts = []
    for dim in range(top_k):
        value = vector[dim]
        if value < 0:
            value = abs(value)  # 处理负数，取绝对值
        highest_digit = str(int(value))[0]  # 提取十进制最高位
        id_parts.append(highest_digit)
    
    # 拼接为唯一标识符
    unique_id = "".join(id_parts)
    return unique_id

# 辅助函数：在向量中嵌入LSB水印
def embed_watermark_single_vector(vector, watermark_bit, position, id, min_pos_fraction=0.5):
    hash_value = hash_re(id)
    position = hash_value % len(vector)
    #print(f"p1 = {position}")
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

def extract_watermark_single_vector(vector, hash_value, position):
    pos = hash_value % len(vector)
    dim_lowest_bit = pos % 2  # 维度最低位
        # 提取向量隐藏的信息
        # hidden_bit = extract_watermark_single_vector(
        #     vector, vector_id, min_pos_fraction=min_pos_fraction
        # )
    hidden_bit = str(hash_re(vector[position]) % 2 ^ dim_lowest_bit) 
    return hidden_bit

def group_vectors_and_classify_bits(train_data, top_k, L, position):
    # 初始化分组容器
    groups = {i: {"express_1": [], "express_0": []} for i in range(L)}

    for i in range(len(train_data)):  # 使用enumerate来获取每个vector的index
        # 根据向量的ID确定分组
        vector = train_data[i]
        vector_id = generate_id_from_selected_dims(vector, top_k)  # 生成向量ID

        hash_value = hash_re(vector_id)
        group_id = hash_value % L  # 哈希分组

        # 提取向量隐藏的信息
        # hidden_bit = extract_watermark_single_vector(
        #     vector, vector_id, min_pos_fraction=min_pos_fraction
        # )
        hidden_bit = extract_watermark_single_vector(vector, hash_value, position)

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
    #     if express_1_count> express_0_count:
    #         w = 1
    #     else:
    #         w = 0
    #     print(f"express {w}")

    return groups

def random_select_vectors(num_need_modi, vectors):
    if num_need_modi > len(vectors):
        raise ValueError("num_need_modi exceeds the number of available vectors.")

    # 使用随机采样从列表中选择向量
    selected_vectors = random.sample(vectors, num_need_modi)
    return selected_vectors

def watermark_embedding(file_path, top_k, strength, position, Standard_e, num_samples=None, watermark="00000"):
    train_data = load_data(file_path, num_samples)
    L = len(watermark)
    vector_num = len(train_data)
    #print(f"Embedding dimensions: {embedding_dims}")
    grouped_data = group_vectors_and_classify_bits(train_data, top_k, L, position)
    E = found_e(train_data, position, Standard_e)

    # # 用于存储修改后的向量
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
        carrier_vectors_id = random_select_vectors(num_need_modi, available_vectors)
        #print(num_need_modi, len(available_vectors), len(carrier_vectors_id))

        for id in carrier_vectors_id:
            all_carrier_vectors.append(id)
            vector = train_data[id]
            vector_id = generate_id_from_selected_dims(vector, top_k)
            hash_value = hash_re(vector_id)

            modified_vector = vector.copy()
            e = E[random.randint(0, vector_num - 1)]
            modified_vector[position] += e
            while extract_watermark_single_vector(modified_vector, hash_value, position) != watermark[i]:
                #print(extract_watermark_single_vector(modified_vector, hash_value), watermark[i])
                e = E[random.randint(0, vector_num - 1)]
                modified_vector[position] += e

            watermarked_data[id] = modified_vector
            modified_vectors.append(modified_vector)

    #print(f"Final data shape: {watermarked_data.shape}")
    watermark_length = len(watermark)
    return watermarked_data, watermark_length, all_carrier_vectors, train_data

def watermark_extraction(watermarked_data, top_k, watermark_length, position):
    L = watermark_length
    #print(embedding_dims)
    # 初始化分组容器
    grouped_data = group_vectors_and_classify_bits(watermarked_data, top_k, L, position)

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
    #print(extracted_watermark)
    return "".join(extracted_watermark)

# def adaptive_dele(data, p, accessibility_indices):
    
#     # 确定要删除的元素数量
#     num_to_remove = int(np.ceil(len(data) * p))
#     # 按照可达性指数降序排列索引
#     sorted_indices = np.argsort(accessibility_indices)[::-1]
    
#     # 获取要保留的索引
#     indices_to_keep = sorted_indices[num_to_remove:]
    
#     # 创建新的数据集，不包含要删除的元素
#     filtered_data = data[indices_to_keep]
    
#     return filtered_data

def random_dele(data, p):
    d = data.shape[1]

    # 确定要删除的元素数量
    num_to_remove = int(np.ceil(len(data) * p))
    
    n = len(data)
    # 获取所有索引
    all_indices = np.arange(len(data))
    
    # 随机选择要删除的索引
    indices_to_remove = np.random.choice(all_indices, size=n - num_to_remove, replace=False)
    
    # # 获取要保留的索引
    # mask = np.ones(len(data), dtype=bool)
    # mask[indices_to_remove] = False
    # indices_to_keep = all_indices[mask]
    
    # 创建新的数据集，不包含要删除的元素
    filtered_data = data[indices_to_remove]
    
    return filtered_data


def BER(wm, ex_wm):
    sum = 0
    for i in range(len(wm)):
        if wm[i] != ex_wm[i]:
            sum += 1
    #print(wm, ex_wm, sum / len(wm))
    return sum/ len(wm)



# position = 100
# top_k = 20
# strength = 0.55
# num_samples = 10000
# Standard_e = 2
# watermark = "001010010101001010010"
# #watermark = "10101"

# train_data = load_data(file_path, num_samples)
# d = train_data.shape[1]
# M = 8   # HNSW参数
# efConstruction = 50
# index = faiss.IndexHNSWFlat(d, M)
# index.hnsw.efConstruction = efConstruction
# index.add(train_data)
# index.hnsw.efSearch = 100
# offsets = faiss.vector_to_array(index.hnsw.offsets)
# neighbors = faiss.vector_to_array(index.hnsw.neighbors)
# accessibility_indices = watermarking.calculate_accessibility_indices_with_index_array(train_data, M, offsets, neighbors)


# watermarked_data, watermark_length, all_carrier_vectors, train_data = watermark_embedding(
#     file_path, top_k, strength, position, Standard_e, num_samples, watermark)
# #print(watermark_length)
# ex_wm = watermark_extraction(watermarked_data, top_k,  watermark_length, position)
# de_data = random_dele(watermarked_data, 0.8)
# ex_wm = watermark_extraction(de_data, top_k,  watermark_length, position)
# #print(watermark, ex_wm, len(watermark), len(ex_wm))
# ber = BER(watermark, ex_wm)
# print(ber)