import h5py
import faiss
import numpy as np
from collections import defaultdict
import math
import random
import hashlib
from collections import Counter
import matplotlib.pyplot as plt
import watermarking

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"
alpha = 1.96

# 1. 导入数据
def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data

def generate_id_from_selected_dims(vector, k):
    id_parts = []
    for dim in range(k):
        value = vector[dim]
        if value < 0:
            value = abs(value)  # 处理负数，取绝对值
        highest_digit = str(int(value))[0]  # 提取十进制最高位
        id_parts.append(highest_digit)
    
    # 拼接为唯一标识符
    #unique_id = "".join(id_parts)
    unique_id = "".join(id_parts).strip()
    #unique_id = int(unique_id_str)
    return unique_id

def generate_key_cell_PK(train_data, nw, k):
    pk = []
    for i in range(nw):
        id = generate_id_from_selected_dims(train_data[i], k)
        pk.append(id)
    return pk

def divide_interval_randomly(p, random_seed):
    # 初始化两类区间的列表和当前长度计数器
    category_1 = []
    category_2 = []
    current_length_1 = 0
    current_length_2 = 0
    start = -p

    random.seed(random_seed)

    while start < p:
        # 随机选择段长，确保段长在合理范围内
        max_possible_length = min(p - start, p)  # 确保不会超出边界
        segment_length = random.uniform(1, max_possible_length)

        end = start + segment_length
        segment = (start, end)

        # 交替分配段落到两类中
        if current_length_1 <= current_length_2:
            category_1.append(segment)
            current_length_1 += segment_length
        else:
            category_2.append(segment)
            current_length_2 += segment_length

        # 检查是否有类的总长度达到或超过 p
        if current_length_1 >= p or current_length_2 >= p:
            break

        start = end

    # 调整达到或超过 p 的那一类的最后一段
    if current_length_1 >= p:
        last_segment = category_1.pop()
        last_start, _ = last_segment
        new_end = last_start + (p - sum(end - start for start, end in category_1))
        adjusted_segment = (last_start, new_end)
        category_1.append(adjusted_segment)

        # 剩余区间全都是另一类的最后一段
        remaining_start = new_end
        remaining_end = p
        category_2.append((remaining_start, remaining_end))

    elif current_length_2 >= p:
        last_segment = category_2.pop()
        last_start, _ = last_segment
        new_end = last_start + (p - sum(end - start for start, end in category_2))
        adjusted_segment = (last_start, new_end)
        category_2.append(adjusted_segment)

        # 剩余区间全都是另一类的最后一段
        remaining_start = new_end
        remaining_end = p
        category_1.append((remaining_start, remaining_end))

    return category_1, category_2

def choose_random_number_from_category(category):
    # 随机选择一个区间
    chosen_segment = random.choice(category)
    start, end = chosen_segment

    # 在选定的区间内随机选择一个数
    random_number = random.uniform(start, end)

    return random_number

def watermark_embed(nw, p, k, position, num_samples):
    train_data = load_data(file_path, num_samples)
    i_tuple = []
    watermarked_data = train_data.copy()
    for i in range(nw):
        id = generate_id_from_selected_dims(watermarked_data[i], k)
        category_1, category_2 = divide_interval_randomly(p, id)
        random_number = choose_random_number_from_category(category_1)
        watermarked_data[i][position] += random_number
        i_tuple.append(i)
    return train_data, i_tuple, watermarked_data

def Matching_vector(data, i_tuple, watermarked_data, k):
    match_index = []
    pk_tuple = []
    for i in i_tuple:
        id = generate_id_from_selected_dims(data[i], k)
        pk_tuple.append([i, id])
        # id1 = generate_id_from_selected_dims(watermarked_data[i], k)
        # f = 0
        # for c in range(len(id)):
        #     print(id[c], id1[c], c)
        #     if id[c] != id1[c]:
        #         f = 1
        #         print(data[i][c], watermarked_data[i][c])
        #         break
        # print(id1, id, i, f, len(id), len(id1))
    #print(len(watermarked_data))
    for i in range(len(watermarked_data)):
        id = generate_id_from_selected_dims(watermarked_data[i], k)
        for pk in pk_tuple:
            if id == pk[1]:
                match_index.append([pk[0], id, i])
                break
        # if len(match_index) == len(i_tuple):
        #     break
    #print(match_index)
    return match_index

def find_category_of_number(number, category_1, category_2):
    # 检查 number 是否在 category_1 的区间中
    for start, end in category_1:
        if start <= number <= end:
            return 0

    # 检查 number 是否在 category_2 的区间中
    for start, end in category_2:
        if start <= number <= end:
            return 1

    # 如果不在任何一个类别中
    return "Not in any category"

def watermark_extrction(train_data, watermarked_data, i_tuple, p, k, position):
    match_index = Matching_vector(train_data, i_tuple, watermarked_data, k)
    ng = 0
    for match in match_index:
        i = match[0]
        id = match[1]
        j = match[2]
        category_1, category_2 = divide_interval_randomly(p, id)
        orig_value = train_data[i][position]
        wm_value = watermarked_data[j][position]
        difference = wm_value - orig_value
        wm = find_category_of_number(difference, category_1, category_2)
        if wm == 0:
            ng += 1
    nw = len(i_tuple)
    z = 2 * (ng - 0.5 * nw) / math.sqrt(nw)
    print(z, nw, ng)
    if z > alpha:
        return True
    else:
        return False

def random_modify(data, p):
    # 确定要修改的元素数量
    num_to_modify = int(np.ceil(len(data) * p))
    
    # 获取所有索引
    all_indices = np.arange(len(data))
    
    # 随机选择要修改的索引
    indices_to_modify = np.random.choice(all_indices, size=num_to_modify, replace=False)
    
    # 创建数据的副本以避免修改原始数据
    modified_data = np.copy(data)
    
    dim_to_modify = np.random.randint(0, data.shape[1])

    min_val = np.min(data[:, dim_to_modify])
    max_val = np.max(data[:, dim_to_modify])
    
    # 对每个选中的向量在指定维度上进行修改
    for idx in indices_to_modify:
        # 在目标维度的取值范围内随机选择一个新值
        new_value = np.random.uniform(min_val, max_val)
        dim_to_modify = np.random.randint(0, data.shape[1])
        for i in range(30):
            #modified_data[idx][dim_to_modify + i] = new_value
            modified_data[idx][(dim_to_modify + i) % data.shape[1]] = new_value  # 可以根据具体需求调整修改方式
        #print(data[idx][dim_to_modify], modified_data[idx][dim_to_modify])
    return modified_data

# nw = 100
# k = 40
# p = 5
# position = 100
# num_samples = 10000
# train_data, i_tuple, watermarked_data = watermark_embed(nw, p, k, position, num_samples)
# watermark_extrction(train_data, watermarked_data, i_tuple, p, k, position)

# it = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
# for i in it:
#     filtered_data = random_modify(watermarked_data, i)
#     #filtered_data = adaptive_modify(watermarked_data, i, accessibility_indices)
#     #print(len(filtered_data))
#     match_index = Matching_vector(train_data, i_tuple, filtered_data, k)
#     print(len(match_index), 1 - len(match_index) / nw)