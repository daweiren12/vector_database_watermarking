import faiss
import numpy as np
import h5py
from collections import defaultdict
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
# 示例水印
wm = "111111111111111111111"
wm = "001010010101001010010"
#wm = "10"

# 设置随机数种子以确保结果可重复
random_seed = 40
np.random.seed(random_seed)

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"

def build_hnsw_index(data, d, M, ef_construction):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index

def load_data_from_back(file_path, data, num_additional_samples):
    new_data = []
    with h5py.File(file_path, 'r') as f:
        for i in range(len(f["train"])):
            print(i)
            if f['train'][i] not in data:
                new_data.append(f['train'][i])
                print(i)
                if len(new_data) == num_additional_samples:
                    break
    return new_data

# 1. 导入数据
def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data

train_data = load_data(file_path, num_samples=10000)

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

def adaptive_modify(data, p, accessibility_indices):
    # 确定要删除的元素数量
    num_to_modify = int(np.ceil(len(data) * p))
    
    # 按照可达性指数降序排列索引
    sorted_indices = np.argsort(accessibility_indices)[::-1]
    
    # 获取前 p 比例的索引
    indices_to_modify = sorted_indices[:num_to_modify]

    # 获取要修改的索引（根据可达性指数排序后的前 num_to_modify 个）
    #indices_to_modify = sorted_indices[:num_to_modify]

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
        #print(np.linalg.norm(data[idx] - modified_data[idx]))
    return modified_data

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
        modified_data[idx][dim_to_modify] = new_value  # 可以根据具体需求调整修改方式
        #print(data[idx][dim_to_modify], modified_data[idx][dim_to_modify])
    return modified_data

def create_hashable(data):
    """将 numpy 数组转换为可哈希的元组"""
    return set(tuple(row) for row in data)

def calculate_overlap_ratio(vec_group1, vec_group2):
    """
    计算两个向量组之间的重叠比例。

    参数:
    - vec_group1: 第一个向量组 (numpy.ndarray)
    - vec_group2: 第二个向量组 (numpy.ndarray)

    返回:
    - float: 重叠比例，范围从 0 到 1，表示完全不重叠到完全重叠
    """
    set1 = create_hashable(vec_group1)
    set2 = create_hashable(vec_group2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return 0.0  # 避免除以零
    
    overlap_ratio = len(intersection) / len(union)
    return overlap_ratio

def generate_vector(data, num_to_insert):
    # 获取数据的形状信息
    num_features = data.shape[1]
    
    # 计算每一列的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # 生成新的向量，假设它们符合原始数据的正态分布
    insert_data = np.random.normal(loc=mean, scale=std, size=(num_to_insert, num_features))
    
    return insert_data

def adaptive_insertion(data, p):
    # 确定要删除的元素数量
    num_to_insert = int(np.ceil(len(data) * (p)))
    insert_data = generate_vector(data, num_to_insert)
    
    #简单合并数据集
    merged_data = np.vstack((data, insert_data))
    #print(num_to_insert)
    overlap_ratio = calculate_overlap_ratio(data, merged_data)
    #print(overlap_ratio)
    return merged_data

#print(len(adaptive_insertion(train_data, 0.2)))
#print(len(adaptive_modify(train_data, 0.1)))

def BER(wm, ex_wm):
    sum = 0
    for i in range(len(wm)):
        if wm[i] != ex_wm[i]:
            sum += 1
    #print(wm, ex_wm, sum / len(wm))
    return sum/ len(wm)


import numpy as np
import matplotlib.pyplot as plt
import pickle

def strength_analysis(file_path, wm, Attack, filename='_roubust.pkl'):
    pl = [0.1, 0.2, 0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8]
    strl = [0.55, 0.6, 0.7, 0.8, 0.9, 1]
    times = 10  # 可以增加测试次数以获得更稳定的结果
    print(f"{Attack}, strength")
    # 存储结果的字典
    results = {strength: {p: [] for p in pl} for strength in strl}
    
    for strength in strl:
        for p in pl:
            ber_sum = 0
            for i in range(times):
                watermarked_data, watermark_length, all_carrier_vectors, train_data, accessibility_indices = watermarking.watermark_embedding_by_ai(
                    file_path, strength, 1, 10000, wm, random_seed=18
                )
                if Attack == "Adaptive Deletion":
                    de_watermarked_data = adaptive_dele(watermarked_data, p, accessibility_indices)
                elif Attack == "Random Deletion":
                    de_watermarked_data = random_dele(watermarked_data, p)
                elif Attack == "Adaptive Modification":
                    de_watermarked_data = adaptive_modify(watermarked_data, p, accessibility_indices)
                elif Attack == "Random Modification":
                    de_watermarked_data = random_modify(watermarked_data, p)
                ex_watermark = watermarking.watermark_extraction(
                    de_watermarked_data, watermark_length, random_seed=18)
                ber = BER(wm, ex_watermark)
                ber_sum += ber
                #print(f"Strength: {strength}, p: {p}, BER: {ber}")

            b = ber_sum / times 
            results[strength][p].append(b)
            print(f"Strength: {strength}, p: {p}, BER: {b}")
    
    # 计算平均BER
    avg_results = {strength: {p: np.mean(results[strength][p]) for p in pl} for strength in strl}
    Attack = Attack.replace(" ", "_")
    with open(Attack + "str" + filename, 'wb') as f:
        pickle.dump(avg_results, f)

def th_analysis(file_path, wm, Attack, filename='_roubust.pkl'):
    print(Attack)
    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pl = [0.1, 0.2, 0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8]
    thl = [0.2, 0.4, 0.6, 0.8, 1]
    times = 50  # 可以增加测试次数以获得更稳定的结果
    
    # 存储结果的字典
    results = {th: {p: [] for p in pl} for th in thl}
    
    for th in thl:
        for p in pl:
            ber_sum = 0
            for i in range(times):
                watermarked_data, watermark_length, all_carrier_vectors, train_data, accessibility_indices = watermarking.watermark_embedding_by_ai(
                    file_path, 0.7, th, 10000, wm, random_seed=18
                )
                if Attack == "Adaptive Deletion":
                    de_watermarked_data = adaptive_dele(watermarked_data, p, accessibility_indices)
                elif Attack == "Adaptive Modification":
                    de_watermarked_data = adaptive_modify(watermarked_data, p, accessibility_indices)
                elif Attack == "Adaptive Insertion":
                    de_watermarked_data = adaptive_insertion(watermarked_data, p)
                ex_watermark = watermarking.watermark_extraction(
                    de_watermarked_data, watermark_length, random_seed=18
                    )
                #print(ex_watermark)
                ber = BER(wm, ex_watermark)
                ber_sum += ber
                #print(f"Strength: {strength}, p: {p}, BER: {ber}")

            b = ber_sum / times 
            results[th][p].append(b)
            print(f"th: {th}, p: {p}, BER: {b}")
    
    # 计算平均BER
    avg_results = {th: {p: np.mean(results[th][p]) for p in pl} for th in thl}
    Attack = Attack.replace(" ", "_")
    with open(Attack + "th" + filename, 'wb') as f:
        pickle.dump(avg_results, f)

def load_ber_results(filename='ber_results.pkl'):
    try:
        with open(filename, 'rb') as f:
            avg_results = pickle.load(f)
        print(f"Successfully loaded results from {filename}")
        return avg_results
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def plot_str(Attack):
    Attack = Attack.replace(" ", "_")
    #filename = "C:\\Users\\USTC\\Desktop\\hnsw1\\roubust\\" + Attack + "_roubust.pkl"
    filename = "C:\\Users\\USTC\\Desktop\\hnsw1\\" + Attack + "str_roubust.pkl"
    #print(filename)
    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pl = [0.1, 0.2, 0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8]
    strl = [0.55, 0.6, 0.7, 0.8, 0.9, 1]
    avg_results = load_ber_results(filename)
    #print(avg_results)
    #plot_ber_results(avg_results, pl, strl)
    plt.figure(figsize=(10, 7.5))
    
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 26,  # 设置标题字体大小
        'axes.labelsize': 25,  # 设置轴标签字体大小
        'xtick.labelsize': 26,  # 设置x轴刻度字体大小
        'ytick.labelsize': 26,  # 设置y轴刻度字体大小
        'legend.fontsize': 20   # 设置图例字体大小
    })

    for strength in strl:
        ber_values = [avg_results[strength][p] for p in pl]
        plt.plot(pl, ber_values, marker='o', label=f's = {strength}')

    Attack = Attack.replace("_", " ")
    plt.title(f'Impact of {Attack} on Bit Error Rate')
    A = Attack.split(' ')[-1]
    plt.xlabel(f'{A} Proportion (p)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{Attack}_str.pdf")
    plt.show()

def plot_th(Attack):
    Attack = Attack.replace(" ", "_")
    #filename = "C:\\Users\\USTC\\Desktop\\hnsw1\\roubust\\" + Attack + "_roubust.pkl"
    filename = "C:\\Users\\USTC\\Desktop\\hnsw1\\" + Attack + "th_roubust.pkl"
    print(filename)
    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thl = [0.2, 0.4, 0.6, 0.8, 1]
    avg_results = load_ber_results(filename)
    plt.figure(figsize=(10, 7.5))
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 26,  # 设置标题字体大小
        'axes.labelsize': 25,  # 设置轴标签字体大小
        'xtick.labelsize': 26,  # 设置x轴刻度字体大小
        'ytick.labelsize': 26,  # 设置y轴刻度字体大小
        'legend.fontsize': 24   # 设置图例字体大小
    })
    for th in thl:
        ber_values = [avg_results[th][p] for p in pl]
        plt.plot(pl, ber_values, marker='o', label=f'th = {th}')

    Attack = Attack.replace("_", " ")
    #plt.title(f'Impact of {Attack} on Bit Error Rate')
    A = Attack.split(' ')[-1]
    plt.xlabel(f'{A} Proportion (p)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{Attack}_th.pdf")
    plt.show()

#Attack = "Adaptive Insertion"
#Attack = "Adaptive Modification"
Attack = "Adaptive Deletion"
#th_analysis(file_path, wm, Attack, filename='_roubust.pkl')
plot_th(Attack)
#strength_analysis(file_path, wm, Attack, filename='_roubust.pkl')
#plot_str(Attack)
