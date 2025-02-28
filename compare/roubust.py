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
import tabularMark
import SCPW
# 假设我们已经定义了前面提到的函数和全局变量
#num_samples = 50000
M = 8 
file_path = parent_directory / "sift-128-euclidean.hdf5"
# 示例水印
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
        dim_to_modify = np.random.randint(0, data.shape[1])
        for i in range(30):
            #modified_data[idx][dim_to_modify + i] = new_value
            modified_data[idx][(dim_to_modify + i) % data.shape[1]] = new_value  # 可以根据具体需求调整修改方式
        #print(data[idx][dim_to_modify], modified_data[idx][dim_to_modify])
    return modified_data

def BER(wm, ex_wm):
    sum = 0
    for i in range(len(wm)):
        if wm[i] != ex_wm[i]:
            sum += 1
    #print(wm, ex_wm, sum / len(wm))
    return sum/ len(wm)

def random_dele(data, p):
    d = data.shape[1]

    # 确定要删除的元素数量
    num_to_remove = int(np.ceil(len(data) * p))
    
    n = len(data)
    # 获取所有索引
    all_indices = np.arange(len(data))
    
    # 随机选择要删除的索引
    indices_to_remove = np.random.choice(all_indices, size=n - num_to_remove, replace=False)
    # 创建新的数据集，不包含要删除的元素
    filtered_data = data[indices_to_remove]
    
    return filtered_data


import numpy as np
import matplotlib.pyplot as plt
import pickle


def dele_analysis(file_path, filename='compare_dele_robust.pkl'):
    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    times = 10  # 可以增加测试次数以获得更稳定的结果
    
    watermark = "001010010101001010010"
    num_samples = 10000
    th = 1

    top_k = 20
    strength = 0.54
    position = 100
    Standard_e = 1.8

    nw = 100
    p = 4
    dim = 40

    # 定义方案名称
    scheme_names = ['TVP', 'RS', 'SCPW', 'TabularMark']

    # 存储结果的字典
    results = {name: {'ber': [], 'match_propotion': []} for name in scheme_names}

    for i, scheme_name in enumerate(scheme_names):
        for p_val in pl:
            ber_sum = 0
            match_propotion_sum = 0
            for j in range(times):
                if scheme_name == 'TVP':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data, accessibility_indices = watermarking.watermark_embedding_by_ai(
                        file_path, strength, th, num_samples, watermark, random_seed=22
                    )
                    de_watermarked_data = adaptive_dele(watermarked_data, p_val, accessibility_indices)
                    ex_watermark = watermarking.watermark_extraction(de_watermarked_data, watermark_length, random_seed=22)

                elif scheme_name == 'RS':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data = watermarking.watermark_embedding(
                        file_path, strength, num_samples, watermark, random_seed=22
                    )
                    de_watermarked_data = random_dele(watermarked_data, p_val)
                    ex_watermark = watermarking.watermark_extraction(de_watermarked_data, watermark_length, random_seed=22)

                elif scheme_name == 'SCPW':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data = SCPW.watermark_embedding(
                        file_path, top_k, strength, position, Standard_e, num_samples, watermark
                    )
                    de_watermarked_data = random_dele(watermarked_data, p_val)
                    ex_watermark = SCPW.watermark_extraction(de_watermarked_data, top_k, watermark_length, position)

                elif scheme_name == 'TabularMark':
                    train_data, all_carrier_vectors, watermarked_data = tabularMark.watermark_embed(nw, p, dim, position, num_samples)
                    de_watermarked_data = random_dele(watermarked_data, p_val)
                    match_index = tabularMark.Matching_vector(train_data, all_carrier_vectors, de_watermarked_data, dim)
                    match_propotion = 1 - len(match_index) / nw
                    match_propotion_sum += match_propotion
                    continue  # Skip BER calculation for TabularMark

                ber = BER(watermark, ex_watermark)
                ber_sum += ber
                print(f"{scheme_name}, p={p_val}, BER: {ber}")

            if scheme_name == 'TabularMark':
                avg_match_propotion = match_propotion_sum / times
                results[scheme_name]['match_propotion'].append(avg_match_propotion)
                print(f"TabularMark, p={p_val}, Average Match Proportion: {avg_match_propotion}")
            else:
                avg_ber = ber_sum / times
                results[scheme_name]['ber'].append(avg_ber)
                print(f"{scheme_name}, p={p_val}, Average BER: {avg_ber}")

    # 保存结果到本地文件
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def modi_analysis(file_path, filename='compare_modi_robust.pkl'):
    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    times = 20  # 可以增加测试次数以获得更稳定的结果
    
    watermark = "001010010101001010010"
    num_samples = 10000
    th = 1

    top_k = 20
    strength = 0.55
    position = 100
    Standard_e = 1.8

    nw = 100
    p = 4
    dim = 40

    # 定义方案名称
    scheme_names = ['TVP', 'RS', 'SCPW', 'TabularMark']

    # 存储结果的字典
    results = {name: {'ber': [], 'match_propotion': []} for name in scheme_names}

    for i, scheme_name in enumerate(scheme_names):
        for p_val in pl:
            ber_sum = 0
            match_propotion_sum = 0
            for j in range(times):
                if scheme_name == 'TVP':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data, accessibility_indices = watermarking.watermark_embedding_by_ai(
                        file_path, strength, th, num_samples, watermark, random_seed=22
                    )
                    de_watermarked_data = adaptive_modify(watermarked_data, p_val, accessibility_indices)
                    ex_watermark = watermarking.watermark_extraction(de_watermarked_data, watermark_length, random_seed=22)

                elif scheme_name == 'RS':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data = watermarking.watermark_embedding(
                        file_path, strength, num_samples, watermark, random_seed=22
                    )
                    de_watermarked_data = random_modify(watermarked_data, p_val)
                    ex_watermark = watermarking.watermark_extraction(de_watermarked_data, watermark_length, random_seed=22)

                elif scheme_name == 'SCPW':
                    watermarked_data, watermark_length, all_carrier_vectors, train_data = SCPW.watermark_embedding(
                        file_path, top_k, strength, position, Standard_e, num_samples, watermark
                    )
                    de_watermarked_data = random_modify(watermarked_data, p_val)
                    ex_watermark = SCPW.watermark_extraction(de_watermarked_data, top_k, watermark_length, position)

                elif scheme_name == 'TabularMark':
                    train_data, all_carrier_vectors, watermarked_data = tabularMark.watermark_embed(nw, p, dim, position, num_samples)
                    de_watermarked_data = random_modify(watermarked_data, p_val)
                    match_index = tabularMark.Matching_vector(train_data, all_carrier_vectors, de_watermarked_data, dim)
                    match_propotion = 1 - len(match_index) / nw
                    match_propotion_sum += match_propotion
                    continue  # Skip BER calculation for TabularMark

                ber = BER(watermark, ex_watermark)
                ber_sum += ber
                #print(f"{scheme_name}, p={p_val}, BER: {ber}")

            if scheme_name == 'TabularMark':
                avg_match_propotion = match_propotion_sum / times
                results[scheme_name]['match_propotion'].append(avg_match_propotion)
                print(f"TabularMark, p={p_val}, Average Match Proportion: {avg_match_propotion}")
            else:
                avg_ber = ber_sum / times
                results[scheme_name]['ber'].append(avg_ber)
                print(f"{scheme_name}, p={p_val}, Average BER: {avg_ber}")

    # 保存结果到本地文件
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def plot_results(filename, Attack):
    # 读取结果文件
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    pl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 16,  # 设置标题字体大小
            'axes.labelsize': 17,  # 设置轴标签字体大小
            'xtick.labelsize': 16,  # 设置x轴刻度字体大小
            'ytick.labelsize': 16,  # 设置y轴刻度字体大小
            'legend.fontsize': 15   # 设置图例字体大小
        })
    # 创建图表
    #fig, ax = plt.subplots(figsize=(10, 7.5))
    fig, ax = plt.subplots(figsize=(8, 6))
    for scheme_name, data in results.items():
        if scheme_name != 'TabularMark':
            ax.plot(pl, data['ber'], marker='o',label=f'{scheme_name} (BER)')
        else:
            ax.plot(pl, data['match_propotion'], marker='o', label=f'{scheme_name} (MP)', linestyle='--')

    ax.set_xlabel(f'{Attack} Proportion (p)')
    ax.set_ylabel('BER or MP')
    ax.set_title(f'Watermark Extraction Performance under {Attack} Attack')
    ax.legend()
    plt.savefig(f"{filename}.pdf")
    plt.show()

Attack = "Modification"
#Attack = "Deletion"
#dele_analysis(file_path)
#modi_analysis(file_path)
plot_results('C:\\Users\\USTC\\Desktop\\hnsw1\\compare_modi_robust.pkl', Attack)