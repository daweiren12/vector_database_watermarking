import faiss
import numpy as np
import h5py
from collections import defaultdict
import sys
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
wm = "1010101010101010"

# 设置随机数种子以确保结果可重复
random_seed = 40
np.random.seed(random_seed)

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"
index_file = "hnsw_index.faiss"
#file_path = "mnist-784-euclidean.hdf5"


import matplotlib.pyplot as plt

import pickle

def save_results(results, filename):
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

def load_results(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def build_hnsw_index(data, d, M, ef_construction):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index

def record_query_results(index, data, all_carrier_vectors, k):
    query_results = defaultdict(list)
    for query_idx, query in enumerate(data):
        #Sprint(query_idx)
        _, I = index.search(query.reshape(1, -1), k)
        for neighbor in I.flatten():
            if neighbor in all_carrier_vectors:
                query_results[neighbor].append(query_idx)
    return query_results

def load_data(file_path, num_samples=None):
    """从HDF5文件中加载数据"""
    with h5py.File(file_path, 'r') as f:
        if num_samples is None:
            train_data = f['train'][:]
        else:
            train_data = f['train'][:num_samples]
    return train_data

def ca(train_data, watermarked_data, all_carrier_vectors, k):
    d = train_data.shape[1]
    M = 8
    ef_construction = 100
    index_train = build_hnsw_index(train_data, d, M, ef_construction)
    index_watermarked = build_hnsw_index(watermarked_data, d, M, ef_construction)

    train_query_results = record_query_results(index_train, train_data, all_carrier_vectors, k)
    #print("done1")
    watermarked_query_results = record_query_results(index_watermarked, watermarked_data, all_carrier_vectors, k)
    #print("done")

    comparison_results = {}

    for neighbor in all_carrier_vectors:
        train_queries = set(train_query_results.get(neighbor, []))
        watermarked_queries = set(watermarked_query_results.get(neighbor, []))

        common_queries = train_queries & watermarked_queries
        missed_queries = train_queries - watermarked_queries
        extra_queries = watermarked_queries - train_queries

        comparison_results[neighbor] = {
            "train_queries": train_queries,
            "watermarked_queries": watermarked_queries,
            "common_queries": common_queries,
            "missed_queries": missed_queries,
            "extra_queries": extra_queries
        }
        # 输出比较结果
    miss = 0
    fals = 0
    miss_sum = 0
    fals_sum = 0
    q_sum = 0
    for neighbor, results in comparison_results.items():
        q_sum += len(results['train_queries'])
        if len(results['train_queries']) != 0:
            mr = len(results['missed_queries']) / len(results['train_queries'])
        else:
            mr = 0
        if len(results['watermarked_queries']) != 0:
            fr = len(results['extra_queries']) / len(results['watermarked_queries'])
        else:
            fr = 0
        miss += mr
        fals += fr
        miss_sum += len(results['missed_queries'])
        fals_sum += len(results['extra_queries'])
    # print(f"average Missed detection rate:{miss / len(all_carrier_vectors)}")
    # print(f"average False positive rate:{fals / len(all_carrier_vectors)}")
    # print(f"average Missed detection sum:{miss_sum / len(all_carrier_vectors)}")
    # print(f"average False positive sum:{fals_sum / len(all_carrier_vectors)}")
    # print(f"average qeury count:{q_sum / len(all_carrier_vectors)}")
    ave_miss = miss_sum / len(all_carrier_vectors)
    ave_false = fals_sum / len(all_carrier_vectors)
    before = len(train_queries)
    after = len(watermarked_queries)
    return ave_miss, ave_false, before, after

def test_random(save_to_file=True, filename='results11.pkl', filename1='results_before.pkl'):
    k = 100
    thl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    # 存储结果的字典
    results = {'ave_miss': {th: [] for th in thl}, 'ave_false': {th: [] for th in thl}}
    results1 = {'ave_before': {th: [] for th in thl}, 'ave_after': {th: [] for th in thl}}
    times = 3
    for th in thl:
            sum_mis = 0
            sum_false = 0
            sum_before = 0
            sum_after = 0
            for i in range(times):
                watermarked_data, watermark_length, all_carrier_vectors, train_data, accessibility_indices = watermarking.watermark_embedding_by_ai(
    file_path, 0.55, th, num_samples = 20000, watermark="10101", random_seed = 201
    )
                ave_miss, ave_false, before, after = ca(train_data, watermarked_data, all_carrier_vectors, 100)
                print(ave_false,ave_miss)
                sum_mis += ave_miss
                sum_false += ave_false
                sum_before += before
                sum_after += after
                #print(f"alpha = {al}, beta = {bl}, ave_false = {ave_false}, ave_miss = {ave_miss}")
            ave_miss = sum_mis / times
            ave_false = sum_false / times
            ave_before = sum_before / times
            ave_after = sum_after / times
            #print(al,bl,ave_false,ave_miss)
            print(f"th = {th}, ave_false = {ave_false}, ave_miss = {ave_miss}, ave_before = {ave_before}, ave_after = {ave_after}")
            results['ave_miss'][th].append(ave_miss)
            results['ave_false'][th].append(ave_false)
            results1['ave_before'][th].append(ave_before)
            results1['ave_after'][th].append(ave_after)
    
    if save_to_file:
        save_results(results, filename)
    
    if save_to_file:
        save_results(results1, filename1)
    
    return thl, results, results1

import matplotlib.pyplot as plt

def plot_results(thl, results):
    # 提取数据用于绘图
    ave_misses = [np.mean(results['ave_miss'][th]) for th in thl]
    ave_falses = [np.mean(results['ave_false'][th]) for th in thl]

    # 创建图表
    plt.figure(figsize=(10.8, 7.5))
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 26,  # 设置标题字体大小
        'axes.labelsize': 25,  # 设置轴标签字体大小
        'xtick.labelsize': 26,  # 设置x轴刻度字体大小
        'ytick.labelsize': 26,  # 设置y轴刻度字体大小
        'legend.fontsize': 24   # 设置图例字体大小
    })
    # 绘制两条线：一条代表遗漏查询次数，另一条代表错误查询次数
    plt.plot(thl, ave_misses, marker='o', label='Average Missed Queries')
    plt.plot(thl, ave_falses, marker='s', label='Average False Queries')

    # 添加标题和标签
    plt.title('Average Missed and False Queries vs Threshold')
    plt.xlabel('Threshold (th)')
    plt.ylabel('Average Query Count')
    
    # 添加图例
    plt.legend()
    
    # 设置网格
    plt.grid(True)
    plt.savefig(f"Threshold1.pdf")
    # 显示图表
    plt.show()

def plot_results_query(thl, results):
    # 提取数据用于绘图
    ave_misses = [np.mean(results['ave_before'][th]) for th in thl]
    ave_falses = [np.mean(results['ave_after'][th]) for th in thl]

    # 创建图表
    plt.figure(figsize=(10.8, 7.5))
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 26,  # 设置标题字体大小
        'axes.labelsize': 25,  # 设置轴标签字体大小
        'xtick.labelsize': 26,  # 设置x轴刻度字体大小
        'ytick.labelsize': 26,  # 设置y轴刻度字体大小
        'legend.fontsize': 24   # 设置图例字体大小
    })
    # 绘制两条线：一条代表遗漏查询次数，另一条代表错误查询次数
    plt.plot(thl, ave_misses, marker='o', label='Average Queries Before embed')
    plt.plot(thl, ave_falses, marker='s', label='Average Queries after embed')

    # 添加标题和标签
    #plt.title('Average Missed and False Queries vs Threshold')
    plt.xlabel('Threshold (th)')
    plt.ylabel('Average Query Count')
    
    # 添加图例
    plt.legend()
    
    # 设置网格
    plt.grid(True)
    plt.savefig(f"QueryCount.pdf")
    # 显示图表
    plt.show()

test_random()
thl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#result = load_results("C:\\Users\\USTC\Desktop\\hnsw1\\impact\\results1.pkl")
result = load_results("C:\\Users\\USTC\Desktop\\hnsw1\\results_before.pkl")
#plot_results(thl, result)
plot_results_query(thl, result)

# watermarked_data, watermark_length, all_carrier_vectors, train_data = watermarking.watermark_embedding_by_ai(
#     file_path, strength = 0.8, th = 0.6, num_samples=1000, watermark="10101", random_seed = 22
#     )
# print(train_data.shape)
# ca(train_data, watermarked_data, all_carrier_vectors, 100)
#test_random()

# thl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# filename = 'results.pkl'
# results = load_results(filename)
# plot_results(thl, results)