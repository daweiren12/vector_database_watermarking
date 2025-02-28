import faiss
import numpy as np
import h5py
from collections import defaultdict
import watermarking

# 示例水印
wm = "1010101010101010"
num_samples = 5000
# 设置随机数种子以确保结果可重复
random_seed = 40
np.random.seed(random_seed)

# 数据文件路径和索引文件路径
file_path = "sift-128-euclidean.hdf5"
index_file = "hnsw_index.faiss"


# 执行水印嵌入的整体流程
#有策略选点
#selected_neighbors_list, wmed_selected_neighbors_list, watermarked_data, train_data = watermark.watermark_embedding_pipeline_with_hl(al, bl,file_path, index_file, max_neighbors_to_find = 50, num_samples=60000, watermark=wm)
#随机选择部分点
watermarked_data, watermark_length, all_carrier_vectors, train_data = watermarking.watermark_embedding_by_M(file_path, strength = 0.8, th = 0.6, num_samples=10000, watermark="10101", random_seed = 22)
#watermarked_data, watermark_length, all_carrier_vectors, train_data  = wm1.watermark_embedding(file_path, strength = 0.7, num_samples=10000, watermark="10101", random_seed = 22)
# 打印结果
# print(len(selected_neighbors_list))
# print(f"选定的含水印真向量ID: {selected_neighbors_list}")
# print(f"提取的最终水印: {extracted_watermark}")


d = train_data.shape[1]

# 2. 构建HNSW索引
M = 8
ef_construction = 100

def build_hnsw_index(data):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index

dis = []
for i in range(len(train_data)):
    dis.append(np.linalg.norm(watermarked_data[i] - train_data[i]))
print(np.mean(dis))

index_train = build_hnsw_index(train_data)
index_watermarked = build_hnsw_index(watermarked_data)

# 3. 查询并记录结果
k = 100  # 最近邻数量

def record_query_results(index, data, vectors):
    query_results = defaultdict(list)
    for query_idx, query in enumerate(data):
        _, I = index.search(query.reshape(1, -1), k)
        for neighbor in I.flatten():
            if neighbor in vectors:
                query_results[neighbor].append(query_idx)
    return query_results

train_query_results = record_query_results(index_train, train_data, all_carrier_vectors)
print("done1")
watermarked_query_results = record_query_results(index_watermarked, watermarked_data, all_carrier_vectors)
print("done")

# 4. 比较结果
comparison_results = {}

for neighbor in all_carrier_vectors:
    train_queries = set(train_query_results.get(neighbor, []))
    watermarked_queries = set(watermarked_query_results.get(neighbor, []))

    common_queries = train_queries & watermarked_queries
    missed_queries = train_queries - watermarked_queries
    extra_queries = watermarked_queries - train_queries

    #print(len(train_queries), len(watermarked_queries))
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
    # print(f"Neighbor {neighbor}:")
    # print(f"  Common Queries: {results['common_queries']}")
    # print(f"  Missed Queries: {results['missed_queries']}")
    # print(f"  Extra Queries: {results['extra_queries']}")
    q_sum += len(results['train_queries'])
    if len(results['train_queries']) != 0:
        mr = len(results['missed_queries']) / len(results['train_queries'])
    else:
        mr = 0
    if len(results['watermarked_queries']) != 0:
        fr = len(results['extra_queries']) / len(results['watermarked_queries'])
    else:
        fr = 0
    # print(f"  Missed detection rate: {mr}")
    # print(f"  False positive rate: {fr}")
    miss += mr
    fals += fr
    miss_sum += len(results['missed_queries'])
    fals_sum += len(results['extra_queries'])
    #print("\n")
print(f"average Missed detection rate:{miss / len(all_carrier_vectors)}")
print(f"average False positive rate:{fals / len(all_carrier_vectors)}")
print(f"average Missed detection sum:{miss_sum / len(all_carrier_vectors)}")
print(f"average False positive sum:{fals_sum / len(all_carrier_vectors)}")
print(f"average qeury count:{q_sum / len(all_carrier_vectors)}")
