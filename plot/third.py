import matplotlib.pyplot as plt
import numpy as np


# 生成随机点数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
num_points = 250  # 点的数量
x = np.random.uniform(0, 10, num_points)  # x坐标
y = np.random.uniform(0, 10, num_points)  # y坐标

# 分组（4组）
group_ids = np.random.choice(4, num_points)

x = x[group_ids == 1]
y = y[group_ids == 1]

print(len(x), len(y))
# 将点随机分为两类（比例接近）
labels = np.random.choice([0, 1], len(x), p=[0.5, 0.5])

# 定义新的颜色，用来区分 1 向量和 0 向量
class_colors = ['blue', 'green']  # 1向量和0向量的颜色
markers = ['s', '^']
# 绘图
plt.figure(figsize=(6, 8))
for label in [0, 1]:
    plt.scatter(x[labels == label], y[labels == label], 
                color=class_colors[label], marker=markers[label], alpha=0.8)

# 添加边框
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# 移除坐标轴
plt.xticks([])
plt.yticks([])

# 添加图例
# plt.legend()
# plt.title('Third Image: Single Group with Two Classes')
plt.savefig(f"third.pdf")
plt.show()
