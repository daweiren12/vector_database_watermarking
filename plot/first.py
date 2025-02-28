import matplotlib.pyplot as plt
import numpy as np

# 生成随机点数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
num_points = 250  # 点的数量
x = np.random.uniform(0, 10, num_points)  # x坐标
y = np.random.uniform(0, 10, num_points)  # y坐标

# 创建二维散点图
plt.figure(figsize=(6, 8))
plt.scatter(x, y, color='black', alpha=0.7)  # 黑色点代表向量
#plt.title("Before Grouping: Random Vector Points", fontsize=18)
# plt.xlabel("Dimension 1", fontsize=12)
# plt.ylabel("Dimension 2", fontsize=12)
plt.grid(alpha=0.3)
# 去掉坐标轴
ax = plt.gca()  # 获取当前轴对象
ax.set_xticks([])  # 去掉x轴刻度
ax.set_yticks([])  # 去掉y轴刻度
ax.spines['top'].set_visible(True)    # 显示上边框
ax.spines['right'].set_visible(True)  # 显示右边框
ax.spines['left'].set_visible(True)   # 显示左边框
ax.spines['bottom'].set_visible(True) # 显示下边框

plt.savefig(f"first.pdf")
plt.show()
