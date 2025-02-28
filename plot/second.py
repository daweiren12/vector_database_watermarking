import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以便复现
np.random.seed(42)

# 生成随机点数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
num_points = 250  # 点的数量
x = np.random.uniform(0, 10, num_points)  # x坐标
y = np.random.uniform(0, 10, num_points)  # y坐标


# 分组（4组）
group_ids = np.random.choice(4, num_points)

# 定义组的颜色
colors = ['red', 'blue', 'green', 'orange']

# 创建二维散点图
plt.figure(figsize=(6, 8))
for group in range(4):
    plt.scatter(x[group_ids == group], y[group_ids == group], color=colors[group], label=f'Group {group+1}', alpha=0.7)

# 添加边框
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# 移除坐标轴
plt.xticks([])
plt.yticks([])

# 添加图例
#plt.legend()
#plt.title('Second Image: Points Divided into 4 Groups')
plt.savefig(f"second.pdf")
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 设置随机种子以便复现
# np.random.seed(42)

# # 生成随机点数据
# num_points = 250  # 点的数量
# x = np.random.uniform(0, 10, num_points)  # x坐标
# y = np.random.uniform(0, 10, num_points)  # y坐标

# # 分组（4组）
# group_ids = np.random.choice(4, num_points)

# # 定义组的标记形状
# markers = ['o', 's', '^', 'v']  # 圆形, 正方形, 向上三角形, 向下三角形

# # 创建二维散点图
# plt.figure(figsize=(6, 8))
# for group in range(4):
#     plt.scatter(x[group_ids == group], 
#                 y[group_ids == group], 
#                 color='black',  # 统一为黑色
#                 marker=markers[group],  # 不同组有不同的标记形状
                 
#                 alpha=0.7)

# # 添加边框
# plt.gca().spines['top'].set_visible(True)
# plt.gca().spines['right'].set_visible(True)
# plt.gca().spines['bottom'].set_visible(True)
# plt.gca().spines['left'].set_visible(True)

# # 移除坐标轴
# plt.xticks([])
# plt.yticks([])

# # 添加图例
# #plt.legend()
# #plt.title('Second Image: Points Divided into 4 Groups with Different Markers')

# # 保存图形
# plt.savefig("second_different_markers.pdf")

# # 显示图形
# plt.show()
