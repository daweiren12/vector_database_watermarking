import random

def divide_interval_randomly(p):
    """
    将区间 [-p, p] 划分为多段，并将这些段分为两类，
    每一类的总长度是区间长度的一半，引入随机性进行划分。
    
    参数:
    - p: 正整数，表示区间的范围
    
    返回:
    - 类别1的区间列表，类别2的区间列表
    """
    if p <= 0 or not isinstance(p, int):
        raise ValueError("p 必须是一个正整数")

    # 初始化两类区间的列表和当前长度计数器
    category_1 = []
    category_2 = []
    current_length_1 = 0
    current_length_2 = 0
    start = -p
    random.seed(20)
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
    """
    从给定的类别区间中随机选择一个数。
    
    参数:
    - category: 包含区间的列表，每个区间是一个 (start, end) 的元组
    
    返回:
    - 随机选择的数
    """
    if not category or not all(isinstance(segment, tuple) and len(segment) == 2 for segment in category):
        raise ValueError("类别必须是非空的区间列表，每个区间是 (start, end) 的元组")

    # 随机选择一个区间
    chosen_segment = random.choice(category)
    start, end = chosen_segment

    # 在选定的区间内随机选择一个数
    random_number = random.uniform(start, end)

    return random_number

# 示例使用
p = 5
category_1, category_2 = divide_interval_randomly(p)
print("Category 1:", category_1)
print("Category 2:", category_2)

# 验证每类总长度是否为 p
print("Total length of Category 1:", sum(end - start for start, end in category_1))
print("Total length of Category 2:", sum(end - start for start, end in category_2))

# 从 category_1 中随机选择一个数
random_number = choose_random_number_from_category(category_1)
print("Random number from Category 1:", random_number)
