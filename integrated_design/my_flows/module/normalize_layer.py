import numpy as np

def normalize(data):
    """
    对数据进行最小-最大归一化，将数据缩放到0到1之间。

    参数:
    data (numpy.ndarray): 需要归一化的数据，假设是二维数组，其中每一行是一个样本，每一列是一个特征。

    返回:
    numpy.ndarray: 归一化后的数据。
    """
    # 计算每列的最小值和最大值
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # 防止除零错误，加入一个很小的值（例如1e-8）
    range_vals = max_vals - min_vals + 1e-8

    # 归一化公式：(data - min_vals) / (max_vals - min_vals)
    normalized_data = (data - min_vals) / range_vals

    return normalized_data

# 测试样例
# data = np.array([[1, 2, 3],
#                  [4, 5, 6],
#                  [7, 8, 9]])
#
# normalized_data = normalize(data)
# print(normalized_data)