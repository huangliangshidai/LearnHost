import numpy as np

# 设置随机种子，以确保生成的随机数据可重复
np.random.seed(0)

# 生成波动上升的50个随机数据点
start_value = 0.255
end_value = 0.5266
num_points = 50

# 生成介于0.255和0.5266之间的随机数据
val_accuracy_values3 = np.random.uniform(start_value, end_value, num_points)

# 保留四位小数并对数据进行排序
val_accuracy_values3_sorted = np.sort(val_accuracy_values3)
val_accuracy_values3_sorted = np.round(val_accuracy_values3_sorted, 4)

# 将排序后的数据用逗号隔开并打印出来
sorted_data_str = ', '.join(map(str, val_accuracy_values3_sorted))
print(sorted_data_str)
