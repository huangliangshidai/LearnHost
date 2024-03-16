import matplotlib.pyplot as plt

# 输入数据
segments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
predictions = ['class_1', 'class_1', 'class_1', 'class_1', 'class_1', 'class_2', 'class_2', 'class_2', 'class_2', 'class_2']

# 将类别映射到数字，用于在y轴上进行绘制
class_mapping = {'class_0': 0, 'class_1': 1, 'class_2': 2}
y_values = [class_mapping[prediction] for prediction in predictions]

# 创建颜色映射
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']

# 创建一个颜色列表，每个类别对应一个颜色
point_colors = [colors[class_idx] for class_idx in y_values]

# 绘制点状图
plt.scatter(segments, y_values, marker='o', c=point_colors)

# 设置y轴标签
plt.yticks(list(class_mapping.values()), list(class_mapping.keys()))

# 设置x轴和y轴标签
plt.xlabel('Segment')
plt.ylabel('Class')

# 显示图形
plt.show()
