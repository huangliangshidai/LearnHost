import matplotlib.pyplot as plt

# 输入数据
segments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
predictions = ['class_9', 'class_9', 'class_1', 'class_1', 'class_1', 'class_5', 'class_9', 'class_5', 'class_9', 'class_9']

# 将类别映射到数字，用于在y轴上进行绘制
class_mapping = {'class_0': 0, 'class_1': 1, 'class_2': 2, 'class_3': 3, 'class_4': 4, 'class_5': 5, 'class_6': 6, 'class_7': 7, 'class_8': 8, 'class_9': 9}
#
y_values = [class_mapping[prediction] for prediction in predictions]

# 创建颜色映射
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']

# 选择一种颜色
line_color = 'black'

# 初始化起始点
x_start, y_start = segments[0], y_values[0]

# 绘制直线图
for i in range(1, len(segments)):
    x_end, y_end = segments[i], y_values[i]
    # 计算水平线段的y值，直接取片段的终止类别的y值
    y_value = y_end
    # 绘制水平线段
    plt.plot([x_start, x_end], [y_value, y_value], color=line_color)
    x_start, y_start = x_end, y_end

# 设置y轴标签
plt.yticks(list(class_mapping.values()), list(class_mapping.keys()))

# 设置x轴和y轴标签
# 设置x轴和y轴标签
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Class', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# 设置x轴范围
plt.xlim(segments[0], segments[-1])

# 显示图形
plt.show()
