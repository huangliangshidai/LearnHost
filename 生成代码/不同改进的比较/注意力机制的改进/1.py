import matplotlib.pyplot as plt
import numpy as np

# 混淆矩阵数据
confusion_matrix = np.array([[200, 0, 0, 11, 0, 1, 50, 16, 0, 0],
                             [0, 265, 6, 0, 0, 0, 0, 0, 0, 11],
                             [0, 15, 244, 1, 0, 0, 1, 0, 0, 28],
                             [10, 1, 2, 236, 0, 0, 25, 0, 0, 3],
                             [0, 0, 0, 0, 265, 10, 0, 1, 4, 0],
                             [0, 0, 1, 0, 48, 205, 0, 21, 28, 0],
                             [30, 0, 0, 15, 0, 0, 230, 2, 0, 0],
                             [14, 0, 0, 9, 7, 25, 37, 168, 14, 0],
                             [0, 0, 0, 0, 14, 8, 0, 5, 266, 0],
                             [0, 30, 24, 0, 1, 1, 0, 0, 0, 234]])

# 类别名称
class_names = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# 创建混淆矩阵的可视化
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=16)  # 增大标题字体大小
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)  # 增大x轴标签字体大小
plt.yticks(tick_marks, class_names, fontsize=12)  # 增大y轴标签字体大小

# 在每个单元格中显示数值，并增大字体大小
thresh = confusion_matrix.max() / 2.
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment="center", fontsize=14,
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()
