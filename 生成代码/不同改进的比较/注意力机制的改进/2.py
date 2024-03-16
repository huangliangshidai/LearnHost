import matplotlib.pyplot as plt
import numpy as np

# 调整后的混淆矩阵数据
confusion_matrix = np.array([[225, 0, 0, 18, 0, 1, 50, 16, 0, 24],
                             [0, 290, 12, 0, 0, 0, 0, 0, 0, 11],
                             [0, 15, 223, 1, 0, 0, 1, 0, 0, 28],
                             [19, 1, 4, 251, 0, 0, 21, 0, 0, 6],
                             [0, 0, 0, 0, 230, 10, 0, 1, 4, 0],
                             [0, 0, 2, 0, 50, 198, 0, 21, 33, 0],
                             [30, 0, 0, 13, 0, 0, 267, 2, 0, 0],
                             [14, 0, 0, 7, 7, 21, 34, 172, 14, 0],
                             [0, 0, 0, 0, 13, 9, 0, 5, 294, 0],
                             [0, 31, 22, 0, 0, 0, 0, 0, 0, 247]])

# 类别名称
class_names = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# 调整字体大小
plt.rcParams.update({'font.size': 16})

# 创建混淆矩阵的可视化
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# 在每个单元格中显示数值
thresh = confusion_matrix.max() / 2.
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
