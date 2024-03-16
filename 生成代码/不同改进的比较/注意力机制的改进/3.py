import matplotlib.pyplot as plt

# 不同设置的准确率数据
settings = ['Setting1', 'Setting2', 'Setting3', 'Setting4']
accuracy_percentage = [74.6, 74.8, 74.2, 74.3]

# 将百分比转换为小数
accuracy = [acc / 100 for acc in accuracy_percentage]

# 创建曲线图
plt.figure(figsize=(8, 6))
plt.plot(settings, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Settings')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy for Different Settings')
plt.grid(True)

# 添加数据标签
for i in range(len(settings)):
    plt.text(settings[i], accuracy[i], f'{accuracy_percentage[i]}%', ha='center', va='bottom')

# 显示图例
plt.legend(['Accuracy'], loc='best')

# 设置y轴范围为0到1
plt.ylim(0.6, 0.8)

# 显示图形
plt.show()
