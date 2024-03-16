import matplotlib.pyplot as plt
import pandas as pd

# 创建数据框
data = {
    "Models": ["Baseline", "Channel", "SE", "CBAM", "SelfAttention"],
    "Airport": [45.0, 53.8, 59.1, 52.5, 76.6],
    "Bus": [62.9, 63.0, 70.3, 86.9, 81.1],
    "Metro": [53.5, 46.3, 49.2, 66.3, 70.8],
    "Metro_station": [53.0, 43.6, 44.5, 77.1, 67.1],
    "Park": [71.3, 75.8, 79.5, 89.9, 87.3],
    "Public square": [44.9, 35.7, 45.0, 52.9, 61.6],
    "Shopping_mall": [48.3, 59.2, 65.5, 65.7, 89.3],
    "Street_pedestrian": [29.8, 33.0, 39.9, 60.3, 68.1],
    "Street_traffic": [79.9, 77.3, 75.5, 89.6, 82.7],
    "Tram": [52.2, 38.6, 36.9, 80.5, 69.7],
    "Average": [54.1, 52.6, 57.6, 72.1, 74.8]
}
# 创建数据框
# data = {
#     "Models": ["wavelet filter-bank", "MFCC", "CQT", "log-mel energies", "DMFCC"],
#     "Airport": [60.7, 65.1, 71.7, 84.1, 76.6],
#     "Bus": [68.3, 69.8, 76.1, 76.1, 81.2],
#     "Metro": [55.27, 65.6, 71.4, 64.5, 70.7],
#     "Metro_station": [59.1, 53.0, 68.8, 65.0, 67.1],
#     "Park": [82.5, 82.3, 79.1, 89.7, 87.3],
#     "Public square": [57.1, 61.7, 60.0, 67.3, 61.6],
#     "Shopping_mall": [67.7, 68.7, 72.7, 80.6, 89.3],
#     "Street_pedestrian": [64.6, 58.2, 59.9, 59.4, 68.1],
#     "Street_traffic": [58.8, 60.0, 71.0, 86.3, 82.7],
#     "Tram": [73.8, 79.9, 68.6, 64.4, 69.7],
#     "Average": [64.8, 66.5, 70.6, 73.7, 74.8]
# }


# 创建Pandas DataFrame
df = pd.DataFrame(data)

# 设置整个图的背景颜色为纯白
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 设置图形风格
plt.style.use("seaborn")

# 绘制曲线图
# 绘制曲线图，加粗"Average"曲线
for scene in df.columns[1:]:
    if scene == "Average":
        plt.plot(df["Models"], df[scene], marker='o', label=scene, color='red', linewidth=2.5)
    else:
        plt.plot(df["Models"], df[scene], marker='o', label=scene)

# 添加"Average"红色注释
average_accuracy = df["Average"]
for i, acc in enumerate(average_accuracy):
    plt.text(i, acc, f"{acc}%", color='red', ha='center', va='bottom')

plt.legend()
plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Accuracy by Scene for Different Models", fontsize=14)

# 显示图形
plt.show()
