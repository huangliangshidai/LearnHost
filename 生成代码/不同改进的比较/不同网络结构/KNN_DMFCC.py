import os
import h5py
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

# 定义数据增强函数
def data_augmentation(audio_data, sample_rate):
    # 确定音频时长
    duration = len(audio_data) / sample_rate

    # 确定随机截取的时间范围
    max_duration = 5.0  # 最大截取时长设置为5秒
    if duration > max_duration:
        start_time = random.uniform(0, duration - max_duration)
        end_time = start_time + max_duration
        augmented_data = audio_data[int(start_time * sample_rate): int(end_time * sample_rate)]
    else:
        augmented_data = audio_data

    # 增加噪声
    noise_amplitude = 0.01  # 设置合适的噪声幅度
    noise = noise_amplitude * np.random.randn(len(augmented_data))
    augmented_data = augmented_data + noise

    return augmented_data

# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
# 数据准备
DATASET_PATH = "E:\桌面\dataset\\train"  # 声学场景数据集的路径
NUM_CLASSES = 10  # 声学场景的类别数

def load_data(dataset_path, num_classes):
    X = []
    y = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() not in SUPPORTED_AUDIO_FORMATS:
                continue
            try:
                for _ in range(1):
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    # augmented_data = data_augmentation(audio_data, sample_rate)
                    feature = extract_feature(audio_data, sample_rate)

                    X.append(feature)
                    y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')

    X = np.array(X).astype(np.float32)
    y = np.array(y)
    y = y.astype(int)

    return X, y

def extract_feature(audio_data, sample_rate):
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=y_resampled, sr=sample_rate, n_mfcc=20)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    return dynamic_mfccs
# 加载完数据集
# 数据集已划分
# 构建KNN模型
# 训练模型
# Test accuracy: 0.4606
# 开始保存模型
# 模型保存成功
# 加载数据集
X, y = load_data(DATASET_PATH, NUM_CLASSES)
print("加载完数据集")

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集已划分")

# 将输入数据从三维降到二维
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# 构建KNN模型
knn_model = KNeighborsClassifier(n_neighbors=5)
print("构建KNN模型")

# 训练模型
knn_model.fit(X_train_flatten, y_train)
print("训练模型")

# 模型评估
y_pred = knn_model.predict(X_test_flatten)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.4f}')

# 保存模型
print("开始保存模型")
with h5py.File("model_knn.h5", "w") as hf:
    hf.create_dataset("weights", data=knn_model._fit_X)  # 存储KNN模型的权重
print("模型保存成功")
