# 导入必要的库
import os
import h5py
import numpy as np
import librosa

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.regularizers import l2
# Epoch 25/80
# 288/288 [==============================] - 194s 673ms/step - loss: 0.0771 - accuracy: 0.9748 - val_loss: 1.7285 - val_accuracy: 0.6696
# Epoch 26/80
# 288/288 [==============================] - 185s 644ms/step - loss: 0.0937 - accuracy: 0.9691 - val_loss: 1.6839 - val_accuracy: 0.6753
# Epoch 27/80
# 288/288 [==============================] - 187s 649ms/step - loss: 0.0967 - accuracy: 0.9699 - val_loss: 1.7625 - val_accuracy: 0.6601
# Epoch 28/80
# 开始模保存型
# 288/288 [==============================] - 190s 659ms/step - loss: 0.0363 - accuracy: 0.9904 - val_loss: 2.3679 - val_accuracy: 0.7013
# Epoch 80/80
# 288/288 [==============================] - 190s 661ms/step - loss: 0.0255 - accuracy: 0.9928 - val_loss: 2.1491 - val_accuracy: 0.6944
# 144/144 [==============================] - 15s 106ms/step - loss: 2.1491 - accuracy: 0.6944
# Test loss: 2.1491
# Test accuracy: 0.5944

# Epoch 25/80
# 288/288 [==============================] - 194s 673ms/step - loss: 0.0771 - accuracy: 0.9748 - val_loss: 1.7285 - val_accuracy: 0.6696
# Epoch 26/80
# 288/288 [==============================] - 185s 644ms/step - loss: 0.0937 - accuracy: 0.9691 - val_loss: 1.6839 - val_accuracy: 0.6753
# Epoch 27/80
# 288/288 [==============================] - 187s 649ms/step - loss: 0.0967 - accuracy: 0.9699 - val_loss: 1.7625 - val_accuracy: 0.6601
# Epoch 28/80
# 开始模保存型
# 288/288 [==============================] - 190s 659ms/step - loss: 0.0363 - accuracy: 0.9904 - val_loss: 2.3679 - val_accuracy: 0.7013
# Epoch 80/80
# 288/288 [==============================] - 190s 661ms/step - loss: 0.0255 - accuracy: 0.9928 - val_loss: 2.1491 - val_accuracy: 0.6944
# 144/144 [==============================] - 15s 106ms/step - loss: 2.1491 - accuracy: 0.6944
# Test loss: 2.1491
# Test accuracy: 0.6944
# 开始模保存型


# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
# 数据准备
DATASET_PATH = "E:\桌面\dataset\\train"  # 声学场景数据集的路径
NUM_CLASSES = 10 # 声学场景的类别数
# 获取类别标签
class_labels = {
    0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
    5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'
}  # 根据您的类别编号修改

def load_data(dataset_path, num_classes):
    # 加载数据集
    X = []
    y = []
    for folder_name in os.listdir(dataset_path):
        print(folder_name)
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 检查文件是否是支持的音频文件类型
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() not in SUPPORTED_AUDIO_FORMATS:
                continue
            try:
                # 数据增强
                for _ in range(1):  # 每个原始样本生成5个增强样本
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    # augmented_data = data_augmentation(audio_data, sample_rate)

                    # 使用librosa库加载音频文件，并提取其特征
                    feature = extract_feature(audio_data,sample_rate)

                    X.append(feature)
                    y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')



    # 将特征和标签转换为NumPy数组
    X = np.array(X)
    y = np.array(y)
    # 转换标签为整数类型
    y = y.astype(int)

    # 将标签转换为one-hot编码
    y = np.eye(num_classes)[y]
    return X, y


def extract_feature(audio_data, sample_rate):
    # 将增强后的音频数据重采样为原始采样率
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)
    # 提取梅尔能量特征
    mel_energy = librosa.feature.melspectrogram(y=y_resampled, sr=sample_rate, n_mels=128)
    # 将梅尔能量转换为对数刻度
    log_mel_energy = librosa.power_to_db(mel_energy)
    # 返回对数刻度的梅尔能量特征
    return log_mel_energy

# poch 79/80
# 288/288 [==============================] - 191s 664ms/step - loss: 0.0295 - accuracy: 0.9929 - val_loss: 2.5431 - val_accuracy: 0.7215
# Epoch 80/80
# 288/288 [==============================] - 191s 664ms/step - loss: 0.0408 - accuracy: 0.9919 - val_loss: 2.3338 - val_accuracy: 0.6933
# 144/144 [==============================] - 18s 122ms/step - loss: 2.3338 - accuracy: 0.6933
# Test loss: 2.3338
# Test accuracy: 0.6933
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.7541
# Class 1 Test Accuracy: 0.8098
# Class 2 Test Accuracy: 0.5833
# Class 3 Test Accuracy: 0.6228
# Class 4 Test Accuracy: 0.8937
# Class 5 Test Accuracy: 0.5499
# Class 6 Test Accuracy: 0.7122
# Class 7 Test Accuracy: 0.5890
# Class 8 Test Accuracy: 0.7943
# Class 9 Test Accuracy: 0.6215



# 加载数据集
X, y = load_data(DATASET_PATH, NUM_CLASSES)
print("加载完数据集")


# 将输入数据转换为四维张量
X= np.expand_dims(X, axis=-1) # 在最后一个维度添加一个维度，表示通道数

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集已划分")

# 构建CNN模型
model = Sequential()
print("构建CNN模型")
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
print("构建神经")
# # 添加自定义的注意力层
# model.add(SelfAttention())
# print("添加自定义的注意力层")
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第一次池化")
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第二次池化")
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第三次池化")
model.add(Flatten())

model.add(Dropout(0.5))#防止过拟合
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.summary()
print("编译模型")
# 编译模型学习率 (learning rate): 默认为0.001。
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("训练模型")
model.fit(X_train, y_train, batch_size=64, epochs=80, validation_data=(X_test, y_test))
# 模型评估
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
print("模型评估")
# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("进行预测")

# 计算每个类别的测试准确率
class_accuracy = {}
for i in range(NUM_CLASSES):
    true_class_samples = y_test_classes[y_test_classes == i]
    pred_class_samples = y_pred_classes[y_test_classes == i]
    class_accuracy[i] = accuracy_score(true_class_samples, pred_class_samples)

#输出每个类别的测试准确率
for class_idx, acc in class_accuracy.items():
    print(f'Class {class_idx} Test Accuracy: {acc:.4f}')

# 保存模型
print("开始保存模型")
# 保存模型为 HDF5 文件
with h5py.File("model_custom11.h5", "w") as hf:
    for layer in model.layers:
        if layer.get_weights():
            g = hf.create_group(layer.name)
            for i, weight in enumerate(layer.get_weights()):
                g.create_dataset(f"weights_layer{i}", data=weight)
print("模型保存成功")
