import os
import h5py
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.regularizers import l2


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
DATASET_PATH = "E:\桌面\dataset2\\train"  # 声学场景数据集的路径
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
            print(filename)
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
    # 将增强后的音频数据重采样为原始采样率
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)
    # 提取梅尔能量特征
    mel_energy = librosa.feature.melspectrogram(y=y_resampled, sr=sample_rate, n_mels=128)
    # 将梅尔能量转换为对数刻度
    log_mel_energy = librosa.power_to_db(mel_energy)
    # 返回对数刻度的梅尔能量特征
    return log_mel_energy
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
# 将输入数据转换为三维张量 (样本数, 时间步长, 特征数)
X = np.swapaxes(X, 1, 2)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集已划分")

# 构建RNN模型
# 构建RNN模型
model = Sequential()

# 添加Masking层以处理可变长度的序列
model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))

# 第一层LSTM，不返回完整的序列
lstm_1 = LSTM(128, return_sequences=False)
model.add(lstm_1)

# 添加第二层LSTM以返回完整的序列
lstm_2 = LSTM(64, return_sequences=True)
model.add(lstm_2)

# 添加第三层LSTM，不返回完整的序列
lstm_3 = LSTM(32)
model.add(lstm_3)

# 设置初始状态
initial_state_1 = [tf.zeros((tf.shape(X_train)[0], lstm_1.units)), tf.zeros((tf.shape(X_train)[0], lstm_1.units))]
initial_state_2 = [tf.zeros((tf.shape(X_train)[0], lstm_2.units)), tf.zeros((tf.shape(X_train)[0], lstm_2.units))]
initial_state_3 = [tf.zeros((tf.shape(X_train)[0], lstm_3.units)), tf.zeros((tf.shape(X_train)[0], lstm_3.units))]

model.build((None, X_train.shape[1], X_train.shape[2]))

# 将初始状态传递给LSTM层
model.layers[1].cell.reset_state(states=initial_state_1)
model.layers[3].cell.reset_state(states=initial_state_2)
model.layers[5].cell.reset_state(states=initial_state_3)

model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))


# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
tensorboard_callback = TensorBoard(log_dir='logs')
epochs = 120

for epoch in range(epochs):
    print(epoch)
    model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test))

print("训练模型")

# 模型评估
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
print("模型评估")

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算每个类别的测试准确率
class_accuracy = {}
for i in range(NUM_CLASSES):
    true_class_samples = y_test[y_test[:, i] == 1]
    pred_class_samples = y_pred_classes[y_test[:, i] == 1]
    class_accuracy[i] = accuracy_score(true_class_samples, pred_class_samples)

# 输出每个类别的测试准确率
for class_idx, acc in class_accuracy.items():
    print(f'Class {class_idx} Test Accuracy: {acc:.4f}')

# 保存模型
print("开始保存模型")
# 保存模型为 HDF5 文件
with h5py.File("model_custom10_rnn.h5", "w") as hf:
    for layer in model.layers:
        if layer.get_weights():
            g = hf.create_group(layer.name)
            for i, weight in enumerate(layer.get_weights()):
                g.create_dataset(f"weights_layer{i}", data=weight)
print("模型保存成功")