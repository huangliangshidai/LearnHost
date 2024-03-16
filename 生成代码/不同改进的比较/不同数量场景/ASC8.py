# 导入必要的库
import os
import h5py
import numpy as np
import librosa
import tensorflow as tf
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 列出所有可用的GPU设备
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # 配置GPU选项
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# 开始模保存型
# 定义数据增强函数
def data_augmentation(audio_data, sample_rate):
    # 确定音频时长
    duration = len(audio_data) / sample_rate

    # 确定随机截取的时间范围
    max_duration = 6.0  # 最大截取时长设置为5秒
    if duration > max_duration:
        start_time = random.uniform(0, duration - max_duration)
        end_time = start_time + max_duration
        augmented_data = audio_data[int(start_time * sample_rate): int(end_time * sample_rate)]
    else:
        augmented_data = audio_data

    # 增加噪声
    # noise_amplitude = 0.01  # 设置合适的噪声幅度
    # noise = noise_amplitude * np.random.randn(len(augmented_data))
    # augmented_data = augmented_data + noise

    return augmented_data

# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
# 数据准备
DATASET_PATH = "E:\桌面\dataset\\train"  # 声学场景数据集的路径
NUM_CLASSES = 8 # 声学场景的类别数
# 获取类别标签
class_labels = {
    0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8'
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
                # audio_data, sample_rate = librosa.load(file_path, sr=None)
                # feature = extract_feature(audio_data, sample_rate)
                # X.append(feature)
                # y.append(label)
                # 数据增强
                for _ in range(3):  # 每个原始样本生成5个增强样本
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    augmented_data = data_augmentation(audio_data, sample_rate)

                    # 使用librosa库加载音频文件，并提取其特征
                    feature = extract_feature(augmented_data,sample_rate)

                    X.append(feature)
                    y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')



    # 将特征和标签转换为NumPy数组
    X = np.array(X).astype(np.float32)
    y = np.array(y)
    # 转换标签为整数类型
    y = y.astype(int)

    # 将标签转换为one-hot编码
    y = np.eye(num_classes)[y]

    return X, y



# 修改 extract_feature 函数
def extract_feature(audio_data, sample_rate):
    # 将增强后的音频数据重采样为原始采样率
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)

    # 提取静态MFCCs特征
    mfccs = librosa.feature.mfcc(y=y_resampled, sr=sample_rate, n_mfcc=15)

    # 计算一阶差分MFCCs特征
    delta_mfccs = librosa.feature.delta(mfccs)

    # 计算二阶差分MFCCs特征
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # 根据权重分配，将三种特征合并在一起，得到动态MFCCs特征
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs,delta2_mfccs])

    # 返回动态MFCCs特征
    return dynamic_mfccs


class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 计算注意力权重
        attention_weights = K.softmax(K.dot(x, self.W) + self.b)
        # 应用注意力权重
        weighted_x = x * attention_weights
        # 融合注意力特征
        output = K.sum(weighted_x, axis=-2)
        # 在输出上添加一个维度，以符合 MaxPooling2D 层的输入维度要求
        output = K.expand_dims(output, axis=-1)
        return output

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
# 添加自定义的注意力层
model.add(SelfAttention())
print("添加自定义的注意力层")
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

print("编译模型")
# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# 训练模型
tensorboard_callback = TensorBoard(log_dir='logs')
epochs=120
# 假设您使用了tensorflow进行训练，且每个epoch都记录了每个类别的准确率
for epoch in range(epochs):
    # 训练模型
    print(epoch)
    model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test))


print("训练模型")
# model.fit(X_train, y_train, batch_size=64, epochs=80, validation_data=(X_test, y_test))
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


# 插入预测代码块
# 替换为未知音频文件路径
unknown_audio_path = "E:\桌面\dataset2\\test\\bus+street8\\bus+street.wav"

# 加载未知音频数据
unknown_audio, unknown_sample_rate = librosa.load(unknown_audio_path, sr=None)

# 分割未知音频为10等份
segment_size = len(unknown_audio) // 10
segments = [unknown_audio[i:i + segment_size] for i in range(0, len(unknown_audio), segment_size)]

# 存储检测结果
results = []

# 对每个片段进行检测
for i, segment in enumerate(segments):
    # 提取特征
    feature = extract_feature(segment, unknown_sample_rate)

    # 将输入数据转换为四维张量
    feature = np.expand_dims(feature, axis=(0, -1))

    # 进行预测
    prediction = model.predict(feature)

    # 获取预测结果的类别索引
    predicted_class_index = np.argmax(prediction)

    # 将类别索引映射到类别标签
    predicted_class_label = class_labels[predicted_class_index]

    # 存储结果
    results.append(predicted_class_label)

    # 输出检测结果
    print(f"Segment {i + 1} Prediction: {predicted_class_label}")

# 输出整体检测结果
print(f"\nOverall Prediction: {max(set(results), key=results.count)}")



# 保存模型
print("开始保存模型")
# 保存模型为 HDF5 文件
# 要删除的文件名
file_to_delete = "your_model.h5"

# 如果文件存在，则删除它
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)

# 保存模型
model.save("your_model.h5")
# 1382/1382 [==============================] - 123s 89ms/step - loss: 0.2863 - accuracy: 0.8957 - val_loss: 0.2526 - val_accuracy: 0.9137
# 109
# 1382/1382 [==============================] - 123s 89ms/step - loss: 0.2909 - accuracy: 0.8949 - val_loss: 0.2565 - val_accuracy: 0.9115
# 110
# 1382/1382 [==============================] - 123s 89ms/step - loss: 0.2860 - accuracy: 0.8974 - val_loss: 0.2399 - val_accuracy: 0.9182
# 111
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2865 - accuracy: 0.8952 - val_loss: 0.2507 - val_accuracy: 0.9163
# 112
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2847 - accuracy: 0.8984 - val_loss: 0.2244 - val_accuracy: 0.9249
# 113
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2816 - accuracy: 0.8976 - val_loss: 0.2349 - val_accuracy: 0.9205
# 114
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2829 - accuracy: 0.8977 - val_loss: 0.2310 - val_accuracy: 0.9238
# 115
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2825 - accuracy: 0.8972 - val_loss: 0.2337 - val_accuracy: 0.9229
# 116
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2828 - accuracy: 0.8966 - val_loss: 0.2562 - val_accuracy: 0.9139
# 117
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2773 - accuracy: 0.9001 - val_loss: 0.2813 - val_accuracy: 0.9047
# 118
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2860 - accuracy: 0.8960 - val_loss: 0.2771 - val_accuracy: 0.9074
# 119
# 1382/1382 [==============================] - 122s 89ms/step - loss: 0.2762 - accuracy: 0.9003 - val_loss: 0.2284 - val_accuracy: 0.9254
# 训练模型
# 346/346 [==============================] - 13s 38ms/step - loss: 0.2284 - accuracy: 0.9254
# Test loss: 0.2284
# Test accuracy: 0.9254
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.9251
# Class 1 Test Accuracy: 0.9652
# Class 2 Test Accuracy: 0.9479
# Class 3 Test Accuracy: 0.8814
# Class 4 Test Accuracy: 0.9630
# Class 5 Test Accuracy: 0.9058
# Class 6 Test Accuracy: 0.9528
# Class 7 Test Accuracy: 0.8618
# WARNING:tensorflow:Model was constructed with shape (None, 45, 517, 1) for input Tensor("conv2d_input:0", shape=(None, 45, 517, 1), dtype=float32), but it was called on an input with incompatible shape (None, 45, 862, 1).
# Segment 1 Prediction: class_1
# Segment 2 Prediction: class_1
# Segment 3 Prediction: class_1
# Segment 4 Prediction: class_1
# Segment 5 Prediction: class_1
# Segment 6 Prediction: class_7
# Segment 7 Prediction: class_7
# Segment 8 Prediction: class_7
# Segment 9 Prediction: class_7
# Segment 10 Prediction: class_2