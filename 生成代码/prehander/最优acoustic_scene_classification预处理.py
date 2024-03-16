# 导入必要的库
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#添加注意力机制和动态MFCC

# dataset
# Test loss: 1.0702
# Test accuracy: 0.5726

# dataset2
# Test loss: 1.0890   NUM_CLASSES = 8 # 声学场景的类别数
# Test accuracy: 0.6138

# dataset3    NUM_CLASSES = 6 # 声学场景的类别数
# Test loss: 0.4123
# Test accuracy: 0.8583

# 数据准备
DATASET_PATH = "E:\桌面\dataset2\\train"  # 声学场景数据集的路径
NUM_CLASSES = 8 # 声学场景的类别数



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
            try:
                # 使用librosa库加载音频文件，并提取其特征
                feature = extract_feature(file_path)

                # 进行数据预处理
                # 静音去除
                # feature = remove_silence(feature)
                # 音频增强
                feature = apply_audio_augmentation(feature)
                # 特征归一化
                # feature = normalize_feature(feature)

                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')

    # 将特征和标签转换为NumPy数组
    X = np.array(X)
    y = np.array(y)

    # 将标签转换为one-hot编码
    y = np.eye(num_classes)[y.astype(int)]

    return X, y
# def load_data(dataset_path, num_classes):
#     # 加载数据集
#     X = []
#     y = []
#     for folder_name in os.listdir(dataset_path):
#         print(folder_name)
#         folder_path = os.path.join(dataset_path, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
#         label = int(folder_name)
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 # 使用librosa库加载音频文件，并提取其特征
#                 feature = extract_feature(file_path)
#                 X.append(feature)
#                 y.append(label)
#             except Exception as e:
#                 print(f'Error loading {file_path}: {e}')
#
#
#
#     # 将特征和标签转换为NumPy数组
#     X = np.array(X)
#     y = np.array(y)
#
#     # 将标签转换为one-hot编码
#     y = np.eye(num_classes)[y]
#
#     return X, y
#数据增强
def apply_audio_augmentation(feature):
    augmented_features = []

    # 添加噪声
    noisy_feature = add_noise(feature)
    augmented_features.append(noisy_feature)

    # 改变音频速度
    # speed_changed_feature = change_audio_speed(feature)
    # augmented_features.append(speed_changed_feature)

    # 将特征列表转换为多维数组
    augmented_features = np.stack(augmented_features)

    return augmented_features

def add_noise(feature, noise_factor=0.02):
    # 生成与特征相同长度的随机噪声
    noise = np.random.randn(*feature.shape)

    # 将噪声与特征叠加
    noisy_feature = feature + noise_factor * noise


    return noisy_feature

def change_audio_speed(feature, speed_factor=1.2):
    # 改变音频的速度
    speed_changed_feature = librosa.effects.time_stretch(feature, speed_factor)

    return speed_changed_feature

#
# 去除静音
def remove_silence(feature, threshold=0.02):
    # 计算特征的能量
    # energy = np.mean(librosa.feature.rms(feature.T), axis=0)
    energy = np.mean(librosa.feature.rms(y=feature.T), axis=0)

    # 找到能量大于阈值的帧
    non_silent_frames = np.where(energy > threshold)[0]

    # 提取非静音帧对应的特征
    feature_no_silence = feature[:, non_silent_frames]

    return feature_no_silence

# 归一化
def normalize_feature(feature):
    scaler = MinMaxScaler()
    normalized_feature = scaler.fit_transform(feature)
    return normalized_feature

def extract_feature(file_path):
    # 提取音频特征
    y, sr = librosa.load(file_path, sr=None)
    # 提取静态MFCCs特征
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # 计算一阶差分MFCCs特征
    delta_mfccs = librosa.feature.delta(mfccs)
    # 计算二阶差分MFCCs特征
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    # 将三种特征合并在一起，得到动态MFCCs特征
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
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
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='SAME', input_shape=(X_train.shape[1], X_train.shape[2], 1)))

print("构建神经")
# 添加自定义的注意力层
model.add(SelfAttention())

print("添加自定义的注意力层")
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第一次池化")
model.add(Conv2D(64, (3, 3), activation='relu', padding='SAME'))


model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第二次池化")
model.add(Conv2D(128, (3, 3), activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第三次池化")
model.add(Flatten())


model.add(Dropout(0.5))#防止过拟合
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("编译模型")
# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
print("训练模型")
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
#
