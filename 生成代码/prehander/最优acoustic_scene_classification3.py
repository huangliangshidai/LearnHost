# 导入必要的库
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py

#添加注意力机制和动态MFCC


# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
# 数据准备
DATASET_PATH = "E:\桌面\dataset2\\train"  # 声学场景数据集的路径
NUM_CLASSES = 2 # 声学场景的类别数


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
                # 使用librosa库加载音频文件，并提取其特征
                feature = extract_feature(file_path)
                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')



    # 将特征和标签转换为NumPy数组
    X = np.array(X)
    y = np.array(y)

    # 将标签转换为one-hot编码
    y = np.eye(num_classes)[y]

    return X, y


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

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("编译模型")
# 训练模型
model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test))
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
# 计算每个类别的测试准确率
class_accuracy = {}
for i in range(NUM_CLASSES):
    true_class_samples = y_test_classes[y_test_classes == i]
    pred_class_samples = y_pred_classes[y_test_classes == i]
    class_accuracy[i] = accuracy_score(true_class_samples, pred_class_samples)

# 输出每个类别的测试准确率
for class_idx, acc in class_accuracy.items():
    print(f'Class {class_idx} Test Accuracy: {acc:.4f}')
# 保存模型
print("开始保存模型")
# 保存模型为 HDF5 文件
with h5py.File("model_custom3.h5", "w") as hf:
    for layer in model.layers:
        if layer.get_weights():
            g = hf.create_group(layer.name)
            for i, weight in enumerate(layer.get_weights()):
                g.create_dataset(f"weights_layer{i}", data=weight)
print("模型保存成功")
