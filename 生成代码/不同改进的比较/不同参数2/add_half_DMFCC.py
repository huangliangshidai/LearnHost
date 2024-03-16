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
#添加注意力机制和动态MFCC，add_DMFCC.py(最新)
#batch_size=32, epochs=120，进行数据增强产生3倍，
# 119
# 1728/1728 [==============================] - 114s 66ms/step - loss: 0.6624 - accuracy: 0.7548 - val_loss: 0.6805 - val_accuracy: 0.7482
# 训练模型
# 432/432 [==============================] - 13s 29ms/step - loss: 0.6805 - accuracy: 0.7482
# Test loss: 0.6805
# Test accuracy: 0.7482
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.7663
# Class 1 Test Accuracy: 0.8112
# Class 2 Test Accuracy: 0.7078
# Class 3 Test Accuracy: 0.6713
# Class 4 Test Accuracy: 0.8730
# Class 5 Test Accuracy: 0.6160
# Class 6 Test Accuracy: 0.8298
# Class 7 Test Accuracy: 0.6816
# Class 8 Test Accuracy: 0.8272
# Class 9 Test Accuracy: 0.6969
#添加注意力机制和动态MFCC，add_DMFCC.py
#batch_size=64, epochs=80，进行数据增强产生3倍，
# 1434/1434 [==============================] - 973s 678ms/step - loss: 0.6059 - accuracy: 0.7769 - val_loss: 0.5528 - val_accuracy: 0.8014
# 训练模型
# 717/717 [==============================] - 70s 97ms/step - loss: 0.5528 - accuracy: 0.8014
# Test loss: 0.5528
# Test accuracy: 0.8014
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.7268
# Class 1 Test Accuracy: 0.7763
# Class 2 Test Accuracy: 0.8996
# Class 3 Test Accuracy: 0.6995
# Class 4 Test Accuracy: 0.9314
# Class 5 Test Accuracy: 0.7422
# Class 6 Test Accuracy: 0.8907
# Class 7 Test Accuracy: 0.7357
# Class 8 Test Accuracy: 0.7357
# Class 9 Test Accuracy: 0.8854




#重复获取5倍相同数据准确率79.35%
# Test loss: 0.6567
# Test accuracy: 0.7935
# 模型评估
# 进行预测
# class 0 Test   Accuracy: 0.7937
# Class 1Test   Accuracy: 0.8440
# class 2Test    Accuracy: 0.9131
# class 3 Test    Accuracy: 0.7254
# Class 4Test     Accuracy: 0.9737
# class 5Test    Accuracy: 0.8080
#  Class 6  Test    Accuracy: 0.8370
# Class 7Test     Accuracy: 0.5429
# class 8 Test   Accuracy: 0.6881
# class 9 Test   Accuracy: 0.8065


#batch_size=32, epochs=50，进行数据增强产生3倍，准确率没有到达最高，下次尝试100
# 1721/1721 [==============================] - 602s 350ms/step - loss: 0.7243 - accuracy: 0.7312 - val_loss: 0.7023 - val_accuracy: 0.7415
# 训练模型
# 431/431 [==============================] - 45s 105ms/step - loss: 0.7023 - accuracy: 0.7415
# Test loss: 0.7023
# Test accuracy: 0.7415
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.6364
# Class 1 Test Accuracy: 0.7420
# Class 2 Test Accuracy: 0.8083
# Class 3 Test Accuracy: 0.6395
# Class 4 Test Accuracy: 0.8840
# Class 5 Test Accuracy: 0.6475
# Class 6 Test Accuracy: 0.8288
# Class 7 Test Accuracy: 0.6395
# Class 8 Test Accuracy: 0.7261
# Class 9 Test Accuracy: 0.8580


#batch_size=32, epochs=80，
# Epoch 100/100
# 1721/1721 [==============================] - 609s 354ms/step - loss: 0.6008 - accuracy: 0.7786 - val_loss: 0.6285 - val_accuracy: 0.7732
# 训练模型
# 431/431 [==============================] - 43s 100ms/step - loss: 0.6285 - accuracy: 0.7732
# Test loss: 0.6285
# Test accuracy: 0.7732
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.6881
# Class 1 Test Accuracy: 0.7549
# Class 2 Test Accuracy: 0.8439
# Class 3 Test Accuracy: 0.6969
# Class 4 Test Accuracy: 0.8712
# Class 5 Test Accuracy: 0.6655
# Class 6 Test Accuracy: 0.8647
# Class 7 Test Accuracy: 0.7134
# Class 8 Test Accuracy: 0.7307
# Class 9 Test Accuracy: 0.8961
# 开始模保存型


# 开始模保存型
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
    mfccs = librosa.feature.mfcc(y=y_resampled, sr=sample_rate, n_mfcc=20)

    # 计算一阶差分MFCCs特征
    delta_mfccs = librosa.feature.delta(mfccs)

    # 计算二阶差分MFCCs特征
    # delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # 根据权重分配，将三种特征合并在一起，得到动态MFCCs特征
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs])

    # 返回动态MFCCs特征
    return dynamic_mfccs
# 118
# 1728/1728 [==============================] - 119s 69ms/step - loss: 0.6118 - accuracy: 0.7742 - val_loss: 0.5709 - val_accuracy: 0.7963
# 119
# 1728/1728 [==============================] - 119s 69ms/step - loss: 0.6122 - accuracy: 0.7770 - val_loss: 0.5772 - val_accuracy: 0.7937
# 训练模型
# 432/432 [==============================] - 13s 30ms/step - loss: 0.5772 - accuracy: 0.7937
# Test loss: 0.5772
# Test accuracy: 0.7937
# 模型评估
# 进行预测
# Class 0 Test Accuracy: 0.7967
# Class 1 Test Accuracy: 0.9023
# Class 2 Test Accuracy: 0.7554
# Class 3 Test Accuracy: 0.7649
# Class 4 Test Accuracy: 0.9106
# Class 5 Test Accuracy: 0.6938
# Class 6 Test Accuracy: 0.8768
# Class 7 Test Accuracy: 0.6713
# Class 8 Test Accuracy: 0.8504
# Class 9 Test Accuracy: 0.7149


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


# # 创建空的图表和子图
# fig, ax = plt.subplots(figsize=(10, 6))
# # 替换NUM_CLASSES为您的声学场景类别数
# class_accuracy = {}
# for i in range(NUM_CLASSES):
#     class_accuracy[i] = [0]  # 初始化每个类别的准确率列表
# # 为每个类别创建一个子图
# for i in range(NUM_CLASSES):
#     ax.plot(class_accuracy[i], label=f'Class {i} Accuracy')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Accuracy')
#     ax.legend()


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
    # 计算整体准确率并传递给update_accuracy函数
    # _, train_acc = model.evaluate(X_train, y_train)
    # update_accuracy(epoch, train_acc)
    # 暂停一段时间，以便观察图表的更新（可选）
    # plt.pause(0.1)



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
# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 将类别索引映射到类别标签
predicted_labels = [class_labels[idx] for idx in y_pred_classes]

# 输出每个片段的预测结果
for i, predicted_label in enumerate(predicted_labels):
    print(f'Segment {i+1} belongs to class: {predicted_label}')

# 保存模型
print("开始保存模型")
# 保存模型为 HDF5 文件
with h5py.File("model_custom10.h5", "w") as hf:
    for layer in model.layers:
        if layer.get_weights():
            g = hf.create_group(layer.name)
            for i, weight in enumerate(layer.get_weights()):
                g.create_dataset(f"weights_layer{i}", data=weight)
print("模型保存成功")
