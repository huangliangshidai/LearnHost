# 导入必要的库
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

## dataset
# Test loss: 2.0293
# Test accuracy: 0.1951

# dataset2
# Test loss: 1.9179
# Test accuracy: 0.2317

# 数据准备
DATASET_PATH = "E:\桌面\dataset2\\train"  # 声学场景数据集的路径
NUM_CLASSES = 8  # 声学场景的类别数


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
            # print(filename)
            file_path = os.path.join(folder_path, filename)
            try:
                # 使用librosa库加载音频文件，并提取其特征
                feature = extract_feature(file_path)
                # print(feature)
                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')

    # 将特征和标签转换为NumPy数组
    X = np.array(X)
    print("获取音频特征")
    print( X)
    y = np.array(y)
    print("获取音频标签")
    print( y)

    # 将标签转换为one-hot编码
    y = np.eye(num_classes)[y]
    print(y)

    return X, y


def extract_feature(file_path):
    # 读取音频文件
    y, sr = librosa.load(file_path, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return mfcc


# 加载数据集
X, y = load_data(DATASET_PATH, NUM_CLASSES)
print("加载完数据集")

# 将输入数据转换为四维张量
X= np.expand_dims(X, axis=-1) # 在最后一个维度添加一个维度，表示通道数
# y= np.expand_dims(y, axis=-1) # 在最后一个维度添加一个维度，表示通道数

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集已划分")
# 构建CNN模型
model = Sequential()
print("构建CNN模型")
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))# 明确指定输入数据的维度，包括高度、宽度和通道数
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第一次池化")
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第二次池化")
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
print("第三次池化")
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
print("全连接")

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("编译模型")
# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test, y_test))
print("训练模型")
# 模型评估
loss, accuracy = model.evaluate(X_test, y_test)
print("模型评估")
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

#
