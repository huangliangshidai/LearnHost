# -*- coding: utf-8 -*-
# 或者使用其他编码方式，比如 GBK
# -*- coding: gbk -*-

import os

import h5py
import numpy as np
import librosa
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
# 在另外的文件中导入模型


# 使用模型进行预测等操作


# 待检测音频路径
AUDIO_PATH = "E:\桌面\dataset\\val\\1.wav"

# 数据准备，类似之前的extract_feature函数
def extract_feature(file_path):
    # 提取音频特征
    print("正在提取音频特征")
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


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 计算注意力权重
        attention_weights = tf.nn.softmax(tf.matmul(x, self.W) + self.b)

        # 应用注意力权重
        weighted_x = x * attention_weights

        # 融合注意力特征
        output = tf.reduce_sum(weighted_x, axis=-2)

        # 在输出上添加一个维度，以符合 MaxPooling2D 层的输入维度要求
        output = tf.expand_dims(output, axis=-1)

        return output


# 加载模型
# model = load_model('E:\桌面\声学场景分类\生成代码\音频检测\trained_model.h5')
# 使用原始字符串表示路径
model_path = r'E:\桌面\声学场景分类\生成代码\\trained_model_weights.h5'

# 加载模型
print("正在加载模型")
# 构建空的 Sequential 模型
loaded_model = Sequential()
# 加载保存的 HDF5 文件
with h5py.File("model_custom3.h5", "r") as hf:
    for layer_name in hf.keys():
        layer_group = hf[layer_name]
        if 'weights' in layer_group.keys():
            layer_weights = layer_group['weights'][...]
            units = layer_weights.shape[-1]  # 获取输出单元数，即权重数组的最后一个维度大小
            # 构建 Dense 层
            layer = Dense(units, input_shape=(layer_weights.shape[0],))
            layer.set_weights([layer_weights])
            loaded_model.add(layer)
# 输出加载的模型结构
# loaded_model.summary()



# 将音频分割为10个片段
def split_audio(audio_path, num_segments=10):
    print("正在将音频分割为10个片段")
    y, sr = librosa.load(audio_path, sr=None)
    segment_length = len(y) // num_segments
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]
    return segments



def classify_feature(feature_file):
    # 加载模型
    # model = load_model("model_name.h5")  # 请替换为你的模型文件名
    # 加载特征
    feature = np.load(feature_file)
    # 对特征进行预处理（如果有必要）
    # 进行分类
    predictions = loaded_model.predict(feature)
    # 获取分类结果（例如取概率最大的类别）
    class_index = np.argmax(predictions)
    # 返回分类结果
    return class_index


# 对每个片段进行场景分类
def classify_segments(segments):
    segment_results = []
    for segment in segments:
        # 提取特征并保存到文件中
        feature = extract_feature(segment)
        feature_file = "segment_feature.npy"  # 指定特征保存的文件名
        np.save(feature_file, feature)
        # 然后将文件路径传递给模型进行分类
        result = classify_feature(feature_file)
        segment_results.append(result)

    return segment_results


# 获取类别标签
class_labels = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 5: 'class_5',6: 'class_6', 7: 'class_7', 8: 'class_8',9: 'class_9', 10: 'class_10'}  # 根据您的类别编号修改

# 将音频分割为10个片段
segments = split_audio(AUDIO_PATH)

# 对每个片段进行场景分类
segment_results = classify_segments(segments)

# 展示每个片段的场景分类结果
for i, segment_result in enumerate(segment_results):
    class_label = class_labels[segment_result]
    print(f'Segment {i+1} belongs to class: {class_label}')
