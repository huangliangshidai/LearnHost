import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf

# 待检测音频路径
AUDIO_PATH = "E:\桌面\dataset3\\val\\tram-barcelona-179-5519-c.wav"

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
loaded_model = tf.keras.models.load_model('trained_model_weights.h5', custom_objects={'SelfAttention': SelfAttention})
# 从JSON文件中加载模型结构
# with open('trained_model.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
# loaded_model = model_from_json(loaded_model_json)
# with open('trained_model.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
#
# # 在custom_objects字典中提供自定义层映射
# custom_objects = {'SelfAttention': SelfAttention}
# # 使用custom_objects加载模型
# with tf.compat.v1.Session() as sess:
#     loaded_model = tf.keras.models.model_from_json(loaded_model_json, custom_objects=custom_objects)
# # 加载权重
# loaded_model.load_weights("trained_model_weights.h5")


# 将音频分割为10个片段
def split_audio(audio_path, num_segments=10):
    print("正在将音频分割为10个片段")
    y, sr = librosa.load(audio_path, sr=None)
    segment_length = len(y) // num_segments
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]
    return segments

# 对每个片段进行场景分类
def classify_segments(segments):
    print("正在对每个片段进行场景分类")
    results = []
    for segment in segments:
        feature = extract_feature(segment)
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=-1)
        prediction = loaded_model.predict(feature)
        class_index = np.argmax(prediction)
        results.append(class_index)
    return results

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
