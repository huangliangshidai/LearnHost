import os
import h5py
import numpy as np
import librosa
import tensorflow as tf
import tensorflow.keras.backend as K


from tensorflow.keras.models import load_model
class_labels = {
    0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
    5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'
}  # Modify based on your class labels

# 定义注意力机制层
class SelfAttention(tf.keras.layers.Layer):
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

# 定义提取特征的函数
def extract_feature(audio_data, sample_rate):
    # 将增强后的音频数据重采样为原始采样率
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)

    # 提取静态MFCCs特征
    mfccs = librosa.feature.mfcc(y=y_resampled, sr=sample_rate, n_mfcc=20)

    # 计算一阶差分MFCCs特征
    delta_mfccs = librosa.feature.delta(mfccs)

    # 计算二阶差分MFCCs特征
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # 根据权重分配，将三种特征合并在一起，得到动态MFCCs特征
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    # 返回动态MFCCs特征
    return dynamic_mfccs

# 替换为模型文件路径
model_path = r'E:/桌面/声学场景分类/生成代码/不同改进的比较/不同参数2/model_custom10.h5'
model = load_model(model_path)

# 替换为未知音频文件路径
unknown_audio_path = "E:\桌面\dataset\\val\\1.wav"

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
