import os
import librosa
import numpy as np
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
# 加载保存的模型
# 创建一个新的 Sequential 模型（模型结构与原模型一致）
loaded_model = Sequential()

# 加载权重
with h5py.File("model_custom10.h5", "r") as hf:
    for layer in loaded_model.layers:
        if layer.get_weights():
            # 获取权重名称
            weight_names = [f"weights_layer{i}" for i in range(len(layer.get_weights()))]
            # 获取权重数据
            weights_data = [hf[layer.name][weight_name][:] for weight_name in weight_names]
            # 设置权重
            layer.set_weights(weights_data)

# 编译模型（确保与原模型编译时的参数一致）
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 定义特征提取函数
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


# 定义预测函数
def predict_scene(model, audio_feature):
    # 使用模型进行预测
    prediction = model.predict(audio_feature)
    # 获取预测类别的索引
    predicted_class_index = np.argmax(prediction, axis=1)
    return predicted_class_index

# 定义待检测音频文件路径
audio_file_path = "E:/桌面/dataset//val//tram-barcelona-179-5519-c.wav"

# 定义音频切割函数
def split_audio(audio_path, num_segments=10):
    print("正在将音频分割为10个片段")
    y, sr = librosa.load(audio_path, sr=None)
    segment_length = len(y) // num_segments
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]
    return segments


# 加载待检测音频文件
audio_data = librosa.load(audio_file_path)

# 切割音频为片段
audio_segments = split_audio(audio_data)

# 存储每个片段的预测结果


# 对每个片段进行预测
for segment in audio_segments:
    # 提取特征
    feature = extract_feature(segment)
    # 进行预测
    predicted_class_index = predict_scene(loaded_model, feature)
    print(f"The detected scene is: {predicted_class_index}")



