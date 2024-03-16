import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# 待检测音频路径
AUDIO_PATH = "E:\桌面\dataset2\\val\tram-barcelona-179-5519-c.wav"

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

# 加载模型
# model = load_model('E:\桌面\声学场景分类\生成代码\音频检测\trained_model.h5')
# 使用原始字符串表示路径
model_path = r'E:\桌面\声学场景分类\生成代码\音频检测\model.h5'
model = load_model(model_path)
print("正在加载模型")
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
        prediction = model.predict(feature)
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
