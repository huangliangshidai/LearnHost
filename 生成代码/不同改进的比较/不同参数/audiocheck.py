import os
import h5py
import numpy as np
import librosa
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import soundfile as sf

# 待检测音频路径
AUDIO_PATH = "E:/桌面/dataset/val/tram-barcelona-179-5519-c.wav"

# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']

# 类别标签
class_labels = {
    0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
    5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'
}  # 根据您的类别编号修改

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

# 加载模型函数
def load_custom_model(model_path):
    # 构建空的 Sequential 模型
    loaded_model = Sequential()
    # 加载保存的 HDF5 文件
    with h5py.File(model_path, "r") as hf:
        print("加载模型0")
        for layer_name in hf.keys():
            print("加载模型1")
            layer_group = hf[layer_name]
            if 'weights' in layer_group.keys():
                print("加载模型2")
                layer_weights = layer_group['weights'][...]
                units = layer_weights.shape[-1]  # 获取输出单元数，即权重数组的最后一个维度大小
                # 构建 Dense 层
                layer = Dense(units, input_shape=(layer_weights.shape[0],), activation='softmax')
                layer.set_weights([layer_weights])
                loaded_model.add(layer)
    return loaded_model

# 将音频分割为指定数量的片段
def split_audio(audio_path, num_segments=10):
    print(f"正在将音频分割为{num_segments}个片段")
    y, sr = librosa.load(audio_path, sr=None)
    segment_length = len(y) // num_segments
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(num_segments)]
    return segments

# 对特征进行分类
def classify_feature(loaded_model, feature):
    # 对特征进行预处理（如果有必要）
    # 进行分类
    predictions = loaded_model.predict(feature)
    # 使用softmax函数获得类别的概率分布
    class_probs = np.squeeze(predictions)  # 去掉维度为1的维度
    # 获取分类结果，即概率最大的类别索引
    class_index = np.argmax(class_probs)
    # 返回分类结果和概率分布
    return class_index, class_probs

if __name__ == "__main__":
    # 使用原始字符串表示路径
    model_path = r'E:\桌面\声学场景分类\生成代码\不同改进的比较\不同参数\model_custom10.h5'

    # 加载模型
    print("正在加载模型")
    loaded_model = load_custom_model(model_path)

    # 将音频分割为10个片段
    segments = split_audio(AUDIO_PATH, num_segments=10)

    # 对每个片段进行场景分类
    segment_results = []
    segment_probs = []
    for i, segment in enumerate(segments):
        # 保存片段为单独的音频文件
        segment_file = f"segment_{i}.wav"
        sf.write(segment_file, segment, 44100)
        # 提取特征并保存到文件中
        feature = extract_feature(segment_file)
        # 然后将特征传递给模型进行分类
        result, probs = classify_feature(loaded_model, feature)
        print(result)
        segment_results.append(result)
        segment_probs.append(probs)
        # 删除临时保存的音频文件
        os.remove(segment_file)


# 展示每个片段的场景分类结果和概率分布
    for i, segment_result in enumerate(segment_results):
        class_label = class_labels[segment_result]
        print(f'Segment {i+1} belongs to class: {class_label}')
        print(f'Class probabilities: {segment_probs[i]}')
