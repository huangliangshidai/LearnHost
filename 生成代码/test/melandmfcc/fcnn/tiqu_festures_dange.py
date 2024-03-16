# extract_features.py

import os
import h5py
import numpy as np
import librosa

# 定义特征保存文件夹的路径
FEATURE_SAVE_FOLDER = "features"

# 数据准备
DATASET_PATH = "E:\桌面\dataset\\train"  # 声学场景数据集的路径
NUM_CLASSES = 10  # 声学场景的类别数

def data_augmentation(audio_data, sample_rate):
    # 数据增强示例（可以根据需求自定义）
    # 增加噪声
    noise_amplitude = 0.01  # 设置合适的噪声幅度
    noise = noise_amplitude * np.random.randn(len(audio_data))
    augmented_data = audio_data + noise
    return augmented_data

def extract_logmel_feature(audio_data, sample_rate):
    # 提取Log-Mel特征示例
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    logmel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return logmel_spec

def generate_feature_file(file_path):
    # 从单个音频文件生成特征文件
    _, file_ext = os.path.splitext(file_path)
    if file_ext.lower() not in ['.wav', '.mp3', '.ogg', '.flac']:
        return

    try:
        for _ in range(5):  # 每个原始样本生成5个增强样本
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            augmented_data = data_augmentation(audio_data, sample_rate)
            feature = extract_logmel_feature(augmented_data, sample_rate)

            # 从文件路径中提取文件名（可以根据需要修改）
            file_name = os.path.basename(file_path)
            feature_filename = os.path.join(FEATURE_SAVE_FOLDER, f"{file_name}_{_}.h5")
            print(feature_filename)


        # 保存特征到特征文件
        with h5py.File(feature_filename, "w") as hf:
            hf.create_dataset("feature", data=feature)
    except Exception as e:
        print(f'Error processing {file_path}: {e}')



if __name__ == "__main__":
    # 创建特征保存文件夹
    os.makedirs(FEATURE_SAVE_FOLDER, exist_ok=True)
    print("文件夹创建成功")

    # 遍历数据集文件夹并生成特征文件
    for folder_name in os.listdir(DATASET_PATH):
        print(folder_name)
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            generate_feature_file(file_path)
