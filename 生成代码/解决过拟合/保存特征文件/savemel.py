import os
import librosa
import pickle
import librosa.display
import numpy as np
import pickle
# 定义音频文件夹路径和特征文件夹路径
audio_folder = 'E:\桌面\dataset\\train'
feat_folder = 'E:\桌面\声学场景分类\生成代码\解决过拟合\\festure'

# 遍历音频文件夹中的每个音频文件
# 遍历每个音频文件夹
for subfolder in os.listdir(audio_folder):
    subfolder_path = os.path.join(audio_folder, subfolder)

    # 检查是否是目录
    if os.path.isdir(subfolder_path):
        # 遍历当前子文件夹中的音频文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.wav'):
                # 构建音频文件的完整路径
                print(filename)
                audio_path = os.path.join(subfolder_path, filename)

                # 读取音频文件
                y, sr = librosa.load(audio_path, sr=None)

                # 提取MFCC特征
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                # 计算一阶差分和二阶差分
                delta1_mfcc = librosa.feature.delta(mfcc)
                delta2_mfcc = librosa.feature.delta(mfcc, order=2)

                # 将MFCC、一阶差分和二阶差分水平堆叠以获得动态MFCC
                dynamic_mfcc = np.vstack([mfcc, delta1_mfcc, delta2_mfcc])


                # 构建特征文件的路径
                feat_path = os.path.join(feat_folder, os.path.splitext(filename)[0] + '.pkl')
                print(feat_path)
                # 保存MFCC特征为Pickle文件
                with open(feat_path, 'wb') as f:
                    pickle.dump(mfcc, f)
