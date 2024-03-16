import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_file = 'E:\桌面\dataset2\\train\\0\\tram-barcelona-179-5519-a.wav'
y, sr = librosa.load(audio_file)

# 计算MFCC特征
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 添加高斯噪声
def add_gaussian_noise(signal, mean=0, std=0.01):
    noise = np.random.normal(mean, std, len(signal))
    return signal + noise

y_noisy = add_gaussian_noise(y)

# 计算添加噪声后的MFCC特征
mfccs_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=13)

# 显示原始MFCC特征图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('Original DMFCC')

# 显示添加噪声后的MFCC特征图
plt.subplot(1, 2, 2)
librosa.display.specshow(mfccs_noisy, x_axis='time')
plt.colorbar()
plt.title('Noisy DMFCC')

plt.tight_layout()
plt.show()
