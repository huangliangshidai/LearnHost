import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_file = 'E:\桌面\dataset2\\train\\0\\tram-barcelona-179-5519-a.wav'
y, sr = librosa.load(audio_file)

# 计算MFCC特征
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# SpecAugment方法
def spec_augment(mfccs2, num_mask=2, freq_masking=0.15, time_masking=0.20):
    # 执行频率遮挡
    for i in range(num_mask):
        freq = np.random.uniform(low=0.0, high=freq_masking)
        freq = int(freq * mfccs2.shape[0])
        f0 = np.random.randint(0, mfccs2.shape[0] - freq)
        mfccs2[f0:f0 + freq, :] = 0

    # 执行时间遮挡
    for i in range(num_mask):
        time = np.random.uniform(low=0.0, high=time_masking)
        time = int(time * mfccs2.shape[1])
        t0 = np.random.randint(0, mfccs2.shape[1] - time)
        mfccs2[:, t0:t0 + time] = 0

    return mfccs2

# 生成SpecAugment后的MFCC特征
# mfccs_spec_aug = spec_augment(mfccs, freq_masking=0.15, time_masking=0.20)

# 显示原始MFCC特征图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('Original DMFCC')

# 显示SpecAugment后的MFCC特征图
plt.subplot(1, 2, 2)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('SpecAugmented DMFCC')

plt.tight_layout()
plt.show()
