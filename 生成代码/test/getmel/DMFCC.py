import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_file = 'E:\桌面\dataset2\\train\\1\\airport-barcelona-0-1-a.wav'
y, sr = librosa.load(audio_file)

# 计算梅尔频谱图
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# 计算MFCC特征
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=13)

# # 计算DMFCC特征
# delta_mfcc = librosa.feature.delta(mfcc)
#
# # 计算Delta Delta MFCC特征
# delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
#
# # 将三种特征合并在一起，得到动态MFCCs特征
# dynamic_mfccs = np.vstack([mfcc, delta_mfcc, delta_delta_mfcc])
# 显示Delta Delta MFCC特征图
plt.figure(figsize=(5, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar(format='%+2.0f')
plt.title('airport')
plt.savefig('delta_delta_mfcc.png')
plt.show()
