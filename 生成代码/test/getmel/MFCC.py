import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_file = 'E:\桌面\dataset2\\train\\0\\tram-barcelona-179-5519-a.wav'
y, sr = librosa.load(audio_file)

# 计算MFCC特征
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 显示MFCC特征图
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.savefig('mfcc.png')

# 显示图形并暂停2分钟
plt.show()
plt.pause(120)  # 暂停2分钟
