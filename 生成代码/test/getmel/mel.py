import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_file = 'E:\桌面\dataset2\\train\\9\\street_traffic-barcelona-161-4901-a.wav'
y, sr = librosa.load(audio_file)

# 计算梅尔频谱图
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# 应用对数变换
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# 显示梅尔频谱图
# 显示对数梅尔特征图
plt.figure(figsize=(6, 4))
librosa.display.specshow(log_mel_spectrogram, y_axis='mel', x_axis='time')
print("开始")
plt.colorbar(format='%+2.0f dB')
plt.title('Street_traffic', fontsize=14)
plt.savefig('log_mel_spectrogram.png')
# 显示图形并暂停2分钟
plt.show()
plt.pause(120)  # 暂停2分钟