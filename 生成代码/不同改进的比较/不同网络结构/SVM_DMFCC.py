import os
import h5py
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 支持的音频文件类型
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
# 数据准备
DATASET_PATH = "E:\\桌面\\dataset\\train"  # 声学场景数据集的路径
NUM_CLASSES = 10  # 声学场景的类别数
class_labels = {
    0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4',
    5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'
}  # 根据您的类别编号修改

def load_data(dataset_path, num_classes):
    # 加载数据集
    X = []
    y = []
    for folder_name in os.listdir(dataset_path):
        print(folder_name)
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 检查文件是否是支持的音频文件类型
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() not in SUPPORTED_AUDIO_FORMATS:
                continue
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                feature = extract_feature(audio_data, sample_rate)
                X.append(feature.flatten())  # 将特征转换为一维数组
                y.append(label)
            except Exception as e:
                print(f'Error loading {file_path}: {e}')

    # 将特征和标签转换为NumPy数组
    X = np.array(X).astype(np.float32)
    y = np.array(y)
    return X, y
def extract_feature(audio_data, sample_rate):
    # 将增强后的音频数据重采样为原始采样率
    y_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=sample_rate)

    # 提取静态MFCCs特征
    mfccs = librosa.feature.mfcc(y=y_resampled, sr=sample_rate, n_mfcc=20)

    # 计算一阶差分MFCCs特征
    delta_mfccs = librosa.feature.delta(mfccs)

    # 计算二阶差分MFCCs特征
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    # delta3_mfccss = librosa.feature.delta(mfccs, order=3)

    # 根据权重分配，将三种特征合并在一起，得到动态MFCCs特征
    dynamic_mfccs = np.vstack([mfccs, delta_mfccs,delta2_mfccs])

    # 返回动态MFCCs特征
    return dynamic_mfccs
# Test accuracy: 0.3260
# Class 0 Test Accuracy: 0.2623
# Class 1 Test Accuracy: 0.1275
# Class 2 Test Accuracy: 0.1944
# Class 3 Test Accuracy: 0.2215
# Class 4 Test Accuracy: 0.3710
# Class 5 Test Accuracy: 0.1220
# Class 6 Test Accuracy: 0.4937
# Class 7 Test Accuracy: 0.1890
# Class 8 Test Accuracy: 0.7527
# Class 9 Test Accuracy: 0.5355

# 数据集划分
X, y = load_data(DATASET_PATH, NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集已划分")

# 构建SVM模型
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1,max_iter=100))
print("构建SVM模型")

# 训练SVM模型
svm_model.fit(X_train, y_train)
print("训练SVM模型")
# 打印模型结构摘要

# 模型评估
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.4f}')

# 输出每个类别的测试准确率
class_accuracy = {}
for i in range(NUM_CLASSES):
    true_class_samples = y_test[y_test == i]
    pred_class_samples = y_pred[y_test == i]
    class_accuracy[i] = accuracy_score(true_class_samples, pred_class_samples)

# 输出每个类别的测试准确率
for class_idx, acc in class_accuracy.items():
    print(f'Class {class_idx} Test Accuracy: {acc:.4f}')

# 保存模型
print("开始保存模型")
with h5py.File("model_svm.h5", "w") as hf:
    hf.create_dataset("support_vectors", data=svm_model.named_steps['svc'].support_vectors_)
    hf.create_dataset("dual_coef", data=svm_model.named_steps['svc'].dual_coef_)
    hf.create_dataset("intercept", data=svm_model.named_steps['svc'].intercept_)
print("模型保存成功")
