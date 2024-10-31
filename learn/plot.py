import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 生成模拟 EEG 信号（可以用你的实际 EEG 数据替换）
fs = 256  # 采样率 (Hz)
t = np.linspace(0, 10, fs * 10)  # 10 秒的时间
# 生成一个包含 Delta 波段的信号
delta_signal = 0.5 * np.sin(2 * np.pi * 2 * t)  # Delta 波 (2 Hz)
# 添加一些随机噪声
noise = 0.1 * np.random.normal(size=t.shape)
eeg_signal = delta_signal + noise

# 设计 Butterworth 带通滤波器
lowcut = 4.0  # 低截止频率 (Hz)
highcut = 8.0  # 高截止频率 (Hz)
order = 4  # 滤波器阶数

# 创建带通滤波器
b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)

# 使用 filtfilt 进行零相位滤波
filtered_signal = filtfilt(b, a, eeg_signal)

# 绘制原始和滤波后的信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, eeg_signal, label='Original EEG Signal', color='b')
plt.title('Original EEG Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Delta Band Signal', color='r')
plt.title('Filtered Delta Band Signal (1-4 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.legend()
plt.show()