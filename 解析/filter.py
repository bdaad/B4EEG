import numpy as np



#coding:utf-8
import wave
import struct
import numpy as np
from pylab import *
import pandas as pd

"""IIRフィルタ"""

def createLPF(fc):
    """IIR版ローパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = (4 * np.pi**2 * fc**2) / denom
    b[1] = (8 * np.pi**2 * fc**2) / denom
    b[2] = (4 * np.pi**2 * fc**2) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createHPF(fc):
    """IIR版ハイパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = 1.0 / denom
    b[1] = -2.0 / denom
    b[2] = 1.0 / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createBPF(fc1, fc2):
    """IIR版バンドパスフィルタ、fc1、fc2:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2
    b[0] = (2 * np.pi * (fc2 - fc1)) / denom
    b[1] = 0.0
    b[2] = - 2 * np.pi * (fc2 - fc1) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    a[2] = (1.0 - 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2) / denom
    return a, b

def createBSF(fc1, fc2):
    """IIR版バンドストップフィルタ、fc1、fc2:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2
    b[0] = (4 * np.pi**2 * fc1 * fc2 + 1) / denom
    b[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    b[2] = (4 * np.pi**2 * fc1 * fc2 + 1) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    a[2] = (1 - 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2) / denom
    return a, b

# まだ未完成(うまく動かない。多分計算ミス)
# def create_1D_BSF(fc1, fc2):
#     """IIR版バンドストップフィルタ、fc1、fc2:カットオフ周波数"""
#     a = [0.0] * 3 # 係数a
#     b = [0.0] * 3 # 係数b
#     denom = 1 + 4 * np.pi ** 2 * fc2 * fc1 + 2 * np.pi * (fc2 - fc1) # 分母
#     b[0] = (4 * np.pi * fc2 * fc1 + 1) / denom
#     b[1] = (8 * np.pi * fc2 * fc1 - 2) / denom
#     b[2] = (4 * np.pi * fc2 * fc1 + 1) / denom
#     a[0] = 1.0
#     a[1] = (8 * np.pi ** 2 * fc1 * fc2 -2) / denom
#     a[2] = (4 * np.pi ** 2 * fc1 * fc2 -2 * np.pi * (fc2 - fc1)) / denom
    
#     return a, b



def iir(x, a, b):
    """IIRフィルタをかける、x:入力信号、a, b:フィルタ係数"""
    y = [0.0] * len(x)  # フィルタの出力信号

    Q = len(a) - 1
    P = len(b) - 1
    for n in range(len(x)):
        for i in range(0, P + 1):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, Q + 1):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
        # print(y[n])
    return y



import scipy.fftpack as fft



def my_freqz(b, a=[1], worN=None):
    lastpoint = np.pi
    N = 512 if worN is None else worN
    w = np.linspace(0.0, lastpoint, N, endpoint=False)
    h = fft.fft(b, 2 * N)[:N] / fft.fft(a, 2 * N)[:N]
    return w, h



def generate_sine_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 正弦波の生成
    x = np.sin(2 * np.pi * frequency * t)

    return x, t


def generate_square_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 1周期の時間
    period = 1 / frequency

    # 矩形波の生成
    x = np.where((t % period) < period / 2, 1, -1)

    return x, t
















import numpy as np
import matplotlib.pyplot as plt


# テスト用にサイン波を生成
def generate_test_tone(frequency, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t), t


def iir_real_time(x, a, b, y_prev, x_prev):
    """1サンプルずつIIRフィルタをかける。y_prev: 直前のフィルタ出力、x_prev: 直前の入力"""
    y = b[0] * x + b[1] * x_prev[0] + b[2] * x_prev[1] - a[1] * y_prev[0] - a[2] * y_prev[1]
    # 直前のサンプルを更新
    x_prev[1], x_prev[0] = x_prev[0], x
    y_prev[1], y_prev[0] = y_prev[0], y
    return y



# フィルタのパラメータ設定
fs = 1000  # サンプリングレート
filter_type = "LPF"  # ここを "HPF", "BPF", "BSF" に変更可能
fc_digital = 10.0  # カットオフ周波数（テスト用）

if filter_type == "LPF":
    fc_analog = np.tan(fc_digital * np.pi / fs) / (2 * np.pi)
    a, b = createLPF(fc_analog)
elif filter_type == "HPF":
    fc_analog = np.tan(fc_digital * np.pi / fs) / (2 * np.pi)
    a, b = createHPF(fc_analog)
# 他のフィルタの種類も同様に設定可能

# 過去の値を保持する配列
y_prev = [0.0, 0.0]
x_prev = [0.0, 0.0]



# 生成したテスト信号
test_frequency = 100  # テスト用サイン波の周波数（5Hz）
duration = 2  # テスト信号の長さ（2秒）
input_signal, t = generate_test_tone(test_frequency, duration, fs)

# フィルタを適用する
output_signal = np.zeros_like(input_signal) # フィルタ適用後の信号
for i in range(len(input_signal)):
    output_signal[i] = iir_real_time(input_signal[i], a, b, y_prev, x_prev)

# 結果をプロットして確認
plt.figure(figsize=(10, 6))

# 入力信号のプロット
plt.subplot(2, 1, 1)
plt.plot(t, input_signal, label="Input Signal (5 Hz sine wave)")
plt.title("Input Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 出力信号のプロット
plt.subplot(2, 1, 2)
plt.plot(t, output_signal, label="Filtered Signal", color='red')
plt.title("Filtered Output Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
