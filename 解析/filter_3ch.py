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

def generate_test_tone(frequency, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t), t

# def iir_real_time_3ch(x, a, b, y_prev, x_prev):
#     """3チャンネル用IIRフィルタをかける"""
#     y1 = b[0] * x[0] + b[1] * x_prev[0,0] + b[2] * x_prev[0,1] - a[1] * y_prev[0,0] - a[2] * y_prev[0,1]
#     y2 = b[0] * x[1] + b[1] * x_prev[1,0] + b[2] * x_prev[1,1] - a[1] * y_prev[1,0] - a[2] * y_prev[1,1]
#     y3 = b[0] * x[2] + b[1] * x_prev[2,0] + b[2] * x_prev[2,1] - a[1] * y_prev[2,0] - a[2] * y_prev[2,1]

#     # 直前のサンプルを更新
#     x_prev[0,1], x_prev[0,0] = x_prev[0,0], x[0]
#     x_prev[1,1], x_prev[1,0] = x_prev[1,0], x[1]
#     x_prev[2,1], x_prev[2,0] = x_prev[2,0], x[2]

#     y_prev[0,1], y_prev[0,0] = y_prev[0,0], y1
#     y_prev[1,1], y_prev[1,0] = y_prev[1,0], y2
#     y_prev[2,1], y_prev[2,0] = y_prev[2,0], y3

#     return [y1, y2, y3]




def iir_real_time_3ch(x, a, b, y_prev, x_prev):
    """3チャンネル用IIRフィルタをかける (NumPyによるベクトル化)"""
    
    # 現在の入力値 x とフィルタ係数を NumPy配列として処理
    x = np.array(x)
    # b = np.array(b)
    # a = np.array(a)
    
    # 3チャンネルのフィルタ適用 (ベクトル化)
    y = (b[0] * x + b[1] * x_prev[:, 0] + b[2] * x_prev[:, 1]
         - a[1] * y_prev[:, 0] - a[2] * y_prev[:, 1])
    
    # 直前のサンプルを更新 (ベクトル化)
    x_prev[:, 1], x_prev[:, 0] = x_prev[:, 0], x
    y_prev[:, 1], y_prev[:, 0] = y_prev[:, 0], y

    return y.tolist()  # リスト形式で返す

# フィルタのパラメータ設定
fs = 1000  # サンプリングレート
a, b = np.array([1.0, -1.8, 0.81]), np.array([0.1, 0.2, 0.1])  # 仮のフィルタ係数

# 過去の値を保持する配列
y_prev = np.zeros((3, 2))  # 3チャンネル、2次フィルタ
x_prev = np.zeros((3, 2))

# テスト用サイン波の生成
test_frequency = 100  # チャンネル1
test_frequency2 = 200  # チャンネル2
test_frequency3 = 300  # チャンネル3
duration = 2  # 2秒間

input_signal, t = generate_test_tone(test_frequency, duration, fs)
input_signal2, t = generate_test_tone(test_frequency2, duration, fs)
input_signal3, t = generate_test_tone(test_frequency3, duration, fs)
input_all = [input_signal, input_signal2, input_signal3]

# フィルタを適用
output_all = [np.zeros_like(input_signal), np.zeros_like(input_signal2), np.zeros_like(input_signal3)]

for i in range(len(input_signal)):
    x = [input_all[0][i], input_all[1][i], input_all[2][i]]  # 3チャンネルの現在のサンプル
    y = iir_real_time_3ch(x, a, b, y_prev, x_prev)
    output_all[0][i], output_all[1][i], output_all[2][i] = y

# 結果をプロットして確認
plt.figure(figsize=(10, 6))

# 入力信号のプロット
plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(t, input_all[i], label=f"Input Signal {i+1}")
plt.title("Input Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# 出力信号のプロット
plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(t, output_all[i], label=f"Filtered Signal {i+1}")
plt.title("Filtered Output Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
