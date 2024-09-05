import numpy as np
import matplotlib.pyplot as plt

analysis_interval = 5  # 分析間隔 (seconds)


def generate_sine_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # 正弦波の生成
    x = np.sin(2 * np.pi * frequency * t)
    return t, x

def generate_chirp_wave(start_freq, end_freq, duration, sampling_rate):
    """
    30Hzからスタートし、徐々に周波数が下がっていき、5秒後には5Hz程度になるような正弦波を生成します。

    Parameters:
    start_freq (float): 開始周波数 (Hz)
    end_freq (float): 終了周波数 (Hz)
    duration (float): 時間の長さ (seconds)
    sampling_rate (int): サンプリングレート (samples per second)

    Returns:
    t (numpy.ndarray): 時間データ
    x (numpy.ndarray): 波形データ
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # 周波数が時間に比例して変化するようにする
    k = (end_freq - start_freq) / duration
    instantaneous_freq = start_freq + k * t
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sampling_rate
    x = np.sin(phase)
    return t, x


if __name__ == '__main__':
    fs = 1000  # サンプリング周波数

    input_signal_frequency = float(input("正弦波の周波数: "))  # 周波数 (Hz)
    input_signal_duration = 5  # 長さ (seconds)

    t, x = generate_sine_wave(input_signal_frequency, input_signal_duration, fs)
    # x = x + np.random.normal(0, 0.1, len(x))  # ノイズを加える
    # t, x = generate_chirp_wave(3, 1, 5, fs) # チャープ波を生成

    
    fig = plt.figure()

    # 全体の波形を表示
    ax1 = fig.add_subplot(311)  # 1行1列の描画領域を確保
    ax1.plot(t, x, c="red")  # t と x を正しくプロット
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")


    # データを1秒ごとに分割して重ね合わせて表示
    ax2 = fig.add_subplot(312)  # 1行1列の描画領域を確保
    for i in range(analysis_interval):
        start_idx = i * fs  # 各1秒間の開始インデックス
        end_idx = (i + 1) * fs  # 各1秒間の終了インデックス
        # 各セグメントを同じ時間範囲にプロット
        ax2.plot(t[:fs], x[start_idx:end_idx], label=f"{i}-{i+1} sec")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    ax2.legend()


    # データを1秒ごとに分割して加算平均して表示
    ax3 = fig.add_subplot(313)  # 1行1列の描画領域を確保

    ave_x = np.zeros(fs)
    for i in range(analysis_interval):
        start_idx = i * fs  # 各1秒間の開始インデックス
        end_idx = (i + 1) * fs  # 各1秒間の終了インデックス
        ave_x = ave_x + x[start_idx:end_idx]
    ave_x = np.array(ave_x) / analysis_interval
    ax3.plot(t[:fs], ave_x, c="blue", label="Average")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.legend()

    plt.show()
