# Description: IIRフィルタをかけるテストプログラム
# a,bの係数は別のプログラムで計算してした物を使っている。(計算の効率化のため事前に係数を計算しておく)



import numpy as np
import matplotlib.pyplot as plt


def generate_sine_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 正弦波の生成
    x = np.sin(2 * np.pi * frequency * t)

    return x, t





if __name__ == '__main__':

    # サンプリング周波数
    fs = 1000 

    # テスト用入力用正弦波1
    input_signal_1_frequency = 50  # 周波数 (Hz)
    input_signal_1_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波2
    input_signal_2_frequency = 1  # 周波数 (Hz)
    input_signal_2_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波3
    input_signal_3_frequency = 10  # 周波数 (Hz)
    input_signal_3_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波4
    input_signal_4_frequency = 100  # 周波数 (Hz)
    input_signal_4_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波5
    input_signal_5_frequency = 1000  # 周波数 (Hz)
    input_signal_5_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波6
    input_signal_6_frequency = 5000  # 周波数 (Hz)
    input_signal_6_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波7
    input_signal_7_frequency = 10000  # 周波数 (Hz)
    input_signal_7_duration = 5  # 長さ (seconds)

    # テスト用入力用正弦波8
    input_signal_8_frequency = 20000  # 周波数 (Hz)
    input_signal_8_duration = 5  # 長さ (seconds)

    #ホワイトノイズ生成
    np.random.seed(0) # 乱数のシードを固定
    white_noise = np.random.randn(44100 * 5) * 0.1 # 5秒分のホワイトノイズ生成






    # 入力用正弦波生成
    x1, t1 = generate_sine_wave(input_signal_1_frequency, input_signal_1_duration, fs)
    x2, t2 = generate_sine_wave(input_signal_2_frequency, input_signal_2_duration, fs)
    x3, t3 = generate_sine_wave(input_signal_3_frequency, input_signal_3_duration, fs)
    x4, t4 = generate_sine_wave(input_signal_4_frequency, input_signal_4_duration, fs)
    x5, t5 = generate_sine_wave(input_signal_5_frequency, input_signal_5_duration, fs)
    x6, t6 = generate_sine_wave(input_signal_6_frequency, input_signal_6_duration, fs)
    x7, t7 = generate_sine_wave(input_signal_7_frequency, input_signal_7_duration, fs)
    x8, t8 = generate_sine_wave(input_signal_8_frequency, input_signal_8_duration, fs)

    # 入力信号の合成
    x = x1*2 + x2 + x3 + x4 + x5 + x6 + x7 + x8
    t = t1

    # グラフ表示
    fig = plt.figure()
    ax = fig.add_subplot(211) # 1行1列の描画領域を確保
    ax.plot(t, x, label='input signal')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.legend()
    # ax.set_xlim([4, 4.3]) #x軸の範囲を4~4.3に指定
    ax.grid()
    plt.title('Input Signal')

    # FFTしたグラフ表示
    ax2 = fig.add_subplot(212)
    # x = x[int(3*fs):int(4*fs)] # 1秒分のデータにする
    # y = y[int(3*fs):int(4*fs)] # 1秒分のデータにする
    ax2.plot(np.abs(np.fft.fft(x)), label='input signal')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.set_xlim([0, fs/2]) #x軸の範囲を0~fs/2に指定
    ax2.grid()


    plt.show()

