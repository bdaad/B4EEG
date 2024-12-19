import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

# 1000hz用

def plot_phase_ana(y_values):
    x = np.linspace(1, 10, 10)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定

    # 各行の最大値を取得
    # max_values_per_row = np.max(y_values, axis=1) # 各行の最大値を取得
    max_indices_per_row = np.argmax(y_values, axis=1) # 各行の最大値のインデックスを取得
    # print(max_values_per_row)
    # plt.scatter(x, max_values_per_row, label='max_values_per_row')
    print(max_indices_per_row)
    plt.scatter(x, max_indices_per_row, label='max_indices_per_row')

    plt.title('Multiple Lines on the Same Graph')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)


    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_path = f'./plt_img/phase_{current_time}.png'
    
    # グラフを保存 (ファイル名は現在の時刻)
    plt.savefig(file_name_path)

    # グラフの表示
    plt.show()

# 関数をテスト
x = np.linspace(0, 5, 100)  # 0から10までの100個の等間隔の点
y_values = [np.sin(x) for i in range(10)]  # 10個の異なるsin波を生成

plot_phase_ana(y_values)
