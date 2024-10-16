import matplotlib.pyplot as plt
import numpy as np

# 1000hz用

def plot_multiple_lines(y_values):
    """
    引数として与えられたデータを基に、同じ線グラフ上に複数の線を描画します。
    
    Parameters:
    y_values (list of arrays): 描画するデータのリスト。各要素はY軸の値を表します。
    """
    x = np.linspace(0, 0.1, 100)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定

    for i, y in enumerate(y_values):
        plt.plot(x, y, label=f'Line {i+1}')

    plt.title('Multiple Lines on the Same Graph')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend(loc='upper right')
    plt.grid(True)

    # グラフの表示
    plt.show()

# 関数をテスト
x = np.linspace(0, 10, 100)  # 0から10までの100個の等間隔の点
y_values = [np.sin(x + i) for i in range(10)]  # 10個の異なるsin波を生成

plot_multiple_lines(y_values)
