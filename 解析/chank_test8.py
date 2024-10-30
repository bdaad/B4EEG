import matplotlib
matplotlib.use('Agg')  # import matplotlib.pyplot as plt の前に設定
import time
from datetime import datetime
from multiprocessing.managers import ListProxy
import sys
import psutil
import datetime
import re
import serial
import serial.tools.list_ports
import time
import multiprocessing
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import math
from PIL import Image
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from concurrent.futures import ProcessPoolExecutor
import time
import glm  # OpenGL Mathematicsライブラリを使用
import copy
import matplotlib.pyplot as plt






def save_2d_array_to_file(data, list_name):
    # 現在の日時を取得してファイル名に使用
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./save_data/{list_name}_{current_datetime}.txt"
    
    # ファイルを作成して二次元配列データを保存
    with open(file_name, "w") as file:
        for row in data:
            # 各要素を文字列に変換してから結合
            file.write(",".join(map(str, row)) + "\n")
    
    print(f"{file_name} に二次元配列データを保存しました。")
    return file_name  # 保存したファイル名を返す



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

# def iir_real_time_3ch(x, a, b, y_prev, x_prev):
#     """3チャンネル用IIRフィルタをかける (NumPyによるベクトル化)"""
    
#     # 現在の入力値 x とフィルタ係数を NumPy配列として処理
#     x = np.array(x)
#     # b = np.array(b)
#     # a = np.array(a)
    
#     # 3チャンネルのフィルタ適用 (ベクトル化)
#     y = (b[0] * x + b[1] * x_prev[:, 0] + b[2] * x_prev[:, 1]
#          - a[1] * y_prev[:, 0] - a[2] * y_prev[:, 1])
    
#     # 直前のサンプルを更新 (ベクトル化)
#     x_prev[:, 1], x_prev[:, 0] = x_prev[:, 0], x
#     y_prev[:, 1], y_prev[:, 0] = y_prev[:, 0], y

#     return y.tolist(),  y_prev, x_prev  # 状態を返す# リスト形式で返す







# 実装したフィルタ関数
def iir_real_time_3ch(x, a, b, y_prev, x_prev):
    x = np.array(x)  # 入力信号（3チャンネル）
    P = len(b) - 1  # 分子の次数
    Q = len(a) - 1  # 分母の次数

    if x_prev.shape != (3, P):
        x_prev = np.zeros((3, P))
    if y_prev.shape != (3, Q):
        y_prev = np.zeros((3, Q))

    # 分子項の計算（フィードフォワード部分）
    x_terms = b[0] * x  # 現在の入力に対する係数適用
    if P > 0:
        x_terms += np.dot(x_prev, b[1:].reshape(-1, 1)).flatten()

    # 分母項の計算（フィードバック部分）
    y_terms = np.zeros_like(x)
    if Q > 0:
        y_terms += np.dot(y_prev, a[1:].reshape(-1, 1)).flatten()

    # 現在の出力を計算
    y = x_terms - y_terms

    # 過去の入力と出力を更新
    if P > 0:
        x_prev = np.roll(x_prev, shift=1, axis=1)
        x_prev[:, 0] = x
    if Q > 0:
        y_prev = np.roll(y_prev, shift=1, axis=1)
        y_prev[:, 0] = y

    return y.tolist(), y_prev, x_prev


# /**************Serial関連**********************************************/


# シリアル通信のボーレート
bitRate = 115200

# COMポートの一覧表示
def list_com():
    ports = serial.tools.list_ports.comports()
    print("利用可能なCOMポート:")
    for port in ports:
        print(port.device)

# COMポートの入力
def input_com():
    com = input("COMポートを入力してください(例: COM3): ")
    return com

# 1000Hzでデータ要求を送信し、受信も行い、データの数をカウントする関数
def communicate_and_count(ser , received_list, receive_value, clock_signal_1, clock_signal_2, lock):

    interval = 1.0 / 1000  # 1000Hz
    next_time = time.perf_counter()  # 高精度タイマーの現在時刻を取得
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    t = 1
    last_data = None # 最後に受信したデータ(補間用)



    while True:
    # for i in range(10000000):
        current_time = time.perf_counter()  # 現在のタイムスタンプを取得

        # 10秒経過したらループを終了
        if current_time - start_time >= 1 * t:
            # 10秒間で受信したデータの数を表示
            print(f"10秒間で受信したデータの数: {data_count}")
            data_count = 0
            t = t + 1


        # 1000Hzでデータ要求を送信
        if current_time >= next_time:
            ser.write(b"req\n")  # Arduinoにデータ要求コマンドを送信
            next_time += interval  # 次の送信時間を設定

        # データを受信しカウント
        if ser.in_waiting > 0:  # 受信データがあるか確認
            result = ser.readline()  # 改行コードまで読み込む
            if result:
                data_count += 1  # データをカウント
                result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除
                # print(result.decode())  # バイト列を文字列に変換
                # received_data.append(result.decode())  # グローバル配列に追加
                try:
                    int_list_data = [int(x) for x in result.decode().split(',')]
                    # int_list_data = iir_real_time_3ch(int_list_data, a, b, y_prev, x_prev)
                    last_data = int_list_data
                except ValueError:
                    print("ValueError")
                    int_list_data = last_data

                with lock:  # ロックを使って排他制御
                    received_list.append(receive_value)
                    # received_list.append(result.decode())  # 共有リストに追加
                with lock:
                    clock_signal_1.value = True
                    clock_signal_2.value = True
                    receive_value[:] = int_list_data
                    # receive_value[:] = iir_real_time(int_list_data, a, b, y_prev, x_prev)
                    # print("receive_value: ", receive_value)
                    # print(type(receive_value))

        # 次のタイムスタンプまでの残り時間を計算
        sleep_time = next_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)  # 必要な場合のみスリープ
    



# 1000Hzでデータ要求を送信しないで、受信も行い、データの数をカウントする関数
def communicate_and_count_test(ser , received_list, receive_value, clock_signal_1, clock_signal_2, lock):
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    t = 1
    last_data = [0,0,0] # 最後に受信したデータ(補間用)
    int_list_data = [0,0,0] # 最後に受信したデータ(補間用)
    
    # フィルタのパラメータ設定
    # fs = 1000  # サンプリングレート
    a_bp = np.array([1.000000000000000000000000000000,   -9.631588364109775923793677065987,41.791433287055802736631449079141, -107.575108377109458501763583626598,181.921819391459621328976936638355, -211.193487264920719326255493797362,170.448765914121622699894942343235,  -94.434625746604908158587932121009,34.373188588382440400437189964578,   -7.422455995324011013281051418744,0.722058567096400594209626433440])
    b_bp = np.array([0.000000274471095589420208285894,  0.000000000000000000000000000000,-0.000001372355477947100935550350,  0.000000000000000000000000000000,0.000002744710955894201871100701,  0.000000000000000000000000000000,-0.000002744710955894201871100701,  0.000000000000000000000000000000,0.000001372355477947100935550350,  0.000000000000000000000000000000,-0.000000274471095589420208285894])

    # # バンドストップフィルタの係数
    # a_bs = np.array([1.0, -1.8464940847417775, 0.9414300888198024])
    # b_bs = np.array([0.9707150444099012, -1.8464940847417775, 0.9707150444099012])


    # 過去の値を保持する配列
    # バンドパスフィルタ用
    # y_prev_bp = np.zeros((3, 2))  # 3チャンネル、2つの過去の出力値
    # x_prev_bp = np.zeros((3, 2))  # 3チャンネル、2つの過去の入力値
    Q = len(a_bp) - 1
    P = len(b_bp) - 1
    y_prev_bp = np.zeros((3, Q))
    x_prev_bp = np.zeros((3, P))







    # # バンドストップフィルタ用
    # y_prev_bs = np.zeros((3, 2))  # 3チャンネル、2つの過去の出力値
    # x_prev_bs = np.zeros((3, 2))  # 3チャンネル、2つの過去の入力値
    # サンプリングレートを入力: 1000
    # フィルタの種類(LPF, HPF, BPF, BSF)を入力: BPF
    # カットオフ周波数下限fc1を入力: 3
    # カットオフ周波数下限fc2を入力: 20
    # テスト用正弦波の周波数: 50
    # a: [1.0, -1.8962594398557984, 0.8985096404962453]
    # b: [0.05074517975187733, 0.0, -0.05074517975187733]




    # サンプリングレートを入力: 1000
    # フィルタの種類(LPF, HPF, BPF, BSF)を入力: BSF 
    # カットオフ周波数下限fc1を入力: 45.4
    # カットオフ周波数下限fc2を入力: 55
    # テスト用正弦波の周波数: 50
    # a: [1.0, -1.8464940847417775, 0.9414300888198024]
    # b: [0.9707150444099012, -1.8464940847417775, 0.9707150444099012]




    # サンプリングレートを入力: 1000
    # フィルタの種類(LPF, HPF, BPF, BSF)を入力: BPF
    # カットオフ周波数下限fc1を入力: 5
    # カットオフ周波数下限fc2を入力: 15
    # テスト用正弦波の周波数: 30
    # a: [1.0, -1.9361916025752066, 0.9390625058174924]
    # b: [0.030468747091253825, 0.0, -0.030468747091253825]


    while True:
    # for i in range(10000000):
        current_time = time.perf_counter()  # 現在のタイムスタンプを取得
        # 1秒経過したらループを終了
        interval_time = current_time - start_time
        if interval_time >= 1* t:
            print(f"1秒間{interval_time}で受信したデータの数: {data_count}")
            data_count = 0
            t = t + 1

        # データを受信しカウント
        if ser.in_waiting > 0:  # 受信データがあるか確認
            result = ser.readline()  # 改行コードまで読み込む
            if result:
                data_count += 1  # データをカウント
                result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除\r\n
                # result = re.sub(rb'\n$', b'', result)  # 改行コードを削除\n      #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                # result = result + b',0,0'
                # print(result)
                #result.decode()の型を出力する.
                # print(type(result.decode()))
                try:
                    int_list_data = [int(x) for x in result.decode().split(',')]
                    # int_list_data = [int(result.decode()), int(0), int(0)] #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

                    # print(int_list_data)
                    # int_list_data = iir_real_time_3ch(int_list_data, a, b, y_prev, x_prev) # フィルタ処理BPF.

                    # **バンドパスフィルタの適用**
                    int_list_data_bp, y_prev_bp, x_prev_bp = iir_real_time_3ch(int_list_data, a_bp, b_bp, y_prev_bp, x_prev_bp) #バンドパスフィルタの適用.
                    # int_list_data_bs, y_prev_bs, x_prev_bs = iir_real_time_3ch(int_list_data_bp, a_bs, b_bs, y_prev_bs, x_prev_bs) #バンドストップフィルタの適用.



                    int_list_data = int_list_data_bp
                    last_data = int_list_data_bp


                except ValueError:
                    print("ValueError")
                    # int_list_data = last_data
                    int_list_data_bp = last_data

                with lock:  # ロックを使って排他制御
                    received_list.append(receive_value)
                # with lock:
                    clock_signal_1.value = True
                    clock_signal_2.value = True
                    # receive_value[:] = int_list_data
                    receive_value[:] = int_list_data

 
# /**************グラフィック関連**********************************************/
# V-Syncの有効化/無効化
def enable_vsync(enable=True):
    if enable:
        glfw.swap_interval(1)  # V-Syncを有効にする
    else:
        glfw.swap_interval(0)  # V-Syncを無効にする
# /***********************************************************/


# シンプルな頂点シェーダー
vertex_shader_code = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 projection;

out vec2 TexCoord;

void main()
{
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# シンプルなフラグメントシェーダー
fragment_shader_code = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texture1;
void main()
{
    FragColor = texture(texture1, TexCoord);
}
"""

# シェーダーをコンパイルしてプログラムを作成
def create_shader_program():
    vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    return shader_program


# 点滅する画像を描画するクラス
class BlinkingImage:
    def __init__(self, position, size, image_path, display_time, frequency, refresh_rate, start_on, projection):
        self.position = position  # 画像の位置 (x, y)
        self.size = size  # 画像のサイズ（幅と高さのタプル）
        self.display_time = display_time  # 表示秒数
        self.frequency = frequency  # 点滅の周波数
        self.refresh_rate = refresh_rate  # 垂直同期のリフレッシュレート（例: 60Hz）
        self.toggle = start_on  # 点滅の初期状態（ON/OFF）
        self.start_time = time.time()  # 開始時刻
        if frequency > 0:
            self.frames_per_blink = refresh_rate / (2 * frequency)  # 1回の点滅（オンまたはオフ）に必要なフレーム数
        else:
            self.frames_per_blink = None  # 点滅なし（常時表示）
        self.frame_count = 0  # フレームカウンタ
        self.frame_count_not_reset = 0  # リセット無しフレームカウンタ
        self.projection = projection # 投影行列
        self.texture_id = self.load_texture(image_path) # 画像のロードとテクスチャの設定
        self.shader_program = create_shader_program() # シェーダープログラムを作成        
        self.vao, self.vbo, self.ebo = self.create_quad() # 頂点データを設定


    # 画像をロードしてテクスチャを設定
    def load_texture(self, image_path):
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 画像を上下反転する
        image_data = np.array(image, np.uint8)

        texture_id = glGenTextures(1) # テクスチャIDを生成
        glBindTexture(GL_TEXTURE_2D, texture_id) # テクスチャIDをバインド
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data) # テクスチャを設定
        glGenerateMipmap(GL_TEXTURE_2D) # ミップマップを生成
        print(f"Texture ID for {image_path}: {texture_id}")  # テクスチャIDを出力
        return texture_id 

    # 四角形の頂点データを設定
    # def create_quad(self):
    #     # 四角形の頂点データ (位置とテクスチャ座標)
    #     x, y = self.position  # 画像の中心位置
    #     w, h = self.size      # 画像の幅と高さ

    #     vertices = np.array([
    #         # 位置            テクスチャ座標
    #         x - w/2, y - h/2, 0.0,  0.0, 0.0,  # 左下
    #         x + w/2, y - h/2, 0.0,  1.0, 0.0,  # 右下
    #         x + w/2, y + h/2, 0.0,  1.0, 1.0,  # 右上
    #         x - w/2, y + h/2, 0.0,  0.0, 1.0   # 左上
    #     ], dtype=np.float32)

    #     indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    #     vao = glGenVertexArrays(1)
    #     vbo = glGenBuffers(1)
    #     ebo = glGenBuffers(1)

    #     glBindVertexArray(vao)

    #     glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #     glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    #     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    #     glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    #     # 頂点の位置属性
    #     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
    #     glEnableVertexAttribArray(0)

    #     # テクスチャ座標属性
    #     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    #     glEnableVertexAttribArray(1)

    #     glBindBuffer(GL_ARRAY_BUFFER, 0)
    #     glBindVertexArray(0)

    #     return vao, vbo, ebo



    # 四角形の頂点データを設定 numpyを使って書き換え..
    def create_quad(self):
        # 四角形の頂点データ (位置とテクスチャ座標)
        x, y = self.position  # 画像の中心位置
        w, h = self.size      # 画像の幅と高さ

        # 頂点座標 (左下、右下、右上、左上)
        positions = np.array([
            [x - w/2, y - h/2, 0.0],  # 左下
            [x + w/2, y - h/2, 0.0],  # 右下
            [x + w/2, y + h/2, 0.0],  # 右上
            [x - w/2, y + h/2, 0.0],  # 左上
        ], dtype=np.float32)

        # テクスチャ座標 (左下、右下、右上、左上)
        tex_coords = np.array([
            [0.0, 0.0],  # 左下
            [1.0, 0.0],  # 右下
            [1.0, 1.0],  # 右上
            [0.0, 1.0],  # 左上
        ], dtype=np.float32)

        # 頂点データを結合 (位置とテクスチャ座標)
        vertices = np.hstack([positions, tex_coords]).flatten()

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # 頂点の位置属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # テクスチャ座標属性
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao, vbo, ebo


    # 画像を描画
    def draw_image(self):
        glUseProgram(self.shader_program)

        # モデル行列を設定 (位置とサイズを反映)
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(self.position[0], self.position[1], 0.0))
        model = glm.scale(model, glm.vec3(self.size[0], self.size[1], 1.0))

        # モデル行列と投影行列をシェーダーに渡す
        model_location = glGetUniformLocation(self.shader_program, "model")
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model))

        # 投影行列をシェーダーに渡す
        projection_location = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(self.projection))

        # テクスチャのバインド
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # 四角形を描画
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def update(self):
        # 点滅のロジック

        #表示時間に関する重要なコード↓(今回は常時表示だしいいか)...
        # current_time = time.time()
        # elapsed_time = current_time - self.start_time  # 経過時間を秒に変換
        # if self.display_time is not None and elapsed_time >= self.display_time:
        #     return False  # 表示秒数が経過したらFalseを返す

        if self.frames_per_blink is not None: # 点滅の周波数が設定されている場合
            # フレームカウンタを更新
            self.frame_count += 1
            self.frame_count_not_reset += 1

            if self.frame_count >= self.frames_per_blink: # 1回の点滅が終了したら
                self.toggle = not self.toggle  # フラグを反転させて点滅を切り替える
                self.frame_count = 0  # カウンタをリセット

        # 点滅がオンのときだけ画像を描画
        if self.toggle or self.frames_per_blink is None: # 点滅がオフのときは描画しない
            self.draw_image()
        # print(f"frame_count_not: {self.frame_count_not_reset}, toggle: {self.toggle}") 
        return True  # 表示継続中
    



# GLFW初期化とウィンドウ作成
def init_glfw(width, height, title):
    if not glfw.init():
        return None
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, title, None, None) # ウィンドウを作成
    if not window:
        glfw.terminate() # ウィンドウが作成できなかった場合はGLFWを終了
        return None
    glfw.make_context_current(window) # 描画対象のウィンドウを設定
    return window

# /**************並列処理関連**********************************************/




def func_visual(priority, flag_blink_1, flag_blink_2, lock, chank_list_1, adjust_chank_list_1, chank_list_2, adjust_chank_list_2, gaze_flag_1):
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_visual) started with priority {priority}")

    if not glfw.init(): # GLFWを初期化
        return

    # プライマリモニターの解像度を取得
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    monitor_width = video_mode.size.width
    monitor_height = video_mode.size.height

    print(f"Monitor Resolution: {monitor_width}x{monitor_height}")
    window = init_glfw(monitor_width, monitor_height, "Blinking Image") # ウィンドウを作成
    refresh_rate = 60 # 垂直同期のリフレッシュレート
    enable_vsync(True)  # V-Syncを有効にする
    
    width, height = glfw.get_framebuffer_size(window) # 画面サイズを取得し、アスペクト比を維持する
    glViewport(0, 0, width, height) # ビューポートをウィンドウ全体に設定
    projection = setup_projection_for_circle(width, height)  # 投影行列を設定

    # シェーダープログラムを使用して投影行列を渡す
    shader_program = create_shader_program() # シェーダープログラムを作成
    glUseProgram(shader_program) # シェーダープログラムを使用
    projection_location = glGetUniformLocation(shader_program, "projection") # uniform変数の場所を取得
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(projection)) # 投影行列をシェーダーに渡す
    print("shader done!!!!")


    blinking_image1 = BlinkingImage(position=(-1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image2 = BlinkingImage(position=(-0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False, projection=projection)
    blinking_image3 = BlinkingImage(position=(0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image4 = BlinkingImage(position=(1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=False, projection=projection)

    blinking_image1_off = BlinkingImage(position=(-1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=0.000000000001, refresh_rate=refresh_rate, start_on=False, projection=projection)
    


    character_image1 = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/a_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image1_2 = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/i_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image1_3 = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/u_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image1_4 = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/e_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image1_5 = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/o_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    
    character_image1_on = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/a_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image1_2_on = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/i_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image1_3_on = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/u_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image1_4_on = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/e_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image1_5_on = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/o_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)



    character_image2 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ka_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image2_2 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ki_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image2_3 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ku_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image2_4 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ke_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image2_5 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ko_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    
    character_image2_on = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ka_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image2_2_on = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ki_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image2_3_on = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ku_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image2_4_on = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ke_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image2_5_on = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ko_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)



    character_image3 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/sa_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image3_2 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/si_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image3_3 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/su_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image3_4 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/se_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    # character_image3_5 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/so_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    
    character_image3_on = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/sa_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image3_2_on = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/si_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image3_3_on = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/su_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image3_4_on = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/se_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image3_5_on = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/so_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)



    character_image4 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ta_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image4_2 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ti_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image4_3 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/tu_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image4_4 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/te_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    # character_image4_5 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/to_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)

    character_image4_on = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ta_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image4_2_on = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ti_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image4_3_on = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/tu_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image4_4_on = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/te_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image4_5_on = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/to_on.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)


    images = [blinking_image1, blinking_image2, blinking_image3, blinking_image4, character_image1, character_image2, character_image3, character_image4]
    # images = [blinking_image, blinking_image2, blinking_image3, blinking_image4]
    # images = [blinking_image]

    previous_time = time.time()
    frame_count = 0
    fullscreen = True  # 現在の状態を管理
    character_count = 0 # 文字の表示を管理
    flag_a = False


    while not glfw.window_should_close(window): # ウィンドウが閉じられるまでループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア

        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?

        # 10Hzの1周期分.. 60/10 = 6
        with lock:
            if blinking_image1.frame_count_not_reset % 6 == 0:            
                if flag_blink_1.value == True:
                    flag_blink_1.value = False
                else:
                    flag_blink_1.value = True

        
        # # 12Hzの1周期分.. 60/12 = 5
        # with lock:
        #     if blinking_image1.frame_count_not_reset % 5 == 0:
        #         if flag_blink_2.value == True:
        #             flag_blink_2.value = False
        #         else:
        #             flag_blink_2.value = True

        #         if blinking_image1.frame_count_not_reset % 6 == 0:            
        #     if flag_blink_1.value == True:
        #         with lock:
        #             flag_blink_1.value = False
        #     else:
        #         with lock:
        #             flag_blink_1.value = True

        # # 12Hzの1周期分.. 60/12 = 5        
        # if blinking_image1.frame_count_not_reset % 5 == 0:
        #     if flag_blink_2.value == True:
        #         with lock:
        #             flag_blink_2.value = False
        #     else:
        #         with lock:
        #             flag_blink_2.value = True






        # # 遅延探しテスト用コード本番では使用しない これは6秒後に点滅をオフにする.
        # if blinking_image1.frame_count_not_reset == 300:
        #     images[0] = blinking_image1_off




        
        if gaze_flag_1.value == True:
            if character_count == 0:
                images[4] = character_image1_on #offあをonあに変更
                flag_a = True
            elif character_count == 60:
                images[4] = character_image1_2_on #onあをonいに変更
            elif character_count == 120:
                images[4] = character_image1_3_on  #onいをonうに変更
            elif character_count == 180:
                images[4] = character_image1_4_on #onうをonえに変更
            elif character_count == 240:
                images[4] = character_image1_5_on #onえをonおに変更
            elif character_count == 300:
                character_count = -1
            character_count += 1

    
        if gaze_flag_1.value == False and flag_a == True:
            if character_count >= 0 and character_count < 60:
                print("あ")
            elif character_count >= 60 and character_count < 120:
                print("い")
            elif character_count >= 120 and character_count < 180:
                print("う")
            elif character_count >= 180 and character_count < 240:
                print("え")
            elif character_count >= 240 and character_count < 300:
                print("お")
            character_count = 0
            flag_a = False
            images[4] = character_image1 #on???をoffあに変更

        # フレームカウンタを更新
        frame_count += 1
        current_time = time.time()

        # 1秒ごとにFPSを計算して出力
        interval = current_time - previous_time
        if interval >= 1.0:
            fps = frame_count / (interval)
            print(f"FPS: {fps:.2f}, frame_count: {frame_count}")
            previous_time = current_time
            frame_count = 0

        # ESCキーで全画面モードを終了し、ウィンドウモードに切り替え
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS and fullscreen:
            glfw.set_window_monitor(window, None, 100, 100, 800, 600, 0)  # ウィンドウモードに切り替え
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (800 / 600), 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -5)
            fullscreen = False

        # TABキーでプログラムを終了
        if glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS:
            # ./save_dataにデータを保存する. セーブするデータは、chank_list_1, adjust_chank_list_1, chank_list_2, adjust_chank_list_2
            with lock:
                save_2d_array_to_file(chank_list_1, "chank_list_1")
                save_2d_array_to_file(adjust_chank_list_1, "adjust_chank_list_1")
                save_2d_array_to_file(chank_list_2, "chank_list_2")
                save_2d_array_to_file(adjust_chank_list_2, "adjust_chank_list_2")
            sys.exit()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()



def setup_projection_for_circle(width, height):
    aspect_ratio = width / height # ウィンドウのアスペクト比を取得
    if aspect_ratio >= 1.0: # アスペクト比に基づいてオルソグラフィック投影の範囲を調整
        # 横長の場合、X軸を拡張
        projection = glm.ortho(-aspect_ratio, aspect_ratio, -1.0, 1.0)
    else:
        # 縦長の場合、Y軸を拡張
        projection = glm.ortho(-1.0, 1.0, -1.0/aspect_ratio, 1.0/aspect_ratio)
    return projection




def func_serial(priority, com, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock):
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_serial) started with priority {priority}")

    while True:
        try:
            ser = serial.Serial(com, bitRate, timeout=None)
            break
        except serial.SerialException:
            print("COMポートが開けませんでした。再度入力してください。")
    
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    communicate_and_count_test(ser, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock)







def func_chank(priority, receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, chank_size, lock):
    """
    1000data / 10Hz = 100data
    1000data / 12Hz = 83.3333data = 83data
    1000data / 15Hz = 66.6666data = 67data
    """
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_chank) started with priority {priority}")
    flag_state = None
    chank_chank_list_1 = [] #buffer1
    chank_chank_list_2 = [] #buffer2
    pretime = time.time()
    current_time = 0;
    # po = 0

    print("chank_size: {chank_size}")
    print("chank_size: {chank_size}")
    print("chank_size: {chank_size}")



    while True:
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime
                    
                    with lock:
                        chank_list.append(chank_chank_list_2)

                        chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                        
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=chank_size)) 
                        analysis_flag.value = True
                    chank_chank_list_2 = []
                    pretime = current_time
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime

                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=chank_size))
                        analysis_flag.value = True
                    chank_chank_list_1 = []
                    print("chank_list len: ", len(chank_list_copy), "interval_time: ", interval_time)
                    pretime = current_time

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False



def func_chank_10hz(priority, receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_chank_10hz) started with priority {priority}")
    flag_state = None
    chank_chank_list_1 = [] #buffer1
    chank_chank_list_2 = [] #buffer2
    pretime = time.time()
    current_time = 0;
    # po = 0

    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")

    while True:
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime
                    
                    with lock:
                        chank_list.append(chank_chank_list_2)

                        chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                        
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100)) #1000data / 10Hz = 100data
                        analysis_flag.value = True
                    chank_chank_list_2 = []
                    pretime = current_time
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime

                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100))
                        analysis_flag.value = True
                    chank_chank_list_1 = []
                    print("chank_list len: ", len(chank_list_copy), "interval_time: ", interval_time)
                    pretime = current_time

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False
    


def func_chank_12hz(priority, receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_chank_12hz) started with priority {priority}")
    flag_state = None
    chank_chank_list_1 = [] #buffer1
    chank_chank_list_2 = [] #buffer2
    pretime = time.time()
    current_time = 0;
    # po = 0

    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")

    while True:
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime
                    
                    with lock:
                        chank_list.append(chank_chank_list_2)

                        chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                        
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=83)) #1000data / 12Hz = 83data
                        analysis_flag.value = True
                    chank_chank_list_2 = []
                    pretime = current_time
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime

                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=83))
                        analysis_flag.value = True
                    chank_chank_list_1 = []
                    print("chank_list len: ", len(chank_list_copy), "interval_time: ", interval_time)
                    pretime = current_time

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False
    




# import win_precise_time


def func_analysis(priority, adjust_chank_list, analysis_flag, gaze_flag, lock):
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_analysis) started with priority {priority}")
    chank_copy = []
    # flag = False
    count = 0
    time.sleep(3)
    print("分析")
    
    while True: # 20個のデータが溜まったら..分析を行う
        if len(adjust_chank_list) >= 20:
            break

    print("分析開始")
        
    # 分析を行う.
    while True:
        if analysis_flag.value == True:
            with lock:
                # chank_copy = copy.deepcopy(list(adjust_chank_list[-20:])) #最後の20個のデータをコピー
                chank_copy = adjust_chank_list[-20:] #最後の20個のデータをコピー
                analysis_flag.value = False

            # chank_copyの要素を出力する
            # print("chank_copy")
            # for row in chank_copy:
            #     print(row)
                    
            # print("行: ", len(chank_copy))#行数
            # print("列: ", len(chank_copy[0]))#列数

            plot_multiple_lines(chank_copy, count, gaze_flag, "10Hz")
            plot_phase_ana(chank_copy, count, gaze_flag, "10Hz")
            # print("11111111111111111111")
            # print(time.time())
            count = count + 1
            # print("count: ", count)
                    # flag = True
        # else:
            # win_precise_time.sleep(0.001)



def func_analysis2(priority, adjust_chank_list_1, analysis_flag_1, gaze_flag_1, adjust_chank_list_2, analysis_flag_2, gaze_flag_2, lock):
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_analysis) started with priority {priority}")
    chank_copy = []
    chank_copy2 = []
    # flag = False
    count = 0
    count2 = 0
    time.sleep(3)
    print("分析")
    
    while True: # 20個のデータが溜まったら..分析を行う
        if len(adjust_chank_list_1) >= 20:
            break

    print("分析開始")
        
    # 分析を行う.
    while True:
        if analysis_flag_1.value == True:
            with lock:
                # chank_copy = copy.deepcopy(list(adjust_chank_list[-20:])) #最後の20個のデータをコピー
                chank_copy = adjust_chank_list_1[-20:] #最後の20個のデータをコピー
                analysis_flag_1.value = False
            plot_multiple_lines(chank_copy, count, gaze_flag_1, "10Hz", 0, 0.1, 100)
            plot_phase_ana(chank_copy, count, gaze_flag_1, "10Hz", 1, 20, 20)
            count = count + 1


        if analysis_flag_2.value == True:
            with lock:
                # chank_copy = copy.deepcopy(list(adjust_chank_list[-20:])) #最後の20個のデータをコピー
                chank_copy2 = adjust_chank_list_2[-20:] #最後の20個のデータをコピー
                analysis_flag_2.value = False
            plot_multiple_lines(chank_copy2, count2, gaze_flag_2, "12Hz", 0, 0.83, 83)
            plot_phase_ana(chank_copy2, count2, gaze_flag_2, "12Hz", 1, 20, 20)
            count2 = count2 + 1




def plot_multiple_lines(y_values, count, gaze_flag, folder, start, end, num_points): #平均値の追加
    """
    start, # 開始値
    end, # 終了値
    num_points # 生成する数値の数
    """
    x = np.linspace(start, end, num_points)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定

    if count % 20 == 0:
        for i, y in enumerate(y_values):
            plt.plot(x, y, label=f'Line {i+1}')

        plt.title('Multiple Lines on the Same Graph')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.legend(loc='upper right')
        plt.grid(True)

        current_time = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")
        file_name_path = f'./plt_img/add_ave/{folder}/{current_time}.png'
        plt.savefig(file_name_path, dpi=70)
        # plt.show() # グラフの表示
        plt.close()




def plot_phase_ana(y_values, count, gaze_flag, folder, start, end, num_points): #位相分析
    x = np.linspace(start, end, num_points)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定


    max_indices_per_row = np.argmax(y_values, axis=1) # 各行の最大値のインデックスを取得. 要素数は20個
    # ここに位相分析の処理を書く
        # None.
    # max_indices_per_rowが10~50に8個以上ある場合、gaze_flagをTrueにする
    if len(max_indices_per_row[(max_indices_per_row >= 10) & (max_indices_per_row <= 50)]) >= 15: #10~50の範囲に16個以上ある場合
        gaze_flag.value = True
        
    else:
        gaze_flag.value = False
        print("gaze_flag: false")


    if count % 20 == 0:
        # max_indices_per_row = np.argmax(y_values, axis=1) # 各行の最大値のインデックスを取得


        # print(max_indices_per_row)
        # plt.scatter(x, max_indices_per_row, label='max_indices_per_row')
        plt.plot(x, max_indices_per_row, 'o', label='max_indices_per_row')  # `scatter`の代わりに`plot`を使用

        plt.title('Multiple Lines on the Same Graph')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.ylim(0, 100)
        plt.legend(loc='upper right')
        plt.grid(True)


        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name_path = f'./plt_img/phase/{folder}/{current_time}.png'
        # dir_path = './plt_img'
        # path = os.path.join(dir_path, file_name)
        # グラフを保存 (ファイル名は現在の時刻)
        plt.savefig(file_name_path, dpi=70)

        # グラフの表示
        # plt.show()
        plt.close()




def adjust_data_to_size(data, target_size):
    # 行数を取得
    row = len(data)
    if row == 0:
        print("No data available")
        return data
    elif row >= 1:
        # 行数が1行の場合でも、データをtarget_sizeに揃える処理
        if len(data) < target_size: #行数がtarget_size未満の場合
            # print(f"len(data) < {target_size}")
            needed_length = target_size - len(data)
            # データが不足している場合、0で埋める
            data = [0] * needed_length + data
        elif len(data) > target_size: #行数がtarget_sizeより大きい場合
            # data = data[:target_size]  # 最初のtarget_size個に切り捨て
            data = data[-target_size:]  # 後ろのtarget_size個を取得
        return data
    # else:
    #     # 最後の行の要素数がtarget_size未満の場合
    #     if len(data) < target_size:
    #         print(f"len(data) < {target_size}")
    #         # 最後の行の要素数がtarget_size未満の場合、最後の行の要素数をtarget_size個にする
    #         needed_length = target_size - len(data)
    #         data = data[row - 2][-needed_length:] + data
    #         return data
    #     elif len(data) > target_size:
    #         print(f"len(data) > {target_size}")
    #         # 最後の行の要素数がtarget_sizeより大きい場合、最後の行の要素数をtarget_size個にする
    #         remainder = len(data) - target_size
    #         data = data[int(remainder/2):int(remainder/2) + target_size]
    #         return data
    #     else:
    #         return data





# /**************main関数**********************************************/


def main():
        # 共有リストとロックを作成
    manager = multiprocessing.Manager()
    shared_receive_list = manager.list()  # 共有リスト
    # lock_receive_list = multiprocessing.Lock()  # ロック
    flag_blink_1 = manager.Value('b', True)
    flag_blink_2 = manager.Value('b', True)
    # lock_flag_blink_1 = multiprocessing.Lock()
    chank_list_1 = manager.list()
    chank_list_2 = manager.list()

    # lock_chank_list = multiprocessing.Lock()
    receive_value = manager.list()  # 共有リスト
    # lock_receive_value = multiprocessing.Lock()
    clock_signal_1 = manager.Value('b', False)
    clock_signal_2 = manager.Value('b', False)
    # lock_clock_signal = multiprocessing.Lock()
    adjust_chank_list_1 = manager.list()
    adjust_chank_list_2 = manager.list()

    analysis_flag_1 = manager.Value('b', False)
    analysis_flag_2 = manager.Value('b', False)

    gaze_flag_1 = manager.Value('b', False)
    gaze_flag_2 = manager.Value('b', False)

    lock = multiprocessing.Lock()



    # psutil.IDLE_PRIORITY_CLASS (64): 最低優先度
    # psutil.BELOW_NORMAL_PRIORITY_CLASS (16384): 通常より低い優先度
    # psutil.NORMAL_PRIORITY_CLASS (32): 通常の優先度
    # psutil.ABOVE_NORMAL_PRIORITY_CLASS (32768): 通常より高い優先度
    # psutil.HIGH_PRIORITY_CLASS (128): 高い優先度
    # psutil.REALTIME_PRIORITY_CLASS (256): 最高優先度
    # プロセスの優先度を設定
    priority1 = psutil.REALTIME_PRIORITY_CLASS
    priority2 = psutil.REALTIME_PRIORITY_CLASS
    priority3 = psutil.REALTIME_PRIORITY_CLASS
    priority4 = psutil.REALTIME_PRIORITY_CLASS
    priority5 = psutil.REALTIME_PRIORITY_CLASS

        
    list_com()# COMポート一覧を表示
    # com = input_com()# COMポート接続の初期化
    com = "COM8"
    # com = input_com()
    # print(com)

    
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     # e.submit(func_1)
    #     e.submit(func_serial, com)
    #     # e.submit(func_visual)
    # 並列処理で実行するプロセスを定義

    
    process1 = multiprocessing.Process(target=func_serial, args=(priority1, com, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock))
    
    # process2 = multiprocessing.Process(target=func_chank_10hz, args=(priority2, receive_value, flag_blink_1, chank_list_1, clock_signal_1, adjust_chank_list_1, analysis_flag_1, lock))
    # process3 = multiprocessing.Process(target=func_chank_12hz, args=(priority3, receive_value, flag_blink_2, chank_list_2, clock_signal_2, adjust_chank_list_2, analysis_flag_2, lock))
    process2 = multiprocessing.Process(target=func_chank, args=(priority2, receive_value, flag_blink_1, chank_list_1, clock_signal_1, adjust_chank_list_1, analysis_flag_1, 100, lock)) #10Hz: 100data / 10Hz = 100
    process3 = multiprocessing.Process(target=func_chank, args=(priority3, receive_value, flag_blink_2, chank_list_2, clock_signal_2, adjust_chank_list_2, analysis_flag_2, 83, lock)) #12Hz: 100data / 12Hz = 83
    
    process4 = multiprocessing.Process(target=func_visual, args=(priority4, flag_blink_1, flag_blink_2, lock, chank_list_1, adjust_chank_list_1, chank_list_2, adjust_chank_list_2, gaze_flag_1))
    
    # process5 = multiprocessing.Process(target=func_analysis, args=(priority5, adjust_chank_list_1 ,analysis_flag_1, gaze_flag_1, lock))
    process5 = multiprocessing.Process(target=func_analysis2, args=(priority5, adjust_chank_list_1 ,analysis_flag_1, gaze_flag_1, adjust_chank_list_2 ,analysis_flag_2, gaze_flag_2,lock))



    # プロセスの開始
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()



    main_process = psutil.Process()  # 自身のプロセスを取得
    print(f"main_process PID: {main_process.pid}")
    print(f"process1 PID: {process1.pid}")
    print(f"process2 PID: {process2.pid}")
    print(f"process3 PID: {process3.pid}")
    print(f"process4 PID: {process4.pid}")
    print(f"process5 PID: {process5.pid}")


    # プロセスの終了を待つ
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
# /***********************************************************/


if __name__ == '__main__':
    main()


# 今日
