# このコードは2024/12/04時点で最新のコード.
# たかあきpcで動作確認済み.
# 実験室pcではうまく動作しない可能性大.

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


# 極大値.
def find_local_maxima(arr, top_n=None):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("1次元の配列のみサポートされています")
    
    maxima = np.zeros_like(arr, dtype=bool)
    # 内部の要素について、前後の要素と比較
    maxima[1:-1] = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])
    
    # 始点と終点は局地的極値とみなさない
    maxima[0] = False
    maxima[-1] = False
    
    # 極大値のインデックスと値を取得
    maxima_indices = np.where(maxima)[0]
    maxima_values = arr[maxima_indices]
    
    # 極大値を大きい順にソート
    sorted_indices = np.argsort(-maxima_values)
    maxima_indices = maxima_indices[sorted_indices]
    maxima_values = maxima_values[sorted_indices]
    
    # 上位n個の極大値を取得
    if top_n is not None:
        maxima_indices = maxima_indices[:top_n]
        maxima_values = maxima_values[:top_n]
    
    return maxima_indices, maxima_values


# 極小値.
def find_local_minima(arr, top_n=None):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("1次元の配列のみサポートされています")
    
    minima = np.zeros_like(arr, dtype=bool)
    # 内部の要素について、前後の要素と比較
    minima[1:-1] = (arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:])
    
    # 始点と終点は局地的極値とみなさない
    minima[0] = False
    minima[-1] = False
    
    # 極小値のインデックスと値を取得
    minima_indices = np.where(minima)[0]
    minima_values = arr[minima_indices]
    
    # 極小値を小さい順にソート
    sorted_indices = np.argsort(minima_values)
    minima_indices = minima_indices[sorted_indices]
    minima_values = minima_values[sorted_indices]
    
    # 上位n個の極小値を取得
    if top_n is not None:
        minima_indices = minima_indices[:top_n]
        minima_values = minima_values[:top_n]
    
    return minima_indices, minima_values


def save_2d_array_to_file(data, list_name):
    # 現在の日時を取得してファイル名に使用
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./save_data/{list_name}_{current_datetime}.txt"
    
    # ファイルを作成して二次元配列データを保存
    with open(file_name, "w") as file:
        # 一次元の場合の特別処理
        if isinstance(data[0], (int, float)):  
            file.write(",".join(map(str, data)) + "\n")
        else:  # 二次元データの場合
            for row in data:
                # 各要素を文字列に変換してから結合
                file.write(",".join(map(str, row)) + "\n")
    
    print(f"{file_name} に配列データを保存しました。")
    return file_name  # 保存したファイル名を返す





# 実装したフィルタ関数
def iir_real_time_3ch(x, a, b, y_prev, x_prev):
    x = np.array(x, dtype=float)  # 入力信号（3チャンネル）
    P = len(b) - 1  # 分子の次数
    Q = len(a) - 1  # 分母の次数

    if x_prev.shape != (3, P):
        x_prev = np.zeros((3, P), dtype=float)
    if y_prev.shape != (3, Q):
        y_prev = np.zeros((3, Q), dtype=float)

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




def setup_projection_for_circle(width, height):
    aspect_ratio = width / height # ウィンドウのアスペクト比を取得
    if aspect_ratio >= 1.0: # アスペクト比に基づいてオルソグラフィック投影の範囲を調整
        # 横長の場合、X軸を拡張
        projection = glm.ortho(-aspect_ratio, aspect_ratio, -1.0, 1.0)
    else:
        # 縦長の場合、Y軸を拡張
        projection = glm.ortho(-1.0, 1.0, -1.0/aspect_ratio, 1.0/aspect_ratio)
    return projection













def func_visual_preparation(priority, measurement_command, lock):
    '''
    measurement_command = 0 : 初期状態(計測なし)
    measurement_command = 1 : インターバル命令(計測なし)

    measurement_command = 2 : 10Hz計測開始命令
    measurement_command = 3 : 10Hz計測終了命令

    measurement_command = 4 : 6Hz計測開始命令
    measurement_command = 5 : 6Hz計測終了命令

    measurement_command = 6 : 非注視計測開始命令
    measurement_command = 7 : 非注視計測終了命令

    measurement_command = 8 : 計測自体を終了する命令.
    '''
    
    
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


    # 画像の初期化
    center_look_point_image = BlinkingImage(position=(0.0, 0.0), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    top_look_point_image = BlinkingImage(position=(0.0, 0.4), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    bottom_look_point_image = BlinkingImage(position=(0.0, -0.4), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    left_look_point_image = BlinkingImage(position=(-0.4, 0.0), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    right_look_point_image = BlinkingImage(position=(0.4, 0.0), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)


    blinking_image1 = BlinkingImage(position=(-0.0, 0.0), size=(0.5, 0.5), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True, projection=projection)
    re_blinking_image1 = BlinkingImage(position=(-0.0, 0.0), size=(0.5, 0.5), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image1_p = BlinkingImage(position=(-1.2, 0.0), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    
    blinking_image2 = BlinkingImage(position=(-0.5, 0.0), size=(0.5, 0.5), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False, projection=projection)
    
    blinking_image3 = BlinkingImage(position=(0.5, 0.0), size=(0.5, 0.5), image_path="./circle.png", display_time=None, frequency=7.5, refresh_rate=refresh_rate, start_on=True, projection=projection)# fre_change_word.
    blinking_image3_p = BlinkingImage(position=(0.7, 0.0), size=(0.2, 0.2), image_path="./img_file/look_point.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)# fre_change_word.


    blinking_image4 = BlinkingImage(position=(1.0, 0.0), size=(0.5, 0.5), image_path="./circle.png", display_time=None, frequency=7.5, refresh_rate=refresh_rate, start_on=False, projection=projection)# fre_change_word.
    blinking_image1_off = BlinkingImage(position=(-1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=0.000000000001, refresh_rate=refresh_rate, start_on=False, projection=projection)
    

    # images = [blinking_image1, blinking_image2, blinking_image3, blinking_image4, look_point_image]
    # images = [blinking_image, blinking_image2, blinking_image3, blinking_image4]
    # images = [blinking_image]


    previous_time = time.time()
    frame_count = 0
    fullscreen = True  # 現在の状態を管理
    character_count = 0 # 文字の表示を管理
    input_character = "None"
    input_character_onece = 0

    images = [center_look_point_image, top_look_point_image, bottom_look_point_image, left_look_point_image, right_look_point_image]
    n= 5 #以下以外定数.
    m=1 #注視点注視、点滅非注視時の表示時間定数.



    #/**********************************************************************************************/
    #10秒間注視点のみ表示.(インターバル要員)
    with lock:
        measurement_command.value = 1 # インターバル命令(計測なし).
    while previous_time + 10/n > time.time(): # 10秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理
    print("1st interval done!!!!")
    #/**********************************************************************************************/
    

    previous_time = time.time()
    with lock:
        measurement_command.value = 2 # 計測開始命令.

     # 15秒間注視点
    print("60秒間注視点")
    while previous_time + 60/n > time.time(): # 60秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理

    with lock:
        measurement_command.value = 3 # 計測終了命令.

    while measurement_command.value != 0: #保存などの処理が終わるまで待機.
        pass
    print("1. done")
    
    

    #/**********************************************************************************************/
    #10秒間注視点のみ表示.(インターバル要員)
    previous_time = time.time()
    with lock:
        measurement_command.value = 1 # インターバル命令(計測なし).
    while previous_time + 10/n > time.time(): # 10秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理
    print("2nd interval done!!!!")
    #/**********************************************************************************************/







    # 10Hz点滅→位相反転
    with lock:
        measurement_command.value = 4 # 6Hz計測開始命令.
    # 15秒間注視点
    images = [center_look_point_image, top_look_point_image, bottom_look_point_image, left_look_point_image, right_look_point_image]
    previous_time = time.time()
    while previous_time + 15/n > time.time(): # 60秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理
    

    # 10秒間10Hz表示.
    images = [top_look_point_image, bottom_look_point_image, left_look_point_image, right_look_point_image, blinking_image1]
    previous_time = time.time()
    while previous_time + 10/n > time.time(): # 60秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理


    # 10秒間反転10Hz表示.
    images = [top_look_point_image, bottom_look_point_image, left_look_point_image, right_look_point_image, re_blinking_image1]
    previous_time = time.time()
    while previous_time + 10/n > time.time(): # 60秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理


    # 15秒間注視点
    images = [center_look_point_image, top_look_point_image, bottom_look_point_image, left_look_point_image, right_look_point_image]
    previous_time = time.time()
    while previous_time + 15/n > time.time(): # 60秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理


    with lock:
        measurement_command.value = 5 # 6Hz計測終了命令.
    
    while measurement_command.value != 0: #保存などの処理が終わるまで待機.
        pass
    print("2. done!!!!")

    


    #/**********************************************************************************************/
    #10秒間注視点のみ表示.(インターバル要員)
    previous_time = time.time()
    with lock:
        measurement_command.value = 1 # インターバル命令(計測なし).
    while previous_time + 10/n > time.time(): # 10秒間ループ
        glClear(GL_COLOR_BUFFER_BIT) # カラーバッファをクリア
        for image in images: # 画像を描画
            if not image.update(): # 表示時間が経過したら
                images.remove(image)  # リストから削除する理由は、リストの要素を削除すると、リストの要素が前に詰められるため、forループが正しく動作するため.本当?
        glfw.swap_buffers(window) # バッファを入れ替え
        glfw.poll_events() # イベントを処理
    print("2nd interval done!!!!")
    #/**********************************************************************************************/





        # # 遅延探しテスト用コード本番では使用しない これは6秒後に点滅をオフにする.
        # if blinking_image1.frame_count_not_reset == 300:
        #     images[0] = blinking_image1_off
    with lock:
        measurement_command.value = 8 # 計測自体を終了する命令.
    print("End of the visual function.")
    glfw.destroy_window(window) # ウィンドウを破棄
    glfw.terminate() # GLFWを終了
    time.sleep(2) # 1秒待つ.









# 1000Hzでデータ要求を送信しないで、受信も行い、データの数をカウントする関数
def communicate_and_count_test_preparation(ser ,lock, measurement_command, threshold_look_10hz_max, threshold_look_6hz_max, threshold_non_look_10hz_max, threshold_non_look_6hz_max, threshold_look_10hz_min, threshold_look_6hz_min, threshold_non_look_10hz_min, threshold_non_look_6hz_min):
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    t = 1
    # 計測データ保存用配列.
    store_list_look_10hz = [] # 10Hz注視計測データ保存用.
    store_list_look_6hz = [] # 6Hz注視計測データ保存用.
    store_list_non_look_10hz = [] # 10Hz非注視計測データ保存用.
    store_list_non_look_6hz = [] # 6Hz非注視計測データ保存用.
    
    # fre_change_word.
    # last_data_highq_10hz = [0,0,0] # 最後に受信したデータ(補間用)
    # last_data_6hz = [0,0,0] # 最後に受信したデータ(補間用)

    # # 10Hz用.
    # a_bp_highq_10hz = np.array([1.000000000000000000000000000000, -3.974428294786210180689067783533, 5.931305335419762236881524586352, -3.939226621802428329743861468160, 0.982364711720531635918973734078])
    # b_bp_highq_10hz = np.array([0.000039222815344601606540007877,  0.000000000000000000000000000000,  -0.000078445630689203213080015753,  0.000000000000000000000000000000, 0.000039222815344601606540007877])
    # # 6Hz用.
    # a_bp_6hz = np.array([1.000000000000000000000000000000, -3.979456903924181165876916566049,5.941304776471029391871070401976, -3.944210692344831681310779458727,0.982364711720532302052788509172])
    # b_bp_6hz = np.array([0.000039222815344601667526380079,  0.000000000000000000000000000000,-0.000078445630689203335052760158,  0.000000000000000000000000000000,0.000039222815344601667526380079])

    # 超高Q--10Hz用.
    a_bp_high_q_10hz = np.array([1.000000000000000000000000000000, -5.986337395899265167997782555176, 14.943525948179951612360127910506, -19.910622476715822415371803799644, 14.934132172565124818675030837767, -5.978813517693076562409260077402, 0.998115330047988802419922649278])
    b_bp_high_q_10hz = np.array([0.000000000104796088903483945995, 0.000000000000000000000000000000, -0.000000000314388266710451837984, 0.000000000000000000000000000000, 0.000000000314388266710451837984, 0.000000000000000000000000000000, -0.000000000104796088903483945995])

    # 超高Q--7.5Hz用.(q=120).
    a_bp_high_q_6hz = np.array([1.000000000000000000000000000000, -5.992046052538612777027537958929, 14.966946335049158278707182034850, -19.947289296362864519096547155641, 14.960665391587113504101580474526, -5.987017931980501472821742936503, 0.998741565358762239945633609750])
    b_bp_high_q_6hz = np.array([0.000000000031178759423254480474, 0.000000000000000000000000000000, -0.000000000093536278269763441422, 0.000000000000000000000000000000, 0.000000000093536278269763441422, 0.000000000000000000000000000000, -0.000000000031178759423254480474])




    # フィルタの次数
    # Q_highq_10hz = len(a_bp_highq_10hz) - 1
    # P_highq_10hz = len(b_bp_highq_10hz) - 1
    # Q_6hz = len(a_bp_6hz) - 1
    # P_6hz = len(b_bp_6hz) - 1
    Q_high_q_10hz = len(a_bp_high_q_10hz) - 1
    P_high_q_10hz = len(b_bp_high_q_10hz) - 1
    Q_high_q_6hz = len(a_bp_high_q_6hz) - 1
    P_high_q_6hz = len(b_bp_high_q_6hz) - 1

    # 過去の値を保持する配列
    # y_prev_bp_highq_10hz = np.zeros((3, Q_highq_10hz))
    # x_prev_bp_highq_10hz = np.zeros((3, P_highq_10hz))
    # y_prev_bp_6hz = np.zeros((3, Q_6hz))
    # x_prev_bp_6hz = np.zeros((3, P_6hz))
    y_prev_bp_high_q_10hz = np.zeros((3, Q_high_q_10hz))
    x_prev_bp_high_q_10hz = np.zeros((3, P_high_q_10hz))
    y_prev_bp_high_q_6hz = np.zeros((3, Q_high_q_6hz))
    x_prev_bp_high_q_6hz = np.zeros((3, P_high_q_6hz))

    last_data_high_q_10hz = [0,0,0] # 最後に受信したデータ(補間用)
    last_data_high_q_6hz = [0,0,0] # 最後に受信したデータ(補間用)



    while measurement_command.value != 8: # 8が終了命令.
    # for i in range(10000000):
        current_time = time.perf_counter()  # 現在のタイムスタンプを取得
        interval_time = current_time - start_time
        if interval_time >= 1* t:
            print(f"1秒間{interval_time}で受信したデータの数: {data_count}, {measurement_command.value}")
            data_count = 0
            t = t + 1

        # データを受信しカウント
        if ser.in_waiting > 0:  # 受信データがあるか確認
            result = ser.readline()  # 改行コードまで読み込む
            if result:
                data_count += 1  # データをカウント
                # result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除\r\n
                result = re.sub(rb'\n$', b'', result)  # 改行コードを削除\n      #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                # result = result + b',0,0'
                # print(result)
                #result.decode()の型を出力する.
                # print(type(result.decode()))
                try:
                    # int_list_data = [int(x) for x in result.decode().split(',')]
                    int_list_data = [int(result.decode()), int(0), int(0)] #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                    # print(int_list_data)
                    # int_list_data = iir_real_time_3ch(int_list_data, a, b, y_prev, x_prev) # フィルタ処理BPF.

                    # **バンドパスフィルタの適用**
                    # int_list_data_bp_highq_10hz, y_prev_bp_highq_10hz, x_prev_bp_highq_10hz = iir_real_time_3ch(int_list_data, a_bp_highq_10hz, b_bp_highq_10hz, y_prev_bp_highq_10hz, x_prev_bp_highq_10hz) #バンドパスフィルタの適用.
                    # int_list_data_bp_6hz, y_prev_bp_6hz, x_prev_bp_6hz = iir_real_time_3ch(int_list_data, a_bp_6hz, b_bp_6hz, y_prev_bp_6hz, x_prev_bp_6hz) #バンドパスフィルタの適用.
                    
                    
                    # フィルタオン.
                    # int_list_data_bp_high_q_10hz, y_prev_bp_high_q_10hz, x_prev_bp_high_q_10hz = iir_real_time_3ch(int_list_data, a_bp_high_q_10hz, b_bp_high_q_10hz, y_prev_bp_high_q_10hz, x_prev_bp_high_q_10hz) #バンドパスフィルタの適用.
                    # int_list_data_bp_high_q_6hz, y_prev_bp_high_q_6hz, x_prev_bp_high_q_6hz = iir_real_time_3ch(int_list_data, a_bp_high_q_6hz, b_bp_high_q_6hz, y_prev_bp_high_q_6hz, x_prev_bp_high_q_6hz) #バンドパスフィルタの適用.
                    
                    # フィルタオフ.
                    int_list_data_bp_high_q_10hz = int_list_data
                    int_list_data_bp_high_q_6hz = int_list_data

                    # データの補間処理
                    # last_data_highq_10hz = int_list_data_bp_highq_10hz
                    # last_data_6hz = int_list_data_bp_6hz
                    last_data_high_q_10hz = int_list_data_bp_high_q_10hz
                    last_data_high_q_6hz = int_list_data_bp_high_q_6hz


                except ValueError: #エラー処理
                    print("ValueError")
                    # int_list_data_bp_highq_10hz = last_data_highq_10hz 
                    # int_list_data_bp_6hz = last_data_6hz
                    int_list_data_bp_high_q_10hz = last_data_high_q_10hz
                    int_list_data_bp_high_q_6hz = last_data_high_q_6hz

                # with lock:  # ロックを使って排他制御
                    # received_list_1.append(receive_value_1) # 受信データをリストに追加
                    # received_list_2.append(receive_value_2) # 受信データをリストに追加
                    # received_list_3.append(receive_value_3)
                    # received_list_4.append(receive_value_4)

                    # receive_value_1[:] = last_data_highq_10hz # 受信データを更新
                    # receive_value_2[:] = last_data_6hz # 受信データを更新
                    # receive_value_3[:] = last_data_high_q_10hz
                    # receive_value_4[:] = last_data_high_q_6hz

                    # clock_signal_1.value = True
                    # clock_signal_2.value = True

                if measurement_command.value == 2: # 点滅真ん中で上下左右に視線を動かす---10Hz計測開始命令
                    store_list_look_10hz.append(int_list_data_bp_high_q_10hz)
                elif measurement_command.value == 4: # 10Hz点滅→位相反転
                    store_list_look_6hz.append(int_list_data_bp_high_q_6hz)
                elif measurement_command.value == 6: # ?開始命令
                    store_list_non_look_10hz.append(int_list_data_bp_high_q_10hz)
                    store_list_non_look_6hz.append(int_list_data_bp_high_q_6hz)

                elif measurement_command.value == 3: #点滅真ん中で上下左右に視線を動かす---10Hz計測終了命令
                    print("measurement_command.value == 3")
                    print(np.array(store_list_look_10hz).shape)
                    print(np.array(store_list_look_10hz).shape)
                    save_2d_array_to_file(store_list_look_10hz, "measurement1")
                    #store_list_look_10hzの極大値を求める.
                    maxima_indices_topn, maxima_values_topn = find_local_maxima(np.array(store_list_look_10hz)[:, 0], top_n=20) #極大値top50を求める.
                    minima_indices_topn, minima_values_topn = find_local_minima(np.array(store_list_look_10hz)[:, 0], top_n=20) #極小値top50を求める.
                    with lock:
                        threshold_look_10hz_max.value = np.mean(maxima_values_topn) #閾値を設定.
                        threshold_look_10hz_min.value = np.mean(minima_values_topn) #閾値を設定.
                    measurement_command.value = 0 #初期状態.

                elif measurement_command.value == 5: #10Hz点滅→位相反転終了命令
                    save_2d_array_to_file(store_list_look_6hz, "measurement2")
                    #store_list_look_6hzの極大値を求める.
                    maxima_indices_topn, maxima_values_topn = find_local_maxima(np.array(store_list_look_6hz)[:, 0], top_n=30) #極大値top50を求める.
                    minima_indices_topn, minima_values_topn = find_local_minima(np.array(store_list_look_6hz)[:, 0], top_n=30) #極小値top50を求める.
                    with lock:
                        threshold_look_6hz_max.value = np.mean(maxima_values_topn) #閾値を設定.
                        threshold_look_6hz_min.value = np.mean(minima_values_topn) #閾値を設定.
                    measurement_command.value = 0 #初期状態.

                elif measurement_command.value == 7: #?終了命令
                    save_2d_array_to_file(store_list_non_look_10hz, "non_look_10hz")
                    save_2d_array_to_file(store_list_non_look_6hz, "non_look_6hz")
                    #store_list_non_look_10hzの極大値を求める.
  
                    maxima_indices_topn, maxima_values_topn = find_local_maxima(np.array(store_list_non_look_10hz)[:, 0], top_n=100) #極大値top50を求める. 60秒-600個
                    minima_indices_topn, minima_values_topn = find_local_minima(np.array(store_list_non_look_10hz)[:, 0], top_n=100) #極小値top50を求める.
                    with lock:
                        threshold_non_look_10hz_max.value = np.mean(maxima_values_topn)
                        threshold_non_look_10hz_min.value = np.mean(minima_values_topn)

                    maxima_indices_topn, maxima_values_topn = find_local_maxima(np.array(store_list_non_look_6hz)[:, 0], top_n=75) #極大値top50を求める. 60秒-450個
                    minima_indices_topn, minima_values_topn = find_local_minima(np.array(store_list_non_look_6hz)[:, 0], top_n=75) #極小値top50を求める.
                    with lock:
                        threshold_non_look_6hz_max.value = np.mean(maxima_values_topn)
                        threshold_non_look_6hz_min.value = np.mean(minima_values_topn)
                    measurement_command.value = 0 #初期状態.






def func_serial_preparation(priority, com, lock, measurement_command, threshold_look_10hz_max, threshold_look_6hz_max, threshold_non_look_10hz_max, threshold_non_look_6hz_max, threshold_look_10hz_min, threshold_look_6hz_min, threshold_non_look_10hz_min, threshold_non_look_6hz_min):
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_serial) started with priority {priority}")
    while True:
        try:
            ser = serial.Serial(com, bitRate, timeout=None)
            break
        except serial.SerialException:
            print("COMポートが開けませんでした。再度入力してください。")
    communicate_and_count_test_preparation(ser, lock, measurement_command, threshold_look_10hz_max, threshold_look_6hz_max, threshold_non_look_10hz_max, threshold_non_look_6hz_max, threshold_look_10hz_min, threshold_look_6hz_min, threshold_non_look_10hz_min, threshold_non_look_6hz_min)
    ser.close() # シリアルポートを閉じる.
   


# /**************main関数**********************************************/



def main():
        # 共有リストとロックを作成
    manager = multiprocessing.Manager()


    measurement_command = manager.Value('i', 0)

    threshold_look_10hz_max = manager.Value('i', 0)
    threshold_look_6hz_max = manager.Value('i', 0)
    threshold_look_10hz_min = manager.Value('i', 0)
    threshold_look_6hz_min = manager.Value('i', 0)

    threshold_non_look_10hz_max = manager.Value('i', 0)
    threshold_non_look_6hz_max = manager.Value('i', 0)
    threshold_non_look_10hz_min = manager.Value('i', 0)
    threshold_non_look_6hz_min = manager.Value('i', 0)



    shared_receive_list_1 = manager.list()  # 共有リスト10Hz
    shared_receive_list_2 = manager.list()  # 共有リスト15Hz

    flag_blink_1 = manager.Value('b', True)
    flag_blink_2 = manager.Value('b', True)

    chank_list_1 = manager.list() # high q 10Hz
    chank_list_1_2 = manager.list() # usual q 10Hz
    chank_list_2 = manager.list() # high q 7.5Hz
    chank_list_2_2 = manager.list() # usual q 7.5Hz

    receive_value_1 = manager.list()  # high q 10Hz
    receive_value_1_2 = manager.list()  # usual q 10Hz
    receive_value_2 = manager.list()  # higi q 7.5Hz
    receive_value_2_2 = manager.list()  # usual q 7.5Hz


    clock_signal_1 = manager.Value('b', False)
    clock_signal_2 = manager.Value('b', False)

    adjust_chank_list_1 = manager.list() # high q 10Hz
    adjust_chank_list_1_2 = manager.list() # usual q 10Hz    
    adjust_chank_list_2 = manager.list() # high q 7.5Hz
    adjust_chank_list_2_2 = manager.list() # usual q 7.5Hz

    analysis_flag_1 = manager.Value('b', False)
    analysis_flag_2 = manager.Value('b', False)

    gaze_flag_1 = manager.Value('b', False) #10Hz『あ』フラグ
    gaze_flag_1_2 = manager.Value('b', False) #10Hz『か』フラグ
    gaze_flag_2 = manager.Value('b', False) #15Hz『さ』フラグ
    gaze_flag_2_2 = manager.Value('b', False) #15Hz『さ』フラグ

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



        #標本データを作成するかどうか. trueの場合、標本データを作成する.
    setup_specimen = True


    # /************ 標本データ作成プロセス  preparation(準備) ****************/
    if setup_specimen == True:
        process0_1 = multiprocessing.Process(target=func_serial_preparation, args=(priority1, com, lock, measurement_command, threshold_look_10hz_max, threshold_look_6hz_max, threshold_non_look_10hz_max, threshold_non_look_6hz_max, threshold_look_10hz_min, threshold_look_6hz_min, threshold_non_look_10hz_min, threshold_non_look_6hz_min))
        process0_2 = multiprocessing.Process(target=func_visual_preparation, args=(priority1, measurement_command,lock))
        # プロセスの開始
        process0_1.start()
        process0_2.start()
        # プロセスの終了を待つ
        process0_1.join()
        process0_2.join()

        print("threshold_look_10hz_max: ", threshold_look_10hz_max.value)
        print("threshold_look_6hz_max: ", threshold_look_6hz_max.value)
        print("threshold_look_10hz_min: ", threshold_look_10hz_min.value)
        print("threshold_look_6hz_min: ", threshold_look_6hz_min.value)
        print("threshold_non_look_10hz_max: ", threshold_non_look_10hz_max.value)
        print("threshold_non_look_6hz_max: ", threshold_non_look_6hz_max.value)
        print("threshold_non_look_10hz_min: ", threshold_non_look_10hz_min.value)
        print("threshold_non_look_6hz_min: ", threshold_non_look_6hz_min.value)

        print("ajustment threshold")
        # 閾値の調整 + 5%up
        gain = 1.5
        gain2 = 1.2
        # gain = 1.0
        with lock:
            threshold_look_10hz_max.value = threshold_look_10hz_max.value * gain
            threshold_look_6hz_max.value = threshold_look_6hz_max.value * gain
            threshold_look_10hz_min.value = threshold_look_10hz_min.value * gain
            threshold_look_6hz_min.value = threshold_look_6hz_min.value * gain
            threshold_non_look_10hz_max.value = threshold_non_look_10hz_max.value * gain
            threshold_non_look_6hz_max.value = threshold_non_look_6hz_max.value * gain2
            threshold_non_look_10hz_min.value = threshold_non_look_10hz_min.value * gain
            threshold_non_look_6hz_min.value = threshold_non_look_6hz_min.value * gain2

        # gain = 0.5
        # with lock:
        #     threshold_non_look_10hz_max.value = threshold_non_look_10hz_max.value + (threshold_look_10hz_max.value - threshold_non_look_10hz_max.value)*gain
        #     threshold_non_look_6hz_max.value = threshold_non_look_6hz_max.value + (threshold_look_6hz_max.value - threshold_non_look_6hz_max.value)*gain

        #     threshold_non_look_10hz_min.value = threshold_non_look_10hz_min.value + (threshold_look_10hz_min.value - threshold_non_look_10hz_min.value)*gain
        #     threshold_non_look_6hz_min.value = threshold_non_look_6hz_min.value + (threshold_look_6hz_min.value - threshold_non_look_6hz_min.value)*gain

        # print("threshold_look_10hz_max: ", threshold_look_10hz_max.value)
        # print("threshold_look_6hz_max: ", threshold_look_6hz_max.value)
        # print("threshold_look_10hz_min: ", threshold_look_10hz_min.value)
        # print("threshold_look_6hz_min: ", threshold_look_6hz_min.value)






        # 配列に値を格納する.
        thresholds = [threshold_look_10hz_max.value, threshold_look_10hz_min.value, threshold_look_6hz_max.value, threshold_look_6hz_min.value, threshold_non_look_10hz_max.value, threshold_non_look_10hz_min.value, threshold_non_look_6hz_max.value, threshold_non_look_6hz_min.value]        # ファイルに保存する.
        save_2d_array_to_file(thresholds, "thresholds")
        time.sleep(10)
    elif setup_specimen == False:
        threshold_look_10hz_max.value = 0
        threshold_look_6hz_max.value = 0
        threshold_look_10hz_min.value = 0
        threshold_look_6hz_min.value = 0
        threshold_non_look_10hz_max.value = 30000
        threshold_non_look_6hz_max.value = 40000
        threshold_non_look_10hz_min.value = -30000
        threshold_non_look_6hz_min.value = -40000

    print("end.")



if __name__ == '__main__':
    main()




# /*******************************************************************************************/

# リフレッシュレート / 点滅周波数 = 1周期のフレーム数       |  10周期の時間 
# 60fps / 15Hz = 4フレーム　　　　　　　　　　　　　　　　 |   666.667ms
# 60fps / 12Hz = 5フレーム　　　　　　　　　　　　　　　　 |   833.333ms      採用
# 60fps / 10Hz = 6フレーム　　　　　　　　　　　　　　　　 |   1000ms      　　採用
# 60fps / 6Hz = 10フレーム　　　　　　　　　　　　　　　　 |   1666.667ms     採用
# 60fps / 5Hz = 12フレーム　　　　　　　　　　　　　　　　 |   2000ms
# 60fps / 3Hz = 20フレーム　　　　　　　　　　　　　　　　 |   3333.333ms


# 点滅刺激追加に必要なコード変更箇所
# plot_phase_ana
# func_visual

# gaze_flag_1は4つ必要なはず.

# 周波数変更による変更箇所のコメントアウトコード↓
# fre_change_word.
# /*******************************************************************************************/