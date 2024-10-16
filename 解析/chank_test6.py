# chank_test2.pyとの違いは、点滅に同期してデータを処理するかどうか。


# 多分うまく動いている..


# グローバル変数として受信データを格納するリスト
# 現在の日時をファイル名に追加
import time
from datetime import datetime
from multiprocessing.managers import ListProxy
import sys

# 現在の日時をファイル名に追加
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
receive_data_txt = f"receive_data_{current_time}.txt"


def append_data_to_file(file_name, list):

    with open(file_name, 'a') as file:
        # 配列1
        for data in list:
            file.write(f"{data}, ")
            # print(f"{data} をファイルに書き込みました")
            # time.sleep(0.1)
        file.write("\n")  # 改行




def iir_real_time(x, a, b, y_prev, x_prev):
    """1サンプルずつIIRフィルタをかける。y_prev: 直前のフィルタ出力、x_prev: 直前の入力"""
    y1 = b[0] * x[0] + b[1] * x_prev[0,0] + b[2] * x_prev[0,1] - a[1] * y_prev[0,0] - a[2] * y_prev[0,1]
    y2 = b[0] * x[1] + b[1] * x_prev[1,0] + b[2] * x_prev[1,1] - a[1] * y_prev[1,0] - a[2] * y_prev[1,1]
    y3 = b[0] * x[2] + b[1] * x_prev[2,0] + b[2] * x_prev[2,1] - a[1] * y_prev[2,0] - a[2] * y_prev[2,1]
    # 直前のサンプルを更新
    x_prev[0,1], x_prev[0,0] = x_prev[0,0], x[0]
    x_prev[1,1], x_prev[1,0] = x_prev[1,0], x[1]
    x_prev[2,1], x_prev[2,0] = x_prev[2,0], x[2]

    y_prev[0,1], y_prev[0,0] = y_prev[0,0], y1
    y_prev[1,1], y_prev[1,0] = y_prev[1,0], y2
    y_prev[2,1], y_prev[2,0] = y_prev[2,0], y3

    y = [y1, y2, y3]
    return y


# /**************Serial関連**********************************************/
import re
import serial
import serial.tools.list_ports
import time
import multiprocessing

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

    # フィルタのパラメータ設定
    fs = 1000  # サンプリングレート
    fc_digital = 10.0  # カットオフ周波数（テスト用）
    a = [1.0, -1.9714359701251216, 0.9751778761806491]  # フィルタの係数a
    b = [0.9875889380903244, -1.9714359701251216, 0.9875889380903244] # フィルタの係数b
    # 過去の値を保持する配列
    y_prev = [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
    x_prev = [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
    # サンプリングレートを入力: 1000
    # フィルタの種類(LPF, HPF, BPF, BSF)を入力: BPF
    # カットオフ周波数下限fc1を入力: 3
    # カットオフ周波数下限fc2を入力: 20
    # テスト用正弦波の周波数: 50
    # a: [1.0, -1.8962594398557984, 0.8985096404962453]
    # b: [0.05074517975187733, 0.0, -0.05074517975187733]


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
                    # receive_value[:] = int_list_data #フィルタ無し
                    receive_value[:] = iir_real_time(int_list_data, a, b, y_prev, x_prev)
                    # print("receive_value: ", receive_value)
                    # print(type(receive_value))

        # 次のタイムスタンプまでの残り時間を計算
        sleep_time = next_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)  # 必要な場合のみスリープ
    



# 1000Hzでデータ要求を送信しないで、受信も行い、データの数をカウントする関数
def communicate_and_count_test(ser , received_list, receive_value, clock_signal_1, clock_signal_2, lock):

    interval = 1.0 / 1000  # 1000Hz
    next_time = time.perf_counter()  # 高精度タイマーの現在時刻を取得
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    data_count_10 = 0  # データのカウント10秒.
    t = 1
    t2 = 1
    last_data = [0,0,0] # 最後に受信したデータ(補間用)

    while True:
    # for i in range(10000000):
        current_time = time.perf_counter()  # 現在のタイムスタンプを取得

        # 10秒経過したらループを終了
        if current_time - start_time >= 1* t:
            print(f"1秒間で受信したデータの数: {data_count}")
            data_count = 0
            t = t + 1

        # if current_time - start_time >= 10 * t2:
        #     print(f"10秒間で受信したデータの数: {data_count_10}")
        #     print("DDDDDDDDtime: ", time.time())
        #     data_count_10 = 0
        #     t2 = t2 + 1

        # データを受信しカウント
        if ser.in_waiting > 0:  # 受信データがあるか確認
            result = ser.readline()  # 改行コードまで読み込む
            if result:
                data_count += 1  # データをカウント
                data_count_10 += 1  # データをカウント
                result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除
                try:
                    int_list_data = [int(x) for x in result.decode().split(',')]
                    last_data = int_list_data
                except ValueError:
                    print("ValueError")
                    int_list_data = last_data

                with lock:  # ロックを使って排他制御
                    received_list.append(receive_value)
                # with lock:
                    clock_signal_1.value = True
                    clock_signal_2.value = True
                    receive_value[:] = int_list_data




 
# /**************グラフィック関連**********************************************/
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import math



def enable_vsync(enable=True):
    if enable:
        glfw.swap_interval(1)  # V-Syncを有効にする
    else:
        glfw.swap_interval(0)  # V-Syncを無効にする




# /***********************************************************/

from PIL import Image
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader


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


        self.projection = projection

        # 画像のロードとテクスチャの設定
        self.texture_id = self.load_texture(image_path)

        # シェーダープログラムを作成
        self.shader_program = create_shader_program()

        # 頂点データを設定
        self.vao, self.vbo, self.ebo = self.create_quad()



    def load_texture(self, image_path):
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 画像を上下反転する
        image_data = np.array(image, np.uint8)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        print(f"Texture ID for {image_path}: {texture_id}")  # テクスチャIDを出力
        return texture_id

    def create_quad(self):
        # 四角形の頂点データ (位置とテクスチャ座標)
        x, y = self.position  # 画像の中心位置
        w, h = self.size      # 画像の幅と高さ

        vertices = np.array([
            # 位置            テクスチャ座標
            x - w/2, y - h/2, 0.0,  0.0, 0.0,  # 左下
            x + w/2, y - h/2, 0.0,  1.0, 0.0,  # 右下
            x + w/2, y + h/2, 0.0,  1.0, 1.0,  # 右上
            x - w/2, y + h/2, 0.0,  0.0, 1.0   # 左上
        ], dtype=np.float32)

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


    def draw_image(self):
        glUseProgram(self.shader_program)

        # モデル行列を設定 (位置とサイズを反映)
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(self.position[0], self.position[1], 0.0))
        model = glm.scale(model, glm.vec3(self.size[0], self.size[1], 1.0))

        # モデル行列と投影行列をシェーダーに渡す
        model_location = glGetUniformLocation(self.shader_program, "model")
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model))

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

        #表示時間に関する重要なコード↓...
        # current_time = time.time()
        # elapsed_time = current_time - self.start_time  # 経過時間を秒に変換
        # if self.display_time is not None and elapsed_time >= self.display_time:
        #     return False  # 表示秒数が経過したらFalseを返す

        if self.frames_per_blink is not None:
            # フレームカウンタを更新
            self.frame_count += 1
            self.frame_count_not_reset += 1
            if self.frame_count >= self.frames_per_blink:
                self.toggle = not self.toggle  # フラグを反転させて点滅を切り替える
                self.frame_count = 0  # カウンタをリセット

        # 点滅がオンのときだけ画像を描画
        if self.toggle or self.frames_per_blink is None:
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
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

# /**************並列処理関連**********************************************/

from concurrent.futures import ProcessPoolExecutor
import time


def func_1():

    while True:
        time.sleep(2)
        print('func_1')

    
import glm  # OpenGL Mathematicsライブラリを使用
def func_visual(flag_blink_1, flag_blink_2, lock):
    if not glfw.init():
        return

    # プライマリモニターの解像度を取得
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    monitor_width = video_mode.size.width
    monitor_height = video_mode.size.height

    #テスト用
    # monitor_width = 800
    # monitor_height = 300

    print(f"Monitor Resolution: {monitor_width}x{monitor_height}")

    window = init_glfw(monitor_width, monitor_height, "Blinking Image")

    # glfw.make_context_current(window)

    # refresh_rate = video_mode.refresh_rate  # 垂直同期のリフレッシュレート
    refresh_rate = 60
    enable_vsync(True)  # V-Syncを有効にする

 # --- ここから修正 ---
    # 画面サイズを取得し、アスペクト比を維持する
    width, height = glfw.get_framebuffer_size(window)
    

    # ビューポートをウィンドウ全体に設定
    glViewport(0, 0, width, height)

    # # 1:1の正射影行列を設定
    # projection = glm.ortho(-1.0, 1.0, -1.0, 1.0)
    # 投影行列を設定（1:1比率を維持）
    projection = setup_projection_for_circle(width, height)

    # シェーダープログラムを使用して投影行列を渡す
    shader_program = create_shader_program()
    glUseProgram(shader_program)
    projection_location = glGetUniformLocation(shader_program, "projection")
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(projection))
    print("shader done!!!!")


    blinking_image = BlinkingImage(position=(-1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image2 = BlinkingImage(position=(-0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False, projection=projection)
    blinking_image3 = BlinkingImage(position=(0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image4 = BlinkingImage(position=(1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=False, projection=projection)


    character_image = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/a_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image2 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ka_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image3 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/sa_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image4 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ta_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)


    # images = [blinking_image, blinking_image2, blinking_image3, blinking_image4, character_image, character_image2, character_image3, character_image4]
    images = [blinking_image, blinking_image2, blinking_image3, blinking_image4]
    # images = [blinking_image]


    previous_time = time.time()
    frame_count = 0
    fullscreen = True  # 現在の状態を管理


    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        for image in images:
            if not image.update():
                images.remove(image)  # 表示時間が経過したらリストから削除

        # 10Hzの1周期分.. 60/10 = 6
        if blinking_image.frame_count_not_reset % 6 == 0:
            with lock:
                if flag_blink_1.value == True:
                    flag_blink_1.value = False
                else:
                    flag_blink_1.value = True
            # print("toggle flag_blink_1", circle1.toggle)
            # print("frame_count_a", circle1.frame_count)
        
        # 12Hzの1周期分.. 60/12 = 5
        if blinking_image.frame_count_not_reset % 5 == 0:
            with lock:
                if flag_blink_2.value == True:
                    flag_blink_2.value = False
                else:
                    flag_blink_2.value = True


        # フレームカウンタを更新
        frame_count += 1
        current_time = time.time()

        # 1秒ごとにFPSを計算して出力
        if current_time - previous_time >= 1.0:
            fps = frame_count / (current_time - previous_time)
            # print(f"FPS: {fps:.2f}")
            # print(f"frame_count: {frame_count}")
            print(f"FPS: {fps:.2f}, frame_count: {frame_count}")
            previous_time = current_time
            frame_count = 0

        if blinking_image.frame_count_not_reset % 600 == 0:
            # print("frame_count: ", blinking_image.frame_count_not_reset)
            print("Ftime :", time.time())

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
            sys.exit()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()



def setup_projection_for_circle(width, height):
    # ウィンドウのアスペクト比を取得
    aspect_ratio = width / height

    # アスペクト比に基づいてオルソグラフィック投影の範囲を調整
    if aspect_ratio >= 1.0:
        # 横長の場合、X軸を拡張
        projection = glm.ortho(-aspect_ratio, aspect_ratio, -1.0, 1.0)
    else:
        # 縦長の場合、Y軸を拡張
        projection = glm.ortho(-1.0, 1.0, -1.0/aspect_ratio, 1.0/aspect_ratio)

    return projection




def func_serial(com, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock):
    # global received_data  # グローバル変数を参照
    while True:
        try:
            # com = input_com()
            # com = "COM3"
            ser = serial.Serial(com, bitRate, timeout=None)
            break
        except serial.SerialException:
            print("COMポートが開けませんでした。再度入力してください。")

    # communicate_and_count(ser, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock)
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    print("communicate_and_count_test")
    communicate_and_count_test(ser, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock)
    # print("shared_receive_list: ", shared_receive_list)
    # print("len of shared_receive_list: ", len(shared_receive_list))


import copy

def func_chank_10hz(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state = None
    chank_chank_list_1 = [] #buffer1
    chank_chank_list_2 = [] #buffer2
    po = 0

    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")
    print("func_chank_10hz")

    while True:
        # if po >= 10000:
        #     break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_2)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100)) #1000data / 10Hz = 100data
                        analysis_flag.value = True
                    chank_chank_list_2 = []
                    po = po + 1
                    
                    if(po % 100 == 0):
                        print("po: ", po)
                    # print("po: ", po)
                    # with lock:
                        # chank_list_copy = copy.deepcopy(list(chank_list[-3:])) #最後の3つのデータをコピー
                        # chank_list_copy = copy.deepcopy(list(chank_list[-1:])) #最後の1つのデータをコピー
                        # adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100)) #1000data / 10Hz = 100data
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100))
                        analysis_flag.value = True
                    chank_chank_list_1 = []
                    po = po + 1
                    if(po % 100 == 0):
                        print("po: ", po)
                    # with lock:
                        # chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
                        # chank_list_copy = copy.deepcopy(list(chank_list[-1:]))
                        # adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False



    # print("len of chank_list 10Hz: ", len(chank_list))               
    # # 各行の列数を出力
    # for i, row in enumerate(chank_list):
    #     if( i > 9500): 
    #         print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力


            
            # print("adjust_chank_list")
            # # 各行の列数を出力
            # for i, row in enumerate(adjust_chank_list):
            #     print(f"Row {i+1} length: {len(row)}")

    


def func_chank_12hz(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state = None
    chank_chank_list_1 = [] #buffer1
    chank_chank_list_2 = [] #buffer2
    po = 0

    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")
    print("func_chank_12hz")

    while True:
        # if po >= 10000:
        #     break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_2)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=83)) #1000data / 12Hz = 83.33333data
                    chank_chank_list_2 = []
                    po = po + 1
                    # print("po: ", po)
                    # with lock:
                        # chank_list_copy = copy.deepcopy(list(chank_list[-3:])) #最後の3つのデータをコピー
                        # chank_list_copy = copy.deepcopy(list(chank_list[-1:])) #最後の1つのデータをコピー
                        # adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100)) #1000data / 10Hz = 83.33333data
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=83))
                    chank_chank_list_1 = []
                    po = po + 1
                    # with lock:
                        # chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
                        # chank_list_copy = copy.deepcopy(list(chank_list[-1:]))
                        # adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=100))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False


    # print("len of chank_list 12Hz: ", len(chank_list))               
    # # 各行の列数を出力
    # for i, row in enumerate(chank_list):
    #     if( i > 9500):
    #         print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
                
            # print("adjust_chank_list")
            # # 各行の列数を出力
            # for i, row in enumerate(adjust_chank_list):
            #     print(f"Row {i+1} length: {len(row)}")




def func_chank_1(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state = None
    chank_chank_list_1 = []
    chank_chank_list_2 = []
    po = 0
    while True:
        if po >= 20:
            break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state is None:
            with lock:
                print("flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_2)
                    chank_chank_list_2 = []
                    po = po + 1
                    print("po: ", po)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list[-3:])) #最後の3つのデータをコピー
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=1000))
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_1)
                    chank_chank_list_1 = []
                    po = po + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=1000))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False
    # print("chank_list: ", chank_list)
    # テキストファイルにデータを追記
    # append_data_to_file(receive_data_txt, adjust_chank_list)
    print("len of chank_list 10Hz: ", len(chank_list))               
    # 各行の列数を出力
    for i, row in enumerate(chank_list):
        print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
    print("adjust_chank_list")
    # 各行の列数を出力
    for i, row in enumerate(adjust_chank_list):
        print(f"Row {i+1} length: {len(row)}")





def func_chank_2(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state = None
    chank_chank_list_1 = []
    chank_chank_list_2 = []
    po = 0
    while True:
        if po >= 20:
            break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state is None:
            with lock:
                print("flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_2)
                    chank_chank_list_2 = []
                    po = po + 1
                    print("po15: ", po)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:    
                    with lock:
                        chank_list.append(chank_chank_list_1)
                    chank_chank_list_1 = []
                    po = po + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        clock_signal.value = False
    # print("chank_list: ", chank_list)
    # テキストファイルにデータを追記
    # append_data_to_file(receive_data_txt, adjust_chank_list)
    print("len of chank_list 15Hz: ", len(chank_list))               
    # 各行の列数を出力
    for i, row in enumerate(chank_list):
        print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
    print("adjust_chank_list")
    # 各行の列数を出力
    for i, row in enumerate(adjust_chank_list):
        print(f"Row {i+1} length: {len(row)}")


def func_chank_all(receive_value, flag_blink_A, flag_blink_B, chank_list_A, chank_list_B, clock_signal_A, clock_signal_B, adjust_chank_list_A, adjust_chank_list_B, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state_A = None
    chank_chank_list_1_A = []
    chank_chank_list_2_A = []
    
    flag_state_B = None
    chank_chank_list_1_B = []
    chank_chank_list_2_B = []


    po = 0
    po2 = 0

    while True:
        if po >= 30:
            break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state_A is None:
            with lock:
                print("flag_blink_A: ", flag_blink_A.value)
                if flag_blink_A.value == True:
                    flag_state_A = True
        else:
            if flag_blink_A.value == True:
                if len(chank_chank_list_2_A) != 0:    
                    with lock:
                        chank_list_A.append(chank_chank_list_2_A)
                    chank_chank_list_2_A = []
                    po = po + 1
                    print("po15: ", po)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_A[-3:]))
                        adjust_chank_list_A.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_A.value == True:
                        chank_chank_list_1_A.append(receive_value[0])
                        clock_signal_A.value = False

            elif flag_blink_A.value == False:
                if len(chank_chank_list_1_A) != 0:    
                    with lock:
                        chank_list_A.append(chank_chank_list_1_A)
                    chank_chank_list_1_A = []
                    po = po + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_A[-3:]))
                        adjust_chank_list_A.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_A.value == True:
                        chank_chank_list_2_A.append(receive_value[0])
                        clock_signal_A.value = False


        if flag_state_B is None:
            with lock:
                print("flag_blink_B: ", flag_blink_B.value)
                if flag_blink_B.value == True:
                    flag_state_B = True
        else:
            if flag_blink_B.value == True:
                if len(chank_chank_list_2_B) != 0:    
                    with lock:
                        chank_list_B.append(chank_chank_list_2_B)
                    chank_chank_list_2_B = []
                    # po2 = po2 + 1
                    # print("po15: ", po2)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_B[-3:]))
                        adjust_chank_list_B.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po2: ", po2)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_B.value == True:
                        chank_chank_list_1_B.append(receive_value[0])
                        clock_signal_B.value = False

            elif flag_blink_B.value == False:
                if len(chank_chank_list_1_B) != 0:    
                    with lock:
                        chank_list_B.append(chank_chank_list_1_B)
                    chank_chank_list_1_B = []
                    # po2 = po2 + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_B[-3:]))
                        adjust_chank_list_B.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po2: ", po2)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_B.value == True:
                        chank_chank_list_2_B.append(receive_value[0])
                        clock_signal_B.value = False
    # print("chank_list_A: ", chank_list_A)
    # テキストファイルにデータを追記
    # append_data_to_file(receive_data_txt, adjust_chank_list_A)
    print("len of chank_list_A 15Hz: ", len(chank_list_A))               
    # 各行の列数を出力
    for i, row in enumerate(chank_list_A):
        print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
    print("adjust_chank_list_A")
    # 各行の列数を出力
    for i, row in enumerate(adjust_chank_list_A):
        print(f"Row {i+1} length: {len(row)}")





def func_analysis(adjust_chank_list, analysis_flag, lock):
    chank_copy = []
    # flag = False
    count = 0
    time.sleep(3)
    print("分析")
    
    while True:
        # print("分析2")
        if len(adjust_chank_list) >= 20:
            while True:
                if analysis_flag.value == True:
                    # print("分析開始")
                    # print("分析開始")
                    # print("分析開始")
                    # print("分析開始")
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

                    plot_multiple_lines(chank_copy, count)
                    plot_phase_ana(chank_copy, count)
                    # print("11111111111111111111")
                    # print(time.time())
                    count = count + 1
                    # flag = True

                
import matplotlib.pyplot as plt


def plot_multiple_lines(y_values, count):
    """
    引数として与えられたデータを基に、同じ線グラフ上に複数の線を描画します。
    
    Parameters:
    y_values (list of arrays): 描画するデータのリスト。各要素はY軸の値を表します。
    """
    x = np.linspace(0, 0.1, 100)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定

    if count % 10 == 0:
        for i, y in enumerate(y_values):
            plt.plot(x, y, label=f'Line {i+1}')

        plt.title('Multiple Lines on the Same Graph')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.legend(loc='upper right')
        plt.grid(True)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name_path = f'./plt_img/add_ave/phase_{current_time}.png'
        plt.savefig(file_name_path)
        # plt.show() # グラフの表示
        plt.close()


import datetime
# import os

def plot_phase_ana(y_values, count):
    x = np.linspace(1, 20, 20)  # 0から10までの100個の等間隔の点

    # グラフの描画
    # plt.figure(figsize=(10, 6)) # グラフのサイズを設定

    if count % 10 == 0:
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


        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name_path = f'./plt_img/phase/phase_{current_time}.png'
        # dir_path = './plt_img'
        # path = os.path.join(dir_path, file_name)
        # グラフを保存 (ファイル名は現在の時刻)
        plt.savefig(file_name_path)

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
    lock = multiprocessing.Lock()


        
    list_com()# COMポート一覧を表示
    # com = input_com()# COMポート接続の初期化
    com = "COM3"
    # com = input_com()
    # print(com)

    
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     # e.submit(func_1)
    #     e.submit(func_serial, com)
    #     # e.submit(func_visual)
    # 並列処理で実行するプロセスを定義

    
    process1 = multiprocessing.Process(target=func_serial, args=(com, shared_receive_list, receive_value, clock_signal_1, clock_signal_2, lock))
    
    
    # process2 = multiprocessing.Process(target=func_chank_1, args=(receive_value, flag_blink_1, chank_list_1, clock_signal_1, adjust_chank_list_1, lock))
    process2 = multiprocessing.Process(target=func_chank_10hz, args=(receive_value, flag_blink_1, chank_list_1, clock_signal_1, adjust_chank_list_1, analysis_flag_1, lock))
    process3 = multiprocessing.Process(target=func_chank_12hz, args=(receive_value, flag_blink_2, chank_list_2, clock_signal_2, adjust_chank_list_2, analysis_flag_2, lock))
    
    
    # process2 = multiprocessing.Process(target=func_chank_all, args=(receive_value, flag_blink_1, flag_blink_2, chank_list_1, chank_list_2, clock_signal_1, clock_signal_2, adjust_chank_list_1, adjust_chank_list_2, lock))
    process4 = multiprocessing.Process(target=func_visual, args=(flag_blink_1, flag_blink_2, lock))
    
    
    
    process5 = multiprocessing.Process(target=func_analysis, args=(adjust_chank_list_1 ,analysis_flag_1, lock))
    # process5 = multiprocessing.Process(target=func_chank_2, args=(receive_value, flag_blink_2, chank_list_2, clock_signal_2, adjust_chank_list_2, lock))

    # プロセスの開始
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()

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
