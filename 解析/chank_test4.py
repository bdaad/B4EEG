# chank_test2.pyとの違いは、点滅に同期してデータを処理するかどうか。


# 多分うまく動いている..


# グローバル変数として受信データを格納するリスト
# 現在の日時をファイル名に追加
import time
from datetime import datetime
from multiprocessing.managers import ListProxy

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
def communicate_and_count(ser , received_list, receive_value, clock_signal, lock):

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
                    last_data = int_list_data
                except ValueError:
                    print("ValueError")
                    int_list_data = last_data

                with lock:  # ロックを使って排他制御
                    received_list.append(receive_value)
                    # received_list.append(result.decode())  # 共有リストに追加
                with lock:
                    clock_signal.value = True
                    receive_value[:] = int_list_data
                    # print("receive_value: ", receive_value)
                    # print(type(receive_value))

        # 次のタイムスタンプまでの残り時間を計算
        sleep_time = next_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)  # 必要な場合のみスリープ
    

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
# def load_texture(image_path):
#     image = Image.open(image_path)
#     image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL用に上下を反転
#     img_data = image.convert("RGB").tobytes()

#     texture_id = glGenTextures(1)
#     glBindTexture(GL_TEXTURE_2D, texture_id)
#     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    
#     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#     return texture_id

# def draw_image(texture_id, x, y, width, height):
#     glEnable(GL_TEXTURE_2D)
#     glBindTexture(GL_TEXTURE_2D, texture_id)
    
#     glBegin(GL_QUADS)
#     # 四角形の頂点とテクスチャ座標
#     glTexCoord2f(0, 0); glVertex2f(x - width / 2, y - height / 2)
#     glTexCoord2f(1, 0); glVertex2f(x + width / 2, y - height / 2)
#     glTexCoord2f(1, 1); glVertex2f(x + width / 2, y + height / 2)
#     glTexCoord2f(0, 1); glVertex2f(x - width / 2, y + height / 2)
#     glEnd()
    
#     glDisable(GL_TEXTURE_2D)

















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
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # 経過時間を秒に変換
        if self.display_time is not None and elapsed_time >= self.display_time:
            return False  # 表示秒数が経過したらFalseを返す

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
def func_visual(flag_blink, lock):
    if not glfw.init():
        return

    # プライマリモニターの解像度を取得
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    monitor_width = video_mode.size.width
    monitor_height = video_mode.size.height

    #テスト用
    monitor_width = 800
    monitor_height = 300

    print(f"Monitor Resolution: {monitor_width}x{monitor_height}")

    # フルスクリーンウィンドウを作成
    # window = glfw.create_window(monitor_width, monitor_height, "V-Sync Blinking Circles with FPS", None, None)
    # if not window:
    #     glfw.terminate()
    #     return
    window = init_glfw(monitor_width, monitor_height, "Blinking Image")

    # glfw.make_context_current(window)

    # refresh_rate = video_mode.refresh_rate  # 垂直同期のリフレッシュレート
    refresh_rate = 60
    enable_vsync(True)  # V-Syncを有効にする

 # --- ここから修正 ---
    # 画面サイズを取得し、アスペクト比を維持する
    width, height = glfw.get_framebuffer_size(window)
    
# ビューポートを1:1の比率で設定（中央に正方形で表示）
    # if width > height:
    #     glViewport((width - height) // 2, 0, height, height)
    # else:
    #     glViewport(0, (height - width) // 2, width, width)

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



    # # OpenGLの視点とカメラ設定
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluPerspective(45, (width / height), 0.1, 50.0)
    
    # # カメラの位置を設定（z方向に-5移動してシーンを表示）
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    # glTranslatef(0.0, 0.0, -5)


    # 画像の読み込み

    blinking_image = BlinkingImage(position=(-1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image2 = BlinkingImage(position=(-0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False, projection=projection)
    blinking_image3 = BlinkingImage(position=(0.5, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=True, projection=projection)
    blinking_image4 = BlinkingImage(position=(1.0, 0.0), size=(0.45, 0.45), image_path="./circle.png", display_time=None, frequency=15, refresh_rate=refresh_rate, start_on=False, projection=projection)


    character_image = BlinkingImage(position=(-1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/a_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image2 = BlinkingImage(position=(-0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/ka_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)
    character_image3 = BlinkingImage(position=(0.5, 0.2), size=(0.45, 0.45), image_path="./img_file/sa_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True, projection=projection)
    character_image4 = BlinkingImage(position=(1.0, 0.2), size=(0.45, 0.45), image_path="./img_file/ta_off.png", display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=False, projection=projection)


    images = [blinking_image, blinking_image2, blinking_image3, blinking_image4, character_image, character_image2, character_image3, character_image4]
    
    # images = [blinking_image]


    previous_time = time.time()
    frame_count = 0
    fullscreen = True  # 現在の状態を管理


    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)


        # 各円を更新して描画
        # for circle in circles:
        #     if not circle.update():
        #         circles.remove(circle)  # 表示秒数が経過したらリストから削除
        # BlinkingImageを更新して描画
        for image in images:
            if not image.update():
                images.remove(image)  # 表示時間が経過したらリストから削除
        # if not blinking_image.update():
        #     break  # 表示時間が経過したら終了
        
        #circleのself.frame_countを出力する
        # print("circle1.frame_count: ", circle1.frame_count_not_reset)
        # print("circle2.frame_count: ", circle2.frame_count)
        # print("frame_count_not: ", blinking_image.frame_count_not_reset)

        if blinking_image.frame_count_not_reset % refresh_rate == 0:
            with lock:
                if flag_blink.value == True:
                    flag_blink.value = False
                else:
                    flag_blink.value = True
            # print("toggle flag_blink", circle1.toggle)
            # print("frame_count_a", circle1.frame_count)


        # フレームカウンタを更新
        frame_count += 1
        current_time = time.time()

        # 1秒ごとにFPSを計算して出力
        if current_time - previous_time >= 1.0:
            fps = frame_count / (current_time - previous_time)
            print(f"FPS: {fps:.2f}")
            print(f"frame_count: {frame_count}")
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




def func_serial(com, shared_receive_list, receive_value, clock_signal, lock):
    # global received_data  # グローバル変数を参照
    ser = serial.Serial(com, bitRate, timeout=None)
    communicate_and_count(ser, shared_receive_list, receive_value, clock_signal, lock)
    # print("shared_receive_list: ", shared_receive_list)
    # print("len of shared_receive_list: ", len(shared_receive_list))


import copy
def func_chank(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..

    flag_state = None
    chank_chank_list_1 = []
    chank_chank_list_2 = []
    # adjust_chank_list = []

    po = 0

    

    while True:
        if po >= 20:
            break
        #計測の最初は、必ずflag_blink=Trueのときにデータを受け取る.
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
                        chank_list_copy = copy.deepcopy(list(chank_list[-3:]))
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


    print("len of chank_list: ", len(chank_list))
                            
    # 各行の列数を出力
    for i, row in enumerate(chank_list):
        print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
            

    print("adjust_chank_list")
    # 各行の列数を出力
    for i, row in enumerate(adjust_chank_list):
        print(f"Row {i+1} length: {len(row)}")





def func_analysis(adjust_chank_list, lock):
    check_row = 3
    current_row = 0   
    while True:
        with lock:
            current_row = len(adjust_chank_list)
        if current_row == check_row:
            #分析処理をここに書く
            check_row = check_row + 1
            print("分析処理", check_row)









# すべての列を100個のデータに揃える処理
# def adjust_data_to_1000(data):
#     # 行数を取得
#     row = len(data)
#     if row == 0:
#         print("No data available")
#         return data
#     elif row == 1:
#         # 行数が1行の場合でも、データを1000に揃える処理
#         if len(data[0]) < 1000:
#             print("len(data[0]) < 1000")
#             needed_length = 1000 - len(data[0])
#             # データが不足している場合、0で埋める
#             data[0] = [0] * needed_length + data[0]
#         elif len(data[0]) > 1000:
#             data[0] = data[0][:1000]  # 最初の1000個に切り捨て
#         return data[0]
#     else:
#         # 最後の行の要素数が1000未満の場合
#         if len(data[row - 1]) < 1000:
#             print("len(data[row - 1]) < 1000")
#             # 最後の行の要素数が1000未満の場合、最後の行の要素数を1000個にする
#             # data[row - 1] = data[row - 2][len(data[row - 1]) : 1000] + data[row - 1]
#             # return data[row - 1]
#             needed_length = 1000 - len(data[row - 1])
#             data[row - 1] = data[row - 2][-needed_length:] + data[row - 1]
#             return data[row - 1]
#         elif len(data[row - 1]) > 1000:
#             print("len(data[row - 1]) > 1000")
#             # 最後の行の要素数が1000より大きい場合、最後の行の要素数を1000個にする
#             remainder = len(data[row - 1]) - 1000
#             data[row - 1] = data[row - 1][int(remainder/2):int(remainder/2) + 1000]
#             return data[row - 1]
#         else:
#             return data[row - 1]



def adjust_data_to_size(data, target_size):
    # 行数を取得
    row = len(data)
    if row == 0:
        print("No data available")
        return data
    elif row == 1:
        # 行数が1行の場合でも、データをtarget_sizeに揃える処理
        if len(data[0]) < target_size:
            print(f"len(data[0]) < {target_size}")
            needed_length = target_size - len(data[0])
            # データが不足している場合、0で埋める
            data[0] = [0] * needed_length + data[0]
        elif len(data[0]) > target_size:
            data[0] = data[0][:target_size]  # 最初のtarget_size個に切り捨て
        return data[0]
    else:
        # 最後の行の要素数がtarget_size未満の場合
        if len(data[row - 1]) < target_size:
            print(f"len(data[row - 1]) < {target_size}")
            # 最後の行の要素数がtarget_size未満の場合、最後の行の要素数をtarget_size個にする
            needed_length = target_size - len(data[row - 1])
            data[row - 1] = data[row - 2][-needed_length:] + data[row - 1]
            return data[row - 1]
        elif len(data[row - 1]) > target_size:
            print(f"len(data[row - 1]) > {target_size}")
            # 最後の行の要素数がtarget_sizeより大きい場合、最後の行の要素数をtarget_size個にする
            remainder = len(data[row - 1]) - target_size
            data[row - 1] = data[row - 1][int(remainder/2):int(remainder/2) + target_size]
            return data[row - 1]
        else:
            return data[row - 1]





# /**************main関数**********************************************/


def main():
        # 共有リストとロックを作成
    manager = multiprocessing.Manager()
    shared_receive_list = manager.list()  # 共有リスト
    # lock_receive_list = multiprocessing.Lock()  # ロック
    flag_blink = manager.Value('b', True)
    # lock_flag_blink = multiprocessing.Lock()
    chank_list = manager.list()
    # lock_chank_list = multiprocessing.Lock()
    receive_value = manager.list()  # 共有リスト
    # lock_receive_value = multiprocessing.Lock()
    clock_signal = manager.Value('b', False)
    # lock_clock_signal = multiprocessing.Lock()
    adjust_chank_list = manager.list()
    lock = multiprocessing.Lock()

        
    list_com()# COMポート一覧を表示
    # com = input_com()# COMポート接続の初期化
    com = "COM3"
    print(com)

    
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     # e.submit(func_1)
    #     e.submit(func_serial, com)
    #     # e.submit(func_visual)
    # 並列処理で実行するプロセスを定義
    process1 = multiprocessing.Process(target=func_serial, args=(com, shared_receive_list, receive_value, clock_signal, lock))
    process2 = multiprocessing.Process(target=func_chank, args=(receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, lock))
    process3 = multiprocessing.Process(target=func_visual, args=(flag_blink, lock))
    process4 = multiprocessing.Process(target=func_analysis, args=(adjust_chank_list, lock))

    # プロセスの開始
    process1.start()
    process2.start()
    process3.start()
    process4.start()

    # プロセスの終了を待つ
    process1.join()
    process2.join()
    process3.join()
    process4.join()
# /***********************************************************/


if __name__ == '__main__':
    main()