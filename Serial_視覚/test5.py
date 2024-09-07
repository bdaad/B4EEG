# 多分うまく動いている..


# グローバル変数として受信データを格納するリスト
received_data = []



# /**************Serial関連**********************************************/
import re
import serial
import serial.tools.list_ports
import time

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
def communicate_and_count(ser):
    global received_data  # グローバル変数を参照

    interval = 1.0 / 1000  # 1000Hz
    next_time = time.perf_counter()  # 高精度タイマーの現在時刻を取得
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    t = 1

    # while True:
    for i in range(10000):
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
                received_data.append(result.decode())  # グローバル配列に追加

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

# 点滅する円を描画するクラス
class BlinkingCircle:
    def __init__(self, position, size, color, display_time, frequency, refresh_rate, start_on=True):
        self.position = position  # 円の位置 (x, y)
        self.size = size  # 円のサイズ（半径）
        self.color = color  # 円の色 (r, g, b)
        self.display_time = display_time  # 表示秒数
        self.frequency = frequency  # 点滅の周波数
        self.refresh_rate = refresh_rate  # 垂直同期のリフレッシュレート（例: 60Hz）
        self.toggle = start_on  # 点滅の初期状態（ON/OFF）
        self.start_time = time.time()  # 開始時刻
        if frequency > 0:
            self.frames_per_blink = refresh_rate / (2 * frequency)  # 1回の点滅（オンまたはオフ）に必要なフレーム数
        else:
            self.frames_per_blink = None  # 点滅なし（常時点灯）
        self.frame_count = 0  # フレームカウンタ

    def draw_circle(self):
        # 塗りつぶされた円を描画するための関数
        glBegin(GL_POLYGON)  # 多角形として描画し、内部を塗りつぶす
        num_segments = 100
        for i in range(num_segments):
            theta = 2.0 * math.pi * i / num_segments  # 角度を計算
            x = self.size * math.cos(theta) + self.position[0]  # x座標
            y = self.size * math.sin(theta) + self.position[1]  # y座標
            glVertex2f(x, y)  # 頂点を設定
        glEnd()

    def update(self):
        # 点滅のロジック
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # 経過時間を秒に変換
        if self.display_time is not None and elapsed_time >= self.display_time:
            return False  # 表示秒数が経過したらFalseを返す

        if self.frames_per_blink is not None:
            # フレームカウンタを更新
            self.frame_count += 1
            if self.frame_count >= self.frames_per_blink:
                self.toggle = not self.toggle  # フラグを反転させて点滅を切り替える
                self.frame_count = 0  # カウンタをリセット

        # 色の設定
        if self.toggle or self.frames_per_blink is None:
            glColor3f(*self.color)
        else:
            glColor3f(0, 0, 0)  # 黒で円を非表示にする

        # 円を描画
        self.draw_circle()

        return True  # 表示継続中

def enable_vsync(enable=True):
    if enable:
        glfw.swap_interval(1)  # V-Syncを有効にする
    else:
        glfw.swap_interval(0)  # V-Syncを無効にする

# /***********************************************************/



# /**************並列処理関連**********************************************/

from concurrent.futures import ProcessPoolExecutor
import time


def func_1():

    while True:
        time.sleep(2)
        print('func_1')

    

def func_visual():
    if not glfw.init():
        return

    # プライマリモニターの解像度を取得
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    monitor_width = video_mode.size.width
    monitor_height = video_mode.size.height
    print(f"Monitor Resolution: {monitor_width}x{monitor_height}")

    # フルスクリーンウィンドウを作成
    window = glfw.create_window(monitor_width, monitor_height, "V-Sync Blinking Circles with FPS", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # refresh_rate = video_mode.refresh_rate  # 垂直同期のリフレッシュレート
    refresh_rate = 60
    enable_vsync(True)  # V-Syncを有効にする

    # 画面サイズを取得
    width, height = glfw.get_framebuffer_size(window)
    
    # OpenGLの視点とカメラ設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    
    # カメラの位置を設定（z方向に-5移動してシーンを表示）
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)

    # 指定された円の設定
    point_center = BlinkingCircle(position=(0.0, 0.0), size=0.03, color=(1.0, 1.0, 0.0), display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True)
    circle1 = BlinkingCircle(position=(-2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=5, frequency=10, refresh_rate=refresh_rate, start_on=True)
    circle2 = BlinkingCircle(position=(2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False)

    circles = [point_center, circle1, circle2]  # 全ての円をリストに追加

    previous_time = time.time()
    frame_count = 0
    fullscreen = True  # 現在の状態を管理

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # 各円を更新して描画
        for circle in circles:
            if not circle.update():
                circles.remove(circle)  # 表示秒数が経過したらリストから削除

        # フレームカウンタを更新
        frame_count += 1
        current_time = time.time()

        # 1秒ごとにFPSを計算して出力
        if current_time - previous_time >= 1.0:
            fps = frame_count / (current_time - previous_time)
            print(f"FPS: {fps:.2f}")
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



def func_serial(com):
    global received_data  # グローバル変数を参照
    ser = serial.Serial(com, bitRate, timeout=None)
    communicate_and_count(ser)
    print(received_data)
    print(len(received_data))



# /**************main関数**********************************************/
import multiprocessing

def main():
        
    list_com()# COMポート一覧を表示
    com = input_com()# COMポート接続の初期化
    print(com)

    
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     # e.submit(func_1)
    #     e.submit(func_serial, com)
    #     # e.submit(func_visual)
    # 並列処理で実行するプロセスを定義
    process1 = multiprocessing.Process(target=func_serial, args=(com,))
    # process2 = multiprocessing.Process(target=func_1)
    process3 = multiprocessing.Process(target=func_visual)

    # プロセスの開始
    process1.start()
    # process2.start()
    process3.start()

    # プロセスの終了を待つ
    process1.join()
    # process2.join()
    process3.join()
# /***********************************************************/


if __name__ == '__main__':
    main()