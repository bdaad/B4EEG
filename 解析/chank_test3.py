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
def communicate_and_count(ser , received_list, lock_receive_list, receive_value, lock_receive_value, clock_signal, lock_clock_signal):

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

                with lock_receive_list:  # ロックを使って排他制御
                    received_list.append(receive_value)
                    # received_list.append(result.decode())  # 共有リストに追加
                with lock_receive_value:
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
        self.frame_count_not_reset = 0  # リセット無しフレームカウンタ

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
            self.frame_count_not_reset += 1
            if self.frame_count == self.frames_per_blink:
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

    

def func_visual(flag_blink, lock_flag_blink):
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
    circle1 = BlinkingCircle(position=(-2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True)
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
        
        #circleのself.frame_countを出力する
        # print("circle1.frame_count: ", circle1.frame_count_not_reset)
        # print("circle2.frame_count: ", circle2.frame_count)

        if circle1.frame_count_not_reset % refresh_rate == 0:
            with lock_flag_blink:
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



def func_serial(com, shared_receive_list, lock_receive_list, receive_value, lock_receive_value, clock_signal, lock_clock_signal):
    # global received_data  # グローバル変数を参照
    ser = serial.Serial(com, bitRate, timeout=None)
    communicate_and_count(ser, shared_receive_list, lock_receive_list, receive_value, lock_receive_value, clock_signal, lock_clock_signal)
    # print("shared_receive_list: ", shared_receive_list)
    # print("len of shared_receive_list: ", len(shared_receive_list))


import copy
def func_chank(receive_value, lock_receive_value, flag_blink, lock_flag_blink, chank_list, lock_chank_list, clock_signal, lock_clock_signal):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..

    flag_state = None
    chank_chank_list_1 = []
    chank_chank_list_2 = []
    adjust_chank_list = []

    po = 0

    while True:
        if po >= 20:
            break
        #計測の最初は、必ずflag_blink=Trueのときにデータを受け取る.
        if flag_state is None:
            with lock_flag_blink:
                print("flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
            
        else:
            # print("flag_blink: ", flag_blink.value)
            if flag_blink.value == True:
                # print("chank_chank_list_2", chank_chank_list_2)
                # print("chank_chank_list_1", chank_chank_list_1)
                if len(chank_chank_list_2) != 0:    
                    with lock_chank_list:
                        chank_list.append(chank_chank_list_2)
                    chank_chank_list_2 = []
                    po = po + 1
                    with lock_chank_list:
                        chank_list_copy = copy.deepcopy(list(chank_list))
                        # adjust_chank_list.append(adjust_data_to_1000(chank_list_copy))
                    # print("po: ", po)
                
                with lock_receive_value:
                    # print(type(receive_value))
                    # print("nowe", isinstance(receive_value, ListProxy))
                    # print("nowe", len(receive_value))
                    # print("nowe", receive_value)
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        clock_signal.value = False
                        # print( "ccl1", len(chank_chank_list_1))

            
            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:    
                    with lock_chank_list:
                        chank_list.append(chank_chank_list_1)
                    chank_chank_list_1 = []
                    po = po + 1
                    with lock_chank_list:
                        chank_list_copy = copy.deepcopy(list(chank_list))
                        # adjust_chank_list.append(adjust_data_to_1000(chank_list_copy))
                    # print("po: ", po)

                with lock_receive_value:
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


# すべての列を100個のデータに揃える処理
def adjust_data_to_1000(data):
    # 行数を取得
    row = len(data)
    if row == 0:
        print("No data available")
        return data
    elif row == 1:
        # 行数が1行の場合でも、データを1000に揃える処理
        if len(data[0]) < 1000:
            print("len(data[0]) < 1000")
            needed_length = 1000 - len(data[0])
            # データが不足している場合、0で埋める
            data[0] = [0] * needed_length + data[0]
        elif len(data[0]) > 1000:
            data[0] = data[0][:1000]  # 最初の1000個に切り捨て
        return data[0]
    else:
        # 最後の行の要素数が1000未満の場合
        if len(data[row - 1]) < 1000:
            print("len(data[row - 1]) < 1000")
            # 最後の行の要素数が1000未満の場合、最後の行の要素数を1000個にする
            # data[row - 1] = data[row - 2][len(data[row - 1]) : 1000] + data[row - 1]
            # return data[row - 1]
            needed_length = 1000 - len(data[row - 1])
            data[row - 1] = data[row - 2][-needed_length:] + data[row - 1]
            return data[row - 1]
        elif len(data[row - 1]) > 1000:
            print("len(data[row - 1]) > 1000")
            # 最後の行の要素数が1000より大きい場合、最後の行の要素数を1000個にする
            remainder = len(data[row - 1]) - 1000
            data[row - 1] = data[row - 1][int(remainder/2):int(remainder/2) + 1000]
            return data[row - 1]
        else:
            return data[row - 1]



# /**************main関数**********************************************/


def main():
        # 共有リストとロックを作成
    manager = multiprocessing.Manager()
    shared_receive_list = manager.list()  # 共有リスト
    lock_receive_list = multiprocessing.Lock()  # ロック
    flag_blink = manager.Value('b', True)
    lock_flag_blink = multiprocessing.Lock()
    chank_list = manager.list()
    lock_chank_list = multiprocessing.Lock()
    receive_value = manager.list()  # 共有リスト
    lock_receive_value = multiprocessing.Lock()
    clock_signal = manager.Value('b', False)
    lock_clock_signal = multiprocessing.Lock()

        
    list_com()# COMポート一覧を表示
    # com = input_com()# COMポート接続の初期化
    com = "COM7"
    print(com)

    
    # with ProcessPoolExecutor(max_workers=2) as e:
    #     # e.submit(func_1)
    #     e.submit(func_serial, com)
    #     # e.submit(func_visual)
    # 並列処理で実行するプロセスを定義
    process1 = multiprocessing.Process(target=func_serial, args=(com, shared_receive_list, lock_receive_list, receive_value, lock_receive_value, clock_signal, lock_clock_signal))
    process2 = multiprocessing.Process(target=func_chank, args=(receive_value, lock_receive_value, flag_blink, lock_flag_blink, chank_list, lock_chank_list, clock_signal, lock_clock_signal))
    process3 = multiprocessing.Process(target=func_visual, args=(flag_blink, lock_flag_blink))

    # プロセスの開始
    process1.start()
    process2.start()
    process3.start()

    # プロセスの終了を待つ
    process1.join()
    process2.join()
    process3.join()
# /***********************************************************/


if __name__ == '__main__':
    main()