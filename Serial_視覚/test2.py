# 2024/09/03 16:22:22 ~


# /******************シリアル通信関連関数******************/
# arduinoでシリアル通信テスト
import re
import serial
import serial.tools.list_ports
import time

read_data = []# 受信したデータを格納するためのリスト（行列）
bitRate = 115200 # 通信速度

# COMポート一覧を表示
def list_com():
    ports = serial.tools.list_ports.comports()
    print("利用可能なCOMポート:")
    for port in ports:
        print(port.device)

# COMポートを入力
def input_com():
    com = input("COMポートを入力してください(例: COM3): ")
    return com

# /******************************************************/


# /******************視覚刺激関連関数******************/
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
import platform
import math

# V-Syncの有効化/無効化を切り替える関数
def set_vsync(enabled):
    system = platform.system()
    if system == "Windows":
        wglSwapIntervalEXT = ctypes.windll.opengl32.wglSwapIntervalEXT
        wglSwapIntervalEXT.restype = ctypes.c_int
        wglSwapIntervalEXT.argtypes = [ctypes.c_int]
        wglSwapIntervalEXT(int(enabled))
    elif system == "Darwin":
        from Cocoa import NSOpenGLContext
        context = NSOpenGLContext.currentContext()
        interval = 1 if enabled else 0
        context.setValues_forParameter_([interval], 222)
    else:
        print("V-Sync control not implemented for this platform")

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
        self.start_time = pygame.time.get_ticks()  # 開始時刻
        if frequency > 0:
            self.frames_per_blink = refresh_rate / (2 * frequency)  # 1回の点滅（オンまたはオフ）に必要なフレーム数
        else:
            self.frames_per_blink = None  # 点滅なし（常時点灯）
        self.frame_count = 0  # フレームカウンタ
        self.update_count = 0  # updateメソッドの呼び出し回数をカウントするための変数

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
        self.update_count += 1  # updateメソッドが呼び出されるたびにカウントを増やす
        print(self.update_count)  # カウントを表示
        # 点滅のロジック
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - self.start_time) / 1000.0  # 経過時間を秒に変換
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

# /******************************************************/






#　/********************並列処理関連関数********************/

from concurrent.futures import ProcessPoolExecutor
import time
# マルチプロセス -> 並列処理
# python3.2以降 concurrent.futuresを使う

def func_serial(ser):
    # ser = serial.Serial(COM, bitRate, timeout=None)
    while True:
    # for _ in range(100):
        result = ser.readline() # 改行コードまで読み込む
        if result:
            result = re.sub(rb'\r\n$', b'', result) # 改行コードを削除
            decoded_result = result.decode()  # バイト列を文字列に変換
            # カンマで分割し、それぞれの要素を整数に変換
            int_values = [int(x) for x in decoded_result.split(',')]
            # print(int_values)  # 整数リストを表示
            read_data.append(int_values)  # 行列に追加


    ser.close()

def func_visual(point_center, circle1, circle2, running, screen, clock, refresh_rate):
    # メインループ
    while running:
        # イベントを処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # エスケープキーが押された場合、全画面表示を終了
                    screen = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)
                    gluPerspective(45, (800 / 600), 0.1, 50.0)
                    glTranslatef(0.0, 0.0, -5)

        # 画面をクリア（カラーバッファと深度バッファ）
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 円を更新し、描画を続けるかどうかを確認
        if not point_center.update():
            running = False  # 表示時間が過ぎたらループを終了

        if not circle1.update():
            running = False  # 表示時間が過ぎたらループを終了

        if not circle2.update():
            running = False  # 表示時間が過ぎたらループを終了

        # 描画バッファをフリップして、画面に反映
        pygame.display.flip()

        # フレームレートを制御
        clock.tick(refresh_rate)

    # Pygameを終了
    pygame.quit()

# /******************************************************/





# /******************メイン関数******************/
def main():
    
    # /*****************セットアップ関連*****************/
    list_com()# COMポート一覧を表示
    while True:
        try:
            com = input_com()
            ser = serial.Serial(com, bitRate, timeout=None)
            break
        except serial.SerialException:
            print("COMポートが開けませんでした。再度入力してください。")


    # Pygameを初期化
    pygame.init()
    # PygameでOpenGL表示用のウィンドウを全画面表示で作成
    screen = pygame.display.set_mode((0, 0), pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN)
    # 時間管理用のClockオブジェクトを作成
    clock = pygame.time.Clock()
    # 背景色を設定（黒）
    glClearColor(0.0, 0.0, 0.0, 1.0)
    # 画面のサイズを取得
    width, height = pygame.display.get_surface().get_size()
    # カメラの視野角や視点の設定
    gluPerspective(45, (width / height), 0.1, 50.0)
    # カメラを後方に移動して、シーン全体が見えるようにする
    glTranslatef(0.0, 0.0, -5)
    # V-Syncを有効化
    set_vsync(True)

    # 垂直同期のリフレッシュレートを指定（例: 60Hz）
    refresh_rate = 20

    # BlinkingCircleクラスのインスタンスを作成
    point_center = BlinkingCircle(position=(0.0, 0.0), size=0.03, color=(1.0, 1.0, 0.0), display_time=10, frequency=0, refresh_rate=refresh_rate, start_on=True)
    circle1 = BlinkingCircle(position=(-2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=True)
    circle2 = BlinkingCircle(position=(2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False)
    # メインループのフラグを初期化
    running = True

    # /******************************************************/



    # # 一秒待つ.
    # time.sleep(1)



    with ProcessPoolExecutor(max_workers=2) as e:
        e.submit(func_visual(point_center, circle1, circle2, running, screen, clock, refresh_rate))
        e.submit(func_serial(ser))
        

# /******************************************************/
        


# メイン関数を実行
if __name__ == '__main__':
    main()
    
    last_index = len(read_data) - 1 # 配列の最後の添え字を取得    
    print("最後の添え字は:", last_index) # 最後の添え字を出力










