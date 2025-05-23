import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
import platform
import math
import time

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
        self.start_time = time.perf_counter()  # 高精度な開始時刻
        if frequency > 0:
            self.frames_per_blink = refresh_rate / (2 * frequency)  # 1回の点滅（オンまたはオフ）に必要なフレーム数
        else:
            self.frames_per_blink = None  # 点滅なし（常時点灯）
        self.frame_count = 0  # フレームカウンタ
        self.update_count = 0  # updateメソッドの呼び出し回数をカウントするための変数
        self.previous_time = self.start_time  # 前回のフレーム更新時間を保存

    def draw_circle(self):
        # 塗りつぶされた円を描画するための関数
        glBegin(GL_POLYGON)  # 多角形として描画し、内部を塗りつぶす
        num_segments = 50  # 描画負荷軽減のためにセグメント数を減少
        for i in range(num_segments):
            theta = 2.0 * math.pi * i / num_segments  # 角度を計算
            x = self.size * math.cos(theta) + self.position[0]  # x座標
            y = self.size * math.sin(theta) + self.position[1]  # y座標
            glVertex2f(x, y)  # 頂点を設定
        glEnd()

    def update(self):
        self.update_count += 1

        # フレームごとのリフレッシュレートを計算して出力
        current_time = time.perf_counter()
        elapsed_time = current_time - self.previous_time
        frame_refresh_rate = 1.0 / elapsed_time if elapsed_time > 0 else 0
        print(f"Frame {self.update_count}: Refresh Rate = {frame_refresh_rate:.2f} Hz")
        self.previous_time = current_time

        # 経過時間に基づいた点滅ロジック（時間ベースの補正）
        total_elapsed_time = current_time - self.start_time
        if self.display_time is not None and total_elapsed_time >= self.display_time:
            return False  # 表示秒数が経過したらFalseを返す

        if self.frames_per_blink is not None:
            # フレームカウンタを更新
            self.frame_count += 1

            # 時間ベースでの補正
            cycle_time = 1.0 / self.frequency if self.frequency > 0 else None
            if cycle_time:
                time_based_toggle = (total_elapsed_time % cycle_time) < (cycle_time / 2)
                if time_based_toggle != self.toggle:
                    self.toggle = time_based_toggle
                    self.frame_count = 0  # カウンタをリセット

            # フレームベースでの点滅制御
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

# Pygameを初期化
pygame.init()

# 描画解像度を下げたウィンドウを作成
screen = pygame.display.set_mode((640, 480), pygame.DOUBLEBUF | pygame.OPENGL)
# PygameでOpenGL表示用のウィンドウをシングルバッファリングで作成
# screen = pygame.display.set_mode((640, 480), pygame.OPENGL)  # DOUBLEBUF を削除


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
refresh_rate =  60

# BlinkingCircleクラスのインスタンスを作成
point_center = BlinkingCircle(position=(0.0, 0.0), size=0.03, color=(1.0, 1.0, 0.0), display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True)
circle1 = BlinkingCircle(position=(-2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=5, frequency=10, refresh_rate=refresh_rate, start_on=True)
circle2 = BlinkingCircle(position=(2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, refresh_rate=refresh_rate, start_on=False)

# メインループのフラグを初期化
running = True

# メインループ
while running:
    # イベントを処理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    # 画面をクリア（カラーバッファと深度バッファ）
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 円を更新し、描画を続けるかどうかを確認
    if not point_center.update():
        running = False  # 表示時間が過ぎたらループを終了

    if not circle1.update():
        running = False  # 表示時間が過ぎたらループを終了

    if not circle2.update():
        running = False  # 表示時間が過ぎたらループを終了

    # 1回目のflipを行う
    pygame.display.flip()
    


    # フレームレートを制御
    clock.tick(refresh_rate)
    

# Pygameを終了
pygame.quit()
