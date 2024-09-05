import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time  # 高精度な時間計測のために追加

# 点滅する円を描画するクラス
class BlinkingCircle:
    def __init__(self, position, size, color, display_time, frequency, start_on=True):
        self.position = position  # 円の位置 (x, y)
        self.size = size  # 円のサイズ（半径）
        self.color = color  # 円の色 (r, g, b)
        self.display_time = display_time  # 表示秒数
        self.frequency = frequency  # 点滅の周波数
        self.start_on = start_on  # 初期状態
        self.toggle = start_on  # 点滅の初期状態（ON/OFF）
        self.start_time = time.perf_counter()  # 高精度な開始時刻
        if frequency > 0:
            self.cycle_time = 1.0 / frequency  # 点滅の周期（秒単位）
        else:
            self.cycle_time = None  # 点滅なし（常時点灯）
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
        current_time = time.perf_counter()
        elapsed_time = current_time - self.start_time

        if self.display_time is not None and elapsed_time >= self.display_time:
            return False  # 表示秒数が経過したらFalseを返す

        # 周期に基づいた点滅の切り替え
        if self.cycle_time is not None:
            # 初期状態に基づいて、点滅のタイミングを変更
            toggle_state = (elapsed_time % self.cycle_time) < (self.cycle_time / 2)
            if self.start_on:
                self.toggle = toggle_state
            else:
                self.toggle = not toggle_state  # 初期状態がOFFの場合は反転

        # 色の設定
        if self.toggle or self.cycle_time is None:
            glColor3f(*self.color)
        else:
            glColor3f(0, 0, 0)  # 黒で円を非表示にする

        # 円を描画
        self.draw_circle()

        return True  # 表示継続中

# Pygameを初期化
pygame.init()

# 描画解像度を下げたウィンドウを作成（例えば、640x480の解像度に設定）
screen = pygame.display.set_mode((640, 480), pygame.DOUBLEBUF | pygame.OPENGL)

# 背景色を設定（黒）
glClearColor(0.0, 0.0, 0.0, 1.0)

# 画面のサイズを取得
width, height = pygame.display.get_surface().get_size()

# カメラの視野角や視点の設定
gluPerspective(45, (width / height), 0.1, 50.0)

# カメラを後方に移動して、シーン全体が見えるようにする
glTranslatef(0.0, 0.0, -5)

# BlinkingCircleクラスのインスタンスを作成
point_center = BlinkingCircle(position=(0.0, 0.0), size=0.03, color=(1.0, 1.0, 0.0), display_time=None, frequency=0, start_on=True)
circle1 = BlinkingCircle(position=(-2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=100, frequency=10, start_on=True)  # 左の円は点灯からスタート
circle2 = BlinkingCircle(position=(2.0, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=10, start_on=False)  # 右の円は消灯からスタート

# メインループのフラグを初期化
running = True

# メインループ
start_time = time.perf_counter()  # 高精度なタイマーの開始時間
frame_duration = 1.0 / 60  # フレーム間隔 (60FPS で描画する場合)

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

    # 高精度タイマーで次のフレームまで待機
    elapsed_time = time.perf_counter() - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)
    
    # フレームの開始時間をリセット
    start_time = time.perf_counter()

# Pygameを終了
pygame.quit()
