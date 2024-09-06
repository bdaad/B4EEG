import glfw
from OpenGL.GL import *
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

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "V-Sync Blinking Circles with FPS", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    refresh_rate = 60  # 垂直同期のリフレッシュレート
    enable_vsync(True)  # V-Syncを有効にする

    # 指定された円の設定
    point_center = BlinkingCircle(position=(0.0, 0.0), size=0.03, color=(1.0, 1.0, 0.0), display_time=None, frequency=0, refresh_rate=refresh_rate, start_on=True)
    circle1 = BlinkingCircle(position=(-0.5, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=20, refresh_rate=refresh_rate, start_on=True)
    circle2 = BlinkingCircle(position=(0.5, 0.0), size=0.2, color=(1.0, 1.0, 1.0), display_time=None, frequency=20, refresh_rate=refresh_rate, start_on=False)

    circles = [point_center, circle1, circle2]  # 全ての円をリストに追加

    previous_time = time.time()
    frame_count = 0

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

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
