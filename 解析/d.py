import glfw
import time

# GLFWを初期化
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

# ウィンドウを作成
window = glfw.create_window(800, 600, "Refresh Rate Measurement", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

# コンテキストを作成
glfw.make_context_current(window)

# 計測用の初期化
start_time = time.time()
frame_count = 0
last_time = start_time

# 計測開始
while not glfw.window_should_close(window):
    # 画面の内容を更新
    glfw.swap_buffers(window)

    # イベント処理
    glfw.poll_events()

    # フレーム数をカウント
    frame_count += 1

    # 現在の時間を取得
    current_time = time.time()

    # 1秒ごとにリフレッシュレートを表示
    if current_time - last_time >= 1.0:
        refresh_rate = frame_count / (current_time - last_time)
        print(f"推定リフレッシュレート: {refresh_rate:.2f} Hz")

        # カウンタをリセット
        last_time = current_time
        frame_count = 0

# GLFWを終了
glfw.terminate()
