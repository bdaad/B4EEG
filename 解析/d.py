import glfw
import time

# GLFWを初期化
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

# ウィンドウを作成
window = glfw.create_window(800, 600, "Frame Rate Measurement", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

# コンテキストを作成
glfw.make_context_current(window)

# V-Syncを有効にする（1で有効、0で無効）
glfw.swap_interval(1)

# 計測用の初期化
frame_count = 0
start_time = time.perf_counter()  # 高精度タイマーを使用
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
    current_time = time.perf_counter()

    # 1秒ごとにフレームレートを表示
    if current_time - last_time >= 10.0:
        elapsed_time = current_time - last_time
        frame_rate = frame_count / elapsed_time
        print(f"フレームレート: {frame_rate:.2f} FPS")

        # カウンタをリセット
        last_time = current_time
        frame_count = 0

# GLFWを終了
glfw.terminate()
