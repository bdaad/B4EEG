import glfw

# GLFWを初期化
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

# デフォルトモニターを取得
monitor = glfw.get_primary_monitor()

# モニターのビデオモードを取得
video_mode = glfw.get_video_mode(monitor)

# リフレッシュレートを取得
refresh_rate = video_mode.refresh_rate
print(f"リフレッシュレート: {refresh_rate} Hz")

# GLFWを終了
glfw.terminate()
