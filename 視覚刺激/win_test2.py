# import glfw
# from OpenGL.GL import *
# import time

# def enable_vsync(enable=True):
#     if enable:
#         glfw.swap_interval(1)  # V-Syncを有効にする
#     else:
#         glfw.swap_interval(0)  # V-Syncを無効にする

# def main():
#     if not glfw.init():
#         return

#     window = glfw.create_window(800, 600, "V-Sync FPS Example", None, None)
#     if not window:
#         glfw.terminate()
#         return

#     glfw.make_context_current(window)
    
#     # V-Syncを有効にする場合
#     enable_vsync(True)

#     previous_time = time.time()
    
#     while not glfw.window_should_close(window):
#         glClear(GL_COLOR_BUFFER_BIT)

#         # OpenGL描画処理をここに追加

#         # バッファをスワップしてフレームの描画を完了
#         glfw.swap_buffers(window)
#         glfw.poll_events()

#         # フレーム描画完了後にフレームレートを計算
#         current_time = time.time()
#         delta_time = current_time - previous_time
#         fps = 1.0 / delta_time if delta_time > 0 else 0
#         print(f"FPS: {fps:.2f}")
        
#         previous_time = current_time

#     glfw.destroy_window(window)
#     glfw.terminate()

# if __name__ == "__main__":
#     main()




import glfw
from OpenGL.GL import *
import time

def enable_vsync(enable=True):
    if enable:
        glfw.swap_interval(1)  # V-Syncを有効にする
    else:
        glfw.swap_interval(0)  # V-Syncを無効にする

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "V-Sync FPS Example", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    # V-Syncを有効にする場合
    enable_vsync(True)

    previous_time = time.time()
    frame_count = 0
    
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # OpenGL描画処理をここに追加

        # バッファをスワップしてフレームの描画を完了
        glfw.swap_buffers(window)
        glfw.poll_events()

        frame_count += 1
        current_time = time.time()

        # 1秒ごとにFPSを計算して出力
        if current_time - previous_time >= 1.0:
            fps = frame_count / (current_time - previous_time)
            print(f"FPS: {fps:.2f}")
            previous_time = current_time
            frame_count = 0

    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
