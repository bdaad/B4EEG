import glfw
from OpenGL.GL import *
import numpy as np
import time

# シェーダープログラムのソースコード
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);  // オレンジ色
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def main():
    # GLFWの初期化
    if not glfw.init():
        return

    # ウィンドウを作成
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(800, 600, "V-Sync Triangle with FPS", None, None)
    if not window:
        glfw.terminate()
        return

    # ウィンドウコンテキストを作成
    glfw.make_context_current(window)

    # V-Syncを有効にする
    glfw.swap_interval(1)

    # シェーダープログラムを作成
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

    # 三角形の頂点データ
    vertices = np.array([
        -0.5, -0.5, 0.0,  # 左下
         0.5, -0.5, 0.0,  # 右下
         0.0,  0.5, 0.0   # 上
    ], dtype=np.float32)

    # VAOとVBOを作成
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    # VAOにバインド
    glBindVertexArray(VAO)

    # VBOにバインドして頂点データを送信
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # 頂点属性を有効にして設定
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)

    # バッファのバインドを解除
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # メインループ
    previous_time = time.time()

    while not glfw.window_should_close(window):
        # 現在の時間を取得
        current_time = time.time()
        delta_time = current_time - previous_time
        if delta_time > 0:  # delta_timeが0より大きい場合のみ計算
            fps = 1.0 / delta_time
            print(f"FPS: {fps:.2f}")
        previous_time = current_time


        # 背景色をクリア
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # シェーダープログラムを使用
        glUseProgram(shader_program)

        # 三角形を描画
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # バッファを交換して描画
        glfw.swap_buffers(window)
        
        
        # イベントを処理
        glfw.poll_events()

    # 終了処理
    glDeleteVertexArrays(1, VAO)
    glDeleteBuffers(1, VBO)
    glfw.terminate()

if __name__ == "__main__":
    main()
