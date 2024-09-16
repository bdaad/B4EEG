import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
import numpy as np
from PIL import Image

# シンプルなシェーダー
vertex_shader_code = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_code = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D texture1;
void main()
{
    FragColor = texture(texture1, TexCoord);
}
"""

# シェーダーをコンパイルしてプログラムを作成
def create_shader_program():
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_code)
    glCompileShader(vertex_shader)

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_code)
    glCompileShader(fragment_shader)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

# 点滅する画像を描画するクラス
class BlinkingImage:
    def __init__(self, position, size, image_path, frequency, refresh_rate):
        self.position = position
        self.size = size
        self.frequency = frequency  # 点滅の周波数
        self.refresh_rate = refresh_rate  # 垂直同期のリフレッシュレート
        self.toggle = True  # 点滅の初期状態（ON/OFF）
        self.start_time = time.time()  # 開始時刻
        self.frames_per_blink = refresh_rate / (2 * frequency)  # 点滅切り替えのフレーム数

        # テスト用に frames_per_blink を固定値に設定
        self.frames_per_blink = 30  # 30フレームごとに切り替わる

        self.frame_count = 0  # フレームカウンタ
        self.frame_count_not_reset = 0  # リセットなしフレームカウンタ

        # 画像のロードとテクスチャの設定
        self.texture_id = self.load_texture(image_path)

        # シェーダープログラムを作成
        self.shader_program = create_shader_program()

        # 頂点データを設定
        self.vao, self.vbo, self.ebo = self.create_quad()

    def load_texture(self, image_path):
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 画像を上下反転する
        image_data = np.array(image, np.uint8)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_id

    def create_quad(self):
        # 四角形の頂点データ (位置とテクスチャ座標)
        vertices = np.array([
            # 位置           テクスチャ座標
            -0.5, -0.5, 0.0,  0.0, 0.0,  # 左下
             0.5, -0.5, 0.0,  1.0, 0.0,  # 右下
             0.5,  0.5, 0.0,  1.0, 1.0,  # 右上
            -0.5,  0.5, 0.0,  0.0, 1.0   # 左上
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # 頂点の位置属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # テクスチャ座標属性
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao, vbo, ebo

    def draw_image(self):
        glUseProgram(self.shader_program)

        # テクスチャのバインド
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # 四角形を描画
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def update(self):
        # 点滅のロジック
        self.frame_count += 1
        self.frame_count_not_reset += 1

        # フレームカウントがframes_per_blinkに達したらtoggleを反転
        if self.frame_count >= self.frames_per_blink:
            self.toggle = not self.toggle
            self.frame_count = 0  # カウンタをリセット

        # 点滅がオンのときだけ画像を描画
        if self.toggle:
            self.draw_image()

        # テスト用のデバッグ出力
        print(f"frame_count_not: {self.frame_count_not_reset}, toggle: {self.toggle}")

# GLFW初期化とウィンドウ作成
def init_glfw(width, height, title):
    if not glfw.init():
        return None
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def main():
    # ウィンドウサイズ
    monitor_width = 800
    monitor_height = 800

    window = init_glfw(monitor_width, monitor_height, "Blinking Image Test")

    refresh_rate = 60

    # 画像の点滅設定
    blinking_image = BlinkingImage(position=(0.0, 0.0), size=(1.0, 1.0), image_path="./circle.png", frequency=5, refresh_rate=refresh_rate)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # 画像を更新して描画
        blinking_image.update()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    main()
