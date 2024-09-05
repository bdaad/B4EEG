import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
import platform

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

pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)
clock = pygame.time.Clock()

glClearColor(0.0, 0.0, 0.0, 1.0)
gluPerspective(45, (800/600), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

set_vsync(True)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(1, 3, 1, 1)
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glColor3f(0, 1, 0)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glColor3f(0, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glColor3f(1, 1, 0)
    glVertex3f(1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glColor3f(0, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glColor3f(1, 0, 1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, -1, -1)
    glEnd()

    pygame.display.flip()

    # フレームレートを計測するためだけにclock.tick()を使用
    clock.tick(60)  # ここでは引数を指定しないか、0を指定することでフレームレートを制限しない

    # フレームレートを表示
    fps = clock.get_fps()
    print(f"FPS: {fps:.2f}")

pygame.quit()
